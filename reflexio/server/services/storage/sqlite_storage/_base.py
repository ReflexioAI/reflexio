"""
Base class and module-level helpers for SQLite storage.

Supports hybrid search combining FTS5 (BM25) with embedding cosine similarity
via Reciprocal Rank Fusion (RRF). Falls back to FTS-only when no embeddings
are available.

"""

import functools
import json
import logging
import math
import re
import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentPlaybookSnapshot,
    AgentPlaybookUpdateEntry,
    AgentSuccessEvaluationResult,
    Citation,
    Interaction,
    PlaybookAggregationChangeLog,
    PlaybookStatus,
    ProfileChangeLog,
    ProfileTimeToLive,
    RegularVsShadow,
    Request,
    Status,
    ToolUsed,
    UserActionType,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    EMBEDDING_DIMENSIONS,
    APIKeyConfig,
    LLMConfig,
    SearchMode,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import (
    ModelRole,
    resolve_model_name,
)
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)
from reflexio.server.services.storage.error import (
    StorageError,
    require_non_empty_session_id,
)
from reflexio.server.services.storage.retention import RetentionTarget
from reflexio.server.services.storage.retention_mixin import (
    RETENTION_DELETE_CHUNK,
    RetentionMixin,
    chunked,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.site_var.site_var_manager import SiteVarManager

from ._stall_state import init_stall_state_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> str | None:
    """Serialize a Python object to a JSON string, or None if the object is None."""
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _json_loads(text: str | None) -> Any:
    """Deserialize a JSON string, returning None for None/empty input."""
    if not text:
        return None
    return json.loads(text)


_FTS5_OPERATORS = frozenset({"OR", "AND", "NOT"})
_FTS5_RESERVED = _FTS5_OPERATORS | {"NEAR"}
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _sanitize_fts_query(text: str) -> str:
    """Sanitize a query string for FTS5, defaulting to OR between tokens.

    Bare (unquoted) tokens preserve Porter stemming. Explicit OR/AND/NOT
    operators are passed through. A trailing ``*`` is appended to the last
    token for prefix matching.

    Args:
        text: Raw user query string (may contain FTS5 boolean operators like OR)

    Returns:
        FTS5-safe query string with stemming enabled and OR default
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return '""'

    has_explicit_operator = any(t in _FTS5_OPERATORS for t in tokens)

    parts: list[str] = []
    for t in tokens:
        if t in _FTS5_OPERATORS:
            if not parts or parts[-1] in _FTS5_OPERATORS:
                continue
            parts.append(t)
        elif t in _FTS5_RESERVED:
            continue
        else:
            if not has_explicit_operator and parts and parts[-1] not in _FTS5_OPERATORS:
                parts.append("OR")
            parts.append(t)

    if parts and parts[-1] in _FTS5_OPERATORS:
        parts.pop()
    if not parts:
        return '""'

    # Append prefix wildcard to last token for partial-word matching
    parts[-1] = parts[-1] + "*"
    return " ".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 for degenerate inputs.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _effective_search_mode(
    mode: SearchMode,
    query_embedding: list[float] | None,
    query: str | None,
) -> SearchMode:
    """Downgrade search mode when the required embedding is unavailable.

    Args:
        mode: Requested search mode.
        query_embedding: Pre-computed query embedding, or None.
        query: Original query text. The fallback warning is suppressed when
            this is falsy, since an empty-query HYBRID/VECTOR request has no
            semantic intent to lose.

    Returns:
        The effective SearchMode — falls back to FTS when HYBRID/VECTOR lacks an embedding.
    """
    if mode in (SearchMode.HYBRID, SearchMode.VECTOR) and not query_embedding:
        if query:
            logger.warning(
                "Search mode '%s' requested but no query embedding provided — falling back to FTS",
                mode,
            )
        return SearchMode.FTS
    return mode


def _vector_rank_rows(
    rows: Sequence[Any],
    query_embedding: list[float],
    match_count: int,
) -> list[Any]:
    """Rank rows by cosine similarity to the query embedding.

    Args:
        rows: Candidate rows with stored embeddings.
        query_embedding: The query's embedding vector.
        match_count: Number of results to return.

    Returns:
        Top ``match_count`` rows sorted by cosine similarity descending.
    """
    scored: list[tuple[Any, float]] = []
    for row in rows:
        raw_emb = row["embedding"] if "embedding" in row.keys() else None  # noqa: SIM118
        emb = _json_loads(raw_emb) if raw_emb else None
        if emb:
            sim = _cosine_similarity(query_embedding, emb)
            scored.append((row, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    # Diagnostic: log the full pre-cut score distribution so retrieval
    # misses are debuggable. Without this, the only signal callers see is
    # "K results returned" with no way to tell whether a relevant row
    # scored 0.39 (close, just below an upstream threshold filter) vs
    # 0.05 (semantic mismatch, no threshold tuning will save it). Top 10
    # is sufficient context; logs at INFO so it shows up in production
    # backend.log without an explicit debug flag. Cost is one log line
    # per vector search call (~200 bytes).
    if scored:
        top = [round(s, 3) for _, s in scored[:10]]
        logger.info(
            "vector_rank: candidates=%d match_count=%d top_scores=%s",
            len(scored),
            match_count,
            top,
        )
    return [row for row, _ in scored[:match_count]]


def _true_rrf_merge(
    fts_rows: Sequence[Any],
    vec_rows: Sequence[Any],
    id_column: str,
    match_count: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 1.0,
) -> list[Any]:
    """Merge independent FTS and vector result sets via Reciprocal Rank Fusion.

    Unlike ``_rrf_rerank`` (which re-ranks FTS results only), this function
    takes two independently-produced result lists and unions them so that
    documents appearing in *either* modality can surface.

    Args:
        fts_rows: Rows from an FTS query, in BM25-ranked order.
        vec_rows: Rows from a vector query, in cosine-similarity order.
        id_column: Column name used as primary key to deduplicate rows.
        match_count: Number of results to return.
        rrf_k: RRF smoothing constant (default 60).
        vector_weight: Weight for vector similarity contribution.
        fts_weight: Weight for FTS contribution.

    Returns:
        Top ``match_count`` rows sorted by combined RRF score.
    """
    if not fts_rows and not vec_rows:
        return []

    # Collect unique rows by ID (first-seen wins for the Row object)
    row_by_id: dict[str | int, Any] = {}
    for row in (*fts_rows, *vec_rows):
        rid = row[id_column]
        if rid not in row_by_id:
            row_by_id[rid] = row

    # Build rank maps (1-based); missing entries get a penalty rank
    fts_rank: dict[str | int, int] = {
        row[id_column]: i + 1 for i, row in enumerate(fts_rows)
    }
    vec_rank: dict[str | int, int] = {
        row[id_column]: i + 1 for i, row in enumerate(vec_rows)
    }
    fts_penalty = len(fts_rows) + 1
    vec_penalty = len(vec_rows) + 1

    scored: list[tuple[Any, float]] = []
    for rid, row in row_by_id.items():
        f_rank = fts_rank.get(rid, fts_penalty)
        v_rank = vec_rank.get(rid, vec_penalty)
        score = fts_weight / (rrf_k + f_rank) + vector_weight / (rrf_k + v_rank)
        scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:match_count]]


# Tombstone statuses: rows with these values are excluded from default reads.
# Tasks 5/9/10 create tombstones; this constant ensures they stay hidden unless
# explicitly requested via include_tombstones=True on by-id getters, or an
# explicit status_filter on list/count methods.
_TOMBSTONE_STATUS_VALUES = (Status.MERGED.value, Status.SUPERSEDED.value)


def _status_value(status: Status | None) -> str | None:
    """Convert a Status enum (or None) to its DB string value."""
    if status is None:
        return None
    if hasattr(status, "value"):
        return status.value
    return None


def _build_status_sql(
    status_filter: list[Status | None],
    col: str = "status",
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment for a list of status values.

    Args:
        status_filter: List of Status enum values (may include None for CURRENT)
        col: Column name to filter on

    Returns:
        Tuple of (SQL fragment, parameter list) ready for AND-chaining
    """
    has_none = False
    values: list[str] = []
    for s in status_filter:
        v = _status_value(s)
        if v is None:
            has_none = True
        else:
            values.append(v)

    if has_none and values:
        placeholders = ",".join("?" for _ in values)
        return f"({col} IS NULL OR {col} IN ({placeholders}))", values
    if has_none:
        return f"{col} IS NULL", []
    if values:
        placeholders = ",".join("?" for _ in values)
        return f"{col} IN ({placeholders})", values
    return "1=1", []


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _epoch_now() -> int:
    """Return current UTC Unix timestamp."""
    return int(datetime.now(UTC).timestamp())


def _iso_to_epoch(iso_str: str | None) -> int:
    """Convert an ISO datetime string to Unix timestamp."""
    if not iso_str:
        return _epoch_now()
    try:
        cleaned = iso_str.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp())
    except (ValueError, TypeError):
        return _epoch_now()


# Bounds that ``datetime.fromtimestamp(tz=UTC)`` can represent (year 1..9999).
# Callers pass sentinel "open" bounds — e.g. ``to_ts=10**12`` for "no upper
# limit" or ``0`` for "from the beginning" — which would otherwise overflow
# ``datetime.fromtimestamp`` with a ``ValueError``. Clamping to these bounds
# yields the same query semantics (the ISO string still sorts before/after every
# stored row) with a valid value.
_MAX_SAFE_EPOCH_TS = 253_402_300_799  # 9999-12-31T23:59:59Z
_MIN_SAFE_EPOCH_TS = 0  # 1970-01-01T00:00:00Z


def _epoch_to_iso(ts: int) -> str:
    """Convert a Unix timestamp (seconds) to an ISO 8601 string.

    Out-of-range sentinel bounds are clamped to the representable range so that
    callers passing "open" window bounds never trigger a ``ValueError``.
    """
    clamped = max(_MIN_SAFE_EPOCH_TS, min(ts, _MAX_SAFE_EPOCH_TS))
    return datetime.fromtimestamp(clamped, tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Row-to-model converters
# ---------------------------------------------------------------------------


def _row_to_profile(row: sqlite3.Row) -> UserProfile:
    d = dict(row)
    return UserProfile(
        profile_id=d["profile_id"],
        user_id=d["user_id"],
        content=d["content"],
        last_modified_timestamp=d["last_modified_timestamp"],
        generated_from_request_id=d["generated_from_request_id"],
        profile_time_to_live=ProfileTimeToLive(d["profile_time_to_live"]),
        expiration_timestamp=d["expiration_timestamp"],
        custom_features=_json_loads(d.get("custom_features")),
        source=d.get("source") or "",
        status=Status(d["status"]) if d.get("status") else None,
        extractor_names=_json_loads(d.get("extractor_names")),
        expanded_terms=d.get("expanded_terms"),
        source_span=d.get("source_span"),
        notes=d.get("notes"),
        reader_angle=d.get("reader_angle"),
        tags=_json_loads(d.get("tags")),
        merged_into=d.get("merged_into"),
        superseded_by=d.get("superseded_by"),
    )


def _row_to_interaction(row: sqlite3.Row) -> Interaction:
    d = dict(row)
    tools_used_raw = _json_loads(d.get("tools_used"))
    tools_used = (
        [ToolUsed(**t) for t in tools_used_raw if isinstance(t, dict)]
        if tools_used_raw and isinstance(tools_used_raw, list)
        else []
    )
    citations_raw = _json_loads(d.get("citations"))
    citations = (
        [Citation(**c) for c in citations_raw if isinstance(c, dict)]
        if citations_raw and isinstance(citations_raw, list)
        else []
    )
    return Interaction(
        interaction_id=d["interaction_id"],
        user_id=d["user_id"],
        content=d["content"],
        request_id=d["request_id"],
        created_at=_iso_to_epoch(d["created_at"]),
        role=d.get("role") or "User",
        user_action=UserActionType(d["user_action"]),
        user_action_description=d["user_action_description"],
        interacted_image_url=d["interacted_image_url"],
        shadow_content=d.get("shadow_content") or "",
        expert_content=d.get("expert_content") or "",
        tools_used=tools_used,
        citations=citations,
    )


def _row_to_request(row: sqlite3.Row) -> Request:
    d = dict(row)
    return Request(
        request_id=d["request_id"],
        user_id=d["user_id"],
        created_at=_iso_to_epoch(d["created_at"]),
        source=d.get("source") or "",
        agent_version=d.get("agent_version") or "",
        session_id=require_non_empty_session_id(d.get("session_id")),
        evaluation_only=bool(d.get("evaluation_only", 0)),
    )


def _row_to_user_playbook(
    row: sqlite3.Row, include_embedding: bool = False
) -> UserPlaybook:
    d = dict(row)
    embedding: list[float] = []
    if include_embedding and d.get("embedding"):
        raw_emb = _json_loads(d["embedding"])
        if isinstance(raw_emb, list):
            embedding = [float(x) for x in raw_emb]
    return UserPlaybook(
        user_playbook_id=d["user_playbook_id"],
        user_id=d.get("user_id"),
        playbook_name=d["playbook_name"],
        created_at=_iso_to_epoch(d["created_at"]),
        request_id=d["request_id"],
        agent_version=d["agent_version"],
        content=d["content"],
        trigger=d.get("trigger"),
        rationale=d.get("rationale"),
        blocking_issue=BlockingIssue(**json.loads(d["blocking_issue"]))
        if d.get("blocking_issue")
        else None,
        status=Status(d["status"]) if d.get("status") else None,
        source=d.get("source"),
        source_interaction_ids=_json_loads(d.get("source_interaction_ids")) or [],
        tags=_json_loads(d.get("tags")),
        embedding=embedding,
        expanded_terms=d.get("expanded_terms"),
        source_span=d.get("source_span"),
        notes=d.get("notes"),
        reader_angle=d.get("reader_angle"),
        merged_into=d.get("merged_into"),
        superseded_by=d.get("superseded_by"),
    )


def _row_to_agent_playbook(row: sqlite3.Row) -> AgentPlaybook:
    d = dict(row)
    return AgentPlaybook(
        agent_playbook_id=d["agent_playbook_id"],
        playbook_name=d["playbook_name"],
        created_at=_iso_to_epoch(d["created_at"]),
        agent_version=d["agent_version"],
        content=d["content"],
        trigger=d.get("trigger"),
        rationale=d.get("rationale"),
        blocking_issue=BlockingIssue(**json.loads(d["blocking_issue"]))
        if d.get("blocking_issue")
        else None,
        playbook_status=PlaybookStatus(d["playbook_status"])
        if d.get("playbook_status")
        else PlaybookStatus.PENDING,
        playbook_metadata=d.get("playbook_metadata") or "",
        tags=_json_loads(d.get("tags")),
        embedding=[],
        status=Status(d["status"]) if d.get("status") else None,
        expanded_terms=d.get("expanded_terms"),
        merged_into=d.get("merged_into"),
        superseded_by=d.get("superseded_by"),
    )


def _row_to_eval_result(row: sqlite3.Row) -> AgentSuccessEvaluationResult:
    d = dict(row)
    return AgentSuccessEvaluationResult(
        result_id=d["result_id"],
        session_id=d["session_id"],
        agent_version=d["agent_version"],
        evaluation_name=d.get("evaluation_name"),
        is_success=bool(d["is_success"]),
        failure_type=d.get("failure_type"),
        failure_reason=d.get("failure_reason"),
        created_at=_iso_to_epoch(d["created_at"]),
        regular_vs_shadow=(
            RegularVsShadow(d["regular_vs_shadow"])
            if d.get("regular_vs_shadow")
            else None
        ),
        number_of_correction_per_session=d.get("number_of_correction_per_session") or 0,
        user_turns_to_resolution=d.get("user_turns_to_resolution"),
        is_escalated=bool(d.get("is_escalated", False)),
        embedding=[],
    )


def _row_to_profile_change_log(row: sqlite3.Row) -> ProfileChangeLog:
    d = dict(row)
    return ProfileChangeLog(
        id=d["id"],
        user_id=d["user_id"],
        request_id=d["request_id"],
        created_at=d["created_at"],
        added_profiles=[
            UserProfile(**p) for p in (_json_loads(d["added_profiles"]) or [])
        ],
        removed_profiles=[
            UserProfile(**p) for p in (_json_loads(d["removed_profiles"]) or [])
        ],
        mentioned_profiles=[
            UserProfile(**p) for p in (_json_loads(d["mentioned_profiles"]) or [])
        ],
    )


def _row_to_playbook_aggregation_change_log(
    row: sqlite3.Row,
) -> PlaybookAggregationChangeLog:
    d = dict(row)
    return PlaybookAggregationChangeLog(
        id=d["id"],
        created_at=d["created_at"],
        playbook_name=d["playbook_name"],
        agent_version=d["agent_version"],
        run_mode=d["run_mode"],
        added_agent_playbooks=[
            AgentPlaybookSnapshot(**fb)
            for fb in (_json_loads(d.get("added_playbooks")) or [])
        ],
        removed_agent_playbooks=[
            AgentPlaybookSnapshot(**fb)
            for fb in (_json_loads(d.get("removed_playbooks")) or [])
        ],
        updated_agent_playbooks=[
            AgentPlaybookUpdateEntry(
                before=AgentPlaybookSnapshot(**entry["before"]),
                after=AgentPlaybookSnapshot(**entry["after"]),
            )
            for entry in (_json_loads(d.get("updated_playbooks")) or [])
        ],
    )


# ---------------------------------------------------------------------------
# SQLiteStorageBase
# ---------------------------------------------------------------------------


class SQLiteStorageBase(RetentionMixin, BaseStorage):
    """SQLite-backed storage base class for local/self-hosted deployments."""

    supports_embedding: ClassVar[bool] = True

    @staticmethod
    def handle_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except StorageError:
                raise
            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                logger.error(
                    "Error in %s: %s\nStack trace:\n%s",
                    func.__name__,
                    str(e),
                    stack_trace,
                )
                raise StorageError(message=f"{e}\nStack trace:\n{stack_trace}") from e

        return wrapper

    def __init__(
        self,
        org_id: str,
        db_path: str | None = None,
        api_key_config: APIKeyConfig | None = None,
        llm_config: LLMConfig | None = None,
        enable_document_expansion: bool = False,
    ) -> None:
        super().__init__(org_id)
        self.api_key_config = api_key_config
        self._enable_document_expansion = enable_document_expansion

        # Resolve db_path: explicit arg > LOCAL_STORAGE_PATH env var > ~/.reflexio/data/
        if db_path is None:
            from reflexio.server import LOCAL_STORAGE_PATH

            db_path = str(Path(LOCAL_STORAGE_PATH) / "reflexio.db")

        self.db_path = db_path
        self._lock = threading.RLock()

        logger.info("SQLite Storage for org %s using db_path: %s", org_id, db_path)

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        # LLM client for embeddings
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}

        self.embedding_model_name = resolve_model_name(
            ModelRole.EMBEDDING,
            site_var_value=site_var.get("embedding_model_name"),
            config_override=llm_config.embedding_model_name if llm_config else None,
            api_key_config=self.api_key_config,
        )
        self.embedding_dimensions = EMBEDDING_DIMENSIONS

        litellm_config = LiteLLMConfig(
            model=self.embedding_model_name,
            temperature=0.0,
            api_key_config=self.api_key_config,
        )
        self.llm_client = LiteLLMClient(litellm_config)

        # Optionally load sqlite-vec for native KNN vector search
        self._has_sqlite_vec = self._try_load_sqlite_vec()

        # Create tables
        self.migrate()

    # ------------------------------------------------------------------
    # DDL / migration
    # ------------------------------------------------------------------

    def migrate(self) -> bool:
        self._migrate_feedback_schema()
        self._migrate_interactions_schema()
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(_DDL)
            self.conn.commit()
        if self._has_sqlite_vec:
            self._create_vec_tables()
            self._migrate_vec_tables()
        # Run after DDL so tables exist on fresh databases
        self._migrate_agent_runs_schema()
        self._migrate_pending_tool_calls_schema()
        self._migrate_expanded_terms()
        self._migrate_tags()
        self._migrate_agentic_signals()
        self._migrate_agent_playbook_source_windows()
        self._migrate_request_evaluation_only()
        self._migrate_request_session_id_required()
        self._migrate_shadow_comparison_verdicts()
        self._migrate_user_playbook_polarity()
        self._migrate_lineage()
        self._migrate_lineage_event_table()
        init_stall_state_table(self.conn)
        return True

    # -- Retention hooks (see RetentionMixin) --

    @handle_exceptions
    def _retention_table_exists(self, table_name: str) -> bool:
        row = self._fetchone(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        )
        return row is not None

    @handle_exceptions
    def _retention_count_rows(self, target: RetentionTarget) -> int:
        row = self._fetchone(f"SELECT COUNT(*) as cnt FROM {target.table_name}")  # noqa: S608
        return int(row["cnt"]) if row else 0

    @handle_exceptions
    def _retention_select_oldest_keys(
        self, target: RetentionTarget, count: int
    ) -> list[tuple[Any, ...]]:
        id_sql = ", ".join(target.id_columns)
        tiebreak_sql = id_sql
        rows = self._fetchall(
            f"SELECT {id_sql} FROM {target.table_name} "  # noqa: S608
            f"ORDER BY {target.order_column} ASC, {tiebreak_sql} ASC LIMIT ?",
            (count,),
        )
        return [tuple(row[col] for col in target.id_columns) for row in rows]

    @handle_exceptions
    def _retention_perform_delete(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        # Wrap dependency + target deletes in a single critical section so
        # concurrent writers see either both or neither.
        with self._lock:
            self._retention_delete_dependencies(target, keys)
            self._retention_delete_target_rows(target, keys)
            self.conn.commit()

    def _retention_delete_dependencies(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        ids = [key[0] for key in keys]
        target_name = target.name
        if target_name == "requests":
            self._delete_interactions_for_request_ids([str(v) for v in ids])
        elif target_name == "interactions":
            self._delete_interaction_search_rows([int(v) for v in ids])
        elif target_name == "profiles":
            self._delete_profile_search_rows([str(v) for v in ids])
        elif target_name == "user_playbooks":
            self._delete_source_windows_for_user_playbook_ids([int(v) for v in ids])
            self._delete_playbook_search_rows("user", [int(v) for v in ids])
        elif target_name == "agent_playbooks":
            self._delete_source_windows_for_agent_playbook_ids([int(v) for v in ids])
            self._delete_playbook_search_rows("agent", [int(v) for v in ids])
        elif target_name == "playbook_optimization_jobs":
            self._delete_optimizer_rows_for_job_ids([int(v) for v in ids])
        elif target_name == "playbook_optimization_candidates":
            self._delete_optimizer_evaluations_for_candidate_ids([int(v) for v in ids])

    def _retention_delete_target_rows(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        if len(target.id_columns) == 1:
            self._delete_in_chunks(
                target.table_name,
                target.id_columns[0],
                [key[0] for key in keys],
            )
            return
        # Composite-key delete: chunk by row to bound parameter count.
        params_per_key = len(target.id_columns)
        rows_per_chunk = max(1, RETENTION_DELETE_CHUNK // params_per_key)
        for chunk in chunked(keys, rows_per_chunk):
            where = " OR ".join(
                "("
                + " AND ".join(f"{column} = ?" for column in target.id_columns)
                + ")"
                for _ in chunk
            )
            params = [value for key in chunk for value in key]
            self.conn.execute(
                f"DELETE FROM {target.table_name} WHERE {where}",  # noqa: S608
                params,
            )

    # -- Chunked-delete primitives shared by the cascade helpers --

    def _delete_in_chunks(
        self, table_name: str, column_name: str, values: list[Any]
    ) -> None:
        """Chunked ``DELETE FROM table WHERE col IN (...)``.

        Chunking keeps parameter count under ``SQLITE_MAX_VARIABLE_NUMBER``
        on older sqlite builds (default 999) and avoids degenerate plans
        on very large IN lists.
        """
        if not values:
            return
        for chunk in chunked(values):
            placeholders = ",".join("?" for _ in chunk)
            self.conn.execute(
                f"DELETE FROM {table_name} WHERE {column_name} IN ({placeholders})",  # noqa: S608
                chunk,
            )

    def _select_in_chunks(self, sql_template: str, values: list[Any]) -> list[Any]:
        """Run ``sql_template`` (containing ``{placeholders}``) over chunks of
        ``values`` and aggregate the result rows."""
        results: list[Any] = []
        for chunk in chunked(values):
            placeholders = ",".join("?" for _ in chunk)
            stmt = sql_template.format(placeholders=placeholders)
            results.extend(self.conn.execute(stmt, chunk).fetchall())
        return results

    def _delete_interactions_for_request_ids(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        rows = self._select_in_chunks(
            "SELECT interaction_id FROM interactions WHERE request_id IN ({placeholders})",
            request_ids,
        )
        self._delete_interaction_search_rows(
            [int(row["interaction_id"]) for row in rows]
        )
        self._delete_in_chunks("interactions", "request_id", request_ids)

    def _delete_interaction_search_rows(self, interaction_ids: list[int]) -> None:
        if not interaction_ids:
            return
        self._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
        for interaction_id in interaction_ids:
            self._vec_delete("interactions_vec", interaction_id)

    def _delete_profile_search_rows(self, profile_ids: list[str]) -> None:
        if not profile_ids:
            return
        rows = self._select_in_chunks(
            "SELECT rowid, profile_id FROM profiles WHERE profile_id IN ({placeholders})",
            profile_ids,
        )
        for row in rows:
            self._fts_delete_profile(row["profile_id"])
            self._vec_delete("profiles_vec", row["rowid"])

    def _delete_playbook_search_rows(self, kind: str, ids: list[int]) -> None:
        if not ids:
            return
        self._delete_in_chunks(f"{kind}_playbooks_fts", "rowid", ids)
        for item_id in ids:
            self._vec_delete(f"{kind}_playbooks_vec", item_id)

    def _delete_source_windows_for_agent_playbook_ids(
        self, agent_playbook_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "agent_playbook_source_user_playbooks",
            "agent_playbook_id",
            agent_playbook_ids,
        )

    def _delete_source_windows_for_user_playbook_ids(
        self, user_playbook_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "agent_playbook_source_user_playbooks",
            "user_playbook_id",
            user_playbook_ids,
        )

    def _delete_optimizer_rows_for_job_ids(self, job_ids: list[int]) -> None:
        if not job_ids:
            return
        for table in (
            "playbook_optimization_evaluations",
            "playbook_optimization_events",
            "playbook_optimization_candidates",
        ):
            self._delete_in_chunks(table, "job_id", job_ids)

    def _delete_optimizer_evaluations_for_candidate_ids(
        self, candidate_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "playbook_optimization_evaluations",
            "candidate_id",
            candidate_ids,
        )

    def _try_load_sqlite_vec(self) -> bool:
        """Attempt to load the sqlite-vec extension for native KNN search.

        Returns:
            True if the extension was loaded successfully, False otherwise.
        """
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            # AttributeError covers Python builds without loadable-extension
            # support (common with pyenv/Homebrew on macOS) — the method
            # itself is absent rather than raising at runtime.
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded — native KNN search enabled")
            return True
        except (ImportError, OSError, AttributeError, sqlite3.OperationalError) as e:
            logger.info("sqlite-vec not available, using Python fallback: %s", e)
            return False

    def _create_vec_tables(self) -> None:
        """Create vec0 virtual tables for each entity that stores embeddings."""
        dim = self.embedding_dimensions
        vec_ddl = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS interactions_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS profiles_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_vec USING vec0(
                embedding float[{dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_vec USING vec0(
                embedding float[{dim}]
            );
        """
        with self._lock:
            self.conn.executescript(vec_ddl)
            self.conn.commit()

    def _migrate_vec_tables(self) -> None:
        """Backfill vec tables from existing embedding TEXT columns (idempotent)."""
        entity_map = [
            ("interactions", "interactions_vec", "interaction_id"),
            ("profiles", "profiles_vec", "profile_id"),
            ("user_playbooks", "user_playbooks_vec", "user_playbook_id"),
            ("agent_playbooks", "agent_playbooks_vec", "agent_playbook_id"),
        ]
        for main_table, vec_table, _id_col in entity_map:
            row = self._fetchone(f"SELECT COUNT(*) as cnt FROM {vec_table}")
            if row and row["cnt"] > 0:
                continue  # Already populated
            rows = self._fetchall(
                f"SELECT rowid AS rid, embedding FROM {main_table} WHERE embedding IS NOT NULL"
            )
            for r in rows:
                emb = _json_loads(r["embedding"])
                if emb:
                    self._vec_upsert(vec_table, r["rid"], emb)

    def _migrate_interactions_schema(self) -> None:
        """Add new columns to existing interactions table if missing."""
        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(interactions)")
            columns = {row[1] for row in cur.fetchall()}

        if not columns:
            return

        if "expert_content" not in columns:
            logger.info("Adding expert_content column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN expert_content TEXT NOT NULL DEFAULT ''"
                )
                self.conn.commit()

        if "citations" not in columns:
            logger.info("Adding citations column to interactions table.")
            with self._lock:
                self.conn.execute("ALTER TABLE interactions ADD COLUMN citations TEXT")
                self.conn.commit()

    def _migrate_feedback_schema(self) -> None:
        """Drop old-schema feedback/playbook tables so _DDL can recreate them.

        Checks for two migration scenarios:
        1. Old column layout (missing ``trigger``) -- drop data tables + FTS.
        2. Old FTS column name (``feedback_content`` instead of ``search_text``)
           -- drop only the FTS tables so they are recreated with the new column.

        Also handles migration from old table names (raw_feedbacks/feedbacks)
        to new names (user_playbooks/agent_playbooks), renames
        feedback_aggregation_change_logs to playbook_aggregation_change_logs,
        and renames columns on related tables (skills, profiles,
        playbook_aggregation_change_logs).

        Since SQLite is used only for local development, data loss is acceptable.
        """
        # Check for old table names and rename if needed
        with self._lock:
            old_tables = {
                row[0]
                for row in self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

        if "raw_feedbacks" in old_tables and "user_playbooks" not in old_tables:
            logger.warning(
                "Detected old table names (raw_feedbacks/feedbacks). "
                "Dropping old tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS raw_feedbacks_fts;
                    DROP TABLE IF EXISTS feedbacks_fts;
                    DROP TABLE IF EXISTS raw_feedbacks;
                    DROP TABLE IF EXISTS feedbacks;
                """)
                self.conn.commit()

        if (
            "feedback_aggregation_change_logs" in old_tables
            and "playbook_aggregation_change_logs" not in old_tables
        ):
            logger.warning(
                "Renaming table feedback_aggregation_change_logs → playbook_aggregation_change_logs."
            )
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE feedback_aggregation_change_logs RENAME TO playbook_aggregation_change_logs"
                )
                self.conn.commit()

        # Migrate renamed columns on related tables (skills, profiles, change_logs)
        self._migrate_renamed_columns()

        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(user_playbooks)")
            columns = {row[1] for row in cur.fetchall()}

        # Table doesn't exist yet -- nothing to migrate
        if not columns:
            return

        # Scenario 1: old data schema (missing trigger column — pre-flattening)
        if "trigger" not in columns:
            logger.warning(
                "Detected old playbook schema (missing trigger column). "
                "Dropping playbook tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS user_playbooks_fts;
                    DROP TABLE IF EXISTS agent_playbooks_fts;
                    DROP TABLE IF EXISTS user_playbooks;
                    DROP TABLE IF EXISTS agent_playbooks;
                """)
                self.conn.commit()
            return

        # Scenario 2: old FTS column name (feedback_content -> search_text)
        with self._lock:
            cur = self.conn.execute("PRAGMA table_info(user_playbooks_fts)")
            fts_columns = {row[1] for row in cur.fetchall()}

        if fts_columns and "search_text" not in fts_columns:
            logger.warning(
                "Detected old FTS column name. "
                "Dropping FTS tables so they can be recreated with the new schema."
            )
            with self._lock:
                self.conn.executescript("""
                    DROP TABLE IF EXISTS user_playbooks_fts;
                    DROP TABLE IF EXISTS agent_playbooks_fts;
                """)
                self.conn.commit()

    def _migrate_renamed_columns(self) -> None:
        """Rename columns on tables affected by the feedback→playbook rename.

        Handles: skills (feedback_name→playbook_name, raw_feedback_ids→user_playbook_ids),
        profiles (profile_content→content), playbook_aggregation_change_logs (feedback_name→playbook_name).

        Since SQLite is used only for local development, we drop and recreate if needed.
        """
        renames = [
            ("skills", "feedback_name", "playbook_name"),
            ("skills", "raw_feedback_ids", "user_playbook_ids"),
            ("profiles", "profile_content", "content"),
            ("playbook_aggregation_change_logs", "feedback_name", "playbook_name"),
        ]

        for table, old_col, new_col in renames:
            with self._lock:
                try:
                    cols = {
                        row[1]
                        for row in self.conn.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()  # noqa: S608
                    }
                except Exception:  # noqa: S112
                    continue  # Table doesn't exist yet

                if not cols:
                    continue  # Table doesn't exist

                if old_col in cols and new_col not in cols:
                    logger.info(
                        "Renaming column %s.%s -> %s",
                        table,
                        old_col,
                        new_col,
                    )
                    try:
                        self.conn.execute(
                            f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}"  # noqa: S608
                        )
                        self.conn.commit()
                    except Exception as e:
                        logger.warning(
                            "Could not rename %s.%s -> %s: %s. "
                            "Dropping table so it can be recreated.",
                            table,
                            old_col,
                            new_col,
                            e,
                        )
                        self.conn.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
                        self.conn.commit()

    def _migrate_expanded_terms(self) -> None:
        """Add expanded_terms column if missing (for databases created before this feature)."""
        for table in ("profiles", "user_playbooks", "agent_playbooks"):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "expanded_terms" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN expanded_terms TEXT")
                logger.info("Added expanded_terms column to %s", table)
        self.conn.commit()

    def _migrate_tags(self) -> None:
        """Add tags column if missing."""
        for table in ("profiles", "user_playbooks", "agent_playbooks"):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "tags" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN tags TEXT")
                logger.info("Added tags column to %s", table)
        self.conn.commit()

    def _migrate_agent_runs_schema(self) -> None:
        """Add resumable-agent run columns if missing from existing SQLite DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(_agent_runs)").fetchall()
        }
        if not cols:
            return
        if "max_steps_remaining" not in cols:
            self.conn.execute(
                "ALTER TABLE _agent_runs ADD COLUMN max_steps_remaining INTEGER"
            )
            logger.info("Added max_steps_remaining column to _agent_runs")
        self.conn.commit()

    def _migrate_pending_tool_calls_schema(self) -> None:
        """Add pending-tool-call columns if missing from existing SQLite DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(_pending_tool_calls)"
            ).fetchall()
        }
        if not cols:
            return
        if "superseded_by" not in cols:
            self.conn.execute(
                "ALTER TABLE _pending_tool_calls ADD COLUMN superseded_by TEXT"
            )
            logger.info("Added superseded_by column to _pending_tool_calls")
        self.conn.commit()

    def _migrate_agentic_signals(self) -> None:
        """Add source_span/notes/reader_angle columns if missing.

        Backfill-safe: columns are nullable with no default. Applies to both
        the profiles and user_playbooks tables — the agentic extraction
        pipeline populates them per-row; classic extraction leaves them NULL.
        """
        for table in ("profiles", "user_playbooks"):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col in ("source_span", "notes", "reader_angle"):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")  # noqa: S608
                    logger.info("Added %s column to %s", col, table)
        self.conn.commit()

    def _migrate_user_playbook_polarity(self) -> None:
        """Drop the legacy ``polarity`` column from ``user_playbooks`` if present.

        Polarity is retired under Option B (orientation lives in rule wording and
        is LLM-judged, never a stored field). This mirrors the Supabase drop
        migration and brings databases created while the column existed back in
        line with the current schema, which no longer defines it.
        """
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(user_playbooks)").fetchall()
        }
        if not cols:
            return
        if "polarity" in cols:
            self.conn.execute("ALTER TABLE user_playbooks DROP COLUMN polarity")
            logger.info("Dropped legacy polarity column from user_playbooks")
        self.conn.commit()

    def _migrate_lineage(self) -> None:
        """Add merged_into/superseded_by forward-pointer columns if missing.

        Backfill-safe: columns are nullable with no default. INTEGER for playbook
        tables (int foreign-key pointers), TEXT for profiles (str profile_id pointers).
        """
        int_tables = {"user_playbooks": "INTEGER", "agent_playbooks": "INTEGER"}
        str_tables = {"profiles": "TEXT"}
        for table, coltype in {**int_tables, **str_tables}.items():
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col in ("merged_into", "superseded_by"):
                if col not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"  # noqa: S608
                    )
                    logger.info("Added %s column to %s", col, table)
        self.conn.commit()

    def _migrate_lineage_event_table(self) -> None:
        """Create the lineage_event table + index for existing databases (idempotent)."""
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS lineage_event (
                    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id           TEXT NOT NULL,
                    entity_type      TEXT NOT NULL,
                    entity_id        TEXT NOT NULL,
                    op               TEXT NOT NULL,
                    prov_relation    TEXT NOT NULL DEFAULT '',
                    source_ids       TEXT NOT NULL DEFAULT '[]',
                    actor            TEXT NOT NULL DEFAULT '',
                    request_id       TEXT NOT NULL DEFAULT '',
                    reason           TEXT NOT NULL DEFAULT '',
                    created_at       INTEGER NOT NULL,
                    UNIQUE (org_id, entity_type, entity_id, op, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_lineage_entity
                    ON lineage_event (entity_type, entity_id);
            """)
            existing_cols = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(lineage_event)"
                ).fetchall()
            }
            for col in ("from_status", "to_status", "status_namespace"):
                if col not in existing_cols:
                    self.conn.execute(
                        f"ALTER TABLE lineage_event ADD COLUMN {col} TEXT"  # noqa: S608
                    )
                    logger.info("Added %s column to lineage_event", col)
            self.conn.commit()

    def _migrate_agent_playbook_source_windows(self) -> None:
        """Add source window snapshots to existing agent source mappings."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(agent_playbook_source_user_playbooks)"
            ).fetchall()
        }
        if not cols:
            return
        if "source_interaction_ids" not in cols:
            self.conn.execute(
                "ALTER TABLE agent_playbook_source_user_playbooks "
                "ADD COLUMN source_interaction_ids TEXT NOT NULL DEFAULT '[]'"
            )
            logger.info(
                "Added source_interaction_ids column to "
                "agent_playbook_source_user_playbooks"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_apsup_user "
            "ON agent_playbook_source_user_playbooks(user_playbook_id)"
        )
        self.conn.commit()

    def _migrate_request_evaluation_only(self) -> None:
        """Add evaluation_only column to requests for learning exclusion."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols:
            return
        if "evaluation_only" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN evaluation_only INTEGER NOT NULL DEFAULT 0"
            )
            logger.info("Added evaluation_only column to requests")
        self.conn.commit()

    def _migrate_request_session_id_required(self) -> None:
        """Require non-empty session ids on ``requests``.

        SQLite cannot add a ``NOT NULL`` or ``CHECK`` constraint to an
        existing column, so existing databases are rebuilt in place. Historical
        null/blank sessions are intentionally backfilled per request to avoid
        inventing conversation groupings that were never recorded.
        """
        cols = {
            row["name"]: row
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols or "session_id" not in cols:
            return

        table_sql_row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'requests'"
        ).fetchone()
        table_sql = (table_sql_row["sql"] if table_sql_row else "") or ""
        blank_count = self.conn.execute(
            "SELECT COUNT(*) FROM requests WHERE session_id IS NULL OR trim(session_id) = ''"
        ).fetchone()[0]
        has_required_schema = (
            bool(cols["session_id"]["notnull"])
            and "CHECK (trim(session_id) != '')" in table_sql
        )
        if has_required_schema and blank_count == 0:
            return

        evaluation_only_expr = (
            "COALESCE(evaluation_only, 0)" if "evaluation_only" in cols else "0"
        )
        # NOTE: this rebuild hardcodes the full `requests` column set. If a
        # future migration adds a column to `requests`, it MUST be added here
        # too (and to the SELECT below) or the rebuild will silently drop it.
        self.conn.executescript(
            f"""
            CREATE TABLE requests_new (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                agent_version TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL CHECK (trim(session_id) != ''),
                evaluation_only INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO requests_new
                (
                    request_id,
                    user_id,
                    created_at,
                    source,
                    agent_version,
                    session_id,
                    evaluation_only
                )
            SELECT
                request_id,
                user_id,
                created_at,
                COALESCE(source, ''),
                COALESCE(agent_version, ''),
                CASE
                    WHEN session_id IS NULL OR trim(session_id) = ''
                    THEN 'legacy-' || lower(hex(randomblob(16)))
                    ELSE trim(session_id)
                END,
                {evaluation_only_expr}
            FROM requests;
            DROP TABLE requests;
            ALTER TABLE requests_new RENAME TO requests;
            CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
            CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
            CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
            """
        )
        self.conn.commit()
        logger.info("Migrated requests.session_id to required non-empty values")

    def _migrate_shadow_comparison_verdicts(self) -> None:
        """F1: create the shadow_comparison_verdicts table if missing.

        Idempotent; safe to run on every startup. The PRAGMA-LBYL guard
        avoids running the CREATE statements on every boot for
        already-migrated DBs. The ``CREATE TABLE IF NOT EXISTS`` in :data:`_DDL` will
        also create this table on a fresh database, so this helper is a
        no-op there; its purpose is explicit symmetry with the per-feature
        migration convention and a single named hook the disk/supabase
        backends in Tasks 6/7 can mirror.
        """
        cur = self.conn.execute("PRAGMA table_info(shadow_comparison_verdicts)")
        cols = cur.fetchall()
        if cols:
            return
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS shadow_comparison_verdicts (
                verdict_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id          TEXT    NOT NULL,
                session_id              TEXT    NOT NULL,
                agent_version           TEXT    NOT NULL,
                reflexio_is_request_1   INTEGER NOT NULL,
                better_request          TEXT    NOT NULL CHECK (better_request IN ('1','2','tie')),
                is_significantly_better INTEGER NOT NULL,
                comparison_reason       TEXT,
                judge_prompt_version    TEXT    NOT NULL,
                created_at              TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_session
                ON shadow_comparison_verdicts (session_id, agent_version);
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_created_at
                ON shadow_comparison_verdicts (created_at);
            CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_v
                ON shadow_comparison_verdicts (judge_prompt_version);
        """)
        self.conn.commit()
        logger.info("Created shadow_comparison_verdicts table (F1 migration)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _fetchone(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _fetchall(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _get_embedding(
        self, text: str, purpose: Literal["document", "query"] = "document"
    ) -> list[float]:
        """Generate an embedding with a purpose-specific prefix.

        Args:
            text: The text to embed.
            purpose: Either ``"document"`` (stored embeddings) or ``"query"``
                (search-time embeddings).  The prefix improves asymmetric
                retrieval quality for models that support it.

        Returns:
            The embedding vector as a list of floats.
        """
        prefix = "search_document: " if purpose == "document" else "search_query: "
        try:
            return self.llm_client.get_embedding(
                prefix + text, self.embedding_model_name, self.embedding_dimensions
            )
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable for %s text; continuing without vector: %s",
                purpose,
                exc,
            )
            return []

    def _should_expand_documents(self) -> bool:
        """Check if document expansion is enabled."""
        return self._enable_document_expansion

    def _expand_document(self, content: str) -> str | None:
        """Expand document content with synonyms for FTS recall.

        Uses DocumentExpander to generate synonym groups. Returns the
        expanded_terms string (e.g., "backup, sync; failure, error")
        or None on failure.

        Args:
            content (str): Document text to expand

        Returns:
            str or None: Expanded terms text, or None if expansion fails/disabled
        """
        if not content:
            return None
        try:
            from reflexio.server.prompt.prompt_manager import PromptManager
            from reflexio.server.services.pre_retrieval import DocumentExpander

            expander = DocumentExpander(
                llm_client=self.llm_client,
                prompt_manager=PromptManager(),
            )
            result = expander.expand(content)
            return result.expanded_text or None
        except Exception:
            logger.warning("Document expansion failed", exc_info=True)
            return None

    def _current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    # FTS helpers
    def _fts_upsert(self, table: str, rowid: int, **text_fields: str | None) -> None:
        """Insert or update an FTS row.  Deletes old entry first to avoid duplicates."""
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            cols = list(text_fields.keys())
            vals = [text_fields[c] or "" for c in cols]
            placeholders = ",".join("?" for _ in cols)
            col_str = ",".join(cols)
            self.conn.execute(
                f"INSERT INTO {table}(rowid, {col_str}) VALUES (?, {placeholders})",
                [rowid, *vals],
            )
            self.conn.commit()

    def _fts_delete(self, table: str, rowid: int) -> None:
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.commit()

    def _fts_upsert_profile(self, profile_id: str, content: str) -> None:
        """FTS for profiles uses profile_id TEXT as key column."""
        with self._lock:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.execute(
                "INSERT INTO profiles_fts(profile_id, content) VALUES (?, ?)",
                (profile_id, content),
            )
            self.conn.commit()

    def _fts_delete_profile(self, profile_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.commit()

    # Vec helpers (sqlite-vec)
    def _vec_upsert(self, table: str, rowid: int, embedding: list[float]) -> None:
        """Insert or update a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.execute(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                (rowid, json.dumps(embedding)),
            )
            self.conn.commit()

    def _vec_delete(self, table: str, rowid: int) -> None:
        """Delete a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.commit()

    def _vec_knn_search(
        self,
        vec_table: str,
        main_table: str,
        query_embedding: list[float],
        match_count: int,
        conditions: list[str] | None = None,
        params: list[Any] | None = None,
    ) -> list[sqlite3.Row]:
        """Run a native KNN search via sqlite-vec and join back to the main table.

        Over-fetches from the KNN index (5x ``match_count``) so that post-filter
        WHERE conditions (org, user, status, etc.) don't silently reduce the
        result set below the requested count.

        Args:
            vec_table: Name of the vec0 virtual table.
            main_table: Name of the main data table.
            query_embedding: Query embedding vector.
            match_count: Number of results to return.
            conditions: Optional WHERE conditions for the main table.
            params: Parameters for the conditions.

        Returns:
            Up to ``match_count`` rows from the main table, ordered by vector
            distance (ascending).
        """
        knn_overfetch = match_count * 5
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT m.* FROM {main_table} m
                  JOIN (
                      SELECT rowid, distance FROM {vec_table}
                      WHERE embedding MATCH ?
                      ORDER BY distance
                      LIMIT ?
                  ) v ON m.rowid = v.rowid
                  WHERE {where_clause}
                  ORDER BY v.distance
                  LIMIT ?"""
        all_params = [
            json.dumps(query_embedding),
            knn_overfetch,
            *(params or []),
            match_count,
        ]
        return self._fetchall(sql, all_params)

    # ------------------------------------------------------------------
    # Per-user data clear
    # ------------------------------------------------------------------

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        """Atomic per-``user_id`` row deletion across all user-scoped tables.

        Overrides the BaseStorage default with a single-transaction SQL
        implementation. Removes interactions, user playbooks, profiles,
        and requests scoped to the user. Intentionally does NOT touch
        ``agent_playbooks`` — they are the cross-project rollup of
        skills and have no ``user_id`` column.

        Also cleans up FTS and vector sidecars for the user's rows so
        subsequent searches don't surface deleted data.

        Args:
            user_id (str): The user id whose rows should be deleted.

        Returns:
            dict[str, int]: Per-entity deletion counts with keys
                ``interactions``, ``user_playbooks``, ``profiles``, and
                ``requests``.
        """
        with self._lock:
            # Snapshot rowids/ids that need FTS or vector cleanup before
            # the DELETE removes them from the main tables.
            interaction_ids = [
                r["interaction_id"]
                for r in self.conn.execute(
                    "SELECT interaction_id FROM interactions WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            user_playbook_ids = [
                r["user_playbook_id"]
                for r in self.conn.execute(
                    "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            profile_rows = self.conn.execute(
                "SELECT rowid, profile_id FROM profiles WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            profile_rowids = [r["rowid"] for r in profile_rows]
            profile_ids = [r["profile_id"] for r in profile_rows]

            # FTS cleanup
            if interaction_ids:
                ph = ",".join("?" for _ in interaction_ids)
                self.conn.execute(
                    f"DELETE FROM interactions_fts WHERE rowid IN ({ph})",
                    interaction_ids,
                )
            if user_playbook_ids:
                ph = ",".join("?" for _ in user_playbook_ids)
                self.conn.execute(
                    f"DELETE FROM user_playbooks_fts WHERE rowid IN ({ph})",
                    user_playbook_ids,
                )
            if profile_ids:
                ph = ",".join("?" for _ in profile_ids)
                self.conn.execute(
                    f"DELETE FROM profiles_fts WHERE profile_id IN ({ph})",
                    profile_ids,
                )

            # Vector index cleanup (best-effort: only if sqlite-vec loaded)
            if self._has_sqlite_vec:
                vec_targets = (
                    ("interactions_vec", interaction_ids),
                    ("user_playbooks_vec", user_playbook_ids),
                    ("profiles_vec", profile_rowids),
                )
                for vec_table, rowids in vec_targets:
                    if rowids:
                        ph = ",".join("?" for _ in rowids)
                        self.conn.execute(
                            f"DELETE FROM {vec_table} WHERE rowid IN ({ph})",
                            rowids,
                        )

            # Main-table deletes. Order matters only for foreign key
            # integrity; SQLite default has FK off for most tables here
            # so order is chosen for readability.
            interactions_cur = self.conn.execute(
                "DELETE FROM interactions WHERE user_id = ?", (user_id,)
            )
            user_playbooks_cur = self.conn.execute(
                "DELETE FROM user_playbooks WHERE user_id = ?", (user_id,)
            )
            profiles_cur = self.conn.execute(
                "DELETE FROM profiles WHERE user_id = ?", (user_id,)
            )
            requests_cur = self.conn.execute(
                "DELETE FROM requests WHERE user_id = ?", (user_id,)
            )
            self.conn.commit()

            return {
                "interactions": interactions_cur.rowcount,
                "user_playbooks": user_playbooks_cur.rowcount,
                "profiles": profiles_cur.rowcount,
                "requests": requests_cur.rowcount,
            }


# ---------------------------------------------------------------------------
# DDL — table and FTS definitions
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    last_modified_timestamp INTEGER NOT NULL,
    generated_from_request_id TEXT NOT NULL DEFAULT '',
    profile_time_to_live TEXT NOT NULL DEFAULT 'infinity',
    expiration_timestamp INTEGER NOT NULL DEFAULT 4102444800,
    custom_features TEXT,
    embedding TEXT,
    source TEXT DEFAULT '',
    status TEXT,
    extractor_names TEXT,
    expanded_terms TEXT,
    tags TEXT,
    source_span TEXT,
    notes TEXT,
    reader_angle TEXT,
    merged_into TEXT,
    superseded_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(status);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'User',
    user_action TEXT NOT NULL DEFAULT 'none',
    user_action_description TEXT NOT NULL DEFAULT '',
    interacted_image_url TEXT NOT NULL DEFAULT '',
    shadow_content TEXT NOT NULL DEFAULT '',
    expert_content TEXT NOT NULL DEFAULT '',
    tools_used TEXT,
    citations TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_request_id ON interactions(request_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL CHECK (trim(session_id) != ''),
    evaluation_only INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_session_created_at_asc
    ON requests(session_id, created_at ASC, request_id ASC);

CREATE TABLE IF NOT EXISTS user_playbooks (
    user_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    trigger TEXT,
    rationale TEXT,
    blocking_issue TEXT,
    source_interaction_ids TEXT,
    status TEXT,
    source TEXT,
    embedding TEXT,
    expanded_terms TEXT,
    tags TEXT,
    source_span TEXT,
    notes TEXT,
    reader_angle TEXT,
    merged_into INTEGER,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_playbook_name ON user_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_agent_version ON user_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_status ON user_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_user_playbooks_created_at ON user_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_playbooks (
    agent_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    trigger TEXT,
    rationale TEXT,
    blocking_issue TEXT,
    playbook_status TEXT NOT NULL DEFAULT 'pending',
    playbook_metadata TEXT NOT NULL DEFAULT '',
    embedding TEXT,
    expanded_terms TEXT,
    tags TEXT,
    status TEXT,
    merged_into INTEGER,
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_playbook_name ON agent_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_agent_version ON agent_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_status ON agent_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_created_at ON agent_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_success_evaluation_result (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    evaluation_name TEXT,
    is_success INTEGER NOT NULL DEFAULT 0,
    failure_type TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    regular_vs_shadow TEXT,
    number_of_correction_per_session INTEGER NOT NULL DEFAULT 0,
    user_turns_to_resolution INTEGER,
    is_escalated INTEGER NOT NULL DEFAULT 0,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version ON agent_success_evaluation_result(agent_version);
CREATE INDEX IF NOT EXISTS idx_eval_created_at ON agent_success_evaluation_result(created_at);
CREATE INDEX IF NOT EXISTS idx_eval_created_at_desc
    ON agent_success_evaluation_result(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version_created_at_desc
    ON agent_success_evaluation_result(agent_version, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_identity_created_at_desc
    ON agent_success_evaluation_result(session_id, evaluation_name, agent_version, created_at DESC);

CREATE TABLE IF NOT EXISTS profile_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    added_profiles TEXT NOT NULL DEFAULT '[]',
    removed_profiles TEXT NOT NULL DEFAULT '[]',
    mentioned_profiles TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_pcl_user_id ON profile_change_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_pcl_created_at ON profile_change_logs(created_at);

CREATE TABLE IF NOT EXISTS playbook_aggregation_change_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    playbook_name TEXT NOT NULL,
    agent_version TEXT NOT NULL,
    run_mode TEXT NOT NULL,
    added_playbooks TEXT NOT NULL DEFAULT '[]',
    removed_playbooks TEXT NOT NULL DEFAULT '[]',
    updated_playbooks TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_pacl_playbook_name ON playbook_aggregation_change_logs(playbook_name);
CREATE INDEX IF NOT EXISTS idx_pacl_agent_version ON playbook_aggregation_change_logs(agent_version);

CREATE TABLE IF NOT EXISTS agent_playbook_source_user_playbooks (
    agent_playbook_id INTEGER NOT NULL,
    user_playbook_id INTEGER NOT NULL,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (agent_playbook_id, user_playbook_id)
);
CREATE INDEX IF NOT EXISTS idx_apsup_agent ON agent_playbook_source_user_playbooks(agent_playbook_id);
CREATE INDEX IF NOT EXISTS idx_apsup_user ON agent_playbook_source_user_playbooks(user_playbook_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    best_candidate_id INTEGER,
    successor_target_id INTEGER,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poj_target ON playbook_optimization_jobs(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_poj_status ON playbook_optimization_jobs(status);

CREATE TABLE IF NOT EXISTS playbook_optimization_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
    aggregate_score REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poc_job ON playbook_optimization_candidates(job_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    scenario_user_playbook_id INTEGER,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    score REAL NOT NULL DEFAULT 0.0,
    verdict TEXT NOT NULL DEFAULT 'tie',
    likert INTEGER NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    asi_json TEXT NOT NULL DEFAULT '{}',
    incumbent_rollout_json TEXT NOT NULL DEFAULT '[]',
    candidate_rollout_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poe_job ON playbook_optimization_evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_poe_candidate ON playbook_optimization_evaluations(candidate_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poev_job ON playbook_optimization_events(job_id);

CREATE TABLE IF NOT EXISTS _operation_state (
    service_name TEXT PRIMARY KEY,
    operation_state TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS _agent_runs (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    extractor_kind TEXT NOT NULL,
    user_id TEXT,
    request_id TEXT NOT NULL,
    agent_version TEXT,
    source TEXT,
    source_interaction_ids TEXT NOT NULL DEFAULT '[]',
    window_start_interaction_id INTEGER,
    window_end_interaction_id INTEGER,
    extractor_config_hash TEXT,
    status TEXT NOT NULL,
    generation_request_snapshot TEXT NOT NULL DEFAULT '{}',
    service_config_snapshot TEXT,
    agent_context_snapshot TEXT,
    committed_output TEXT,
    pending_tool_call_ids TEXT NOT NULL DEFAULT '[]',
    max_steps_remaining INTEGER,
    resume_attempts INTEGER NOT NULL DEFAULT 0,
    finalization_attempts INTEGER NOT NULL DEFAULT 0,
    next_resume_at TEXT,
    claimed_by TEXT,
    claimed_at TEXT,
    agent_completed_at TEXT,
    finalized_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_ready ON _agent_runs(status, next_resume_at, updated_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_binding ON _agent_runs(org_id, extractor_kind, user_id);

CREATE TABLE IF NOT EXISTS _pending_tool_calls (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT,
    scope TEXT NOT NULL DEFAULT '{}',
    scope_hash TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    status TEXT NOT NULL,
    question_text TEXT NOT NULL,
    answer_format TEXT,
    args TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    embedding TEXT,
    superseded_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT,
    expires_at TEXT NOT NULL,
    cache_until TEXT NOT NULL,
    valid_until TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_active ON _pending_tool_calls(org_id, scope_hash, tool_name, dedup_key, status, cache_until);
CREATE INDEX IF NOT EXISTS idx_pending_tool_calls_prior ON _pending_tool_calls(org_id, scope_hash, tool_name, status, valid_until);

CREATE TABLE IF NOT EXISTS _run_tool_dependencies (
    run_id TEXT NOT NULL,
    pending_tool_call_id TEXT NOT NULL,
    dependency_kind TEXT NOT NULL DEFAULT 'followup',
    resolved_at TEXT,
    consumed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (run_id, pending_tool_call_id),
    FOREIGN KEY (run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (pending_tool_call_id) REFERENCES _pending_tool_calls(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_tool_dependencies_pending ON _run_tool_dependencies(pending_tool_call_id, resolved_at, consumed_at);
CREATE INDEX IF NOT EXISTS idx_run_tool_dependencies_ready ON _run_tool_dependencies(run_id, resolved_at, consumed_at);

-- FTS5 virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts USING fts5(
    content, user_action_description,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS profiles_fts USING fts5(
    profile_id, content,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_fts USING fts5(
    search_text,
    tokenize="porter unicode61"
);

CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_fts USING fts5(
    search_text,
    tokenize="porter unicode61"
);

CREATE TABLE IF NOT EXISTS share_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    created_by_email TEXT
);
CREATE INDEX IF NOT EXISTS idx_share_links_resource ON share_links(resource_type, resource_id);

-- ============================================================================
-- Braintrust connector (Plan C-backend)
-- ============================================================================

CREATE TABLE IF NOT EXISTS braintrust_connection (
    org_id TEXT PRIMARY KEY,
    api_key_enc TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    workspace_name TEXT NOT NULL DEFAULT '',
    project_ids TEXT NOT NULL DEFAULT '[]',
    last_sync_ts INTEGER,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS imported_score (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    session_id TEXT,
    scorer_name TEXT NOT NULL,
    value REAL NOT NULL,
    ts INTEGER NOT NULL,
    UNIQUE (org_id, source, source_run_id, scorer_name)
);
CREATE INDEX IF NOT EXISTS idx_imported_score_session
    ON imported_score (org_id, session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_imported_score_ts ON imported_score (org_id, ts);

-- ============================================================================
-- Per-turn shadow comparison verdicts (F1)
-- ============================================================================

CREATE TABLE IF NOT EXISTS shadow_comparison_verdicts (
    verdict_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id          TEXT    NOT NULL,
    session_id              TEXT    NOT NULL,
    agent_version           TEXT    NOT NULL,
    reflexio_is_request_1   INTEGER NOT NULL,
    better_request          TEXT    NOT NULL CHECK (better_request IN ('1','2','tie')),
    is_significantly_better INTEGER NOT NULL,
    comparison_reason       TEXT,
    judge_prompt_version    TEXT    NOT NULL,
    created_at              TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_session
    ON shadow_comparison_verdicts (session_id, agent_version);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_created_at
    ON shadow_comparison_verdicts (created_at);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_v
    ON shadow_comparison_verdicts (judge_prompt_version);
CREATE INDEX IF NOT EXISTS idx_shadow_verdicts_prompt_created_at_desc
    ON shadow_comparison_verdicts (judge_prompt_version, created_at DESC);

-- ============================================================================
-- Append-only, content-free lineage event log
-- ============================================================================

CREATE TABLE IF NOT EXISTS lineage_event (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id           TEXT NOT NULL,
    entity_type      TEXT NOT NULL,
    entity_id        TEXT NOT NULL,
    op               TEXT NOT NULL,
    prov_relation    TEXT NOT NULL DEFAULT '',
    source_ids       TEXT NOT NULL DEFAULT '[]',
    actor            TEXT NOT NULL DEFAULT '',
    request_id       TEXT NOT NULL DEFAULT '',
    reason           TEXT NOT NULL DEFAULT '',
    created_at       INTEGER NOT NULL,
    from_status      TEXT,
    to_status        TEXT,
    status_namespace TEXT,
    UNIQUE (org_id, entity_type, entity_id, op, request_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_entity ON lineage_event (entity_type, entity_id);

"""
