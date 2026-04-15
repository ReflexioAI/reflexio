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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentPlaybookSnapshot,
    AgentPlaybookUpdateEntry,
    AgentSuccessEvaluationResult,
    Interaction,
    PlaybookAggregationChangeLog,
    PlaybookStatus,
    ProfileChangeLog,
    ProfileTimeToLive,
    RegularVsShadow,
    Request,
    Status,
    StructuredData,
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
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.site_var.site_var_manager import SiteVarManager

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
) -> SearchMode:
    """Downgrade search mode when the required embedding is unavailable.

    Args:
        mode: Requested search mode.
        query_embedding: Pre-computed query embedding, or None.

    Returns:
        The effective SearchMode — falls back to FTS when HYBRID/VECTOR lacks an embedding.
    """
    if mode in (SearchMode.HYBRID, SearchMode.VECTOR) and not query_embedding:
        logger.warning(
            "Search mode '%s' requested but no query embedding provided — falling back to FTS",
            mode,
        )
        return SearchMode.FTS
    return mode


def _vector_rank_rows(
    rows: list[sqlite3.Row],
    query_embedding: list[float],
    match_count: int,
) -> list[sqlite3.Row]:
    """Rank rows by cosine similarity to the query embedding.

    Args:
        rows: Candidate rows with stored embeddings.
        query_embedding: The query's embedding vector.
        match_count: Number of results to return.

    Returns:
        Top ``match_count`` rows sorted by cosine similarity descending.
    """
    scored: list[tuple[sqlite3.Row, float]] = []
    for row in rows:
        raw_emb = row["embedding"] if "embedding" in row.keys() else None  # noqa: SIM118
        emb = _json_loads(raw_emb) if raw_emb else None
        if emb:
            sim = _cosine_similarity(query_embedding, emb)
            scored.append((row, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:match_count]]


def _true_rrf_merge(
    fts_rows: list[sqlite3.Row],
    vec_rows: list[sqlite3.Row],
    id_column: str,
    match_count: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 1.0,
) -> list[sqlite3.Row]:
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
    row_by_id: dict[str | int, sqlite3.Row] = {}
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

    scored: list[tuple[sqlite3.Row, float]] = []
    for rid, row in row_by_id.items():
        f_rank = fts_rank.get(rid, fts_penalty)
        v_rank = vec_rank.get(rid, vec_penalty)
        score = fts_weight / (rrf_k + f_rank) + vector_weight / (rrf_k + v_rank)
        scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:match_count]]


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
        return int(datetime.fromisoformat(cleaned).timestamp())
    except (ValueError, TypeError):
        return _epoch_now()


def _epoch_to_iso(ts: int) -> str:
    """Convert a Unix timestamp to ISO 8601 string."""
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


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
    )


def _row_to_interaction(row: sqlite3.Row) -> Interaction:
    d = dict(row)
    tools_used_raw = _json_loads(d.get("tools_used"))
    tools_used = (
        [ToolUsed(**t) for t in tools_used_raw if isinstance(t, dict)]
        if tools_used_raw and isinstance(tools_used_raw, list)
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
    )


def _row_to_request(row: sqlite3.Row) -> Request:
    d = dict(row)
    return Request(
        request_id=d["request_id"],
        user_id=d["user_id"],
        created_at=_iso_to_epoch(d["created_at"]),
        source=d.get("source") or "",
        agent_version=d.get("agent_version") or "",
        session_id=d.get("session_id"),
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
        structured_data=StructuredData(**json.loads(d.get("structured_data") or "{}")),
        status=Status(d["status"]) if d.get("status") else None,
        source=d.get("source"),
        source_interaction_ids=_json_loads(d.get("source_interaction_ids")) or [],
        embedding=embedding,
        expanded_terms=d.get("expanded_terms"),
    )


def _row_to_agent_playbook(row: sqlite3.Row) -> AgentPlaybook:
    d = dict(row)
    return AgentPlaybook(
        agent_playbook_id=d["agent_playbook_id"],
        playbook_name=d["playbook_name"],
        created_at=_iso_to_epoch(d["created_at"]),
        agent_version=d["agent_version"],
        content=d["content"],
        structured_data=StructuredData(**json.loads(d.get("structured_data") or "{}")),
        playbook_status=PlaybookStatus(d["playbook_status"])
        if d.get("playbook_status")
        else PlaybookStatus.PENDING,
        playbook_metadata=d.get("playbook_metadata") or "",
        embedding=[],
        status=Status(d["status"]) if d.get("status") else None,
        expanded_terms=d.get("expanded_terms"),
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


class SQLiteStorageBase(BaseStorage):
    """SQLite-backed storage base class for local/self-hosted deployments."""

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

        # Resolve db_path: explicit arg > SQLITE_FILE_DIRECTORY env var > reflexio/data/
        if db_path is None:
            from reflexio.server import SQLITE_FILE_DIRECTORY

            db_path = str(Path(SQLITE_FILE_DIRECTORY) / "reflexio.db")

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
        self._migrate_expanded_terms()
        return True

    def _try_load_sqlite_vec(self) -> bool:
        """Attempt to load the sqlite-vec extension for native KNN search.

        Returns:
            True if the extension was loaded successfully, False otherwise.
        """
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded — native KNN search enabled")
            return True
        except (ImportError, OSError, sqlite3.OperationalError) as e:
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

    def _migrate_feedback_schema(self) -> None:
        """Drop old-schema feedback/playbook tables so _DDL can recreate them.

        Checks for two migration scenarios:
        1. Old column layout (missing ``structured_data``) -- drop data tables + FTS.
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

        # Scenario 1: old data schema (missing structured_data column)
        if "structured_data" not in columns:
            logger.warning(
                "Detected old playbook schema (missing structured_data column). "
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
        return self.llm_client.get_embedding(
            prefix + text, self.embedding_model_name, self.embedding_dimensions
        )

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
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);

CREATE TABLE IF NOT EXISTS user_playbooks (
    user_playbook_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    playbook_name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    structured_data TEXT,
    source_interaction_ids TEXT,
    status TEXT,
    source TEXT,
    embedding TEXT,
    expanded_terms TEXT
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
    structured_data TEXT,
    playbook_status TEXT NOT NULL DEFAULT 'pending',
    playbook_metadata TEXT NOT NULL DEFAULT '',
    embedding TEXT,
    expanded_terms TEXT,
    status TEXT
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

CREATE TABLE IF NOT EXISTS _operation_state (
    service_name TEXT PRIMARY KEY,
    operation_state TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

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

"""
