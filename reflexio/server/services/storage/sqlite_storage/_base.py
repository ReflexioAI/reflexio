"""
Base class and module-level helpers for SQLite storage.

Supports hybrid search combining FTS5 (BM25) with embedding cosine similarity
via Reciprocal Rank Fusion (RRF). Falls back to FTS-only when no embeddings
are available.

"""

import contextlib
import functools
import json
import logging
import math
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Generator, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Literal

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain import SessionOutcomeKind
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Citation,
    Interaction,
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    Request,
    RetrievedLearning,
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
    resolve_service_configured_model,
)
from reflexio.server.services.embedding_text import embedding_input
from reflexio.server.services.storage.error import (
    StorageError,
    require_non_empty_session_id,
)
from reflexio.server.services.storage.retention_mixin import RetentionMixin
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
    canonical_json_bytes,
    canonical_session_trajectory,
    outcome_contract_digest,
    trajectory_digest,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.site_var.site_var_manager import SiteVarManager

from ._governance import init_governance_tables
from ._stall_state import init_stall_state_table

logger = logging.getLogger(__name__)

_OUTCOME_ALLOWED_VALUES = tuple(kind.value for kind in SessionOutcomeKind)
_MINIMUM_SQLITE_VERSION = (3, 35, 0)
_SQLITE_INITIALIZATION_LOCK_STRIPES = 64
_sqlite_initialization_locks = tuple(
    threading.Lock() for _ in range(_SQLITE_INITIALIZATION_LOCK_STRIPES)
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_sqlite_initialization_lock(db_path: str) -> threading.Lock:
    """Return a bounded process-local lock for one SQLite database path."""
    normalized_path = str(Path(db_path).resolve())
    return _sqlite_initialization_locks[
        hash(normalized_path) % _SQLITE_INITIALIZATION_LOCK_STRIPES
    ]


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


def _canonical_session_snapshot(
    conn: sqlite3.Connection, session_id: str
) -> CanonicalSessionTrajectory:
    """Return the durable session state used to bind an outcome finalization."""
    request_rows = conn.execute(
        """SELECT request_id, user_id, created_at, source, agent_version, session_id,
                  evaluation_only, retrieval_experiment_id, retrieval_experiment_arm
           FROM requests WHERE session_id = ?
           ORDER BY created_at ASC, request_id ASC""",
        (session_id,),
    ).fetchall()
    request_payloads = [dict(request) for request in request_rows]
    interactions_by_request: dict[str, list[dict[str, object]]] = {
        str(request["request_id"]): [] for request in request_rows
    }
    interaction_rows = conn.execute(
        """SELECT interaction_id, user_id, request_id, created_at, content, role,
                  token_count, user_action, user_action_description,
                  interacted_image_url, image_encoding, shadow_content,
                  expert_content, tools_used, citations, retrieved_learnings
           FROM interactions
           WHERE request_id IN (
               SELECT request_id FROM requests WHERE session_id = ?
           )
           ORDER BY request_id ASC, created_at ASC, interaction_id ASC""",
        (session_id,),
    ).fetchall()
    for interaction in interaction_rows:
        interactions_by_request[str(interaction["request_id"])].append(
            dict(interaction)
        )
    return canonical_session_trajectory(
        session_id, request_payloads, interactions_by_request
    )


def _legacy_session_outcome_id(user_id: str, session_id: str) -> str:
    """Return a delimiter-safe immutable identity for a migrated legacy outcome."""
    return sha256(canonical_json_bytes([user_id, session_id])).hexdigest()


_FTS5_OPERATORS = frozenset({"OR", "AND", "NOT"})
_FTS5_RESERVED = _FTS5_OPERATORS | {"NEAR"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_UNICODE_LEXICAL_MAX_TERMS = 16
_UNICODE_LEXICAL_MAX_CANDIDATES = 250
_UNICODE_LEXICAL_INDEX_VERSION = 1


def _normalize_lexical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _unicode_lexical_runs(text: str) -> tuple[str, ...]:
    """Split normalized text at punctuation and ASCII/non-ASCII boundaries."""
    normalized = _normalize_lexical_text(text)
    runs: list[str] = []
    current: list[str] = []
    current_is_ascii: bool | None = None
    for char in normalized:
        if not char.isalnum():
            if current:
                runs.append("".join(current))
                current = []
            current_is_ascii = None
            continue
        is_ascii = char.isascii()
        if current and is_ascii != current_is_ascii:
            runs.append("".join(current))
            current = []
        current.append(char)
        current_is_ascii = is_ascii
    if current:
        runs.append("".join(current))
    return tuple(runs)


@functools.lru_cache(maxsize=512)
def _unicode_lexical_terms(query: str) -> tuple[str, ...]:
    """Return bounded Unicode-aware terms used for query coverage scoring."""
    terms: list[str] = []
    for run in _unicode_lexical_runs(query):
        if run.isascii() or len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tuple(dict.fromkeys(terms))[:_UNICODE_LEXICAL_MAX_TERMS]


def _unicode_lexical_index_text(document: str | None) -> str:
    """Build normalized unigram/bigram tokens for the Unicode FTS sidecars."""
    if not document:
        return ""
    terms: list[str] = []
    for run in _unicode_lexical_runs(document):
        if run.isascii():
            terms.append(run)
            continue
        terms.extend(run)
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return " ".join(dict.fromkeys(terms))


def register_unicode_lexical_index_function(conn: sqlite3.Connection) -> None:
    """Register the Unicode lexical trigger function on a writer connection."""
    conn.create_function(
        "reflexio_unicode_lexical_index",
        1,
        _unicode_lexical_index_text,
        deterministic=True,
    )


def _unicode_lexical_fts_query(query: str) -> str:
    """Return an OR query over normalized Unicode lexical terms."""
    return " OR ".join(f'"{term}"' for term in _unicode_lexical_terms(query))


def _unicode_lexical_candidate_limit(result_limit: int) -> int:
    """Bound the indexed candidate pool before Python-side coverage scoring."""
    return min(max(result_limit * 5, 50), _UNICODE_LEXICAL_MAX_CANDIDATES)


def _uses_unicode_lexical_fallback(query: str) -> bool:
    return any(char.isalnum() and not char.isascii() for char in query)


def _unicode_lexical_score(document: str | None, query: str | None) -> float:
    """Score query-term coverage in a Unicode-normalized document."""
    if not document or not query:
        return 0.0
    terms = _unicode_lexical_terms(query)
    if not terms:
        return 0.0
    normalized_document = _normalize_lexical_text(document)
    matched = sum(term in normalized_document for term in terms)
    return matched / len(terms)


def _rank_unicode_lexical_rows(
    rows: Sequence[Any],
    query: str,
    text_columns: Sequence[str],
    limit: int,
) -> list[Any]:
    """Rank an already-bounded indexed candidate set by query-term coverage."""
    scored: list[tuple[float, int, Any]] = []
    for index, row in enumerate(rows):
        document = " ".join(str(row[column] or "") for column in text_columns)
        score = _unicode_lexical_score(document, query)
        if score > 0:
            scored.append((score, index, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _score, _index, row in scored[:limit]]


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
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text))
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
    *,
    threshold: float,
) -> list[Any]:
    """Rank rows by cosine similarity to the query embedding.

    Args:
        rows: Candidate rows with stored embeddings.
        query_embedding: The query's embedding vector.
        match_count: Number of results to return.
        threshold: Strict minimum cosine similarity for the vector arm.

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
        passing = [(row, score) for row, score in scored if score > threshold]
        logger.info(
            "vector_rank: candidates=%d passing=%d threshold=%.3f "
            "match_count=%d top_scores=%s",
            len(scored),
            len(passing),
            threshold,
            match_count,
            top,
        )
        return [row for row, _ in passing[:match_count]]
    return []


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

# Profile reads also treat EXPIRED (TTL-expired tombstone) as non-current. Kept
# separate from _TOMBSTONE_STATUS_VALUES because that tuple is shared with playbook
# queries (which never carry EXPIRED) via hardcoded placeholders.
_PROFILE_TOMBSTONE_STATUS_VALUES = (
    Status.MERGED.value,
    Status.SUPERSEDED.value,
    Status.EXPIRED.value,
)


def parse_status(value: str | None) -> Status | None:
    """Parse a stored status string into a Status, tolerating unknown values.

    Unknown non-null values (e.g. a status written by a newer build after a
    rollback) map to a non-current tombstone sentinel and log an anomaly, rather
    than raising ValueError. Never maps to None (which models CURRENT/live).

    Args:
        value: Raw status string from the database, or None/empty for CURRENT.

    Returns:
        The matching Status enum member, Status.SUPERSEDED for unknown values,
        or None for falsy input (representing the CURRENT/live state).
    """
    if not value:
        return None
    try:
        return Status(value)
    except ValueError:
        from reflexio.server.error_reporting import capture_anomaly

        capture_anomaly(
            "storage.status.unknown_value", level="warning", status_value=value
        )
        return Status.SUPERSEDED


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
_MIN_CONTEMPORARY_MILLISECOND_EPOCH_TS = 1_500_000_000_000


def _epoch_to_iso(ts: int) -> str:
    """Convert a Unix timestamp (seconds) to an ISO 8601 string.

    Plausible contemporary millisecond timestamps are normalized to seconds.
    Lower values stay in seconds-space so documented sentinel bounds such as
    ``to_ts=10**12`` still clamp to the representable range instead of being
    treated as a 2001 millisecond timestamp.
    """
    if _MIN_CONTEMPORARY_MILLISECOND_EPOCH_TS <= ts <= _MAX_SAFE_EPOCH_TS * 1000:
        ts = ts // 1000
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
        status=parse_status(d.get("status")),
        extractor_names=_json_loads(d.get("extractor_names")),
        expanded_terms=d.get("expanded_terms"),
        source_span=d.get("source_span"),
        notes=d.get("notes"),
        reader_angle=d.get("reader_angle"),
        tags=_json_loads(d.get("tags")),
        source_interaction_ids=_json_loads(d.get("source_interaction_ids")) or [],
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
    retrieved_learnings_raw = _json_loads(d.get("retrieved_learnings"))
    retrieved_learnings = (
        [RetrievedLearning(**c) for c in retrieved_learnings_raw if isinstance(c, dict)]
        if retrieved_learnings_raw and isinstance(retrieved_learnings_raw, list)
        else []
    )
    return Interaction(
        interaction_id=d["interaction_id"],
        user_id=d["user_id"],
        content=d["content"],
        request_id=d["request_id"],
        created_at=_iso_to_epoch(d["created_at"]),
        role=d.get("role") or "User",
        token_count=d.get("token_count"),
        user_action=UserActionType(d["user_action"]),
        user_action_description=d["user_action_description"],
        interacted_image_url=d["interacted_image_url"],
        image_encoding=d.get("image_encoding") or "",
        shadow_content=d.get("shadow_content") or "",
        expert_content=d.get("expert_content") or "",
        tools_used=tools_used,
        citations=citations,
        retrieved_learnings=retrieved_learnings,
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
        retrieval_experiment_id=d.get("retrieval_experiment_id"),
        retrieval_experiment_arm=d.get("retrieval_experiment_arm"),
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
        status=parse_status(d.get("status")),
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
        status=parse_status(d.get("status")),
        expanded_terms=d.get("expanded_terms"),
        merged_into=d.get("merged_into"),
        superseded_by=d.get("superseded_by"),
    )


def _row_to_eval_result(row: sqlite3.Row) -> AgentSuccessEvaluationResult:
    d = dict(row)
    return AgentSuccessEvaluationResult(
        result_id=d["result_id"],
        user_id=d.get("user_id") or "",
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
        tags=_json_loads(d.get("tags")),
        embedding=[],
    )


# ---------------------------------------------------------------------------
# SQLiteStorageBase
# ---------------------------------------------------------------------------


class SQLiteStorageBase(RetentionMixin, BaseStorage):
    """SQLite-backed storage base class for local/self-hosted deployments."""

    supports_embedding: ClassVar[bool] = True

    # Chunked bulk-delete helpers provided by SQLiteDeletionMixin via the
    # composed SQLiteStorage MRO; declared here for clear_user_data's benefit.
    _delete_in_chunks: Any
    _delete_source_windows_for_user_playbook_ids: Any
    _subject_ref_for_user_id: Callable[[str], str]

    # FTS/vec index helpers provided by SQLiteFtsVecMixin via the composed
    # SQLiteStorage MRO; declared here for _migrate_vec_tables's benefit.
    _vec_upsert: Any
    _flush_index_op: Any

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
        self._scope_depth = 0
        self._deferred_index_ops: list[
            tuple[str, Any]
        ] = []  # (kind, args) flushed post-commit

        logger.info("SQLite Storage for org %s using db_path: %s", org_id, db_path)

        if sqlite3.sqlite_version_info < _MINIMUM_SQLITE_VERSION:
            detected_version = ".".join(map(str, sqlite3.sqlite_version_info))
            raise RuntimeError(
                f"SQLite 3.35.0 or newer is required; detected {detected_version}"
            )

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        initialization_lock = _get_sqlite_initialization_lock(db_path)

        # SQLite's journal-mode negotiation can fail immediately when another
        # connection is cold-starting the same file. Serialize that setup by
        # database path while allowing unrelated SQLite files to initialize in
        # parallel.
        with initialization_lock:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            register_unicode_lexical_index_function(self.conn)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")

        # LLM client for embeddings
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}

        self.embedding_model_name = resolve_service_configured_model(
            resolve_model_name(
                ModelRole.EMBEDDING,
                site_var_value=site_var.get("embedding_model_name"),
                config_override=(
                    llm_config.embedding_model_name if llm_config else None
                ),
                api_key_config=self.api_key_config,
            )
        )
        self.embedding_dimensions = EMBEDDING_DIMENSIONS
        # Text-generation model for storage-time document expansion. The
        # shared self.llm_client is pinned to the EMBEDDING model, which is
        # not a chat model — expansion calls must override the model or they
        # fail (e.g. local/minilm-l6-v2 cannot serve completions).
        self._expansion_model_name = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            config_override=llm_config.generation_model_name if llm_config else None,
            api_key_config=self.api_key_config,
        )

        litellm_config = LiteLLMConfig(
            model=self.embedding_model_name,
            temperature=0.0,
            api_key_config=self.api_key_config,
        )
        self.llm_client = LiteLLMClient(litellm_config)

        # Optionally load sqlite-vec for native KNN vector search
        self._has_sqlite_vec = self._try_load_sqlite_vec()

        # Migrations use an instance-local lock, so separate storage instances
        # for the same file also need the shared path lock.
        with initialization_lock:
            self.migrate()

    # ------------------------------------------------------------------
    # Transaction scope
    # ------------------------------------------------------------------

    def _own_transaction(self) -> bool:
        """True when the caller is NOT inside a commit_scope (owns its BEGIN/commit)."""
        return self._scope_depth == 0

    @contextlib.contextmanager
    def commit_scope(self) -> Generator[None, None, None]:
        """Group writes into one atomic commit; defer FTS/vec index ops.

        Nested scopes join the outermost: only the outer scope commits. On
        exception the whole transaction rolls back and deferred index ops
        are discarded.
        """
        with self._lock:
            if self._scope_depth > 0:
                self._scope_depth += 1
                try:
                    yield
                finally:
                    self._scope_depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._scope_depth = 1
            try:
                yield
                self.conn.commit()
                ops, self._deferred_index_ops = self._deferred_index_ops, []
                for kind, args in ops:
                    self._flush_index_op(kind, args)  # self-commits — fine post-commit
            except Exception:
                self.conn.rollback()
                self._deferred_index_ops = []
                raise
            finally:
                self._scope_depth = 0

    # ------------------------------------------------------------------
    # DDL / migration
    # ------------------------------------------------------------------

    def migrate(self) -> bool:
        # Import lazily: the aggregation mixin shares this base and importing its
        # package while ``_base`` is still initializing would create a cycle.
        from .playbook._aggregation import init_playbook_aggregation_tables

        self._migrate_feedback_schema()
        self._migrate_interactions_schema()
        # Backfill columns that _DDL indexes depend on BEFORE running _DDL.
        # _DDL builds idx_eval_identity_created_at_desc on
        # agent_success_evaluation_result(user_id, ...). On a pre-existing DB that
        # table predates user_id, so executescript(_DDL) raises "no such column:
        # user_id" and migrate() never reaches the backfill below — leaving the DB
        # permanently stuck. The helper is guarded (no-ops when the table is
        # absent), so running it before _DDL is safe on fresh databases too.
        self._migrate_eval_result_user_id()
        self._migrate_retrieved_learning_interaction_identity()
        self._migrate_playbook_optimization_job_columns()
        # _DDL creates an index over these columns. Upgrade legacy request tables
        # before executescript so index creation cannot fail on missing columns.
        self._migrate_request_retrieval_experiment()
        with self._lock:
            session_outcome_columns = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(session_outcomes)"
                ).fetchall()
            }
            if (
                session_outcome_columns
                and "governance_subject_ref" not in session_outcome_columns
            ):
                self.conn.execute(
                    "ALTER TABLE session_outcomes "
                    "ADD COLUMN governance_subject_ref TEXT"
                )
            cur = self.conn.cursor()
            cur.executescript(_DDL)
            init_governance_tables(self.conn)
            init_playbook_aggregation_tables(self.conn)
            self.conn.commit()
        self._migrate_session_outcomes_schema()
        if self._has_sqlite_vec:
            self._create_vec_tables()
            self._migrate_vec_tables()
            self._migrate_playbook_aggregation_agent_centroids()
        # Run after DDL so tables exist on fresh databases
        self._migrate_agent_runs_schema()
        self._migrate_pending_tool_calls_schema()
        self._migrate_expanded_terms()
        self._migrate_unicode_lexical_indexes()
        self._migrate_tags()
        self._migrate_profile_source_interaction_ids()
        self._migrate_interaction_window_indexes()
        self._migrate_agentic_signals()
        self._migrate_agent_playbook_source_windows()
        self._migrate_request_evaluation_only()
        self._migrate_request_session_id_required()
        # _migrate_eval_result_user_id() runs before _DDL (see above).
        self._migrate_shadow_comparison_verdicts()
        self._migrate_user_playbook_polarity()
        self._migrate_lineage()
        self._migrate_retired_at()
        self._migrate_lineage_event_table()
        self._migrate_playbook_optimization_candidate_metadata()
        self._migrate_user_playbook_publication_staging_columns()
        self._classify_legacy_playbook_optimization_jobs()
        self._enforce_playbook_optimization_job_constraints()
        self._migrate_retire_profile_change_logs()
        self._migrate_retire_playbook_aggregation_change_logs()
        init_stall_state_table(self.conn)
        self._migrate_learning_jobs()
        return True

    def _migrate_unicode_lexical_indexes(self) -> None:
        """Backfill the trigger-maintained Unicode FTS sidecars exactly once."""
        version_sentinel_rowid = -_UNICODE_LEXICAL_INDEX_VERSION
        with self._lock:
            version_row = self.conn.execute(
                "SELECT rowid FROM interactions_unicode_fts WHERE rowid = ?",
                (version_sentinel_rowid,),
            ).fetchone()
            if version_row is not None:
                return

            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for index_table in (
                    "interactions_unicode_fts",
                    "profiles_unicode_fts",
                    "user_playbooks_unicode_fts",
                    "agent_playbooks_unicode_fts",
                ):
                    self.conn.execute(f"DELETE FROM {index_table}")  # noqa: S608

                self.conn.execute(
                    """INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
                       SELECT i.rowid, reflexio_unicode_lexical_index(
                           COALESCE(i.content, '') || ' ' ||
                           COALESCE(i.user_action_description, '')
                       )
                       FROM interactions i"""
                )
                self.conn.execute(
                    """INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
                       SELECT p.rowid, reflexio_unicode_lexical_index(
                           COALESCE(p.content, '') || ' ' ||
                           COALESCE(p.expanded_terms, '')
                       )
                       FROM profiles p"""
                )
                self.conn.execute(
                    """INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
                       SELECT up.rowid, reflexio_unicode_lexical_index(
                           COALESCE(up.trigger, '') || ' ' ||
                           COALESCE(up.content, '') || ' ' ||
                           COALESCE(up.rationale, '') || ' ' ||
                           COALESCE(up.source, '')
                       )
                       FROM user_playbooks up"""
                )
                self.conn.execute(
                    """INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
                       SELECT ap.rowid, reflexio_unicode_lexical_index(
                           COALESCE(ap.trigger, '') || ' ' ||
                           COALESCE(ap.content, '') || ' ' ||
                           COALESCE(ap.rationale, '')
                       )
                       FROM agent_playbooks ap"""
                )
                self.conn.execute(
                    "INSERT INTO interactions_unicode_fts(rowid, search_ngrams) "
                    "VALUES (?, '')",
                    (version_sentinel_rowid,),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

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
            CREATE VIRTUAL TABLE IF NOT EXISTS playbook_aggregation_clusters_vec USING vec0(
                embedding float[{dim}] distance_metric=cosine
            );
        """
        with self._lock:
            self.conn.executescript(vec_ddl)
            self.conn.commit()

    def _migrate_playbook_aggregation_agent_centroids(self) -> None:
        """Replace legacy user-vector means with canonical agent embeddings."""
        rows = self.conn.execute(
            "SELECT c.cluster_id, c.index_rowid, a.embedding "
            "FROM playbook_aggregation_cluster c JOIN agent_playbooks a "
            "ON a.agent_playbook_id=c.agent_playbook_id "
            "WHERE c.vector_sum IS NOT NULL AND c.index_rowid IS NOT NULL "
            "AND a.embedding IS NOT NULL "
            "AND trim(a.embedding) NOT IN ('', '[]')"
        ).fetchall()
        with self._lock:
            for row in rows:
                try:
                    embedding = json.loads(row[2])
                except (json.JSONDecodeError, TypeError):
                    continue
                if (
                    not isinstance(embedding, list)
                    or len(embedding) != self.embedding_dimensions
                    or not all(isinstance(value, int | float) for value in embedding)
                ):
                    continue
                self.conn.execute(
                    "UPDATE playbook_aggregation_cluster SET centroid=?, "
                    "vector_sum=NULL, embedding_dimension=? WHERE cluster_id=?",
                    (json.dumps(embedding), len(embedding), str(row[0])),
                )
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    (int(row[1]),),
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                    "VALUES (?, ?)",
                    (int(row[1]), json.dumps(embedding)),
                )
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

    def _migrate_session_outcomes_schema(self) -> None:
        """Rebuild legacy outcome rows with immutable v1 finalization identities."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                columns = {
                    row["name"]
                    for row in self.conn.execute(
                        "PRAGMA table_info(session_outcomes)"
                    ).fetchall()
                }
                table = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'session_outcomes'"
                ).fetchone()
                table_sql = (table["sql"] if table is not None else "") or ""
                required = {
                    "outcome_id",
                    "outcome_revision",
                    "outcome_contract_digest",
                    "finalized_trajectory_digest",
                }
                if required.issubset(columns) and "'unknown'" in table_sql:
                    rows_missing_subject_ref = self.conn.execute(
                        """SELECT user_id, session_id FROM session_outcomes
                           WHERE governance_subject_ref IS NULL
                              OR trim(governance_subject_ref) = ''"""
                    ).fetchall()
                    self.conn.executemany(
                        """UPDATE session_outcomes SET governance_subject_ref = ?
                           WHERE user_id = ? AND session_id = ?""",
                        [
                            (
                                self._subject_ref_for_user_id(str(row["user_id"])),
                                row["user_id"],
                                row["session_id"],
                            )
                            for row in rows_missing_subject_ref
                        ],
                    )
                    self.conn.commit()
                    return

                legacy_rows = self.conn.execute(
                    "SELECT * FROM session_outcomes"
                ).fetchall()
                self.conn.execute(
                    "ALTER TABLE session_outcomes RENAME TO session_outcomes_legacy"
                )
                self.conn.execute(
                    """CREATE TABLE session_outcomes (
                        outcome_id TEXT NOT NULL UNIQUE,
                        outcome_revision INTEGER NOT NULL CHECK (outcome_revision >= 1),
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
                        occurred_at INTEGER NOT NULL,
                        source TEXT NOT NULL,
                        label TEXT,
                        value REAL,
                        metadata TEXT,
                        outcome_contract_digest TEXT NOT NULL,
                        finalized_trajectory_digest TEXT NOT NULL,
                        governance_subject_ref TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        PRIMARY KEY (user_id, session_id)
                    )"""
                )
                for row in legacy_rows:
                    source = str(row["source"])
                    subject_ref = row["governance_subject_ref"]
                    if subject_ref is None or not str(subject_ref).strip():
                        subject_ref = self._subject_ref_for_user_id(str(row["user_id"]))
                    self.conn.execute(
                        """INSERT INTO session_outcomes (
                            outcome_id, outcome_revision, user_id, session_id, outcome,
                            occurred_at, source, label, value, metadata,
                            outcome_contract_digest, finalized_trajectory_digest,
                            governance_subject_ref, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            _legacy_session_outcome_id(
                                str(row["user_id"]), str(row["session_id"])
                            ),
                            1,
                            row["user_id"],
                            row["session_id"],
                            row["outcome"],
                            row["occurred_at"],
                            source,
                            row["label"],
                            row["value"],
                            row["metadata"],
                            outcome_contract_digest(
                                source=source,
                                schema_version=1,
                                allowed_values=_OUTCOME_ALLOWED_VALUES,
                                finalization_rule="first_write",
                            ),
                            trajectory_digest(
                                _canonical_session_snapshot(
                                    self.conn, str(row["session_id"])
                                )
                            ),
                            subject_ref,
                            row["created_at"],
                        ),
                    )
                self.conn.execute("DROP TABLE session_outcomes_legacy")
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_occurred_at "
                    "ON session_outcomes(occurred_at)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_session_id "
                    "ON session_outcomes(session_id)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_source_outcome "
                    "ON session_outcomes(source, outcome)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_label ON session_outcomes(label)"
                )
                self.conn.execute(
                    "CREATE INDEX idx_session_outcomes_subject_ref "
                    "ON session_outcomes(governance_subject_ref)"
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

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

        if "image_encoding" not in columns:
            logger.info("Adding image_encoding column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN image_encoding TEXT NOT NULL DEFAULT ''"
                )
                self.conn.commit()

        if "retrieved_learnings" not in columns:
            logger.info("Adding retrieved_learnings column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN retrieved_learnings TEXT"
                )
                self.conn.commit()

        if "token_count" not in columns:
            logger.info("Adding token_count column to interactions table.")
            with self._lock:
                self.conn.execute(
                    "ALTER TABLE interactions ADD COLUMN token_count INTEGER"
                )
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
        for table in (
            "profiles",
            "user_playbooks",
            "agent_playbooks",
            "agent_success_evaluation_result",
        ):
            cols = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "tags" not in cols:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN tags TEXT")
                logger.info("Added tags column to %s", table)
        self.conn.commit()

    def _migrate_profile_source_interaction_ids(self) -> None:
        """Add profile source interaction ids for provenance on existing DBs."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(profiles)").fetchall()
        }
        if "source_interaction_ids" not in cols:
            self.conn.execute(
                "ALTER TABLE profiles ADD COLUMN source_interaction_ids TEXT"
            )
            logger.info("Added source_interaction_ids column to profiles")
        self.conn.commit()

    def _migrate_interaction_window_indexes(self) -> None:
        """Add composite indexes used by sliding-window provenance lookups."""
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_user_created_at_desc "
            "ON interactions(user_id, created_at DESC, interaction_id DESC)"
        )
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

    def _migrate_retired_at(self) -> None:
        """Add nullable ``retired_at INTEGER`` GC column to tombstone-bearing tables.

        Backfill-safe: column is nullable with no default. Existing tombstones
        will have ``retired_at = NULL`` (conservative — GC T2 uses ``retired_at``
        as the age signal, so old tombstones without it are simply not yet eligible
        by the new clock; ops can backfill via T4 if needed).
        Also creates the covering index for GC queries (idempotent).
        """
        with self._lock:
            for table in ("profiles", "user_playbooks", "agent_playbooks"):
                cols = {
                    row["name"]
                    for row in self.conn.execute(
                        f"PRAGMA table_info({table})"  # noqa: S608
                    ).fetchall()
                }
                if "retired_at" not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN retired_at INTEGER"  # noqa: S608
                    )
                    logger.info("Added retired_at column to %s", table)
                self.conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_retired_at "  # noqa: S608
                    f"ON {table}(status, retired_at)"
                )
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
                    model_name       TEXT,
                    provider         TEXT,
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
            for col in (
                "from_status",
                "to_status",
                "status_namespace",
                "model_name",
                "provider",
            ):
                if col not in existing_cols:
                    self.conn.execute(
                        f"ALTER TABLE lineage_event ADD COLUMN {col} TEXT"  # noqa: S608
                    )
                    logger.info("Added %s column to lineage_event", col)
            self.conn.commit()

    def _migrate_playbook_optimization_candidate_metadata(self) -> None:
        """Add metadata_json to legacy optimizer candidate tables when missing."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_candidates)"
            ).fetchall()
        }
        if not cols:
            return
        if "metadata_json" not in cols:
            self.conn.execute(
                "ALTER TABLE playbook_optimization_candidates "
                "ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
            logger.info(
                "Added metadata_json column to playbook_optimization_candidates"
            )
        self.conn.commit()

    def _migrate_playbook_optimization_job_columns(self) -> None:
        """Add durable optimizer identity, lease, stage, and digest columns."""
        existing_cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_jobs)"
            ).fetchall()
        }
        if not existing_cols:
            return
        columns = {
            "optimizer_kind": "TEXT",
            "discovery_key": "TEXT",
            "attempt_key": "TEXT",
            "lease_owner": "TEXT",
            "lease_fence": "INTEGER NOT NULL DEFAULT 0",
            "lease_expires_at": "INTEGER",
            "stage": "TEXT",
            "terminal_outcome": "TEXT",
            "expected_population_manifest_digest": "TEXT",
            "generation_selection_manifest_digest": "TEXT",
            "replay_manifest_digest": "TEXT",
            "candidate_content_digest": "TEXT",
            "search_projection_digest": "TEXT",
            "publication_scope_digest": "TEXT",
        }
        for column, definition in columns.items():
            if column not in existing_cols:
                self.conn.execute(
                    f"ALTER TABLE playbook_optimization_jobs "
                    f"ADD COLUMN {column} {definition}"  # noqa: S608
                )
        self.conn.commit()

    def _migrate_user_playbook_publication_staging_columns(self) -> None:
        """Add frozen incumbent CAS fields to legacy publication staging."""
        existing_cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(user_playbook_publication_staging)"
            ).fetchall()
        }
        if not existing_cols:
            return
        columns = {
            "incumbent_content_digest": "TEXT",
            "incumbent_trigger": "TEXT",
            "incumbent_semantic_digest": "TEXT",
        }
        for column, definition in columns.items():
            if column not in existing_cols:
                self.conn.execute(
                    "ALTER TABLE user_playbook_publication_staging "
                    f"ADD COLUMN {column} {definition}"  # noqa: S608
                )
        self.conn.commit()

    def _classify_legacy_playbook_optimization_jobs(self) -> None:
        """Classify legacy optimizer history without choosing ambiguous rule order."""
        columns = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(playbook_optimization_jobs)"
            ).fetchall()
        }
        if "optimizer_kind" not in columns:
            return
        for index_name in (
            "uq_poj_active_discovery",
            "uq_poj_active_attempt",
            "uq_poj_active_target",
        ):
            self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")  # noqa: S608
        self.conn.execute(
            """
            WITH signatures AS (
                SELECT
                    jobs.job_id,
                    (
                        EXISTS (
                            SELECT 1
                            FROM playbook_optimization_events AS events
                            WHERE events.job_id = jobs.job_id
                              AND events.event_type LIKE 'offline_tuner_%'
                        )
                        OR (
                            json_valid(jobs.metadata_json)
                            AND json_type(jobs.metadata_json, '$.offline_tuner')
                                IS NOT NULL
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM playbook_optimization_candidates AS candidates
                            WHERE candidates.job_id = jobs.job_id
                              AND json_valid(candidates.metadata_json)
                              AND (
                                  json_type(
                                      candidates.metadata_json,
                                      '$.offline_tuner_metrics'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.rollback_baseline'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.frozen_selection_set'
                                  ) IS NOT NULL
                                  OR json_type(
                                      candidates.metadata_json,
                                      '$.proposed_edit'
                                  ) IS NOT NULL
                              )
                        )
                    ) AS tuner_signature,
                    (
                        json_valid(jobs.metadata_json)
                        AND json_type(jobs.metadata_json, '$.source_window_count')
                            IS NOT NULL
                        AND json_type(jobs.metadata_json, '$.train_window_count')
                            IS NOT NULL
                        AND json_type(
                            jobs.metadata_json,
                            '$.validation_window_count'
                        ) IS NOT NULL
                    ) AS gepa_signature
                FROM playbook_optimization_jobs AS jobs
                WHERE jobs.optimizer_kind IS NULL
            )
            UPDATE playbook_optimization_jobs
            SET optimizer_kind = CASE
                WHEN tuner_signature AND NOT gepa_signature
                    THEN 'offline_tuner_legacy'
                WHEN gepa_signature AND NOT tuner_signature THEN 'gepa'
                ELSE 'optimizer_legacy_unknown'
            END
            FROM signatures
            WHERE playbook_optimization_jobs.job_id = signatures.job_id
            """
        )
        self.conn.execute(
            """
            WITH ranked AS (
                SELECT
                    job_id,
                    row_number() OVER (
                        PARTITION BY optimizer_kind, target_kind, target_id
                        ORDER BY created_at, job_id
                    ) AS target_rank,
                    CASE WHEN discovery_key IS NOT NULL THEN row_number() OVER (
                        PARTITION BY optimizer_kind, discovery_key
                        ORDER BY created_at, job_id
                    ) END AS discovery_rank,
                    CASE WHEN attempt_key IS NOT NULL THEN row_number() OVER (
                        PARTITION BY optimizer_kind, attempt_key
                        ORDER BY created_at, job_id
                    ) END AS attempt_rank
                FROM playbook_optimization_jobs
                WHERE status IN ('pending', 'running')
            )
            UPDATE playbook_optimization_jobs
            SET status = 'skipped',
                decision_reason = 'retired_duplicate_legacy_active_job',
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = CAST(strftime('%s', 'now') AS INTEGER)
            WHERE job_id IN (
                SELECT job_id
                FROM ranked
                WHERE target_rank > 1
                   OR discovery_rank > 1
                   OR attempt_rank > 1
            )
            """
        )
        self.conn.execute(
            """
            UPDATE playbook_optimization_jobs
            SET status = 'skipped',
                decision_reason = 'retired_by_replay_redesign',
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = CAST(strftime('%s', 'now') AS INTEGER)
            WHERE optimizer_kind IN (
                    'offline_tuner_legacy',
                    'optimizer_legacy_unknown'
                )
              AND status IN ('pending', 'running')
            """
        )
        self.conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_discovery
               ON playbook_optimization_jobs(optimizer_kind, discovery_key)
               WHERE status IN ('pending', 'running')
                 AND discovery_key IS NOT NULL"""
        )
        self.conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_attempt
               ON playbook_optimization_jobs(optimizer_kind, attempt_key)
               WHERE status IN ('pending', 'running')
                 AND attempt_key IS NOT NULL"""
        )
        self.conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_target
               ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
               WHERE status IN ('pending', 'running')"""
        )
        self.conn.commit()

    def _enforce_playbook_optimization_job_constraints(self) -> None:
        """Rebuild upgraded optimizer tables so legacy values receive fresh checks."""
        table_sql_row = self.conn.execute(
            """SELECT sql FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_jobs'"""
        ).fetchone()
        if table_sql_row is None:
            return
        table_sql = table_sql_row["sql"]
        required_checks = (
            "CHECK (optimizer_kind IN",
            "CHECK (stage IS NULL OR stage IN",
            "CHECK (terminal_outcome IS NULL OR terminal_outcome IN",
            "'governance_erased'",
        )
        if all(check in table_sql for check in required_checks):
            return
        foreign_keys_enabled = bool(
            self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        )
        self.conn.commit()
        if foreign_keys_enabled:
            self.conn.execute("PRAGMA foreign_keys=OFF")
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute("DROP TABLE IF EXISTS playbook_optimization_jobs_new")
            self.conn.execute(
                """
            CREATE TABLE playbook_optimization_jobs_new (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimizer_kind TEXT NOT NULL DEFAULT 'optimizer_legacy_unknown'
                    CHECK (optimizer_kind IN (
                        'gepa',
                        'offline_tuner_replay',
                        'offline_tuner_legacy',
                        'optimizer_legacy_unknown'
                    )),
                target_kind TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                best_candidate_id INTEGER,
                successor_target_id INTEGER,
                decision_reason TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                discovery_key TEXT,
                attempt_key TEXT,
                lease_owner TEXT,
                lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
                lease_expires_at INTEGER,
                stage TEXT CHECK (stage IS NULL OR stage IN (
                    'evidence_frozen',
                    'candidate_generated',
                    'replay_running',
                    'replay_evaluated',
                    'publishing',
                    'applied',
                    'abstained',
                    'failed'
                )),
                terminal_outcome TEXT CHECK (terminal_outcome IS NULL OR terminal_outcome IN (
                    'applied',
                    'insufficient_negative_evidence',
                    'insufficient_positive_evidence',
                    'insufficient_coverage',
                    'replay_unsupported',
                    'deployment_unsupported',
                    'incomplete_replay_scope',
                    'insufficient_replay_cases',
                    'replay_inconclusive',
                    'candidate_regressed',
                    'candidate_did_not_improve',
                    'incumbent_changed',
                    'generation_failed',
                    'replay_failed',
                    'publication_failed',
                    'governance_erased'
                )),
                expected_population_manifest_digest TEXT,
                generation_selection_manifest_digest TEXT,
                replay_manifest_digest TEXT,
                candidate_content_digest TEXT,
                search_projection_digest TEXT,
                publication_scope_digest TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
            )
            self.conn.execute(
                """
            INSERT INTO playbook_optimization_jobs_new (
                job_id, optimizer_kind, target_kind, target_id, status,
                best_candidate_id, successor_target_id, decision_reason,
                metadata_json, discovery_key, attempt_key, lease_owner,
                lease_fence, lease_expires_at, stage, terminal_outcome,
                expected_population_manifest_digest,
                generation_selection_manifest_digest, replay_manifest_digest,
                candidate_content_digest, search_projection_digest,
                publication_scope_digest, created_at, updated_at
            ) SELECT
                job_id, optimizer_kind, target_kind, target_id, status,
                best_candidate_id, successor_target_id, decision_reason,
                metadata_json, discovery_key, attempt_key, lease_owner,
                lease_fence, lease_expires_at, stage, terminal_outcome,
                expected_population_manifest_digest,
                generation_selection_manifest_digest, replay_manifest_digest,
                candidate_content_digest, search_projection_digest,
                publication_scope_digest, created_at, updated_at
            FROM playbook_optimization_jobs;
            """
            )
            self.conn.execute("DROP TABLE playbook_optimization_jobs")
            self.conn.execute(
                "ALTER TABLE playbook_optimization_jobs_new "
                "RENAME TO playbook_optimization_jobs"
            )
            self.conn.execute(
                "CREATE INDEX idx_poj_target "
                "ON playbook_optimization_jobs(target_kind, target_id)"
            )
            self.conn.execute(
                "CREATE INDEX idx_poj_status ON playbook_optimization_jobs(status)"
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_discovery
                ON playbook_optimization_jobs(optimizer_kind, discovery_key)
                WHERE status IN ('pending', 'running') AND discovery_key IS NOT NULL"""
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_attempt
                ON playbook_optimization_jobs(optimizer_kind, attempt_key)
                WHERE status IN ('pending', 'running') AND attempt_key IS NOT NULL"""
            )
            self.conn.execute(
                """CREATE UNIQUE INDEX uq_poj_active_target
                ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
                WHERE status IN ('pending', 'running')"""
            )
            violations = self.conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(
                    f"foreign key check failed after optimizer migration: {violations}"
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.execute(
                f"PRAGMA foreign_keys={'ON' if foreign_keys_enabled else 'OFF'}"
            )

    def _migrate_retire_profile_change_logs(self) -> None:
        """Retire the frozen ``profile_change_logs`` table via a reversible RENAME.

        Lineage B3 Task 8: the legacy change log is fully de-referenced (no
        readers, writers, or GDPR delete callers remain) and the view is served
        from reconstruction. We rename the table out of the way now and DROP it
        in a later migration after the recovery window — keeping the data
        recoverable in the interim.

        Idempotent: SQLite has no ``RENAME ... IF EXISTS``, so we guard on the
        source table's presence and no-op once it has been renamed. The
        ``CREATE TABLE`` for ``profile_change_logs`` was deleted from ``_DDL`` so
        ``executescript(_DDL)`` does not recreate an empty table after the rename.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("profile_change_logs",),
            ).fetchone()
            if row is None:
                return
            target_row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("profile_change_logs_retired_20260623",),
            ).fetchone()
            if target_row is not None:
                logger.info(
                    "Retired target profile_change_logs_retired_20260623 already exists;"
                    " skipping rename (idempotent no-op)."
                )
                return
            self.conn.execute(
                "ALTER TABLE profile_change_logs "
                "RENAME TO profile_change_logs_retired_20260623"
            )
            logger.info(
                "Renamed profile_change_logs to "
                "profile_change_logs_retired_20260623 (B3 Task 8 retirement)"
            )
            self.conn.commit()

    def _migrate_retire_playbook_aggregation_change_logs(self) -> None:
        """Retire the frozen ``playbook_aggregation_change_logs`` table via a reversible RENAME.

        Lineage Track B Task 4: the legacy change log is fully de-referenced (no
        readers, writers, or GDPR delete callers remain) and the view is served
        from reconstruction. We rename the table out of the way now and DROP it
        in a later migration after the recovery window — keeping the data
        recoverable in the interim.

        Idempotent: SQLite has no ``RENAME ... IF EXISTS``, so we guard on the
        source table's presence and no-op once it has been renamed. The
        ``CREATE TABLE`` for ``playbook_aggregation_change_logs`` was deleted from
        ``_DDL`` so ``executescript(_DDL)`` does not recreate an empty table after
        the rename.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("playbook_aggregation_change_logs",),
            ).fetchone()
            if row is None:
                return
            target_row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                ("playbook_aggregation_change_logs_retired_20260624",),
            ).fetchone()
            if target_row is not None:
                logger.info(
                    "Retired target playbook_aggregation_change_logs_retired_20260624"
                    " already exists; skipping rename (idempotent no-op)."
                )
                return
            self.conn.execute(
                "ALTER TABLE playbook_aggregation_change_logs "
                "RENAME TO playbook_aggregation_change_logs_retired_20260624"
            )
            logger.info(
                "Renamed playbook_aggregation_change_logs to "
                "playbook_aggregation_change_logs_retired_20260624 "
                "(Track B Task 4 retirement)"
            )
            self.conn.commit()

    def _migrate_learning_jobs(self) -> None:
        """Create the learning_jobs table + partial indexes if missing (idempotent).

        Runs ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS`` only —
        both are no-ops when the table / indexes already exist.  New columns added in
        subsequent tasks are backfilled via ``PRAGMA table_info`` + ``ALTER TABLE … ADD
        COLUMN`` (mirroring ``_migrate_lineage_event_table``), because
        ``CREATE TABLE IF NOT EXISTS`` silently skips the DDL on existing databases.

        Called at the end of migrate() so the table is always present on startup.
        """
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS learning_jobs (
                    job_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    job_type TEXT NOT NULL DEFAULT 'learning',
                    latest_request_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    claimed_by TEXT,
                    claim_token TEXT,
                    claim_expires_at TEXT,
                    covers_through TEXT,
                    force_extraction INTEGER NOT NULL DEFAULT 0,
                    skip_aggregation INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS learning_jobs_coalesce
                    ON learning_jobs (org_id, user_id, job_type) WHERE status = 'pending';
                -- Recreate the poll index with 'failed' in the predicate. CREATE
                -- INDEX IF NOT EXISTS is a no-op on an existing index with the old
                -- ('pending','claimed') predicate, so DROP first to migrate it.
                DROP INDEX IF EXISTS learning_jobs_poll;
                CREATE INDEX IF NOT EXISTS learning_jobs_poll
                    ON learning_jobs (created_at) WHERE status IN ('pending','failed','claimed');
            """)
            # Backfill columns added after the initial release (existing DBs skip
            # CREATE TABLE IF NOT EXISTS so these must be applied separately).
            existing_cols = {
                row["name"]
                for row in self.conn.execute(
                    "PRAGMA table_info(learning_jobs)"
                ).fetchall()
            }
            if "force_extraction" not in existing_cols:
                self.conn.execute(
                    "ALTER TABLE learning_jobs ADD COLUMN force_extraction INTEGER NOT NULL DEFAULT 0"
                )
            if "skip_aggregation" not in existing_cols:
                self.conn.execute(
                    "ALTER TABLE learning_jobs ADD COLUMN skip_aggregation INTEGER NOT NULL DEFAULT 0"
                )
            self.conn.commit()

    def learning_jobs_columns(self) -> list[str]:
        """Return the column names of the learning_jobs table."""
        with self._lock:
            rows = self.conn.execute("PRAGMA table_info(learning_jobs)").fetchall()
        return [row["name"] for row in rows]

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

    def _migrate_request_retrieval_experiment(self) -> None:
        """Add nullable retrieval experiment attribution to existing requests."""
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(requests)").fetchall()
        }
        if not cols:
            return
        if "retrieval_experiment_id" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN retrieval_experiment_id TEXT"
            )
        if "retrieval_experiment_arm" not in cols:
            self.conn.execute(
                "ALTER TABLE requests ADD COLUMN retrieval_experiment_arm TEXT"
            )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment "
            "ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id)"
        )
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
        retrieval_experiment_id_expr = (
            "retrieval_experiment_id" if "retrieval_experiment_id" in cols else "NULL"
        )
        retrieval_experiment_arm_expr = (
            "retrieval_experiment_arm" if "retrieval_experiment_arm" in cols else "NULL"
        )
        governance_subject_ref_expr = (
            "governance_subject_ref" if "governance_subject_ref" in cols else "NULL"
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
                evaluation_only INTEGER NOT NULL DEFAULT 0,
                retrieval_experiment_id TEXT,
                retrieval_experiment_arm TEXT,
                governance_subject_ref TEXT
            );
            INSERT INTO requests_new
                (
                    request_id,
                    user_id,
                    created_at,
                    source,
                    agent_version,
                    session_id,
                    evaluation_only,
                    retrieval_experiment_id,
                    retrieval_experiment_arm,
                    governance_subject_ref
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
                {evaluation_only_expr},
                {retrieval_experiment_id_expr},
                {retrieval_experiment_arm_expr},
                {governance_subject_ref_expr}
            FROM requests;
            DROP TABLE requests;
            ALTER TABLE requests_new RENAME TO requests;
            CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
            CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
            CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
            CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment
                ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id);
            CREATE INDEX IF NOT EXISTS idx_requests_governance_subject_ref
                ON requests(governance_subject_ref);
            """
        )
        self.conn.commit()
        logger.info("Migrated requests.session_id to required non-empty values")

    def _migrate_eval_result_user_id(self) -> None:
        """Add user_id to session evaluation results for per-user identity."""
        cols = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(agent_success_evaluation_result)"
            ).fetchall()
        }
        if not cols:
            return
        if "user_id" not in cols:
            self.conn.execute(
                "ALTER TABLE agent_success_evaluation_result "
                "ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
            )
            logger.info("Added user_id column to agent_success_evaluation_result")
        self.conn.execute("DROP INDEX IF EXISTS idx_eval_identity_created_at_desc")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_identity_created_at_desc "
            "ON agent_success_evaluation_result"
            "(user_id, session_id, evaluation_name, agent_version, created_at DESC)"
        )
        self.conn.commit()

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

    def _migrate_retrieved_learning_interaction_identity(self) -> None:
        """Rebuild legacy session-learning verdicts for occurrence identity.

        Existing rows are preserved with nullable interaction fields. Fresh
        evaluator runs replace a session snapshot with fully attributed rows.
        """
        columns = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(retrieved_learning_evaluation)"
            ).fetchall()
        }
        if not columns or {
            "interaction_id",
            "interaction_created_at",
        }.issubset(columns):
            return
        self.conn.executescript("""
            CREATE TABLE retrieved_learning_evaluation_new (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                agent_version TEXT NOT NULL DEFAULT '',
                interaction_id INTEGER,
                interaction_created_at INTEGER,
                kind TEXT NOT NULL,
                learning_id TEXT NOT NULL,
                is_relevant INTEGER,
                relevance_reason TEXT NOT NULL DEFAULT '',
                impact TEXT,
                impact_reason TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                governance_subject_ref TEXT,
                UNIQUE (
                    user_id, session_id, interaction_id, kind, learning_id
                )
            );
            INSERT INTO retrieved_learning_evaluation_new (
                result_id, user_id, session_id, agent_version, kind,
                learning_id, is_relevant, relevance_reason, impact,
                impact_reason, created_at, governance_subject_ref
            )
            SELECT
                result_id, user_id, session_id, agent_version, kind,
                learning_id, is_relevant, relevance_reason, impact,
                impact_reason, created_at, governance_subject_ref
            FROM retrieved_learning_evaluation;
            DROP TABLE retrieved_learning_evaluation;
            ALTER TABLE retrieved_learning_evaluation_new
                RENAME TO retrieved_learning_evaluation;
            CREATE INDEX IF NOT EXISTS idx_rle_created_at_result_id
                ON retrieved_learning_evaluation(created_at DESC, result_id DESC);
            CREATE INDEX IF NOT EXISTS idx_rle_interaction_created_at
                ON retrieved_learning_evaluation(
                    interaction_created_at DESC, interaction_id DESC, result_id DESC
                );
            CREATE INDEX IF NOT EXISTS idx_rle_session_id
                ON retrieved_learning_evaluation(session_id);
            CREATE INDEX IF NOT EXISTS idx_rle_subject_ref
                ON retrieved_learning_evaluation(governance_subject_ref);
        """)
        self.conn.commit()
        logger.info(
            "Migrated retrieved-learning verdicts to interaction occurrence identity"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            if self._own_transaction():
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
        """Generate an embedding with model-specific input formatting.

        Args:
            text: The text to embed.
            purpose: Either ``"document"`` (stored embeddings) or ``"query"``
                (search-time embeddings).

        Returns:
            The embedding vector as a list of floats.
        """
        try:
            return self.llm_client.get_embedding(
                embedding_input(
                    text,
                    model_name=self.embedding_model_name,
                    purpose=purpose,
                ),
                self.embedding_model_name,
                self.embedding_dimensions,
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
                model_name=self._expansion_model_name,
            )
            result = expander.expand(content)
            return result.expanded_text or None
        except Exception:
            logger.warning("Document expansion failed", exc_info=True)
            return None

    def _current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    # ------------------------------------------------------------------
    # Per-user data clear
    # ------------------------------------------------------------------

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        """Per-``user_id`` row deletion across all user-scoped tables.

        Overrides the BaseStorage default with an optimized SQL implementation.
        Removes interactions, user playbooks, profiles, and requests scoped to
        the user. Intentionally does NOT touch ``agent_playbooks`` — they are
        the cross-project rollup of skills and have no ``user_id`` column.

        Also cleans up FTS and vector sidecars for the user's rows so
        subsequent searches don't surface deleted data.

        **Lineage-aware erasure for profiles and user_playbooks:**
        Rows that are tombstones (``merged_into`` or ``superseded_by`` is set)
        *or* are pointed-to by another row (``has_inbound_lineage_refs`` returns
        True) are **content-purged** (skeleton kept, body blanked) rather than
        hard-deleted. This preserves chain resolution across user erasures.
        Standalone rows with no lineage involvement are hard-deleted as before.

        The purge/delete decision delegates to
        ``BaseStorage._partition_purge_vs_delete`` so the logic is defined once
        and shared with the default ``clear_user_data`` implementation used by
        Supabase/Postgres backends.

        **Commit ordering (atomicity invariant):**
        The hard-deletes for interactions, requests, and the delete-sets of
        profiles/user_playbooks are committed in one transaction first. Then
        ``purge_content`` is called for each purge-eligible row — each call
        commits atomically on its own. This two-phase approach is required
        because ``purge_content`` issues its own ``conn.commit()``, and nesting
        it inside the outer transaction would prematurely flush the still-pending
        hard-DELETEs.

        Args:
            user_id (str): The user id whose rows should be deleted.

        Returns:
            dict[str, int]: Per-entity counts with keys ``session_outcomes``,
                ``interactions``, ``user_playbooks``, ``profiles``, ``requests``,
                ``purged_profiles``, and ``purged_user_playbooks``.
                ``profiles`` and ``user_playbooks`` reflect hard-deleted counts;
                purged rows are counted separately.
        """
        with self._lock:
            # ------------------------------------------------------------------
            # Phase 1: snapshot all user-scoped ids before any mutations.
            # ------------------------------------------------------------------
            interaction_ids = [
                r["interaction_id"]
                for r in self.conn.execute(
                    "SELECT interaction_id FROM interactions WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            raw_upb_ids = [
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
            subject_ref = self._subject_ref_for_user_id(user_id)

            # Build a rowid lookup for FTS/vec cleanup (SQLite-specific need).
            profile_rowid_by_id: dict[str, int] = {
                r["profile_id"]: r["rowid"] for r in profile_rows
            }
            all_profile_ids = list(profile_rowid_by_id.keys())

            # ------------------------------------------------------------------
            # Phase 2: partition profiles and user_playbooks into purge vs delete.
            # Delegates to the shared BaseStorage helper so the decision logic
            # is defined once and reused by all backends.
            # ------------------------------------------------------------------
            purge_profile_ids, delete_profile_ids = self._partition_purge_vs_delete(
                "profile", all_profile_ids
            )
            purge_upb_str_ids, delete_upb_str_ids = self._partition_purge_vs_delete(
                "user_playbook", [str(uid) for uid in raw_upb_ids]
            )
            purge_upb_ids = [int(s) for s in purge_upb_str_ids]
            delete_upb_ids = [int(s) for s in delete_upb_str_ids]

            # Rowids for the delete-set only (purge_content handles its own cleanup).
            delete_profile_rowids = [
                profile_rowid_by_id[pid]
                for pid in delete_profile_ids
                if pid in profile_rowid_by_id
            ]

            # ------------------------------------------------------------------
            # Phase 3: FTS and vector cleanup — only for the delete-sets
            # (purge_content handles its own index cleanup for purged rows).
            # Use _delete_in_chunks to stay under SQLite's SQLITE_MAX_VARIABLE_NUMBER
            # limit for large user datasets.
            # ------------------------------------------------------------------
            self._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
            self._delete_in_chunks("user_playbooks_fts", "rowid", delete_upb_ids)
            self._delete_in_chunks("profiles_fts", "profile_id", delete_profile_ids)

            if self._has_sqlite_vec:
                self._delete_in_chunks("interactions_vec", "rowid", interaction_ids)
                self._delete_in_chunks("user_playbooks_vec", "rowid", delete_upb_ids)
                self._delete_in_chunks("profiles_vec", "rowid", delete_profile_rowids)

            # ------------------------------------------------------------------
            # Phase 4: hard-delete the delete-sets and all interactions/requests.
            # ------------------------------------------------------------------
            interactions_cur = self.conn.execute(
                "DELETE FROM interactions WHERE user_id = ?", (user_id,)
            )
            session_outcomes_cur = self.conn.execute(
                """DELETE FROM session_outcomes
                   WHERE user_id = ? OR governance_subject_ref = ?""",
                (user_id, subject_ref),
            )
            requests_cur = self.conn.execute(
                "DELETE FROM requests WHERE user_id = ?", (user_id,)
            )
            upb_deleted_count = 0
            if delete_upb_ids:
                # Clean up source-window join rows before deleting the parent rows.
                self._delete_source_windows_for_user_playbook_ids(delete_upb_ids)
                self._delete_in_chunks(
                    "user_playbooks", "user_playbook_id", delete_upb_ids
                )
                # rowcount not available from _delete_in_chunks; derive from list length
                # (all ids came from a pre-snapshot so they exist at delete time).
                upb_deleted_count = len(delete_upb_ids)
            profile_deleted_count = 0
            if delete_profile_ids:
                self._delete_in_chunks("profiles", "profile_id", delete_profile_ids)
                profile_deleted_count = len(delete_profile_ids)

            # Commit the hard-deletes before calling purge_content, because
            # purge_content issues its own conn.commit() and nesting it here
            # would prematurely flush the still-pending deletes.
            self.conn.commit()

            # Phase 5: content-purge the purge-sets WITHOUT releasing the lock,
            # so erase-eligible rows are never observable by another thread with
            # PII still intact between the hard-delete commit and the purge. Each
            # purge_content call self-commits; self._lock is an RLock so its
            # internal ``with self._lock`` re-acquires cleanly, and the commit
            # above already closed the outer transaction (no flush hazard).
            for pid in purge_profile_ids:
                self.purge_content(entity_type="profile", entity_id=str(pid))
            for upid in purge_upb_ids:
                self.purge_content(entity_type="user_playbook", entity_id=str(upid))

        return {
            "session_outcomes": session_outcomes_cur.rowcount,
            "interactions": interactions_cur.rowcount,
            "user_playbooks": upb_deleted_count,
            "profiles": profile_deleted_count,
            "requests": requests_cur.rowcount,
            "purged_profiles": len(purge_profile_ids),
            "purged_user_playbooks": len(purge_upb_ids),
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
    source_interaction_ids TEXT,
    source_span TEXT,
    notes TEXT,
    reader_angle TEXT,
    merged_into TEXT,
    superseded_by TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    retired_at INTEGER
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
    token_count INTEGER,
    user_action TEXT NOT NULL DEFAULT 'none',
    user_action_description TEXT NOT NULL DEFAULT '',
    interacted_image_url TEXT NOT NULL DEFAULT '',
    image_encoding TEXT NOT NULL DEFAULT '',
    shadow_content TEXT NOT NULL DEFAULT '',
    expert_content TEXT NOT NULL DEFAULT '',
    tools_used TEXT,
    citations TEXT,
    retrieved_learnings TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_request_id ON interactions(request_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_user_created_at_desc
    ON interactions(user_id, created_at DESC, interaction_id DESC);

CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_version TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL CHECK (trim(session_id) != ''),
    evaluation_only INTEGER NOT NULL DEFAULT 0,
    retrieval_experiment_id TEXT,
    retrieval_experiment_arm TEXT,
    CHECK (
        (retrieval_experiment_id IS NULL AND retrieval_experiment_arm IS NULL)
        OR
        (retrieval_experiment_id IS NOT NULL AND retrieval_experiment_arm IN ('treatment', 'holdout'))
    )
);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_session_id ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_session_created_at_asc
    ON requests(session_id, created_at ASC, request_id ASC);
CREATE INDEX IF NOT EXISTS idx_requests_retrieval_experiment
    ON requests(retrieval_experiment_id, user_id, session_id, created_at, request_id);

CREATE TABLE IF NOT EXISTS session_outcomes (
    outcome_id TEXT NOT NULL UNIQUE,
    outcome_revision INTEGER NOT NULL CHECK (outcome_revision >= 1),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
    occurred_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    label TEXT,
    value REAL,
    metadata TEXT,
    outcome_contract_digest TEXT NOT NULL,
    finalized_trajectory_digest TEXT NOT NULL,
    governance_subject_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_occurred_at ON session_outcomes(occurred_at);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_session_id ON session_outcomes(session_id);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_source_outcome ON session_outcomes(source, outcome);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_label ON session_outcomes(label);
CREATE INDEX IF NOT EXISTS idx_session_outcomes_subject_ref ON session_outcomes(governance_subject_ref);

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
    superseded_by INTEGER,
    retired_at INTEGER
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
    superseded_by INTEGER,
    retired_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_playbook_name ON agent_playbooks(playbook_name);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_agent_version ON agent_playbooks(agent_version);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_status ON agent_playbooks(status);
CREATE INDEX IF NOT EXISTS idx_agent_playbooks_created_at ON agent_playbooks(created_at);

CREATE TABLE IF NOT EXISTS agent_success_evaluation_result (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
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
    tags TEXT,
    embedding TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version ON agent_success_evaluation_result(agent_version);
CREATE INDEX IF NOT EXISTS idx_eval_created_at ON agent_success_evaluation_result(created_at);
CREATE INDEX IF NOT EXISTS idx_eval_created_at_desc
    ON agent_success_evaluation_result(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_agent_version_created_at_desc
    ON agent_success_evaluation_result(agent_version, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_identity_created_at_desc
    ON agent_success_evaluation_result(user_id, session_id, evaluation_name, agent_version, created_at DESC);

CREATE TABLE IF NOT EXISTS retrieved_learning_evaluation (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    agent_version TEXT NOT NULL DEFAULT '',
    interaction_id INTEGER,
    interaction_created_at INTEGER,
    kind TEXT NOT NULL,
    learning_id TEXT NOT NULL,
    is_relevant INTEGER,
    relevance_reason TEXT NOT NULL DEFAULT '',
    impact TEXT,
    impact_reason TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    governance_subject_ref TEXT,
    UNIQUE (user_id, session_id, interaction_id, kind, learning_id)
);
CREATE INDEX IF NOT EXISTS idx_rle_created_at_result_id
    ON retrieved_learning_evaluation(created_at DESC, result_id DESC);
CREATE INDEX IF NOT EXISTS idx_rle_interaction_created_at
    ON retrieved_learning_evaluation(
        interaction_created_at DESC, interaction_id DESC, result_id DESC
    );
CREATE INDEX IF NOT EXISTS idx_rle_session_id
    ON retrieved_learning_evaluation(session_id);
CREATE INDEX IF NOT EXISTS idx_rle_subject_ref
    ON retrieved_learning_evaluation(governance_subject_ref);

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
    optimizer_kind TEXT NOT NULL DEFAULT 'optimizer_legacy_unknown'
        CHECK (optimizer_kind IN (
            'gepa',
            'offline_tuner_replay',
            'offline_tuner_legacy',
            'optimizer_legacy_unknown'
        )),
    target_kind TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    best_candidate_id INTEGER,
    successor_target_id INTEGER,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    discovery_key TEXT,
    attempt_key TEXT,
    lease_owner TEXT,
    lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
    lease_expires_at INTEGER,
    stage TEXT CHECK (stage IS NULL OR stage IN (
        'evidence_frozen',
        'candidate_generated',
        'replay_running',
        'replay_evaluated',
        'publishing',
        'applied',
        'abstained',
        'failed'
    )),
    terminal_outcome TEXT CHECK (terminal_outcome IS NULL OR terminal_outcome IN (
        'applied',
        'insufficient_negative_evidence',
        'insufficient_positive_evidence',
        'insufficient_coverage',
        'replay_unsupported',
        'deployment_unsupported',
        'incomplete_replay_scope',
        'insufficient_replay_cases',
        'replay_inconclusive',
        'candidate_regressed',
        'candidate_did_not_improve',
        'incumbent_changed',
        'generation_failed',
        'replay_failed',
        'publication_failed',
        'governance_erased'
    )),
    expected_population_manifest_digest TEXT,
    generation_selection_manifest_digest TEXT,
    replay_manifest_digest TEXT,
    candidate_content_digest TEXT,
    search_projection_digest TEXT,
    publication_scope_digest TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_poj_target ON playbook_optimization_jobs(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_poj_status ON playbook_optimization_jobs(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_discovery
    ON playbook_optimization_jobs(optimizer_kind, discovery_key)
    WHERE status IN ('pending', 'running') AND discovery_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_attempt
    ON playbook_optimization_jobs(optimizer_kind, attempt_key)
    WHERE status IN ('pending', 'running') AND attempt_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_poj_active_target
    ON playbook_optimization_jobs(optimizer_kind, target_kind, target_id)
    WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS case_first_exposure (
    equivalence_group_id TEXT PRIMARY KEY,
    first_role TEXT NOT NULL CHECK (first_role IN (
        'generation', 'improvement', 'preservation', 'safety'
    )),
    owner_discovery_key TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS case_exposure_uses (
    job_id INTEGER NOT NULL,
    equivalence_group_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN (
        'generation', 'improvement', 'preservation', 'safety'
    )),
    ownership TEXT NOT NULL CHECK (ownership IN (
        'self_generation', 'self_improvement', 'self_preservation',
        'self_safety', 'foreign'
    )),
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (job_id, equivalence_group_id, role),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_case_exposure_uses_group
    ON case_exposure_uses(equivalence_group_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
        'expected_population_manifest',
        'generation_selection',
        'replay_manifest',
        'candidate',
        'candidate_search_projection'
    )),
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (job_id, artifact_kind),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_poa_job
    ON playbook_optimization_artifacts(job_id);

CREATE TABLE IF NOT EXISTS playbook_optimization_candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
    aggregate_score REAL,
    is_winner INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS user_playbook_publication_claims (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    owner TEXT NOT NULL,
    publication_fence INTEGER NOT NULL CHECK (publication_fence >= 1),
    worker_fence INTEGER NOT NULL CHECK (worker_fence >= 1),
    consumed INTEGER NOT NULL DEFAULT 0 CHECK (consumed IN (0, 1)),
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_playbook_publication_staging (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    attempt_key TEXT NOT NULL,
    claim_owner TEXT NOT NULL,
    publication_fence INTEGER NOT NULL,
    worker_fence INTEGER NOT NULL,
    incumbent_user_playbook_id INTEGER NOT NULL,
    incumbent_content_digest TEXT NOT NULL,
    incumbent_trigger TEXT,
    incumbent_semantic_digest TEXT NOT NULL,
    revised_content TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    projection_json TEXT NOT NULL,
    projection_digest TEXT NOT NULL,
    proof_json TEXT NOT NULL,
    proof_digest TEXT NOT NULL,
    subject_epochs_json TEXT NOT NULL,
    request_id TEXT NOT NULL,
    staging_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_playbook_publication_results (
    optimizer_kind TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('applied', 'incumbent_changed')),
    successor_user_playbook_id INTEGER,
    staging_digest TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (optimizer_kind, job_id),
    FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
        ON DELETE CASCADE
);

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

CREATE TABLE IF NOT EXISTS _agent_run_finalization_receipts (
    run_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('profile', 'user_playbook')),
    learning_ids TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (run_id) REFERENCES _agent_runs(id) ON DELETE CASCADE
);

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

-- Unicode lexical candidate indexes. Main-table triggers keep these normalized
-- n-gram sidecars current so non-ASCII searches never scan every document.
CREATE VIRTUAL TABLE IF NOT EXISTS interactions_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS profiles_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS user_playbooks_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);
CREATE VIRTUAL TABLE IF NOT EXISTS agent_playbooks_unicode_fts USING fts5(
    search_ngrams,
    tokenize="unicode61 remove_diacritics 0"
);

CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_ai
AFTER INSERT ON interactions BEGIN
    INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.user_action_description, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_au
AFTER UPDATE OF content, user_action_description ON interactions BEGIN
    DELETE FROM interactions_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO interactions_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.user_action_description, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS interactions_unicode_fts_ad
AFTER DELETE ON interactions BEGIN
    DELETE FROM interactions_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_ai
AFTER INSERT ON profiles BEGIN
    INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.expanded_terms, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_au
AFTER UPDATE OF content, expanded_terms ON profiles BEGIN
    DELETE FROM profiles_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO profiles_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.expanded_terms, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS profiles_unicode_fts_ad
AFTER DELETE ON profiles BEGIN
    DELETE FROM profiles_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_ai
AFTER INSERT ON user_playbooks BEGIN
    INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '') || ' ' ||
            COALESCE(new.source, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_au
AFTER UPDATE OF trigger, content, rationale, source ON user_playbooks BEGIN
    DELETE FROM user_playbooks_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO user_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '') || ' ' ||
            COALESCE(new.source, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS user_playbooks_unicode_fts_ad
AFTER DELETE ON user_playbooks BEGIN
    DELETE FROM user_playbooks_unicode_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_ai
AFTER INSERT ON agent_playbooks BEGIN
    INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_au
AFTER UPDATE OF trigger, content, rationale ON agent_playbooks BEGIN
    DELETE FROM agent_playbooks_unicode_fts WHERE rowid = old.rowid;
    INSERT INTO agent_playbooks_unicode_fts(rowid, search_ngrams)
    VALUES (
        new.rowid,
        reflexio_unicode_lexical_index(
            COALESCE(new.trigger, '') || ' ' ||
            COALESCE(new.content, '') || ' ' ||
            COALESCE(new.rationale, '')
        )
    );
END;
CREATE TRIGGER IF NOT EXISTS agent_playbooks_unicode_fts_ad
AFTER DELETE ON agent_playbooks BEGIN
    DELETE FROM agent_playbooks_unicode_fts WHERE rowid = old.rowid;
END;

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
    model_name       TEXT,
    provider         TEXT,
    UNIQUE (org_id, entity_type, entity_id, op, request_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_entity ON lineage_event (entity_type, entity_id);

-- ============================================================================
-- Durable learning pipeline — cross-org job queue
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_jobs (
    job_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    job_type TEXT NOT NULL DEFAULT 'learning',
    latest_request_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    claimed_by TEXT,
    claim_token TEXT,
    claim_expires_at TEXT,
    covers_through TEXT,
    force_extraction INTEGER NOT NULL DEFAULT 0,
    skip_aggregation INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS learning_jobs_coalesce
    ON learning_jobs (org_id, user_id, job_type) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS learning_jobs_poll
    ON learning_jobs (created_at) WHERE status IN ('pending','failed','claimed');

"""
