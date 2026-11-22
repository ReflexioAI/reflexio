"""JSON/time helpers and row-to-model mappers for SQLite storage."""

import json
import logging
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.common import BlockingIssue
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
from reflexio.server.services.storage.error import (
    require_non_empty_session_id,
)

logger = logging.getLogger(__name__)


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


def _row_to_interaction(row: sqlite3.Row | Mapping[str, Any]) -> Interaction:
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
        [
            RetrievedLearning.from_stored_payload(c)
            for c in retrieved_learnings_raw
            if isinstance(c, dict)
        ]
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


def _row_to_request(row: sqlite3.Row | Mapping[str, Any]) -> Request:
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
        governance_subject_ref=d.get("governance_subject_ref"),
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


def _row_to_eval_result(
    row: sqlite3.Row, *, include_embedding: bool = True
) -> AgentSuccessEvaluationResult:
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
        embedding=(_json_loads(d.get("embedding")) or [] if include_embedding else []),
        trajectory_through_request_id=d.get("trajectory_through_request_id"),
        trajectory_interaction_count=d.get("trajectory_interaction_count"),
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
