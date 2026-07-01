"""Shared row/datetime helpers for SQLite resumable extraction agent runs.

Helpers-only residual for the ``_agent_run.py`` decomposition: all twenty-three
agent-run methods now live in the three ``agent_run`` sub-mixins
(``SQLiteAgentRunStoreMixin`` / ``SQLitePendingToolCallStoreMixin`` /
``SQLiteRunToolDependencyStoreMixin``), which import the ``_row_to_*`` /
``_dt`` / ``_dt_str`` helpers below rather than duplicating them.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    PriorAnswerMatch,
    RunToolDependencyKind,
    RunToolDependencyRecord,
    embedding_similarity,
)

from ._base import _json_loads


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _dt_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _row_to_agent_run(row: sqlite3.Row) -> AgentRunRecord:
    data = dict(row)
    binding = AgentBinding(
        org_id=data["org_id"],
        extractor_kind=data["extractor_kind"],
        user_id=data.get("user_id"),
        request_id=data["request_id"],
        agent_version=data.get("agent_version"),
        source=data.get("source"),
        source_interaction_ids=_json_loads(data.get("source_interaction_ids")) or [],
        window_start_interaction_id=data.get("window_start_interaction_id"),
        window_end_interaction_id=data.get("window_end_interaction_id"),
        extractor_config_hash=data.get("extractor_config_hash"),
    )
    return AgentRunRecord(
        id=data["id"],
        binding=binding,
        status=AgentRunStatus(data["status"]),
        generation_request_snapshot=_json_loads(data.get("generation_request_snapshot"))
        or {},
        service_config_snapshot=_json_loads(data.get("service_config_snapshot")),
        agent_context_snapshot=data.get("agent_context_snapshot"),
        committed_output=_json_loads(data.get("committed_output")),
        pending_tool_call_ids=_json_loads(data.get("pending_tool_call_ids")) or [],
        max_steps_remaining=(
            int(data["max_steps_remaining"])
            if data.get("max_steps_remaining") is not None
            else None
        ),
        resume_attempts=int(data.get("resume_attempts") or 0),
        finalization_attempts=int(data.get("finalization_attempts") or 0),
        next_resume_at=_dt(data.get("next_resume_at")),
        claimed_by=data.get("claimed_by"),
        claimed_at=_dt(data.get("claimed_at")),
        agent_completed_at=_dt(data.get("agent_completed_at")),
        finalized_at=_dt(data.get("finalized_at")),
        created_at=_dt(data.get("created_at")),
        updated_at=_dt(data.get("updated_at")),
        expires_at=_dt(data.get("expires_at")),
        last_error=data.get("last_error"),
    )


def _row_to_pending_tool_call(row: sqlite3.Row) -> PendingToolCallRecord:
    data = dict(row)
    return PendingToolCallRecord(
        id=data["id"],
        org_id=data["org_id"],
        scope=_json_loads(data.get("scope")) or {},
        scope_hash=data["scope_hash"],
        tool_name=data["tool_name"],
        dedup_key=data["dedup_key"],
        status=PendingToolCallStatus(data["status"]),
        question_text=data["question_text"],
        args=_json_loads(data.get("args")) or {},
        tags=_json_loads(data.get("tags")) or [],
        user_id=data.get("user_id"),
        answer_format=data.get("answer_format"),
        result=_json_loads(data.get("result")),
        embedding=_json_loads(data.get("embedding")),
        superseded_by=data.get("superseded_by"),
        created_at=_dt(data.get("created_at")),
        resolved_at=_dt(data.get("resolved_at")),
        expires_at=_dt(data.get("expires_at")),
        cache_until=_dt(data.get("cache_until")),
        valid_until=_dt(data.get("valid_until")),
    )


def _record_to_prior_answer_match(
    record: PendingToolCallRecord,
    *,
    query_embedding: list[float] | None = None,
) -> PriorAnswerMatch:
    return PriorAnswerMatch(
        pending_tool_call_id=record.id,
        status=record.status,
        question_text=record.question_text,
        result=record.result,
        valid_until=record.valid_until,
        answer_format=record.answer_format,
        created_at=record.created_at,
        resolved_at=record.resolved_at,
        expires_at=record.expires_at,
        similarity=embedding_similarity(query_embedding, record.embedding),
    )


def _row_to_run_tool_dependency(row: sqlite3.Row) -> RunToolDependencyRecord:
    data = dict(row)
    return RunToolDependencyRecord(
        run_id=data["run_id"],
        pending_tool_call_id=data["pending_tool_call_id"],
        dependency_kind=RunToolDependencyKind(data["dependency_kind"]),
        resolved_at=_dt(data.get("resolved_at")),
        consumed_at=_dt(data.get("consumed_at")),
        created_at=_dt(data.get("created_at")),
    )
