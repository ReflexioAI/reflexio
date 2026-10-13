"""Pin complete human answers and verify their lifetime at the write boundary."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_scope_hash,
    human_feedback_scope,
)

if TYPE_CHECKING:
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )


ANSWER_FIELDS = (
    "id",
    "org_id",
    "scope_hash",
    "scope",
    "answer_format",
    "args",
    "tool_name",
    "status",
    "question_text",
    "result",
    "resolved_at",
    "valid_until",
    "superseded_by",
)


def answer_witness(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize the backend representation without dropping answer qualifiers."""
    import json

    witness = {field: values[field] for field in ANSWER_FIELDS}
    for field in ("resolved_at", "valid_until"):
        value = witness[field]
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value is not None:
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            witness[field] = value.astimezone(UTC).isoformat()
    for field in ("result", "scope", "args"):
        if isinstance(witness[field], str):
            witness[field] = json.loads(witness[field])
    return witness


def require_current_answer(witness: dict[str, Any], org_id: str, now: datetime) -> None:
    if (
        witness["org_id"] != org_id
        or witness["scope_hash"] != build_scope_hash(human_feedback_scope(org_id))
        or witness["scope"] != human_feedback_scope(org_id)
        or witness["tool_name"] != "ask_human"
        or witness["status"] != PendingToolCallStatus.RESOLVED
        or witness["resolved_at"] is None
        or witness["superseded_by"] is not None
        or (
            witness["valid_until"] is not None
            and datetime.fromisoformat(witness["valid_until"]) <= now
        )
    ):
        raise InputBudgetConfigurationError("Extraction human answer is unavailable")


def pin_overflow_answer(record: PendingToolCallRecord) -> None:
    from reflexio.server.services.extraction.overflow_coordinator import (
        current_overflow_coordinator,
    )

    coordinator = current_overflow_coordinator.get()
    if coordinator is None:
        return
    witness = answer_witness(asdict(record))
    require_current_answer(witness, coordinator.storage.org_id, datetime.now(UTC))
    checkpoint = coordinator.checkpoint
    answers = checkpoint.setdefault("answer_witnesses", {})
    if record.id in answers and answers[record.id] != witness:
        raise InputBudgetConfigurationError("Extraction human answer changed")
    if record.id not in answers:
        if len(answers) >= 128:
            raise InputBudgetConfigurationError(
                "Extraction answer witness limit exceeded"
            )
        answers[record.id] = witness
        coordinator.save_checkpoint(checkpoint)


def dependency_witness(values: dict[str, Any]) -> dict[str, Any]:
    witness = {
        field: values[field]
        for field in ("run_id", "pending_tool_call_id", "resolved_at", "consumed_at")
    }
    for field in ("resolved_at", "consumed_at"):
        value = witness[field]
        if value is not None:
            parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
            witness[field] = parsed.astimezone(UTC).isoformat()
    return witness


def resume_waiting_child(coordinator: OverflowCoordinator, key: str) -> None:
    """Stage answers once; the leased window worker resumes the saved child."""
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowSuspendedError,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        append_resolved_tool_result_context,
    )

    checkpoint = coordinator.checkpoint
    waiting = checkpoint.get("waiting_child")
    if waiting is None:
        return
    active = checkpoint.get("active_child")
    ids = waiting.get("pending_tool_call_ids", [])
    if (
        active is None
        or waiting.get("key") != key
        or active.get("key") != key
        or not ids
        or len(ids) > 128
        or len(set(ids)) != len(ids)
    ):
        raise InputBudgetConfigurationError(
            "Extraction child continuation binding changed"
        )
    records: list[PendingToolCallRecord] = []
    for call_id in ids:
        record = coordinator.storage.get_pending_tool_call(call_id)
        if record is None or record.status not in (
            PendingToolCallStatus.PENDING,
            PendingToolCallStatus.RESOLVED,
        ):
            raise InputBudgetConfigurationError(
                "Extraction child answer is unavailable"
            )
        if (
            record.status == PendingToolCallStatus.PENDING
            and record.expires_at is not None
            and record.expires_at <= datetime.now(UTC)
        ):
            raise InputBudgetConfigurationError("Extraction child question expired")
        records.append(record)
    if any(record.status == PendingToolCallStatus.PENDING for record in records):
        raise OverflowSuspendedError("Extraction child awaits human/tool input")
    with coordinator.storage.commit_scope():
        coordinator.storage.fence_user_extraction(
            coordinator.window.user_id, coordinator.token
        )
        coordinator.storage.validate_extraction_source_digest(
            coordinator.window, coordinator.source_digest
        )
        for record in records:
            pin_overflow_answer(record)
        checkpoint = coordinator.checkpoint
        witnesses = checkpoint["answer_witnesses"]
        coordinator.storage.validate_extraction_answers(witnesses)
        consumed = coordinator.storage.consume_extraction_answer_dependencies(
            coordinator.window,
            coordinator.token,
            {call_id: witnesses[call_id]["resolved_at"] for call_id in ids},
        )
        checkpoint.setdefault("answer_dependencies", {}).update(consumed)
        checkpoint["active_child"]["messages"] = append_resolved_tool_result_context(
            active["messages"], records
        )
        checkpoint.pop("waiting_child")
        coordinator.save_checkpoint(checkpoint)
