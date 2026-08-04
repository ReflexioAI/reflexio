"""Optional usage metrics hook.

This module intentionally has no storage or vendor dependency. Deployments that
want usage metrics can register a recorder; deployments that do not register one
pay only a cheap function-call/no-op cost.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class UsageEvent:
    org_id: str
    event_name: str
    event_category: str
    user_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    pipeline: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    event_key: str | None = None
    extractor_name: str | None = None
    playbook_name: str | None = None
    source: str | None = None
    agent_version: str | None = None
    backend: str | None = None
    outcome: str | None = None
    count_value: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    billing_input_tokens: int | None = None
    platform_llm: bool | None = None
    platform_storage: bool | None = None
    caller_type: str | None = None
    duration_ms: int | None = None
    error_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class UsageEventDeliveryStatus(Enum):
    """Recorder-reported durability outcome for one usage event."""

    APPENDED = "appended"
    DUPLICATE = "duplicate"
    EXEMPT = "exempt"
    UNKNOWN = "unknown"
    FAILED = "failed"
    REJECTED = "rejected"


class UsageEventDeliveryError(RuntimeError):
    """Raised when strict usage delivery was not durably accepted."""

    def __init__(self, status: UsageEventDeliveryStatus) -> None:
        self.status = status
        super().__init__(f"usage event delivery {status.value}")


UsageEventRecorder = Callable[[UsageEvent], UsageEventDeliveryStatus | None]

_recorder: UsageEventRecorder | None = None


def configure_usage_event_recorder(recorder: UsageEventRecorder | None) -> None:
    """Set the process-global usage metrics recorder.

    Args:
        recorder: Callable that accepts UsageEvent, or None to disable metrics.
    """
    global _recorder
    _recorder = recorder


def exempt_usage_event_recorder(_event: UsageEvent) -> UsageEventDeliveryStatus:
    """Explicitly exempt a deployment from durable usage-event delivery."""
    return UsageEventDeliveryStatus.EXEMPT


def record_usage_event(
    *,
    org_id: str,
    event_name: str,
    event_category: str,
    user_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    pipeline: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    event_key: str | None = None,
    extractor_name: str | None = None,
    playbook_name: str | None = None,
    source: str | None = None,
    agent_version: str | None = None,
    backend: str | None = None,
    outcome: str | None = None,
    count_value: int = 1,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    billing_input_tokens: int | None = None,
    platform_llm: bool | None = None,
    platform_storage: bool | None = None,
    caller_type: str | None = None,
    duration_ms: int | None = None,
    error_kind: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: float | None = None,
) -> None:
    """Record one usage event if a recorder is configured.

    The product path must never fail because metrics failed, so this function
    catches and logs all recorder errors.
    """
    try:
        record_usage_event_strict(
            org_id=org_id,
            event_name=event_name,
            event_category=event_category,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            pipeline=pipeline,
            entity_type=entity_type,
            entity_id=entity_id,
            event_key=event_key,
            extractor_name=extractor_name,
            playbook_name=playbook_name,
            source=source,
            agent_version=agent_version,
            backend=backend,
            outcome=outcome,
            count_value=count_value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            billing_input_tokens=billing_input_tokens,
            platform_llm=platform_llm,
            platform_storage=platform_storage,
            caller_type=caller_type,
            duration_ms=duration_ms,
            error_kind=error_kind,
            metadata=metadata,
            created_at=created_at,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Usage metrics recorder failed: %s", exc)


def record_usage_event_strict(
    *,
    org_id: str,
    event_name: str,
    event_category: str,
    user_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    pipeline: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    event_key: str | None = None,
    extractor_name: str | None = None,
    playbook_name: str | None = None,
    source: str | None = None,
    agent_version: str | None = None,
    backend: str | None = None,
    outcome: str | None = None,
    count_value: int = 1,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    billing_input_tokens: int | None = None,
    platform_llm: bool | None = None,
    platform_storage: bool | None = None,
    caller_type: str | None = None,
    duration_ms: int | None = None,
    error_kind: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: float | None = None,
) -> UsageEventDeliveryStatus:
    """Deliver one event and fail unless the recorder accepted it durably.

    Only an explicit append, duplicate, or deployment exemption proves the
    receipt's billing obligation was handled. Missing and legacy recorders are
    unknown delivery, which strict receipt-backed callers must retry.
    """
    recorder = _recorder
    if recorder is None:
        raise UsageEventDeliveryError(UsageEventDeliveryStatus.UNKNOWN)
    delivery_outcome = recorder(
        UsageEvent(
            org_id=str(org_id),
            event_name=event_name,
            event_category=event_category,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            pipeline=pipeline,
            entity_type=entity_type,
            entity_id=entity_id,
            event_key=event_key,
            extractor_name=extractor_name,
            playbook_name=playbook_name,
            source=source,
            agent_version=agent_version,
            backend=backend,
            outcome=outcome,
            count_value=count_value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            billing_input_tokens=billing_input_tokens,
            platform_llm=platform_llm,
            platform_storage=platform_storage,
            caller_type=caller_type,
            duration_ms=duration_ms,
            error_kind=error_kind,
            metadata=metadata or {},
            created_at=time.time() if created_at is None else created_at,
        )
    )
    status = (
        UsageEventDeliveryStatus.UNKNOWN
        if delivery_outcome is None
        else delivery_outcome
    )
    if not isinstance(status, UsageEventDeliveryStatus):
        raise TypeError(
            f"usage event recorder returned an invalid delivery status: {status!r}"
        )
    if status not in {
        UsageEventDeliveryStatus.APPENDED,
        UsageEventDeliveryStatus.DUPLICATE,
        UsageEventDeliveryStatus.EXEMPT,
    }:
        raise UsageEventDeliveryError(status)
    return status
