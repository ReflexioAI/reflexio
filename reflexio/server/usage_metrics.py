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
    extractor_name: str | None = None
    playbook_name: str | None = None
    source: str | None = None
    agent_version: str | None = None
    backend: str | None = None
    outcome: str | None = None
    count_value: int = 1
    duration_ms: int | None = None
    error_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


UsageEventRecorder = Callable[[UsageEvent], None]

_recorder: UsageEventRecorder | None = None


def configure_usage_event_recorder(recorder: UsageEventRecorder | None) -> None:
    """Set the process-global usage metrics recorder.

    Args:
        recorder: Callable that accepts UsageEvent, or None to disable metrics.
    """
    global _recorder
    _recorder = recorder


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
    extractor_name: str | None = None,
    playbook_name: str | None = None,
    source: str | None = None,
    agent_version: str | None = None,
    backend: str | None = None,
    outcome: str | None = None,
    count_value: int = 1,
    duration_ms: int | None = None,
    error_kind: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Record one usage event if a recorder is configured.

    The product path must never fail because metrics failed, so this function
    catches and logs all recorder errors.
    """
    recorder = _recorder
    if recorder is None:
        return
    try:
        recorder(
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
                extractor_name=extractor_name,
                playbook_name=playbook_name,
                source=source,
                agent_version=agent_version,
                backend=backend,
                outcome=outcome,
                count_value=count_value,
                duration_ms=duration_ms,
                error_kind=error_kind,
                metadata=metadata or {},
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Usage metrics recorder failed: %s", exc)
