from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.playbook.aggregation_scheduler import (
    ensure_local_playbook_aggregation_scheduler,
)
from reflexio.server.services.playbook.aggregation_scheduler import (
    logger as aggregation_progress_logger,
)

logger = logging.getLogger(__name__)

AggregationTriggerStatus = Literal[
    "scheduled",
    "skipped_no_config",
    "failed",
]


@dataclass(frozen=True)
class AggregationTriggerResult:
    status: AggregationTriggerStatus
    reason: str
    agent_version: str
    error_type: str | None = None


def _failed_result(
    *,
    exc: Exception,
    reason: str,
    agent_version: str,
    message: str,
) -> AggregationTriggerResult:
    logger.exception(
        "%s agent_version=%s reason=%s",
        message,
        agent_version,
        reason,
    )
    return AggregationTriggerResult(
        status="failed",
        reason=reason,
        agent_version=agent_version,
        error_type=type(exc).__name__,
    )


def has_user_playbook_aggregation_config(request_context: RequestContext) -> bool:
    root_config = request_context.configurator.get_config()
    playbook_config = getattr(root_config, "user_playbook_extractor_config", None)
    return bool(playbook_config and playbook_config.aggregation_config)


def maybe_trigger_user_playbook_aggregation(
    *,
    request_context: RequestContext,
    llm_client: Any,  # noqa: ARG001 - retained for the public trigger contract
    agent_version: str,
    reason: str,
) -> AggregationTriggerResult:
    try:
        if not has_user_playbook_aggregation_config(request_context):
            return AggregationTriggerResult(
                status="skipped_no_config",
                reason=reason,
                agent_version=agent_version,
            )

        logger.info(
            "Triggering user playbook aggregation agent_version=%s reason=%s",
            agent_version,
            reason,
        )
        storage = request_context.storage
        if storage is None:
            raise RuntimeError("playbook aggregation requires configured storage")
        storage.schedule_playbook_aggregation(agent_version)
        ensure_local_playbook_aggregation_scheduler(request_context)
        aggregation_progress_logger.info(
            "event=playbook_aggregation_progress state=scheduled org_id=%s "
            "agent_version=%s reason=%s pending=true",
            request_context.org_id,
            agent_version,
            reason,
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_result(
            exc=exc,
            reason=reason,
            agent_version=agent_version,
            message="User playbook aggregation setup failed",
        )

    return AggregationTriggerResult(
        status="scheduled",
        reason=reason,
        agent_version=agent_version,
    )
