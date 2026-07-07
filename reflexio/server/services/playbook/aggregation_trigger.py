from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.extensions import get_service
from reflexio.server.operation_limiter import run_with_operation_limit
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AGGREGATION_PROMPT_PROCESSOR,
)
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)

logger = logging.getLogger(__name__)

AggregationTriggerStatus = Literal[
    "triggered",
    "skipped_no_config",
    "skipped_saturated",
    "failed",
]


@dataclass(frozen=True)
class AggregationTriggerResult:
    status: AggregationTriggerStatus
    reason: str
    agent_version: str
    error_type: str | None = None


def maybe_trigger_user_playbook_aggregation(
    *,
    request_context: RequestContext,
    llm_client: Any,
    agent_version: str,
    reason: str,
) -> AggregationTriggerResult:
    root_config = request_context.configurator.get_config()
    playbook_config = getattr(root_config, "user_playbook_extractor_config", None)
    if not playbook_config or not playbook_config.aggregation_config:
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
    aggregator_kwargs: dict[str, Any] = {}
    aggregation_prompt_processor = get_service(AGGREGATION_PROMPT_PROCESSOR)
    if aggregation_prompt_processor is not None:
        aggregator_kwargs["aggregation_prompt_processor"] = aggregation_prompt_processor

    aggregator = PlaybookAggregator(
        llm_client=llm_client,
        request_context=request_context,
        agent_version=agent_version,
        **aggregator_kwargs,
    )
    request = PlaybookAggregatorRequest(agent_version=agent_version)
    try:
        run_with_operation_limit(
            org_id=request_context.org_id,
            operation="aggregation",
            fn=lambda: aggregator.run(request),
        )
    except TimeoutError:
        logger.info(
            "Skipping inline aggregation for agent_version=%s reason=%s: aggregation limiter is saturated",
            agent_version,
            reason,
        )
        return AggregationTriggerResult(
            status="skipped_saturated",
            reason=reason,
            agent_version=agent_version,
            error_type="TimeoutError",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "User playbook aggregation failed after successor commit agent_version=%s reason=%s",
            agent_version,
            reason,
        )
        return AggregationTriggerResult(
            status="failed",
            reason=reason,
            agent_version=agent_version,
            error_type=type(exc).__name__,
        )

    return AggregationTriggerResult(
        status="triggered",
        reason=reason,
        agent_version=agent_version,
    )
