"""Execute explicit full aggregations on the existing scheduler and lease."""

import logging
from collections.abc import Callable

from reflexio.lib.generation_client import create_generation_litellm_client
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.background_work import (
    report_background_failure,
    report_background_success,
)
from reflexio.server.extensions import get_service
from reflexio.server.operation_limiter import operation_limit
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AGGREGATION_PROMPT_PROCESSOR,
)
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.storage_base.playbook import (
    AGGREGATION_RETRY_BASE_SECONDS,
    AGGREGATION_RETRY_MAX_SECONDS,
)

logger = logging.getLogger(__name__)


def run_explicit_operation(
    context: RequestContext,
    *,
    owner: str,
    project_id: str | None = None,
    on_work_claimed: Callable[[RequestContext], None] | None = None,
    on_claim_completed: Callable[[], None] | None = None,
) -> bool:
    """Return whether explicit work owns this scheduler turn (including contention).

    Notify the scheduler immediately after the claim query returns, including
    contention, so it can reset its org streak before execution or finalization.
    A failed claim query does not notify it. Operation and explicit-finalization
    outcomes retain their independent failure scopes here.
    """
    from .aggregation_scheduler import (
        _REPAIR_INTERVAL_SECONDS,
        AGGREGATION_BACKLOG_RETRY_SECONDS,
        AGGREGATION_LEASE_SECONDS,
        AGGREGATION_RETRY_SECONDS,
        AggregationLeaseHeartbeat,
        _FinalizationReportedError,
        _is_unqualified_success,
        aggregation_min_interval_seconds,
    )

    storage = context.storage
    if storage is None or not getattr(
        storage, "supports_durable_playbook_aggregation_operations", False
    ):
        return False
    operation = storage.next_playbook_aggregation_operation()
    if operation is None:
        return False
    claim = storage.claim_due_playbook_aggregation(
        owner=owner,
        lease_seconds=AGGREGATION_LEASE_SECONDS,
        agent_version=operation.agent_version,
    )
    if on_claim_completed is not None:
        on_claim_completed()
    if claim is None:
        return True
    exhausted = operation.attempts >= 5
    heartbeat = AggregationLeaseHeartbeat(
        storage, claim, org_id=context.org_id, project_id=project_id
    )
    heartbeat.start()
    run_scope = (
        f"playbook-aggregation-operation:{context.org_id}:{project_id}:"
        f"{operation.operation_id}"
    )
    success = False
    try:
        if on_work_claimed is not None:
            try:
                on_work_claimed(context)
            except Exception:
                logger.exception(
                    "event=playbook_aggregation_claim_notification_failed org_id=%s",
                    context.org_id,
                )
        operation = storage.begin_playbook_aggregation_operation(
            operation.operation_id, claim
        )
        with operation_limit(context.org_id, "aggregation"):
            if exhausted:
                raise ValueError("attempts_exhausted")
            config = context.configurator.get_config().user_playbook_extractor_config
            if (
                config is None
                or config.aggregation_config is None
                or config.aggregation_config.min_cluster_size < 2
            ):
                raise ValueError("invalid_aggregation_config")
            aggregator = PlaybookAggregator(
                llm_client=create_generation_litellm_client(context),
                request_context=context,
                agent_version=operation.agent_version,
                aggregation_claim=claim,
                explicit_operation_id=operation.operation_id,
                aggregation_prompt_processor=get_service(AGGREGATION_PROMPT_PROCESSOR),
            )
            result = aggregator.run(
                PlaybookAggregatorRequest(
                    agent_version=operation.agent_version,
                    rerun=True,
                    operation_key=operation.operation_id,
                )
            )
            if not _is_unqualified_success(result):
                raise RuntimeError("explicit aggregation returned incomplete work")
        heartbeat.require_live()
        # Early successful no-ops have no output transaction. Complete them under
        # the same fence; output-producing paths already committed the receipt.
        with storage.commit_scope():
            current = storage.get_playbook_aggregation_operation(operation.operation_id)
            if current is None:
                raise RuntimeError("aggregation operation disappeared")
            if current.status != "succeeded":
                storage.complete_playbook_aggregation_operation(
                    operation.operation_id, heartbeat.claim, result
                )
        success = True
        report_background_success(run_scope)
    except Exception as exc:
        terminal = (
            isinstance(exc, ValueError)
            or "safety cap" in str(exc).lower()
            or operation.attempts >= 5
        )
        # Do not expose provider messages, credentials, or customer content in status.
        error = (
            "attempts_exhausted"
            if exhausted or operation.attempts >= 5
            else ("invalid_configuration_or_input" if terminal else "execution_failed")
        )
        storage.fail_playbook_aggregation_operation(
            operation.operation_id,
            heartbeat.claim,
            error=error,
            retry_seconds=None
            if terminal
            else min(
                AGGREGATION_RETRY_MAX_SECONDS,
                AGGREGATION_RETRY_BASE_SECONDS * 2 ** max(0, operation.attempts - 1),
            ),
        )
        report_background_failure(
            logger,
            "playbook_aggregation_progress",
            exc,
            scope=run_scope,
            state="failed" if terminal else "retryable_failed",
            org_id=context.org_id,
            project_id=project_id,
            operation_id=operation.operation_id,
            attempt=operation.attempts,
        )
    finally:
        heartbeat.stop()
        finalize_scope = (
            f"playbook-aggregation-operation-finalize:{context.org_id}:{project_id}:"
            f"{claim.agent_version}"
        )
        try:
            finished = storage.finish_playbook_aggregation_claim(
                heartbeat.claim,
                success=success,
                retry_after_seconds=AGGREGATION_RETRY_SECONDS,
                backlog_retry_after_seconds=AGGREGATION_BACKLOG_RETRY_SECONDS,
                min_interval_seconds=aggregation_min_interval_seconds(),
            )
        except Exception as exc:
            report_background_failure(
                logger,
                "playbook_aggregation_scheduler_org_failed",
                exc,
                scope=finalize_scope,
                org_id=context.org_id,
                project_id=project_id,
                stage="finalization",
                retry_after_seconds=_REPAIR_INTERVAL_SECONDS,
            )
            raise _FinalizationReportedError from exc
        if finished:
            report_background_success(finalize_scope)
    return True
