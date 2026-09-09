"""Billing money-path metering helpers shared by search + playbook routes (Tier3 A2).

These emit the ③ Application / search-request billing signals. They resolve
``get_reflexio`` through the ``reflexio_cache`` module (not a bound import) so a
single ``patch("reflexio.server.cache.reflexio_cache.get_reflexio")`` at the
source intercepts the lookup regardless of which route module calls them.
"""

import logging
import time

from fastapi import Request

from reflexio.server.cache import reflexio_cache
from reflexio.server.error_reporting import capture_anomaly

logger = logging.getLogger(__name__)


def _stamp_search_dependencies_done(request: Request) -> None:
    """Stamp when dependency resolution reaches its final search dependency."""
    request.state.search_deps_done_monotonic = time.monotonic()


def _meter_applied_learnings(
    *,
    org_id: str,
    caller_type: str,
    surfaced_count: int,
    request_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Emit the ③ Application event via the OSS emission helper.

    No-op unless a production-agent caller surfaced >= 1 result.  The cheap
    caller-type and count guard runs first so the get_reflexio / config lookup
    is skipped on the free paths (dashboard / empty result).

    Args:
        org_id: Organization ID for the requesting caller.
        caller_type: Resolved caller classification (e.g. ``"production_agent"``).
        surfaced_count: Total number of learnings returned to the caller.
        request_id: Optional request correlation ID from the payload.
        session_id: Optional session ID from the payload.
    """
    if caller_type != "production_agent" or surfaced_count <= 0:
        return False
    try:
        from reflexio.server.billing_meter import record_applied_learnings
        from reflexio.server.billing_signals import platform_llm_from_config

        config = reflexio_cache.get_reflexio(
            org_id=org_id
        ).request_context.configurator.get_config()
        record_applied_learnings(
            org_id=org_id,
            surfaced_count=surfaced_count,
            caller_type=caller_type,
            platform_llm=platform_llm_from_config(config),
            platform_storage=None,  # resolved enterprise-side at rollup (Phase 1)
            request_id=request_id,
            session_id=session_id,
        )
        return True
    except Exception:
        # ALARM, not just a log line. This is a billable event being dropped, and
        # the drop is otherwise invisible: the caller ignores the return value,
        # and `_worker_loop`'s own `search.metering.job_failed` capture can never
        # fire because this handler swallows first. Note the asymmetry that makes
        # it dangerous -- `_meter_search_request` below performs NO config lookup,
        # so a config-store blip drops `learning_applied` while `search_request`
        # keeps flowing, and the two meters silently diverge.
        logger.warning(
            "applied-learnings metering failed for org %s", org_id, exc_info=True
        )
        capture_anomaly(
            "search.metering.emit_failed",
            level="error",
            meter="applied_learnings",
            org_id=org_id,
        )
        return False


def _meter_search_request(
    *,
    org_id: str,
    caller_type: str,
    request_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Emit one production-agent search request metric.

    Args:
        org_id: Organization ID for the requesting caller.
        caller_type: Resolved caller classification.
        request_id: Optional request correlation ID from the payload.
        session_id: Optional session ID from the payload.
    """
    if caller_type != "production_agent":
        return False
    try:
        from reflexio.server.billing_meter import record_search_request

        record_search_request(
            org_id=org_id,
            caller_type=caller_type,
            request_id=request_id,
            session_id=session_id,
        )
        return True
    except Exception:
        # Same reasoning as `_meter_applied_learnings`: a swallowed emit is lost
        # billable usage, so it must reach the error reporter and not only a log.
        logger.warning(
            "search-request metering failed for org %s", org_id, exc_info=True
        )
        capture_anomaly(
            "search.metering.emit_failed",
            level="error",
            meter="search_requests",
            org_id=org_id,
        )
        return False
