"""Bounded housekeeping on the scheduler thread, independent of model capacity."""

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Event

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.background_work import (
    report_background_success,
    report_failure_at_interval,
)
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_disabled_reason,
)
from reflexio.server.services.storage.storage_base._user_work_position import (
    ResumeCursor,
)
from reflexio.server.work_scope import WorkScope, bind_work_scope

logger = logging.getLogger(__name__)


def _failure_detail(exc: BaseException) -> str:
    from reflexio.server.services.durable_learning.worker import exception_detail

    return exception_detail(exc)


@dataclass(frozen=True)
class _ExpiryPosition:
    route: object
    cursor: ResumeCursor = ResumeCursor()
    disabled_reason: str | None = None


class ExpiryHousekeeping:
    """One fixed metadata candidate per org/tick, with no context lifetime owner."""

    def __init__(self, factory: Callable[[str], RequestContext], stop_event: Event):
        self._factory = factory
        self._stop_event = stop_event
        self._positions: dict[str, _ExpiryPosition] = {}

    def run(self, org_ids: Iterable[str]) -> None:
        orgs = set(org_ids)
        self._positions = {
            org: position for org, position in self._positions.items() if org in orgs
        }
        for org_id in sorted(orgs):
            if self._stop_event.is_set():
                break
            scope = f"durable-expiry:{org_id}"
            try:
                self._run_org(org_id)
            except Exception as exc:
                report_failure_at_interval(
                    logger,
                    "durable_expiry_failed",
                    exc,
                    scope=scope,
                    interval_seconds=None,
                    detail=_failure_detail,
                    org_id=org_id,
                )
            else:
                report_background_success(scope)

    def _run_org(self, org_id: str) -> None:
        context = self._factory(org_id)
        storage = context.storage
        if context.org_id != org_id or storage is None or storage.org_id != org_id:
            raise ValueError("Expiry source org mismatch")
        route = storage._durable_work_route()
        position = self._positions.get(org_id)
        if position is None or position.route != route:
            position = _ExpiryPosition(route)
        cursor, item = storage._expired_question_page(cursor=position.cursor)
        self._positions[org_id] = _ExpiryPosition(
            route, cursor, position.disabled_reason
        )
        if item is None:
            return
        if item["org_id"] != org_id or not isinstance(item["project_id"], str):
            raise ValueError("Original expiry scope unavailable")
        reason = pending_tool_calls_disabled_reason(context)
        if reason is not None:
            if reason != position.disabled_reason:
                logger.info(
                    "event=durable_expiry_skipped_gate_disabled org_id=%s reason=%s",
                    org_id,
                    reason,
                )
            self._positions[org_id] = _ExpiryPosition(route, cursor, reason)
            return
        self._positions[org_id] = _ExpiryPosition(route, cursor)
        with bind_work_scope(WorkScope(org_id, item["project_id"] or None)):
            storage._expire_pending_question(str(item["id"]))
