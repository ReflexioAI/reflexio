"""Admitted user selection and persisted opportunities for existing work classes."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reflexio.server.background_work import (
    report_background_success,
    report_failure_at_interval,
)
from reflexio.server.operational_metrics import record_health
from reflexio.server.services.playbook.review_window import (
    infer_playbook_review_user_id,
)
from reflexio.server.services.storage.error import execution_input_is_busy
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    BaseStorage,
    LeaseLostError,
)
from reflexio.server.services.storage.storage_base._user_work_position import (
    UserWorkPosition,
    WorkClass,
)
from reflexio.server.work_scope import WorkScope, bind_work_scope

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext


logger = logging.getLogger(__name__)


def _failure_detail(exc: BaseException) -> str:
    from reflexio.server.services.durable_learning.worker import exception_detail

    return exception_detail(exc)


@dataclass(frozen=True)
class _FleetPosition:
    route: object
    # Start ordinary fleet acquisition with learning. User-owned progress has
    # its own persisted default, which correctly starts with finalization.
    progress: UserWorkPosition = field(
        default_factory=lambda: UserWorkPosition(last_work_class="continuation")
    )


def _descriptor_scope(storage: BaseStorage, item: dict[str, Any]) -> WorkScope:
    if item["org_id"] != storage.org_id or not isinstance(item["project_id"], str):
        raise ValueError("Original resume scope unavailable")
    return WorkScope(storage.org_id, item["project_id"] or None)


class ResumeSelection:
    """Only fleet inspection is local; user/class progress lives in the writer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: dict[str, _FleetPosition] = {}

    def claim_user(
        self, context: RequestContext, *, worker_id: str, lease_seconds: int
    ) -> tuple[str, str] | None:
        storage = context.storage
        if storage is None:
            return None
        route = storage._durable_work_route()
        # At most one traversal of the fixed classes. Failed/empty/busy work
        # never blocks an independent class; no DB/local lock spans execution.
        for _ in ("learning", "finalization", "continuation"):
            with self._lock:
                position = self._positions.get(storage.org_id)
                if position is None or position.route != route:
                    position = _FleetPosition(route)
                    self._positions[storage.org_id] = position
            kind = position.progress.next_work_class
            cursor = None if kind == "learning" else getattr(position.progress, kind)
            item = None
            scope = f"durable-resume-discovery:{storage.org_id}:{kind}"
            failed = False
            if kind != "learning":
                try:
                    cursor, item = storage._resume_candidate_page(kind, cursor=cursor)
                    report_background_success(scope)
                except Exception as exc:
                    report_failure_at_interval(
                        logger,
                        "resume_selection_failed",
                        exc,
                        scope=scope,
                        interval_seconds=None,
                        detail=_failure_detail,
                        org_id=storage.org_id,
                    )
                    failed = True
            with self._lock:
                if self._positions.get(storage.org_id) is not position:
                    continue
                self._positions[storage.org_id] = _FleetPosition(
                    route, position.progress.selected(kind, cursor=cursor)
                )
            if kind == "learning":
                claim = storage.claim_extraction(worker_id, lease_seconds)
                if claim is not None:
                    return claim
            elif item is not None and not failed:
                key = hashlib.sha256(str(item["id"]).encode()).hexdigest()[:12]
                admission_scope = f"durable-resume-admission:{storage.org_id}:{key}"
                try:
                    # Validate configuration before claiming, without using
                    # permission for new pending tools as a backlog veto.
                    with bind_work_scope(_descriptor_scope(storage, item)):
                        if context.configurator.get_config() is None:
                            raise ValueError("Accepted-work configuration unavailable")
                        run = storage.get_agent_run(str(item["id"]))
                        if run is None:
                            continue
                        if run.binding.org_id != storage.org_id:
                            raise ValueError("Resume run org changed")
                        user = run.binding.user_id
                        if not user or not user.strip():
                            if run.binding.extractor_kind != "playbook":
                                raise ValueError("Resume user unavailable")
                            user = infer_playbook_review_user_id(
                                storage=storage,
                                source_interaction_ids=run.binding.source_interaction_ids,
                                subject="Resume selection",
                            )
                        token = storage.claim_user_extraction(
                            user, worker_id, lease_seconds
                        )
                        report_background_success(admission_scope)
                        if token is not None:
                            return user, token
                        try:
                            storage._observe_live_user_work_parent(
                                expected=run, user_id=user
                            )
                        except Exception as exc:
                            if not execution_input_is_busy(exc):
                                raise
                            # Independent observation rolled back; no logical
                            # mutation or authority was acquired by inspection.
                            record_health("resume.selection_refused", 1, kind="counter")
                except Exception as exc:
                    report_failure_at_interval(
                        logger,
                        "resume_admission_failed",
                        exc,
                        scope=admission_scope,
                        interval_seconds=None,
                        detail=_failure_detail,
                        org_id=storage.org_id,
                    )
        return None

    def select_owned(
        self, context: RequestContext, *, user_id: str, token: str, worker_id: str
    ) -> tuple[WorkClass | None, AgentRunRecord | None]:
        from reflexio.server.services.durable_learning.user_lease import (
            _require_user_work_owner,
        )

        storage = context.storage
        if storage is None:
            raise ValueError("Owned resume storage unavailable")
        owner = _require_user_work_owner(storage)
        for _ in ("learning", "finalization", "continuation"):
            owner.fence(storage)
            position = storage._read_owned_work_position(user_id, token)
            kind = position.next_work_class
            if kind == "learning":
                selected, _ = storage._commit_owned_work_selection(
                    user_id=user_id,
                    token=token,
                    expected_position=position,
                    work_class=kind,
                    worker_id=worker_id,
                )
                return (kind if selected else None), None
            cursor = getattr(position, kind)
            item = None
            user_ref = hashlib.sha256(user_id.encode()).hexdigest()[:12]
            discovery_scope = (
                f"durable-owned-resume-discovery:{storage.org_id}:{user_ref}:{kind}"
            )
            try:
                cursor, item = storage._resume_candidate_page(
                    kind, cursor=cursor, user_id=user_id, token=token
                )
                report_background_success(discovery_scope)
            except Exception as exc:
                storage._commit_owned_work_selection(
                    user_id=user_id,
                    token=token,
                    expected_position=position,
                    work_class=kind,
                    cursor=cursor,
                    worker_id=worker_id,
                )
                report_failure_at_interval(
                    logger,
                    "owned_resume_discovery_failed",
                    exc,
                    scope=discovery_scope,
                    interval_seconds=None,
                    detail=_failure_detail,
                    org_id=storage.org_id,
                )
                return kind, None
            key = hashlib.sha256(
                str(item["id"] if item is not None else kind).encode()
            ).hexdigest()[:12]
            scope = f"durable-owned-resume-selection:{storage.org_id}:{key}"
            try:
                if item is None:
                    selected, _ = storage._commit_owned_work_selection(
                        user_id=user_id,
                        token=token,
                        expected_position=position,
                        work_class=kind,
                        cursor=cursor,
                        worker_id=worker_id,
                    )
                else:
                    with bind_work_scope(_descriptor_scope(storage, item)):
                        # Revalidate after waiting for the real user owner.
                        if context.configurator.get_config() is None:
                            raise ValueError("Accepted-work configuration unavailable")
                        run = storage.get_agent_run(str(item["id"]))
                        selected, adopted = storage._commit_owned_work_selection(
                            user_id=user_id,
                            token=token,
                            expected_position=position,
                            work_class=kind,
                            cursor=cursor,
                            expected_run=run,
                            worker_id=worker_id,
                        )
                    report_background_success(scope)
                    return kind, adopted
            except Exception as exc:
                if isinstance(exc, LeaseLostError):
                    raise
                # The failed APP writer has completely rolled back. Consume
                # only private progress under the STILL-live actual owner.
                storage._commit_owned_work_selection(
                    user_id=user_id,
                    token=token,
                    expected_position=position,
                    work_class=kind,
                    cursor=cursor,
                    worker_id=worker_id,
                )
                if execution_input_is_busy(exc):
                    record_health("worker.retries", phase="resume_input_busy")
                else:
                    report_failure_at_interval(
                        logger,
                        "owned_resume_selection_failed",
                        exc,
                        scope=scope,
                        interval_seconds=None,
                        detail=_failure_detail,
                        org_id=storage.org_id,
                    )
                return kind, None
            if not selected:
                return None, None
        return None, None
