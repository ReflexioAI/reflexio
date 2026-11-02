"""Discover durable work fairly; never queue claims behind busy workers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from reflexio import durable_learning_policy
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.background_work import (
    report_background_failure,
    report_background_success,
)
from reflexio.server.scheduling import ThreadedScheduler
from reflexio.server.services.durable_learning.discovery import (
    IncompleteWorkDiscoveryError,
    WorkDiscovery,
    discover_work,
)
from reflexio.server.services.durable_learning.expiry import ExpiryHousekeeping
from reflexio.server.services.durable_learning.worker import (
    DurableLearningWorker,
    exception_detail,
    worker_count,
)

logger = logging.getLogger(__name__)


class DurableLearningScheduler(ThreadedScheduler):
    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        org_ids_provider: Callable[[], Iterable[str]],
        instance_id: str | None = None,
    ):
        timing = durable_learning_policy.resolve_discovery_timing()
        super().__init__(thread_name="reflexio-durable-learning-scheduler")
        self._worker = DurableLearningWorker(
            request_context_factory, instance_id=instance_id
        )
        self._org_ids_provider = org_ids_provider
        self._housekeeping = ExpiryHousekeeping(
            request_context_factory, self._stop_event
        )
        self._poll = timing.poll_seconds
        self._lease = timing.lease_seconds
        self._last_org: str | None = None

    def _run_once(self) -> float:
        failure: Exception | None = None
        try:
            try:
                result = self._org_ids_provider()
                orgs = sorted(set(result))
                expiry_orgs = (
                    result.expiry_org_ids if isinstance(result, WorkDiscovery) else orgs
                )
            except IncompleteWorkDiscoveryError as exc:
                orgs = list(exc.org_ids)
                failure = exc
                expiry_orgs = exc.expiry_org_ids
            self._housekeeping.run(expiry_orgs)
            if orgs:
                if self._last_org in orgs:
                    offset = orgs.index(self._last_org) + 1
                    orgs = orgs[offset:] + orgs[:offset]
                for index in range(worker_count()):
                    if self._stop_event.is_set():
                        break
                    org_id = orgs[index % len(orgs)]
                    if not self._worker.start_org(org_id, self._lease):
                        break
                    self._last_org = org_id
        except Exception as exc:
            failure = exc
        if failure is not None:
            report_background_failure(
                logger,
                "durable_extraction_discovery_failed",
                failure,
                scope="durable-learning-discovery",
                detail=exception_detail,
            )
        else:
            report_background_success("durable-learning-discovery")
        return self._poll


def maybe_start_durable_learning(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_ids_provider: Callable[[], Iterable[str]] | None = None,
) -> DurableLearningScheduler:
    def default_provider() -> WorkDiscovery:
        context = request_context_factory(bootstrap_org_id)
        storage = context.storage
        if context.org_id != bootstrap_org_id:
            raise ValueError("Default discovery source org mismatch")
        if storage is None:
            return WorkDiscovery()
        if storage.org_id != bootstrap_org_id:
            raise ValueError("Default discovery source org mismatch")
        return discover_work(storage)

    scheduler = DurableLearningScheduler(
        request_context_factory=request_context_factory,
        org_ids_provider=org_ids_provider or default_provider,
    )
    scheduler.start()
    from reflexio.server.services.durable_learning.local import adopt_server_scheduler

    adopt_server_scheduler(scheduler)
    return scheduler
