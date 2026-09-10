"""Discover durable work fairly; never queue claims behind busy workers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_str
from reflexio.server.scheduling import ThreadedScheduler
from reflexio.server.services.durable_learning.worker import (
    DurableLearningWorker,
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
        super().__init__(thread_name="reflexio-durable-learning-scheduler")
        self._worker = DurableLearningWorker(
            request_context_factory, instance_id=instance_id
        )
        self._org_ids_provider = org_ids_provider
        self._poll = max(
            0.01, float(env_str("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "2.0"))
        )
        self._lease = max(
            3, int(env_str("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", "300"))
        )
        self._last_org: str | None = None

    def _run_once(self) -> float:
        try:
            orgs = sorted(set(self._org_ids_provider()))
            if not orgs:
                return self._poll
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
        except Exception:
            logger.exception("Durable extraction discovery failed")
        return self._poll


def maybe_start_durable_learning(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_ids_provider: Callable[[], Iterable[str]] | None = None,
) -> DurableLearningScheduler:
    def default_provider() -> list[str]:
        storage = request_context_factory(bootstrap_org_id).storage
        return storage.list_extraction_orgs() if storage is not None else []

    scheduler = DurableLearningScheduler(
        request_context_factory=request_context_factory,
        org_ids_provider=org_ids_provider or default_provider,
    )
    scheduler.start()
    from reflexio.server.services.durable_learning.local import adopt_server_scheduler

    adopt_server_scheduler(scheduler)
    return scheduler
