"""Process-local scheduler for the durable learning-job queue (Task 6).

Each tick discovers the orgs with actionable learning jobs (via an injected
``org_ids_provider``) and drains each through a :class:`DurableLearningWorker`.
The worker's claims are org-scoped and the fenced complete/fail transitions make
the drain exactly-once, so racing scheduler instances across processes are safe.

Gating: startup is gated on ``REFLEXIO_DURABLE_LEARNING_QUEUE`` (a rollout flag,
scalability design 2026-07-04 §8 stage 2). When off, :meth:`start` is a no-op.

Multi-ref capability (design §3.1, per-ref polling): the mechanism supports
per-data-ref fan-out — inject an ``org_ids_provider`` that yields orgs across
refs plus a ``request_context_factory`` that routes each org to its ref's
storage. The default provider (single-ref: bootstrap-storage discovery) is wired
in :func:`maybe_start_durable_learning`; the enterprise cross-ref enumerator is a
deferred follow-up (tracked with the pre-Task-8 gates).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_str, env_truthy
from reflexio.server.scheduling import ThreadedScheduler
from reflexio.server.services.durable_learning.worker import DurableLearningWorker

logger = logging.getLogger(__name__)


class DurableLearningScheduler(ThreadedScheduler):
    """Polling daemon that drains the durable learning queue per org.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
            Passed straight to the worker; a multi-ref deployment routes each org
            to its data ref's storage here.
        org_ids_provider: Called once per tick; yields the org_ids to drain. The
            single-ref default (see :func:`maybe_start_durable_learning`) reads
            them from the bootstrap storage's cross-org discovery query.
        instance_id: Stable label written to ``claimed_by`` on each claim.
    """

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        org_ids_provider: Callable[[], Iterable[str]],
        instance_id: str | None = None,
    ) -> None:
        super().__init__(thread_name="reflexio-durable-learning-scheduler")
        self._worker = DurableLearningWorker(
            request_context_factory, instance_id=instance_id
        )
        self._org_ids_provider = org_ids_provider
        # Read env INSIDE __init__ so tests can set the values before construction.
        self._poll = float(env_str("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "2.0"))
        self._batch = int(env_str("REFLEXIO_DURABLE_LEARNING_BATCH", "16"))
        self._lease = int(env_str("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", "300"))

    def _should_start(self) -> bool:
        if env_truthy(env_str("REFLEXIO_DURABLE_LEARNING_QUEUE", "false")):
            return True
        logger.info(
            "event=durable_learning_scheduler_disabled "
            "REFLEXIO_DURABLE_LEARNING_QUEUE not set — not starting"
        )
        return False

    def _on_started(self) -> None:
        logger.info("event=durable_learning_scheduler_started")

    def _on_stopped(self) -> None:
        logger.info("event=durable_learning_scheduler_stopped")

    def _run_once(self) -> float:
        """Drain every discovered org once; per-org error isolation.

        A single org raising never aborts the tick, and a failure to enumerate
        orgs never kills the daemon thread. Returns the poll interval.
        """
        try:
            for org_id in self._org_ids_provider():
                if self._stop_event.is_set():
                    break
                try:
                    self._worker.drain_org(org_id, self._batch, self._lease)
                except Exception:
                    logger.exception(
                        "event=durable_learning_drain_failed org_id=%s", org_id
                    )
        except Exception:
            logger.exception("event=durable_learning_tick_failed")
        return self._poll


def maybe_start_durable_learning(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_ids_provider: Callable[[], Iterable[str]] | None = None,
) -> DurableLearningScheduler | None:
    """Start the durable-learning scheduler when the rollout flag is on.

    Returns ``None`` when ``REFLEXIO_DURABLE_LEARNING_QUEUE`` is off (mirrors
    ``maybe_start_resume_scheduler``). The default ``org_ids_provider`` performs
    single-ref discovery: it queries the bootstrap org's storage for every org
    with actionable work (covers OSS/local and the shared managed DATA ref). The
    enterprise cross-ref enumerator is a deferred follow-up.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
        bootstrap_org_id: Org used to reach a storage for cross-org discovery.
        org_ids_provider: Optional explicit provider (e.g. a future enterprise
            cross-ref enumerator); when ``None`` the single-ref default is used.
    """
    if not env_truthy(env_str("REFLEXIO_DURABLE_LEARNING_QUEUE", "false")):
        return None

    def _default_provider() -> list[str]:
        try:
            ctx = request_context_factory(bootstrap_org_id)
            storage = getattr(ctx, "storage", None)
            if storage is None:
                return []
            return list(storage.list_org_ids_with_pending_learning_jobs())
        except NotImplementedError:
            return []
        except Exception:
            logger.exception(
                "event=durable_learning_discovery_failed bootstrap_org_id=%s",
                bootstrap_org_id,
            )
            return []

    scheduler = DurableLearningScheduler(
        request_context_factory=request_context_factory,
        org_ids_provider=org_ids_provider or _default_provider,
    )
    scheduler.start()
    return scheduler
