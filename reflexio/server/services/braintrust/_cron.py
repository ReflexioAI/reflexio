"""Recurring 15-minute sync scheduler for the Braintrust connector.

Unlike `GroupEvaluationScheduler` (which is event-driven — fires once per
session after a delay), `BraintrustSyncScheduler` is a fixed-interval
daemon: every N seconds, walk every connected org and call
`BraintrustConnectorService.sync_once`.

Singleton per process. Started on first call to `get_instance()`. Idempotent
construction guarded by a class-level lock; the worker thread is a daemon
so it doesn't block process shutdown.

The scheduler discovers connected orgs by asking storage — `BaseStorage`'s
default no-op returns []; concrete backends override (currently SQLite
returns the single org it belongs to, via `org_id` attribute).

`BRAINTRUST_BASE_URL` env override is honored via the `BraintrustClient`
factory: when set, all sync calls hit that URL instead of api.braintrust.dev.
Useful for mocking in dev / staging.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from reflexio.server.services.braintrust.client import (
    DEFAULT_BASE_URL,
    BraintrustClient,
)

logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL_SECONDS = 15 * 60  # 15 min
_TEST_INTERVAL_SECONDS = 5  # IS_TEST_ENV shortcut for fast tests


def _interval_seconds() -> int:
    """Pick the recurring interval based on `IS_TEST_ENV`."""
    if os.environ.get("IS_TEST_ENV", "").strip().lower() == "true":
        return _TEST_INTERVAL_SECONDS
    return _DEFAULT_INTERVAL_SECONDS


def _resolve_base_url() -> str:
    """Honor the `BRAINTRUST_BASE_URL` env override (used in dev/staging)."""
    raw = os.environ.get("BRAINTRUST_BASE_URL", "").strip()
    return raw or DEFAULT_BASE_URL


def make_client_factory_with_base_url() -> Callable[[str], BraintrustClient]:
    """Return a client factory that honors `BRAINTRUST_BASE_URL`.

    Use this anywhere a `client_factory` is needed so the env override
    flows through uniformly (sync scheduler, manual /sync endpoint).
    """
    base_url = _resolve_base_url()

    def factory(api_key: str) -> BraintrustClient:
        return BraintrustClient(api_key, base_url=base_url)

    return factory


@dataclass
class BraintrustSyncScheduler:
    """Singleton recurring scheduler for Braintrust sync_once.

    Args:
        interval_seconds (int): How often to poll. Defaults to 15 min
            (5s under IS_TEST_ENV).
        list_connected_orgs (Callable[[], list[str]]): Discovery hook.
        run_sync_for_org (Callable[[str], None]): Per-org sync action.
            Provided by the caller so the scheduler stays decoupled from
            BraintrustConnectorService construction.
    """

    interval_seconds: int = field(default_factory=_interval_seconds)
    list_connected_orgs: Callable[[], list[str]] = field(default=lambda: [])
    run_sync_for_org: Callable[[str], None] = field(default=lambda _org: None)

    _instance: BraintrustSyncScheduler | None = field(
        default=None, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Spawn the daemon thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="braintrust-sync-scheduler"
            )
            self._thread.start()
            logger.info(
                "BraintrustSyncScheduler started (interval=%ss)",
                self.interval_seconds,
            )

    def stop(self) -> None:
        """Signal the daemon to exit. Returns immediately; thread joins lazily."""
        self._stop.set()

    def _loop(self) -> None:
        """Main loop: every `interval_seconds`, sync every connected org."""
        while not self._stop.is_set():
            try:
                org_ids = self.list_connected_orgs()
            except Exception:  # noqa: BLE001
                logger.exception("BraintrustSyncScheduler: org discovery failed")
                org_ids = []
            for org_id in org_ids:
                if self._stop.is_set():
                    break
                try:
                    self.run_sync_for_org(org_id)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "BraintrustSyncScheduler: sync failed for org=%s", org_id
                    )
            # Sleep interruptibly so stop() takes effect promptly.
            self._stop.wait(timeout=self.interval_seconds)


# Module-level singleton accessor. Tests reset via `_reset_for_test`.
_INSTANCE: BraintrustSyncScheduler | None = None
_GET_LOCK = threading.Lock()


def get_instance(
    *,
    list_connected_orgs: Callable[[], list[str]] | None = None,
    run_sync_for_org: Callable[[str], None] | None = None,
) -> BraintrustSyncScheduler:
    """Get or create the process-wide singleton.

    Args:
        list_connected_orgs (Callable[[], list[str]] | None): Override
            org discovery (defaults to no-op returning []).
        run_sync_for_org (Callable[[str], None] | None): Override the
            per-org sync action (defaults to no-op).

    Returns:
        BraintrustSyncScheduler: The shared instance.
    """
    global _INSTANCE
    with _GET_LOCK:
        if _INSTANCE is None:
            _INSTANCE = BraintrustSyncScheduler(
                list_connected_orgs=list_connected_orgs or (lambda: []),
                run_sync_for_org=run_sync_for_org or (lambda _org: None),
            )
            _INSTANCE.start()
        return _INSTANCE


def _reset_for_test() -> None:
    """Reset the singleton — for tests only."""
    global _INSTANCE
    with _GET_LOCK:
        if _INSTANCE is not None:
            _INSTANCE.stop()
            if _INSTANCE._thread is not None:
                _INSTANCE._thread.join(timeout=2.0)
        _INSTANCE = None


def trigger_tick_now(scheduler: BraintrustSyncScheduler) -> None:
    """Force one immediate sync tick — for tests.

    Tests that don't want to wait for the interval call this to run one
    pass of the loop body synchronously. Mirrors the loop's per-org
    exception handling so a single failing org doesn't abort the tick.
    """
    org_ids = scheduler.list_connected_orgs()
    for org_id in org_ids:
        try:
            scheduler.run_sync_for_org(org_id)
        except Exception:  # noqa: BLE001
            logger.exception("trigger_tick_now: sync failed for org=%s", org_id)


# Tiny no-op `_` reference so unused-import lint doesn't trip on `time`.
_ = time
