"""Library lifecycle registration for the same durable extraction scheduler."""

import threading

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning._context_registry import (
    WeakContextRegistry,
)
from reflexio.server.services.durable_learning.discovery import (
    IncompleteWorkDiscoveryError,
    WorkDiscovery,
    discover_work,
)
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler

_lock = threading.RLock()

# One ordered weak catalog answers both current identity and liveness. A newer
# same-key handle never erases an older live handle; GC reveals that predecessor
# instead of leaving a running scheduler with an empty discovery roster.
_contexts = WeakContextRegistry()
_schedulers: dict[str | None, DurableLearningScheduler] = {}
_server_scheduler: DurableLearningScheduler | None = None


def _start_scheduler(directory: str | None) -> DurableLearningScheduler:
    """Build and start the scheduler that polls one storage directory.

    Both closures capture only ``directory``, never a context, so the registry
    stays the sole owner of context lifetime.

    Args:
        directory (str | None): Storage base directory the scheduler serves.

    Returns:
        DurableLearningScheduler: The started scheduler.
    """

    def factory(org_id: str) -> RequestContext:
        with _lock:
            cached = _contexts.current(org_id, directory)
        return cached or RequestContext(org_id=org_id, storage_base_dir=directory)

    def discover() -> WorkDiscovery:
        with _lock:
            contexts = _contexts.snapshot(directory)
        orgs: set[str] = set()
        expiry: set[str] = set()
        failure: Exception | None = None
        for ctx in contexts:
            if ctx.storage is None:
                continue
            try:
                if ctx.storage.org_id != ctx.org_id:
                    raise ValueError("Local discovery source org mismatch")
                result = discover_work(ctx.storage)
                orgs.update(result.org_ids)
                expiry.update(result.expiry_org_ids)
            except IncompleteWorkDiscoveryError as exc:
                orgs.update(exc.org_ids)
                expiry.update(exc.expiry_org_ids)
                if failure is None:
                    failure = exc
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise IncompleteWorkDiscoveryError(orgs, expiry_org_ids=expiry) from failure
        return WorkDiscovery(tuple(sorted(orgs)), tuple(sorted(expiry)))

    scheduler = DurableLearningScheduler(
        request_context_factory=factory, org_ids_provider=discover
    )
    scheduler.start()
    return scheduler


def _take_orphan_schedulers() -> list[DurableLearningScheduler]:
    """Unregister the schedulers whose directory has no live context left.

    Must be called with ``_lock`` held. The caller stops the returned
    schedulers *after* releasing the lock: ``stop()`` joins the scheduler
    thread, whose discovery callback takes ``_lock`` itself, so stopping under
    the lock would stall until the join timed out and leave the thread alive.

    Returns:
        list[DurableLearningScheduler]: Schedulers removed from the registry,
        which the caller owns and must stop.
    """
    live_directories = _contexts.directories()
    return [
        _schedulers.pop(directory)
        for directory in list(_schedulers)
        if directory not in live_directories
    ]


def ensure_local_extraction(context: RequestContext) -> None:
    """Keep a durable extraction scheduler polling this context's directory.

    Registers ``context`` weakly and starts a scheduler for its directory if
    one is not already running, then retires any scheduler whose directory has
    no reachable context left.

    Args:
        context (RequestContext): Live context whose storage directory needs
            local durable extraction.
    """
    if context.storage is None:
        return
    directory = context.storage_base_dir
    with _lock:
        if _server_scheduler is not None and _server_scheduler.is_running():
            _server_scheduler.wake()
            return
        _contexts.register(context)
        orphans = _take_orphan_schedulers()
        if directory not in _schedulers:
            _schedulers[directory] = _start_scheduler(directory)
        _schedulers[directory].wake()
    for orphan in orphans:
        orphan.stop()


def wake_extraction(context: RequestContext) -> None:
    """Signal an existing lifecycle owner after a committed storage change.

    A callback never creates a consumer, registers a request context, loads
    model configuration or claims work. A stopped/missing owner leaves the
    persisted work for normal discovery on restart.
    """
    with _lock:
        scheduler = _server_scheduler
        if scheduler is None or not scheduler.is_running():
            scheduler = _schedulers.get(context.storage_base_dir)
    if scheduler is not None and scheduler.is_running():
        scheduler.wake()


def adopt_server_scheduler(scheduler: DurableLearningScheduler) -> None:
    """The server owns discovery once its lifespan starts; library fallback stops."""
    global _server_scheduler
    with _lock:
        previous = list(_schedulers.values())
        _schedulers.clear()
        _contexts.clear()
        _server_scheduler = scheduler
    for local in previous:
        local.stop()
