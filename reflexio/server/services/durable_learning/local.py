"""Library lifecycle registration for the same durable extraction scheduler."""

import threading
import weakref

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler

_lock = threading.RLock()

# Lifecycle choice: contexts are held WEAKLY, and a directory's scheduler is
# retired once every context for it has been collected.
#
# A library-local scheduler exists to serve live ``Reflexio`` handles, so "a
# handle pointed at this directory is still reachable" is the only honest
# signal that one is still needed. Reference counting would have said the same
# thing but needs an explicit ``close()`` on a public API that has none today —
# every caller would have to remember it, and the ones that forgot would leak
# exactly as before. A weak registry gets the same answer for free and cannot
# be forgotten.
#
# Retirement runs on the *caller's* thread inside ``ensure_local_extraction``
# rather than from inside a tick: ``stop()`` joins the scheduler thread, and a
# thread cannot join itself. Nothing is lost by waiting for the next caller —
# durable work is persisted, and a dropped handle means nobody in this process
# is waiting for its results; a later ``Reflexio`` on the same directory picks
# the backlog up through the usual restart-recovery path.
_contexts: "weakref.WeakValueDictionary[tuple[str, str | None], RequestContext]" = (
    weakref.WeakValueDictionary()
)
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
            cached = _contexts.get((org_id, directory))
        return cached or RequestContext(org_id=org_id, storage_base_dir=directory)

    def discover() -> list[str]:
        with _lock:
            contexts = [
                ctx for (_, path), ctx in _contexts.items() if path == directory
            ]
        orgs = set()
        for ctx in contexts:
            if ctx.storage is not None:
                orgs.update(ctx.storage.list_extraction_orgs())
        return sorted(orgs)

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
    live = {directory for _, directory in _contexts}
    return [_schedulers.pop(d) for d in [*_schedulers] if d not in live]


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
            return
        _contexts[(context.org_id, directory)] = context
        orphans = _take_orphan_schedulers()
        if directory not in _schedulers:
            _schedulers[directory] = _start_scheduler(directory)
    for orphan in orphans:
        orphan.stop()


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
