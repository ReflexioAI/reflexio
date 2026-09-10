"""Library lifecycle registration for the same durable extraction scheduler."""

import threading

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler

_lock = threading.RLock()
_contexts: dict[tuple[str, str | None], RequestContext] = {}
_schedulers: dict[str | None, DurableLearningScheduler] = {}
_server_scheduler: DurableLearningScheduler | None = None


def ensure_local_extraction(context: RequestContext) -> None:
    if context.storage is None:
        return
    directory = context.storage_base_dir
    with _lock:
        if _server_scheduler is not None and _server_scheduler.is_running():
            return
        _contexts[(context.org_id, directory)] = context
        if directory in _schedulers:
            return

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
        _schedulers[directory] = scheduler
        scheduler.start()


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
