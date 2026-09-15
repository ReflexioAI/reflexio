"""Optional best-effort observation of completed searches, without response content."""

import logging
from dataclasses import dataclass
from typing import Protocol

from reflexio.server.callback_executor import submit_callback
from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.work_scope import WorkScope, bind_work_scope, current_project_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletedSearch:
    org_id: str
    caller_type: str
    user_id: str | None
    session_id: str | None
    request_id: str | None
    profile_ids: tuple[str, ...]
    user_playbook_ids: tuple[str, ...]


class SearchObserver(Protocol):
    def observe(self, search: CompletedSearch) -> None: ...


SEARCH_OBSERVER = ServiceKey[SearchObserver]("completed_search_observer")


def observe_completed_search(search: CompletedSearch) -> None:
    """Queue observation without waiting; saturation drops the oldest callback.

    Capture only the tenant identifiers needed to restore scope on the worker.
    A dropped or failed observation can be retried by running the search again.
    """
    observer = get_service(SEARCH_OBSERVER)
    if observer is None:
        return
    try:
        scope = WorkScope(org_id=search.org_id, project_id=current_project_id())

        def observe() -> None:
            with bind_work_scope(scope):
                observer.observe(search)

        # The shared executor bounds workers/queue, logs dropped work and catches
        # callback errors. Never wait for database work on the response path.
        submit_callback("completed_search_observer", observe)
    except Exception:
        logger.warning("Completed search observation enqueue failed", exc_info=True)
