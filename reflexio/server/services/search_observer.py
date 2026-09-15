"""Optional best-effort observation of completed searches, without response content."""

import logging
from dataclasses import dataclass
from typing import Protocol

from reflexio.server.extensions import ServiceKey, get_service

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
    observer = get_service(SEARCH_OBSERVER)
    if observer is not None:
        try:
            observer.observe(search)
        except Exception:
            # An ancillary progress write must never break a successful retrieval.
            logger.warning("Completed search observation failed", exc_info=True)
