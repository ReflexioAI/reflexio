"""Deployment-owned selection and ordering within final native USER results."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.storage.error import ReadinessUnavailableError

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.server.services.storage.storage_base import BaseStorage


class UserPlaybookSelector(Protocol):
    def select(
        self,
        *,
        items: list[UserPlaybook],
        storage: BaseStorage,
        request: UnifiedSearchRequest,
        org_id: str,
        degraded: bool,
        created_after: datetime | None,
        created_before: datetime | None,
        excluded_ids: frozenset[int],
    ) -> list[UserPlaybook]:
        """Return only original item instances; reordering and removal are allowed."""
        ...


USER_PLAYBOOK_SELECTOR = ServiceKey[UserPlaybookSelector](
    "final_user_playbook_selector"
)


def select_user_playbooks(
    *,
    items: list[UserPlaybook],
    storage: BaseStorage,
    request: UnifiedSearchRequest,
    org_id: str,
    degraded: bool,
    created_after: datetime | None,
    created_before: datetime | None,
    excluded_ids: frozenset[int],
) -> list[UserPlaybook]:
    """Keep native ownership and identity while permitting a selected ordering."""
    provider = get_service(USER_PLAYBOOK_SELECTOR)
    if provider is None or not items:
        return items
    originals = tuple(items)
    selected = provider.select(
        items=list(originals),
        storage=storage,
        request=request,
        org_id=org_id,
        degraded=degraded,
        created_after=created_after,
        created_before=created_before,
        excluded_ids=excluded_ids,
    )
    identities = {id(item) for item in originals}
    if (
        not isinstance(selected, list)
        or len({id(item) for item in selected}) != len(selected)
        or any(id(item) not in identities for item in selected)
    ):
        raise ReadinessUnavailableError(
            "User selection must return distinct original native items"
        )
    return selected
