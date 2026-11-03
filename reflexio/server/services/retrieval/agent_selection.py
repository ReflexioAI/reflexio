"""Deployment-owned selection of final native agent-playbook results.

The native pipeline owns retrieval, ordering, scope and source suppression.
A selected provider may only remove final agent results. Bare OSS installs no
provider; enterprise supplies one at its existing composition root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.storage.error import ReadinessUnavailableError

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.models.api_schema.service_schemas import AgentPlaybook
    from reflexio.server.services.storage.storage_base import BaseStorage


class AgentPlaybookSelector(Protocol):
    def select(
        self,
        *,
        items: list[AgentPlaybook],
        query_embedding: list[float] | None,
        storage: BaseStorage,
        request: UnifiedSearchRequest,
        org_id: str,
        degraded: bool,
    ) -> list[AgentPlaybook]:
        """Return original, unmodified item instances in their original relative order.

        The provider owns source immutability, model compatibility and missing-feature refusals.
        ``None`` is passed unchanged for explicit FTS or degraded retrieval;
        the provider must distinguish those policies rather than inventing a
        query vector or silently replacing failed selection with native output.
        """
        ...


AGENT_PLAYBOOK_SELECTOR = ServiceKey[AgentPlaybookSelector](
    "final_agent_playbook_selector"
)


def select_agent_playbooks(
    *,
    items: list[AgentPlaybook],
    query_embedding: list[float] | None,
    storage: BaseStorage,
    request: UnifiedSearchRequest,
    org_id: str,
    degraded: bool,
) -> list[AgentPlaybook]:
    """Apply the selected provider without changing native retrieval or order."""
    provider = get_service(AGENT_PLAYBOOK_SELECTOR)
    if provider is None or not items:
        return items
    native_items = tuple(items)
    selected = provider.select(
        items=list(native_items),
        query_embedding=query_embedding,
        storage=storage,
        request=request,
        org_id=org_id,
        degraded=degraded,
    )
    # The seam authorizes selection, never rewriting, reranking or backfill.
    # Compare object identities so no embedding/source serialization is added
    # to the caller path merely to enforce that boundary.
    remaining = iter(native_items)
    if not isinstance(selected, list) or any(
        not any(original is item for original in remaining) for item in selected
    ):
        raise ReadinessUnavailableError(
            "Agent selection must return a stable subsequence of native items"
        )
    return selected
