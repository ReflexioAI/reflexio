"""One deployment-owned decision over the native eligible playbook pools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from reflexio.server import search_runtime
from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.storage.error import ReadinessUnavailableError

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
    from reflexio.server.services.storage.storage_base import BaseStorage

PLAYBOOK_CANDIDATE_LIMIT = 20
PLAYBOOK_RESULT_LIMIT = 5


class JointPlaybookSelector(Protocol):
    def select(
        self,
        *,
        agent_playbooks: list[AgentPlaybook],
        user_playbooks: list[UserPlaybook],
        request: UnifiedSearchRequest,
        org_id: str,
        storage: BaseStorage,
        degraded: bool,
    ) -> tuple[list[AgentPlaybook], list[UserPlaybook]]:
        """Order or remove original eligible objects, without mutating their source."""
        ...


JOINT_PLAYBOOK_SELECTOR = ServiceKey[JointPlaybookSelector]("joint_playbook_selector")


def joint_playbook_selector(
    request: UnifiedSearchRequest,
) -> JointPlaybookSelector | None:
    """Resolve only for requests that include a playbook type."""
    if request.entity_types and not {
        "agent_playbooks",
        "user_playbooks",
    }.intersection(request.entity_types):
        return None
    return get_service(JOINT_PLAYBOOK_SELECTOR)


def select_playbooks(
    *,
    provider: JointPlaybookSelector,
    agent_playbooks: list[AgentPlaybook],
    user_playbooks: list[UserPlaybook],
    request: UnifiedSearchRequest,
    org_id: str,
    storage: BaseStorage,
    degraded: bool,
) -> tuple[list[AgentPlaybook], list[UserPlaybook]]:
    """Enforce native identity at the shared selection boundary."""
    originals = (tuple(agent_playbooks), tuple(user_playbooks))
    search_runtime.increment(
        "search.playbooks.selector_input_agents", len(originals[0])
    )
    search_runtime.increment("search.playbooks.selector_input_users", len(originals[1]))
    if any(len(items) > PLAYBOOK_CANDIDATE_LIMIT for items in originals):
        raise ReadinessUnavailableError("Playbook candidate limit exceeded")
    if not any(originals):
        search_runtime.increment("search.playbooks.selected_agents", 0)
        search_runtime.increment("search.playbooks.selected_users", 0)
        return [], []
    original_identifiers = [
        (item, attribute, getattr(item, attribute))
        for items, attribute in zip(
            originals, ("agent_playbook_id", "user_playbook_id"), strict=True
        )
        for item in items
    ]
    selected = provider.select(
        agent_playbooks=list(originals[0]),
        user_playbooks=list(originals[1]),
        request=request,
        org_id=org_id,
        storage=storage,
        degraded=degraded,
    )
    for item, attribute, original_identifier in original_identifiers:
        identifier = getattr(item, attribute, None)
        if (
            type(identifier) is not type(original_identifier)
            or identifier != original_identifier
        ):
            raise ReadinessUnavailableError(
                "Playbook selection changed a native identifier"
            )
    if not isinstance(selected, tuple) or len(selected) != 2:
        raise ReadinessUnavailableError("Playbook selection requires two result lists")
    for items, native in zip(selected, originals, strict=True):
        identities = {id(item) for item in native}
        if (
            not isinstance(items, list)
            or len({id(item) for item in items}) != len(items)
            or any(id(item) not in identities for item in items)
        ):
            raise ReadinessUnavailableError(
                "Playbook selection requires distinct original items"
            )
    search_runtime.increment("search.playbooks.selected_agents", len(selected[0]))
    search_runtime.increment("search.playbooks.selected_users", len(selected[1]))
    return selected
