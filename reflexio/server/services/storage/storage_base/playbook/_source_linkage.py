"""Abstract agent playbook source-linkage declarations."""

from abc import abstractmethod
from collections.abc import Sequence

from reflexio.models.api_schema.domain import AgentPlaybookSourceWindow


class PlaybookSourceLinkageMixin:
    """Abstract agent playbook source-linkage methods."""

    @abstractmethod
    def set_source_user_playbook_ids_for_agent_playbook(
        self, agent_playbook_id: int, user_playbook_ids: list[int]
    ) -> None:
        """Persist the source user playbook ids that produced an agent playbook."""
        raise NotImplementedError

    @abstractmethod
    def get_source_user_playbook_ids_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[int]:
        """Return source user playbook ids for an agent playbook."""
        raise NotImplementedError

    @abstractmethod
    def get_source_user_playbook_ids_for_agent_playbooks(
        self, agent_playbook_ids: Sequence[int]
    ) -> dict[int, list[int]]:
        """Return source user playbook ids keyed by agent playbook id."""
        raise NotImplementedError

    @abstractmethod
    def set_source_windows_for_agent_playbook(
        self,
        agent_playbook_id: int,
        source_windows: list[AgentPlaybookSourceWindow],
    ) -> None:
        """Persist replayable source windows that produced an agent playbook."""
        raise NotImplementedError

    @abstractmethod
    def get_source_windows_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[AgentPlaybookSourceWindow]:
        """Return replayable source windows for an agent playbook."""
        raise NotImplementedError
