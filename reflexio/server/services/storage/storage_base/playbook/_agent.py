"""Abstract agent playbook CRUD + search declarations."""

from abc import abstractmethod

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    PlaybookStatus,
    Status,
)
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
)
from reflexio.models.config_schema import SearchOptions


class AgentPlaybookStoreMixin:
    """Abstract agent playbook CRUD + search methods."""

    @abstractmethod
    def save_agent_playbooks(
        self,
        agent_playbooks: list[AgentPlaybook],
        *,
        lineage_contexts: list[LineageContext] | None = None,
    ) -> list[AgentPlaybook]:
        """Save agent playbooks and their origin lineage events atomically.

        Args:
            agent_playbooks (list[AgentPlaybook]): List of agent playbook objects to save
            lineage_contexts: Optional per-row create or aggregate attribution.
                When omitted, storage emits a create event with null model/provider.

        Returns:
            list[AgentPlaybook]: Saved agent playbooks with agent_playbook_id populated from storage
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_playbooks(
        self,
        limit: int = 100,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
        tags: list[str] | None = None,
        agent_playbook_id: int | None = None,
        query: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        offset: int = 0,
    ) -> list[AgentPlaybook]:
        """Get agent playbooks from storage.

        Args:
            limit (int): Maximum number of agent playbooks to return
            agent_playbook_id (int, optional): Exact agent playbook ID to retrieve.
            query (str, optional): Case-insensitive text filter across visible fields.
            playbook_name (str, optional): The playbook name to filter by. If None, returns all agent playbooks.
            agent_version (str, optional): The agent version to filter by. If None, returns all versions.
            start_time (int, optional): Unix timestamp. Only return playbooks created at or after this time.
            end_time (int, optional): Unix timestamp. Only return playbooks created at or before this time.
            status_filter (list[Optional[Status]], optional): List of Status values to filter by. None in the list means CURRENT status.
            playbook_status_filter (Optional[list[PlaybookStatus]]): List of PlaybookStatus values to filter by.
                If None, returns all playbook statuses.
            tags (list[str], optional): Match playbooks having any of these tags.
            offset (int): Number of matching rows to skip. Defaults to 0.

        Returns:
            list[AgentPlaybook]: List of agent playbook objects
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_playbook_by_id(
        self, agent_playbook_id: int, *, include_tombstones: bool = False
    ) -> AgentPlaybook | None:
        """Fetch one agent playbook by primary key.

        Args:
            agent_playbook_id: The agent_playbook_id to look up.
            include_tombstones: When False (default), MERGED/SUPERSEDED rows
                return None. Set to True for lineage resolution (resolve_current).

        Returns:
            The AgentPlaybook if found and not filtered, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_playbooks_by_ids(
        self,
        agent_playbook_ids: list[int],
        *,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
        include_inactive: bool = False,
    ) -> list[AgentPlaybook]:
        """Fetch agent playbooks in bulk with lifecycle filters.

        Args:
            agent_playbook_ids (list[int]): Playbook ids to fetch. Empty list
                returns ``[]`` without hitting storage.
            status_filter (list[Status | None] | None): Lifecycle statuses to
                include. ``None`` (default) means CURRENT only.
            playbook_status_filter (list[PlaybookStatus] | None): Approval
                statuses to include. ``None`` (default) means no approval filter.
            include_inactive (bool): Return every matching row regardless of
                lifecycle status *or* approval status. This is the *historical
                resolution* mode (see ``RetrievedLearningEvaluator``): it answers
                "what did this id point at", not "what is retrievable now", so an
                archived or never-approved playbook that was actually served is
                still returned. It is a strict superset of ``include_tombstones``
                on ``get_agent_playbook_by_id``, which only unhides
                MERGED/SUPERSEDED for lineage walks. The default preserves
                retrieval behavior.

        Returns:
            list[AgentPlaybook]: Matching playbooks. Order is unspecified.

        Raises:
            StorageError: If ``include_inactive`` is combined with an explicit
                ``status_filter`` or ``playbook_status_filter`` — the two are
                contradictory.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_agent_playbooks(self) -> None:
        """Delete all agent playbooks from storage."""
        raise NotImplementedError

    @abstractmethod
    def delete_agent_playbook(self, agent_playbook_id: int) -> None:
        """Delete an agent playbook by ID.

        Args:
            agent_playbook_id (int): The ID of the agent playbook to delete
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        """Delete all agent playbooks by playbook name from storage.

        Args:
            playbook_name (str): The playbook name to delete
            agent_version (str, optional): The agent version to filter by. If None, deletes all agent versions.
        """
        raise NotImplementedError

    @abstractmethod
    def update_agent_playbook_status(
        self, agent_playbook_id: int, playbook_status: PlaybookStatus
    ) -> None:
        """Update the status of a specific agent playbook.

        Args:
            agent_playbook_id (int): The ID of the agent playbook to update
            playbook_status (PlaybookStatus): The new status to set

        Raises:
            ValueError: If agent playbook with the given ID is not found
        """
        raise NotImplementedError

    @abstractmethod
    def update_agent_playbook(
        self,
        agent_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
        blocking_issue: BlockingIssue | None = None,
        playbook_status: PlaybookStatus | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Update editable fields of an agent playbook. Only non-None fields are updated.

        Args:
            agent_playbook_id (int): The ID of the agent playbook to update
            playbook_name (str, optional): New playbook name
            content (str, optional): New content text
            trigger (str, optional): New trigger text
            rationale (str, optional): New rationale text
            blocking_issue (BlockingIssue, optional): New blocking issue
            playbook_status (PlaybookStatus, optional): New playbook status
            tags (list[str], optional): Replacement tags

        Raises:
            ValueError: If agent playbook with the given ID is not found
        """
        raise NotImplementedError

    @abstractmethod
    def archive_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        """Archive non-APPROVED agent playbooks by setting their status field to 'archived'.
        APPROVED agent playbooks are left untouched to preserve user-approved playbooks.

        Args:
            playbook_name (str): The playbook name to archive
            agent_version (str, optional): The agent version to filter by. If None, archives all agent versions.
        """
        raise NotImplementedError

    @abstractmethod
    def archive_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        """Archive non-APPROVED agent playbooks by IDs, setting their status field to 'archived'.
        APPROVED agent playbooks are left untouched. No-op if agent_playbook_ids is empty.

        Args:
            agent_playbook_ids (list[int]): List of agent playbook IDs to archive
        """
        raise NotImplementedError

    @abstractmethod
    def supersede_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int], request_id: str
    ) -> int:
        """Soft-delete agent playbooks by setting status to SUPERSEDED, emitting set-based lineage.

        For each eligible id (not APPROVED, not already tombstoned), updates status to
        SUPERSEDED and emits one status_change event under the shared request_id.
        Atomic: mutation and event in one commit, guarded on rowcount.
        FTS/vec rows are NOT removed.

        Args:
            agent_playbook_ids (list[int]): Agent playbook ids to supersede.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of agent playbooks actually updated.
        """
        raise NotImplementedError

    @abstractmethod
    def supersede_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None, request_id: str
    ) -> int:
        """Soft-delete archived agent playbooks by name/version via SUPERSEDED status.

        Selects rows with playbook_name matching and status='archived', then
        soft-supersedes each one with a status_change lineage event under request_id.
        Atomic: one commit at the end.
        FTS/vec rows are NOT removed.

        Args:
            playbook_name (str): Playbook name to supersede.
            agent_version (str | None): Agent version filter. None matches all versions.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of agent playbooks actually updated.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        """Restore archived agent playbooks by setting their status field to null.

        Args:
            playbook_name (str): The playbook name to restore
            agent_version (str, optional): The agent version to filter by. If None, restores all agent versions.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_archived_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> None:
        """Restore archived agent playbooks by IDs, setting their status field to null.
        No-op if agent_playbook_ids is empty.

        Args:
            agent_playbook_ids (list[int]): List of agent playbook IDs to restore
        """
        raise NotImplementedError

    @abstractmethod
    def delete_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        """Permanently delete agent playbooks that have status='archived'.

        Args:
            playbook_name (str): The playbook name to delete
            agent_version (str, optional): The agent version to filter by. If None, deletes all agent versions.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int], *, emit_hard_delete: bool = True
    ) -> None:
        """Permanently delete agent playbooks by their IDs.
        No-op if agent_playbook_ids is empty.

        Args:
            agent_playbook_ids (list[int]): List of agent playbook IDs to delete
            emit_hard_delete: When True (default), append a ``hard_delete``
                lineage event per id (genuine erasure). Set False for rollback
                cleanup of a never-live row (e.g. a lost supersede CAS), so no
                spurious audit event is recorded.
        """
        raise NotImplementedError

    @abstractmethod
    def search_agent_playbooks(
        self,
        request: SearchAgentPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[AgentPlaybook]:
        """Search agent playbooks with advanced filtering including semantic search.

        Args:
            request (SearchAgentPlaybookRequest): Search request with query, filters, and pagination
            options (SearchOptions, optional): Engine-level search parameters (e.g. pre-computed embedding)

        Returns:
            list[AgentPlaybook]: List of matching agent playbook objects
        """
        raise NotImplementedError
