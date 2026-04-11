from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    PlaybookStatus,
    Status,
    StructuredData,
    UserPlaybook,
)
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
)
from reflexio.models.config_schema import SearchOptions


class PlaybookMixin:
    """Mixin for playbook and agent success evaluation methods."""

    # ==============================
    # User Playbook methods
    # ==============================

    @abstractmethod
    def save_user_playbooks(self, user_playbooks: list[UserPlaybook]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_user_playbooks(
        self,
        limit: int = 100,
        user_id: str | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        include_embedding: bool = False,
    ) -> list[UserPlaybook]:
        """Get user playbooks from storage.

        Args:
            limit (int): Maximum number of playbooks to return
            user_id (str, optional): The user ID to filter by. If None, returns playbooks for all users.
            playbook_name (str, optional): The playbook name to filter by. If None, returns all user playbooks.
            agent_version (str, optional): The agent version to filter by. If None, returns all agent versions.
            status_filter (list[Optional[Status]], optional): List of status values to filter by.
                Can include None (current), Status.PENDING (from rerun), Status.ARCHIVED (old).
                If None, returns playbooks with all statuses.
            start_time (int, optional): Unix timestamp. Only return playbooks created at or after this time.
            end_time (int, optional): Unix timestamp. Only return playbooks created at or before this time.
            include_embedding (bool): If True, fetch and parse embedding vectors. Defaults to False.

        Returns:
            list[UserPlaybook]: List of user playbook objects
        """
        raise NotImplementedError

    @abstractmethod
    def count_user_playbooks(
        self,
        user_id: str | None = None,
        playbook_name: str | None = None,
        min_user_playbook_id: int | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
    ) -> int:
        """Count user playbooks in storage efficiently.

        Args:
            user_id (str, optional): The user ID to filter by. If None, counts playbooks for all users.
            playbook_name (str, optional): The playbook name to filter by. If None, counts all user playbooks.
            min_user_playbook_id (int, optional): Only count playbooks with user_playbook_id greater than this value.
            agent_version (str, optional): The agent version to filter by. If None, counts all agent versions.
            status_filter (list[Optional[Status]], optional): List of status values to filter by.
                Can include None (current), Status.PENDING (from rerun), Status.ARCHIVED (old).
                If None, returns playbooks with all statuses.

        Returns:
            int: Count of user playbooks matching the filters
        """
        raise NotImplementedError

    @abstractmethod
    def count_user_playbooks_by_session(self, session_id: str) -> int:
        """Count user playbooks linked to a session via request_id -> requests.session_id.

        Args:
            session_id (str): The session ID to count user playbooks for

        Returns:
            int: Count of user playbooks linked to the session
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_user_playbooks(self) -> None:
        """Delete all user playbooks from storage."""
        raise NotImplementedError

    @abstractmethod
    def delete_all_user_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        """Delete all user playbooks by playbook name from storage.

        Args:
            playbook_name (str): The playbook name to delete
            agent_version (str, optional): The agent version to filter by. If None, deletes all agent versions.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_user_playbook(self, user_playbook_id: int) -> None:
        """Delete a user playbook by ID.

        Args:
            user_playbook_id (int): The ID of the user playbook to delete
        """
        raise NotImplementedError

    @abstractmethod
    def update_all_user_playbooks_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        """Update all user playbooks with old_status to new_status atomically.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            agent_version: Optional filter by agent version
            playbook_name: Optional filter by playbook name

        Returns:
            int: Number of user playbooks updated
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_user_playbooks_by_status(
        self,
        status: Status,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        """Delete all user playbooks with the given status atomically.

        Args:
            status: The status of user playbooks to delete
            agent_version: Optional filter by agent version
            playbook_name: Optional filter by playbook name

        Returns:
            int: Number of user playbooks deleted
        """
        raise NotImplementedError

    @abstractmethod
    def delete_user_playbooks_by_ids(self, user_playbook_ids: list[int]) -> int:
        """Delete user playbooks by their IDs.

        Args:
            user_playbook_ids: List of user_playbook_id values to delete

        Returns:
            int: Number of user playbooks deleted
        """
        raise NotImplementedError

    @abstractmethod
    def has_user_playbooks_with_status(
        self,
        status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> bool:
        """Check if any user playbooks exist with given status and filters.

        Args:
            status: The status to check for (None for CURRENT)
            agent_version: Optional filter by agent version
            playbook_name: Optional filter by playbook name

        Returns:
            bool: True if any matching user playbooks exist
        """
        raise NotImplementedError

    # ==============================
    # Agent Playbook methods
    # ==============================

    @abstractmethod
    def save_agent_playbooks(
        self, agent_playbooks: list[AgentPlaybook]
    ) -> list[AgentPlaybook]:
        """Save agent playbooks with embeddings.

        Args:
            agent_playbooks (list[AgentPlaybook]): List of agent playbook objects to save

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
    ) -> list[AgentPlaybook]:
        """Get agent playbooks from storage.

        Args:
            limit (int): Maximum number of agent playbooks to return
            playbook_name (str, optional): The playbook name to filter by. If None, returns all agent playbooks.
            agent_version (str, optional): The agent version to filter by. If None, returns all versions.
            status_filter (list[Optional[Status]], optional): List of Status values to filter by. None in the list means CURRENT status.
            playbook_status_filter (Optional[list[PlaybookStatus]]): List of PlaybookStatus values to filter by.
                If None, returns all playbook statuses.

        Returns:
            list[AgentPlaybook]: List of agent playbook objects
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
        structured_data: StructuredData | None = None,
        playbook_status: PlaybookStatus | None = None,
    ) -> None:
        """Update editable fields of an agent playbook. Only non-None fields are updated.

        Args:
            agent_playbook_id (int): The ID of the agent playbook to update
            playbook_name (str, optional): New playbook name
            content (str, optional): New content text
            structured_data (StructuredData, optional): New structured data
            playbook_status (PlaybookStatus, optional): New playbook status

        Raises:
            ValueError: If agent playbook with the given ID is not found
        """
        raise NotImplementedError

    @abstractmethod
    def update_user_playbook(
        self,
        user_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        structured_data: StructuredData | None = None,
    ) -> None:
        """Update editable fields of a user playbook. Only non-None fields are updated.

        Args:
            user_playbook_id (int): The ID of the user playbook to update
            playbook_name (str, optional): New playbook name
            content (str, optional): New content text
            structured_data (StructuredData, optional): New structured data

        Raises:
            ValueError: If user playbook with the given ID is not found
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
    def delete_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        """Permanently delete agent playbooks by their IDs.
        No-op if agent_playbook_ids is empty.

        Args:
            agent_playbook_ids (list[int]): List of agent playbook IDs to delete
        """
        raise NotImplementedError

    # ==============================
    # Search methods
    # ==============================

    @abstractmethod
    def search_user_playbooks(
        self,
        request: SearchUserPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[UserPlaybook]:
        """Search user playbooks with advanced filtering including semantic search.

        Args:
            request (SearchUserPlaybookRequest): Search request with query, filters, and pagination
            options (SearchOptions, optional): Engine-level search parameters (e.g. pre-computed embedding)

        Returns:
            list[UserPlaybook]: List of matching user playbook objects
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

    # ==============================
    # Agent Success Evaluation methods
    # ==============================

    @abstractmethod
    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        """Save agent success evaluation results to storage.

        Args:
            results (list[AgentSuccessEvaluationResult]): List of agent success evaluation results to save
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        """Get agent success evaluation results from storage.

        Args:
            limit (int): Maximum number of results to return
            agent_version (str, optional): The agent version to filter by. If None, returns all results.

        Returns:
            list[AgentSuccessEvaluationResult]: List of agent success evaluation result objects
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_agent_success_evaluation_results(self) -> None:
        """Delete all agent success evaluation results from storage."""
        raise NotImplementedError
