"""Abstract user playbook CRUD + search declarations."""

from abc import abstractmethod

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain import Status, UserPlaybook
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.config_schema import SearchOptions


class UserPlaybookStoreMixin:
    """Abstract user playbook CRUD + search methods."""

    @abstractmethod
    def save_user_playbooks(
        self,
        user_playbooks: list[UserPlaybook],
        *,
        skip_embedding: bool = False,
    ) -> None:
        """Insert user playbooks, assigning survivor ids.

        Args:
            user_playbooks: Playbooks to insert.
            skip_embedding: When ``False`` (default — what every current caller
                gets), the embedding (and, when document expansion is enabled,
                ``expanded_terms``) is recomputed unconditionally at write time,
                exactly as before. Callers that ``model_copy`` a DB-loaded row
                with changed content but the old embedding preserved rely on
                this recompute, so it must stay the default. When ``True``, the
                embedding step is skipped because ``.embedding`` was already
                populated up front by
                :meth:`precompute_user_playbook_embeddings` (the durable
                compute/persist split — the only caller that opts in). A bare
                ``if not up.embedding`` guard is deliberately NOT used: it would
                persist a stale vector for the ``model_copy`` callers' changed
                content (silent search corruption).
        """
        raise NotImplementedError

    @abstractmethod
    def precompute_user_playbook_embeddings(
        self, playbooks: list[UserPlaybook]
    ) -> None:
        """Populate ``.embedding`` (and ``.expanded_terms`` when document
        expansion is enabled) on each playbook in place, issuing NO DB write.

        Lets the durable learning worker do the (slow, LLM/embedding) compute
        outside the writer transaction, then pass ``skip_embedding=True`` to
        :meth:`save_user_playbooks` so the fenced persist writes the
        pre-computed vector without re-embedding.
        """
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
        tags: list[str] | None = None,
        offset: int = 0,
        user_playbook_id: int | None = None,
        request_id: str | None = None,
        query: str | None = None,
    ) -> list[UserPlaybook]:
        """Get user playbooks from storage.

        Args:
            limit (int): Maximum number of playbooks to return
            user_playbook_id (int, optional): Exact user playbook ID to retrieve.
            user_id (str, optional): The user ID to filter by. If None, returns playbooks for all users.
            request_id (str, optional): Request ID that generated the playbook.
            query (str, optional): Case-insensitive text filter across visible fields.
            playbook_name (str, optional): The playbook name to filter by. If None, returns all user playbooks.
            agent_version (str, optional): The agent version to filter by. If None, returns all agent versions.
            status_filter (list[Optional[Status]], optional): List of status values to filter by.
                Can include None (current), Status.PENDING (from rerun), Status.ARCHIVED (old).
                If None, returns playbooks with all statuses.
            start_time (int, optional): Unix timestamp. Only return playbooks created at or after this time.
            end_time (int, optional): Unix timestamp. Only return playbooks created at or before this time.
            include_embedding (bool): If True, fetch and parse embedding vectors. Defaults to False.
            tags (list[str], optional): Match playbooks having any of these tags.
            offset (int): Number of matching rows to skip. Defaults to 0.

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
    def delete_user_playbooks_by_ids(
        self, user_playbook_ids: list[int], *, emit_hard_delete: bool = True
    ) -> int:
        """Delete user playbooks by their IDs.

        Args:
            user_playbook_ids: List of user_playbook_id values to delete
            emit_hard_delete: When True (default), append a ``hard_delete``
                lineage event per id (genuine erasure). Set False for rollback
                cleanup of a never-live row (e.g. a lost supersede CAS), so no
                spurious audit event is recorded.

        Returns:
            int: Number of user playbooks deleted
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_playbooks_by_ids(
        self,
        user_id: str,
        user_playbook_ids: list[int],
        status_filter: list[Status | None] | None = None,
    ) -> list[UserPlaybook]:
        """Fetch the subset of a user's playbooks whose ids are in the list.

        Server-side filter on (``user_id``, ``user_playbook_id IN (...)``)
        so callers (e.g. the reflection service resolving a small set of
        cited playbook ids) avoid scanning every playbook for the user.

        Args:
            user_id (str): Owning user id.
            user_playbook_ids (list[int]): Playbook ids to fetch. Empty
                list returns ``[]`` without hitting storage.
            status_filter (list[Status | None] | None): Statuses to
                include. ``None`` (default) means CURRENT only — same
                default as ``get_user_playbooks`` for consistency.

        Returns:
            list[UserPlaybook]: Matching playbooks. Order is unspecified.
                Ids that do not exist (or do not match the user / status
                filter) are silently omitted.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_playbook_by_id(
        self, user_playbook_id: int, *, include_tombstones: bool = False
    ) -> UserPlaybook | None:
        """Fetch one user playbook by primary key.

        Args:
            user_playbook_id: The user_playbook_id to look up.
            include_tombstones: When False (default), MERGED/SUPERSEDED rows
                return None. Set to True for lineage resolution (resolve_current).

        Returns:
            The UserPlaybook if found and not filtered, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_playbooks_by_ids_any_user(
        self,
        user_playbook_ids: list[int],
        status_filter: list[Status | None] | None = None,
    ) -> list[UserPlaybook]:
        """Fetch user playbooks by ids without requiring a single owner id."""
        raise NotImplementedError

    @abstractmethod
    def archive_user_playbook_by_id(self, user_id: str, user_playbook_id: int) -> bool:
        """Atomically archive a single user playbook by id, only if CURRENT.

        Flips the row's ``status`` from ``None`` (CURRENT) to
        ``Status.ARCHIVED``. No-op when the playbook does not exist, has
        a different ``user_id``, or is already non-current.

        Args:
            user_id (str): Owning user id; used as a guard so callers
                cannot accidentally archive another user's playbook.
            user_playbook_id (int): The user_playbook_id to archive.

        Returns:
            bool: True if a row was archived; False otherwise.
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

    @abstractmethod
    def update_user_playbook(
        self,
        user_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
        blocking_issue: BlockingIssue | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Update editable fields of a user playbook. Only non-None fields are updated.

        Args:
            user_playbook_id (int): The ID of the user playbook to update
            playbook_name (str, optional): New playbook name
            content (str, optional): New content text
            trigger (str, optional): New trigger text
            rationale (str, optional): New rationale text
            blocking_issue (BlockingIssue, optional): New blocking issue
            tags (list[str], optional): Replacement tags

        Raises:
            ValueError: If user playbook with the given ID is not found
        """
        raise NotImplementedError

    @abstractmethod
    def supersede_user_playbooks_by_ids(
        self, user_playbook_ids: list[int], request_id: str
    ) -> int:
        """Soft-delete user playbooks by setting status to SUPERSEDED.

        Eligible rows (CURRENT, PENDING, or ARCHIVED; not already MERGED /
        SUPERSEDED) are transitioned to SUPERSEDED and emit one status_change
        lineage event under the shared request id. This is the user-playbook
        analogue of the existing agent/profile soft-supersede helpers and
        preserves dead-source content for point-in-time attribution reads.

        Args:
            user_playbook_ids (list[int]): User playbook ids to supersede.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of user playbooks actually updated.
        """
        raise NotImplementedError

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
