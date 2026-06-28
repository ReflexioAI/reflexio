import logging
from abc import abstractmethod
from collections.abc import Sequence

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    AgentSuccessEvaluationResult,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
    PlaybookStatus,
    Status,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.entities import LineageEvent
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_AGGREGATE_EVENT_EMIT_ATTEMPTS = 3

# Shared prefix for aggregate lineage event reasons.
# Consumers: storage_base (here), sqlite_storage/_playbook.py, and lib/_agent_playbook.py
# (which imports this constant to keep the parser and producers in sync).
AGGREGATE_REASON_PREFIX = "aggregate:"


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
        tags: list[str] | None = None,
        offset: int = 0,
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

    def save_agent_playbook_with_aggregate_event(
        self,
        agent_playbook: AgentPlaybook,
        *,
        source_ids: list[str],
        request_id: str,
        run_mode: str,
    ) -> AgentPlaybook:
        """Persist an agent playbook AND its ``op=aggregate`` lineage event.

        Backends SHOULD override this so the row insert and the event commit in ONE
        transaction (the event is the sole record of the run->playbook membership for
        reconstruction). This base default is a non-atomic save-then-emit fallback
        with bounded retry + loud (level=error) on final failure.

        Args:
            agent_playbook (AgentPlaybook): The playbook to persist.
            source_ids (list[str]): IDs of the source entities that produced this playbook.
            request_id (str): The aggregation run ID (used as the lineage event request_id).
            run_mode (str): The aggregation run mode (e.g. ``full_archive`` or ``incremental``).

        Returns:
            AgentPlaybook: The saved playbook with ``agent_playbook_id`` populated.

        Raises:
            ValueError: If ``request_id`` is empty (would produce an unreconstructable event).
        """
        if not request_id or not request_id.strip():
            raise ValueError(
                "save_agent_playbook_with_aggregate_event requires a non-empty request_id"
            )
        saved = self.save_agent_playbooks([agent_playbook])[0]
        event = LineageEvent(
            org_id=self.org_id,  # type: ignore[attr-defined]
            entity_type="agent_playbook",
            entity_id=str(saved.agent_playbook_id),
            op="aggregate",
            prov_relation="wasDerivedFrom",
            source_ids=source_ids,
            actor="aggregator",
            request_id=request_id,
            reason=f"{AGGREGATE_REASON_PREFIX}{run_mode}",
        )
        # The row is already committed; this default is non-atomic (SQLite overrides it to
        # make the INSERT + event one transaction). The event is the sole reconstruction signal
        # for the run, so make the emit durable: bounded retry (idempotent on retrying the
        # same row's emit — entity_id is a fresh autoincrement per run, so this is NOT
        # cross-run idempotency), and on final failure fail LOUD at level=error so the gap
        # is paged + backfillable rather than silently lost. Never raise — the playbook
        # itself is saved and must not be lost.
        for attempt in range(_AGGREGATE_EVENT_EMIT_ATTEMPTS):
            try:
                self.append_lineage_event(event)  # type: ignore[attr-defined]
                return saved
            except Exception:  # noqa: BLE001
                logger.warning(
                    "aggregate lineage event append failed (attempt %d/%d) for agent_playbook %s",
                    attempt + 1,
                    _AGGREGATE_EVENT_EMIT_ATTEMPTS,
                    saved.agent_playbook_id,
                    exc_info=True,
                )
        capture_anomaly(
            "lineage.aggregate.append_failed",
            level="error",
            entity_id=str(saved.agent_playbook_id),
            org_id=self.org_id,  # type: ignore[attr-defined]
            request_id=request_id,
        )
        return saved

    @abstractmethod
    def get_agent_playbooks(
        self,
        limit: int = 100,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
        tags: list[str] | None = None,
    ) -> list[AgentPlaybook]:
        """Get agent playbooks from storage.

        Args:
            limit (int): Maximum number of agent playbooks to return
            playbook_name (str, optional): The playbook name to filter by. If None, returns all agent playbooks.
            agent_version (str, optional): The agent version to filter by. If None, returns all versions.
            status_filter (list[Optional[Status]], optional): List of Status values to filter by. None in the list means CURRENT status.
            playbook_status_filter (Optional[list[PlaybookStatus]]): List of PlaybookStatus values to filter by.
                If None, returns all playbook statuses.
            tags (list[str], optional): Match playbooks having any of these tags.

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

    # ==============================
    # Playbook optimization methods
    # ==============================

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

    @abstractmethod
    def create_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        """Persist a playbook optimization job and return it with id populated."""
        raise NotImplementedError

    @abstractmethod
    def update_playbook_optimization_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        best_candidate_id: int | None = None,
        successor_target_id: int | None = None,
        decision_reason: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        """Update mutable fields on a playbook optimization job."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_candidate(
        self, candidate: PlaybookOptimizationCandidate
    ) -> PlaybookOptimizationCandidate:
        """Persist an optimizer candidate and return it with id populated."""
        raise NotImplementedError

    @abstractmethod
    def list_playbook_optimization_candidates(
        self, job_id: int
    ) -> list[PlaybookOptimizationCandidate]:
        """List optimizer candidates for a job."""
        raise NotImplementedError

    @abstractmethod
    def update_playbook_optimization_candidate(
        self,
        candidate_id: int,
        *,
        aggregate_score: float | None = None,
        is_winner: bool | None = None,
    ) -> None:
        """Update mutable optimizer candidate result fields."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_evaluation(
        self, evaluation: PlaybookOptimizationEvaluation
    ) -> PlaybookOptimizationEvaluation:
        """Persist an optimizer evaluation and return it with id populated."""
        raise NotImplementedError

    @abstractmethod
    def list_playbook_optimization_evaluations(
        self, job_id: int
    ) -> list[PlaybookOptimizationEvaluation]:
        """List optimizer evaluations for a job."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_event(
        self, event: PlaybookOptimizationEvent
    ) -> PlaybookOptimizationEvent:
        """Persist an optimizer callback/event and return it with id populated."""
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

    def get_agent_success_evaluation_results_in_window(
        self,
        from_ts: int,
        to_ts: int,
        agent_version: str | None = None,
        limit: int | None = None,
    ) -> list[AgentSuccessEvaluationResult]:
        """Return eval results in ``[from_ts, to_ts]``.

        Default implementation filters the existing latest-results method.
        SQL backends should override so callers do not depend on an arbitrary
        latest-row cap.
        """
        rows = self.get_agent_success_evaluation_results(
            limit=limit or 10_000,
            agent_version=agent_version,
        )
        return [r for r in rows if from_ts <= r.created_at <= to_ts]

    def get_agent_success_evaluation_result_ids(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> list[int]:
        """Return result ids for one eval identity tuple."""
        rows = self.get_agent_success_evaluation_results(
            limit=10_000,
            agent_version=agent_version,
        )
        return [
            r.result_id
            for r in rows
            if r.user_id == user_id
            and r.session_id == session_id
            and r.evaluation_name == evaluation_name
        ]

    @abstractmethod
    def delete_all_agent_success_evaluation_results(self) -> None:
        """Delete all agent success evaluation results from storage."""
        raise NotImplementedError

    @abstractmethod
    def delete_agent_success_evaluation_results_for_session(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> int:
        """Delete stored results for (user_id, session_id, evaluation_name, agent_version).

        Args:
            user_id (str): User whose session results to clear.
            session_id (str): Session whose results to clear.
            evaluation_name (str): Which evaluator's results to clear.
            agent_version (str): Agent version scope.

        Returns:
            int: Number of rows deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_agent_success_evaluation_results_by_ids(
        self, result_ids: list[int]
    ) -> int:
        """Delete agent success eval results matching specific result_ids.

        Used by the regenerate flow to remove only the prior-run rows after the
        new rows have been saved durably (so an LLM/save failure cannot leave
        the session with zero rows).

        Args:
            result_ids (list[int]): Primary-key result_ids to delete.

        Returns:
            int: Number of rows deleted.
        """
        raise NotImplementedError
