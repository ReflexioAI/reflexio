from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    DeleteUserInteractionRequest,
    DeleteUserProfileRequest,
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)


class ProfileMixin:
    """Mixin for profile and interaction CRUD methods."""

    # read methods
    @abstractmethod
    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        raise NotImplementedError

    @abstractmethod
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_user_profile(
        self,
        user_id: str,
        status_filter: list[Status | None] | None = None,
        tags: list[str] | None = None,
    ) -> list[UserProfile]:
        raise NotImplementedError

    @abstractmethod
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        raise NotImplementedError

    # create or update methods
    @abstractmethod
    def add_user_profile(self, user_id: str, user_profiles: list[UserProfile]) -> None:
        """Add the user profile for a given user id."""
        raise NotImplementedError

    @abstractmethod
    def add_user_interaction(self, user_id: str, interaction: Interaction) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_user_interactions_bulk(
        self, user_id: str, interactions: list[Interaction]
    ) -> None:
        """Add multiple user interactions with batched embedding generation.

        Args:
            user_id: The user ID
            interactions: List of interactions to add
        """
        raise NotImplementedError

    # delete methods
    @abstractmethod
    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_user_profile(self, request: DeleteUserProfileRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_user_profile_by_id(
        self, user_id: str, profile_id: str, new_profile: UserProfile
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_user_profile_tags(
        self, user_id: str, profile_id: str, tags: list[str]
    ) -> None:
        """Replace only the tags of a profile, leaving content and embedding untouched."""
        raise NotImplementedError

    @abstractmethod
    def delete_all_interactions_for_user(self, user_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_profiles_for_user(self, user_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_profiles(self) -> None:
        """Delete all profiles across all users."""
        raise NotImplementedError

    @abstractmethod
    def delete_all_interactions(self) -> None:
        """Delete all interactions across all users."""
        raise NotImplementedError

    @abstractmethod
    def count_all_interactions(self) -> int:
        """Count total interactions across all users.

        Returns:
            int: Total number of interactions
        """
        raise NotImplementedError

    @abstractmethod
    def count_all_profiles(self) -> int:
        """Count total profiles across all users without hydrating rows.

        Cheap alternative to ``len(get_all_profiles(...))`` — avoids
        loading every profile row (including embedding BLOBs) just to
        take a count. Used by publish-path diagnostics that snapshot
        totals before and after a run.

        Returns:
            int: Total number of profiles across all users
        """
        raise NotImplementedError

    @abstractmethod
    def delete_oldest_interactions(self, count: int) -> int:
        """Delete the oldest N interactions based on created_at timestamp.

        Args:
            count (int): Number of oldest interactions to delete

        Returns:
            int: Number of interactions actually deleted
        """
        raise NotImplementedError

    @abstractmethod
    def update_all_profiles_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        user_ids: list[str] | None = None,
    ) -> int:
        """Update all profiles with old_status to new_status atomically.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            user_ids: Optional list of user_ids to filter updates. If None, updates all users.

        Returns:
            int: Number of profiles updated
        """
        raise NotImplementedError

    @abstractmethod
    def get_profiles_by_ids(
        self,
        user_id: str,
        profile_ids: list[str],
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        """Fetch the subset of a user's profiles whose ids are in the list.

        Server-side filter on (``user_id``, ``profile_id IN (...)``) so
        callers (e.g. the reflection service resolving a small set of
        cited profile ids) avoid scanning every profile for the user.

        Args:
            user_id (str): Owning user id.
            profile_ids (list[str]): Profile ids to fetch. Empty list
                returns ``[]`` without hitting storage.
            status_filter (list[Status | None] | None): Statuses to
                include. ``None`` (default) means CURRENT only — same
                default as ``get_user_profile`` for consistency.

        Returns:
            list[UserProfile]: Matching profiles. Order is unspecified.
                Ids that do not exist (or do not match the user / status
                filter) are silently omitted.
        """
        raise NotImplementedError

    @abstractmethod
    def get_profile_by_id(
        self, profile_id: str, *, include_tombstones: bool = False
    ) -> UserProfile | None:
        """Fetch a single profile by primary key.

        Args:
            profile_id: The profile's primary key.
            include_tombstones: When False (default), MERGED/SUPERSEDED profiles
                return None. Set to True for lineage resolution (resolve_current).

        Returns:
            The UserProfile if found and not filtered, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def archive_profile_by_id(self, user_id: str, profile_id: str) -> bool:
        """Atomically archive a single profile by id, only if currently CURRENT.

        Flips the row's ``status`` from ``None`` (CURRENT) to
        ``Status.ARCHIVED``. No-op when the profile does not exist or is
        already non-current.

        Args:
            user_id (str): Owning user id.
            profile_id (str): The profile id to archive.

        Returns:
            bool: True if a row was archived; False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_profiles_by_status(self, status: Status) -> int:
        """Delete all profiles with the given status atomically.

        Args:
            status: The status of profiles to delete

        Returns:
            int: Number of profiles deleted
        """
        raise NotImplementedError

    @abstractmethod
    def get_user_ids_with_status(self, status: Status | None) -> list[str]:
        """Get list of unique user_ids that have profiles with the given status.

        Args:
            status: The status to filter by (None for CURRENT)

        Returns:
            list[str]: List of unique user_ids
        """
        raise NotImplementedError

    @abstractmethod
    def delete_profiles_by_ids(
        self, profile_ids: list[str], *, emit_hard_delete: bool = True
    ) -> int:
        """Delete profiles by their IDs.

        Args:
            profile_ids (list[str]): List of profile IDs to delete
            emit_hard_delete (bool): When True (default), emit a ``hard_delete``
                lineage event for each id before removing the row.  Pass ``False``
                from rollback callers that cleaned up a never-CURRENT successor to
                avoid poisoning the audit log with phantom erasures.

        Returns:
            int: Number of profiles deleted
        """
        raise NotImplementedError

    @abstractmethod
    def get_distinct_generated_from_request_ids(self) -> list[str]:
        """Return the DISTINCT non-empty generated_from_request_id values present on profiles.

        Scoped to the org. Includes profiles of any status (tombstones included) so
        that an add-only run whose profiles were later tombstoned is still discoverable.
        Empty-string values are excluded by the query (they must never form a group).

        Used by ``reconstruct_profile_change_log`` to discover add-only dedup runs
        (runs that added new profiles but superseded nothing, which emit no lineage
        event and would otherwise be invisible).

        Returns:
            list[str]: Distinct non-empty ``generated_from_request_id`` values.
        """
        raise NotImplementedError

    @abstractmethod
    def get_profiles_by_generated_from_request_id(
        self,
        request_id: str,
    ) -> list[UserProfile]:
        """Return all profiles (any status, including tombstones) for a given generated_from_request_id.

        Scoped to the org. Used by reconstruct_profile_change_log to find the
        "added" side of a dedup run without depending on mutable status columns.
        The column is set at profile creation and never changes — it is the
        time-travel-stable signal for "added in run R".

        Args:
            request_id (str): The generated_from_request_id value to filter on.

        Returns:
            list[UserProfile]: All profiles (live or tombstone) whose
                ``generated_from_request_id`` matches, scoped by org.
        """
        raise NotImplementedError

    def get_all_generated_profiles(self) -> list[UserProfile]:
        """Return every profile (any status, incl. tombstones) with a non-empty
        ``generated_from_request_id``, scoped to the org.

        Bulk equivalent of :meth:`get_profiles_by_generated_from_request_id` — it
        lets ``reconstruct_profile_change_log`` resolve the "added" side of every
        run in a single query instead of one read per candidate request_id (an
        org-history-sized fan-out on network-backed storage).

        The default implementation loops the per-id reads, so every backend is
        correct without an override; backends SHOULD override it with a single
        ``WHERE generated_from_request_id <> ''`` query for the performance win.

        Returns:
            list[UserProfile]: All profiles with a non-empty
                ``generated_from_request_id``.
        """
        out: list[UserProfile] = []
        for request_id in self.get_distinct_generated_from_request_ids():
            out.extend(self.get_profiles_by_generated_from_request_id(request_id))
        return out

    @abstractmethod
    def supersede_profiles_by_ids(
        self,
        user_id: str,
        profile_ids: list[str],
        request_id: str,
    ) -> list[str]:
        """Soft-delete profiles by setting status to SUPERSEDED, emitting set-based lineage.

        For each profile id that matches (user_id, current status in {NULL/CURRENT,
        PENDING}), updates status to SUPERSEDED and emits one ``status_change``
        lineage event under the shared ``request_id``.  Rows are NOT physically
        deleted — reads that exclude tombstones will simply filter them out by status.

        Args:
            user_id (str): Owning user id. Predicate scoped to this user.
            profile_ids (list[str]): Profile ids to supersede. Already-superseded or
                non-existent ids are silently skipped.
            request_id (str): Shared request id to stamp on all emitted events so the
                entire dedup run is reconstructible from a single id.

        Returns:
            list[str]: The profile ids that were actually superseded by this call, in
                input order. Already-superseded or non-existent ids are omitted (so an
                empty list means nothing committed). Callers needing a count use
                ``len(...)``. This is the commit-accurate set used to build a
                commit-atomic legacy change-log "removed" entry.
        """
        raise NotImplementedError

    # Search methods
    @abstractmethod
    def search_interaction(
        self,
        search_interaction_request: SearchInteractionRequest,
        query_embedding: list[float] | None = None,
    ) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def search_user_profile(
        self,
        search_user_profile_request: SearchUserProfileRequest,
        status_filter: list[Status | None] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[UserProfile]:
        raise NotImplementedError
