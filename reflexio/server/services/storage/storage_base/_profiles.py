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
    def delete_profiles_by_ids(self, profile_ids: list[str]) -> int:
        """Delete profiles by their IDs.

        Args:
            profile_ids (list[str]): List of profile IDs to delete

        Returns:
            int: Number of profiles deleted
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
