from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    DeleteUserInteractionRequest,
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)


class ProfileMixin:
    """Mixin for interaction CRUD + search methods."""

    @abstractmethod
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
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

    @abstractmethod
    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_interactions_for_user(self, user_id: str) -> None:
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
    def delete_oldest_interactions(self, count: int) -> int:
        """Delete the oldest N interactions based on created_at timestamp.

        Args:
            count (int): Number of oldest interactions to delete

        Returns:
            int: Number of interactions actually deleted
        """
        raise NotImplementedError

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
