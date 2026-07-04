from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    DeleteUserInteractionRequest,
    Interaction,
)


class InteractionStoreMixin:
    """Mixin for interaction-store CRUD methods."""

    @abstractmethod
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        raise NotImplementedError

    @abstractmethod
    def get_all_user_ids(self) -> list[str]:
        """Return distinct user IDs that have stored interactions."""
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
