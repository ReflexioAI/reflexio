from abc import abstractmethod

from reflexio.models.api_schema.domain import Interaction
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel


class OperationMixin:
    """Mixin for operation state methods."""

    @abstractmethod
    def create_operation_state(self, service_name: str, operation_state: dict) -> None:
        """Create operation state for a service.

        Args:
            service_name (str): Name of the service
            operation_state (dict): Operation state data as a dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def upsert_operation_state(self, service_name: str, operation_state: dict) -> None:
        """Create or update operation state for a service.

        Args:
            service_name (str): Name of the service
            operation_state (dict): Operation state data as a dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_operation_state(self, service_name: str) -> dict | None:
        """Get operation state for a specific service.

        Args:
            service_name (str): Name of the service

        Returns:
            Optional[dict]: Operation state data or None if not found
        """
        raise NotImplementedError

    @abstractmethod
    def get_operation_state_with_new_request_interaction(
        self,
        service_name: str,
        user_id: str | None,
        sources: list[str] | None = None,
    ) -> tuple[dict, list[RequestInteractionDataModel]]:
        """Get the last operation state and retrieve new interactions since last processing,
        grouped by request.

        Args:
            service_name (str): Name of the service
            user_id (Optional[str]): User identifier to filter interactions.
                If None, returns interactions across all users (for non-user-scoped extractors).
            sources (Optional[list[str]]): Optional list of sources to filter interactions by

        Returns:
            tuple[dict, list[RequestInteractionDataModel]]: Operation state payload and list of
                RequestInteractionDataModel objects containing new interactions grouped by request
        """
        raise NotImplementedError

    @abstractmethod
    def get_last_k_interactions_grouped(
        self,
        user_id: str | None,
        k: int,
        sources: list[str] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        agent_version: str | None = None,
    ) -> tuple[list[RequestInteractionDataModel], list[Interaction]]:
        """Get the last K interactions ordered by time (most recent first), grouped by request.

        This method retrieves the most recent K interactions for a user and groups them
        by their associated requests. Used for sliding window extraction where we want
        to process the last K interactions regardless of whether they were previously processed.

        Args:
            user_id (Optional[str]): User identifier to filter interactions.
                If None, returns interactions across all users (for non-user-scoped extractors).
            k (int): Maximum number of interactions to retrieve
            sources (Optional[list[str]]): Optional list of sources to filter interactions by.
                If provided, only interactions from requests with source in this list are returned.
            start_time (Optional[int]): Unix timestamp. Only return interactions created at or after this time.
            end_time (Optional[int]): Unix timestamp. Only return interactions created at or before this time.
            agent_version (Optional[str]): Filter by agent_version on the request.
                If provided, only interactions from requests with this agent_version are returned.

        Returns:
            tuple[list[RequestInteractionDataModel], list[Interaction]]:
                - List of RequestInteractionDataModel objects (grouped by request/session)
                - Flat list of all interactions sorted by created_at DESC
        """
        raise NotImplementedError

    @abstractmethod
    def update_operation_state(self, service_name: str, operation_state: dict) -> None:
        """Update operation state for a specific service.

        Args:
            service_name (str): Name of the service
            operation_state (dict): Operation state data as a dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_operation_states(self) -> list[dict]:
        """Get all operation states.

        Returns:
            list[dict]: List of all operation state records
        """
        raise NotImplementedError

    @abstractmethod
    def delete_operation_state(self, service_name: str) -> None:
        """Delete operation state for a specific service.

        Args:
            service_name (str): Name of the service
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_operation_states(self) -> None:
        """Delete all operation states."""
        raise NotImplementedError

    @abstractmethod
    def try_acquire_in_progress_lock(
        self, state_key: str, request_id: str, stale_lock_seconds: int = 300
    ) -> dict:
        """Atomically try to acquire an in-progress lock.

        This method should use atomic operations to either:
        1. Acquire the lock if no active lock exists (or lock is stale)
        2. Update pending_request_id if an active lock is held by another request

        Args:
            state_key (str): The operation state key (e.g., "profile_generation_in_progress::3::user_id")
            request_id (str): The current request's unique identifier
            stale_lock_seconds (int): Seconds after which a lock is considered stale (default 300)

        Returns:
            dict: Result with keys:
                - 'acquired' (bool): True if lock was acquired, False if blocked
                - 'state' (dict): The current operation state after the operation
        """
        raise NotImplementedError
