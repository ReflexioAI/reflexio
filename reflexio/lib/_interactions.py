from collections.abc import Callable

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.lib._storage_labels import describe_storage
from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    GetInteractionsResponse,
    SearchInteractionRequest,
    SearchInteractionResponse,
)
from reflexio.models.api_schema.service_schemas import (
    BulkDeleteResponse,
    DeleteRequestRequest,
    DeleteRequestResponse,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DeleteUserInteractionRequest,
    DeleteUserInteractionResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
)
from reflexio.server.services.generation_service import GenerationService


def _safe_count(counter: Callable[[], int]) -> int:
    """Return a storage count, or 0 on failure.

    Used to snapshot profile/playbook counts around a publish so the CLI
    can report deltas. Never raises — a storage hiccup during counting
    shouldn't block the publish itself from returning a useful response.

    Args:
        counter (Callable[[], int]): Zero-arg thunk returning an integer
            count (e.g. ``storage.count_all_profiles``).

    Returns:
        int: The count, or 0 if the thunk raised.
    """
    try:
        return counter()
    except Exception:  # noqa: BLE001
        return 0


class InteractionsMixin(ReflexioBase):
    def publish_interaction(
        self,
        request: PublishUserInteractionRequest | dict,
    ) -> PublishUserInteractionResponse:
        """Publish user interactions.

        Args:
            request (Union[PublishUserInteractionRequest, dict]): The publish user interaction request

        Returns:
            PublishUserInteractionResponse: Response containing success status and message
        """
        if not self._is_storage_configured():
            return PublishUserInteractionResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        # Snapshot profile + playbook totals before the run so we can report
        # how many rows the extraction actually produced. These are best-effort
        # counts — we don't distinguish added vs. updated, only net deltas.
        storage = self._get_storage()
        before_profiles = _safe_count(storage.count_all_profiles)
        before_playbooks = _safe_count(storage.count_user_playbooks)

        generation_service = GenerationService(
            llm_client=self.llm_client,
            request_context=self.request_context,
        )
        # Describe the storage the server is actually writing to so the CLI
        # can surface it. Computed once, up-front, so the response reflects
        # the resolved config at the moment of the call.
        storage_type, storage_label = describe_storage(
            self.request_context.configurator.get_current_storage_configuration()
        )
        try:
            # Convert dict to PublishUserInteractionRequest if needed
            if isinstance(request, dict):
                request = PublishUserInteractionRequest(**request)
            result = generation_service.run(request)
            after_profiles = _safe_count(storage.count_all_profiles)
            after_playbooks = _safe_count(storage.count_user_playbooks)
            # Don't concatenate warnings into the message field — they
            # already travel through ``result.warnings`` and the CLI
            # renders them separately. Embedding multi-line error
            # strings (e.g. server-side Supabase stack traces) into
            # ``message`` turns the CLI's one-line header into a
            # terminal dump.
            if result.warnings:
                message = (
                    f"Interaction published successfully with "
                    f"{len(result.warnings)} warning(s)"
                )
            else:
                message = "Interaction published successfully"
            return PublishUserInteractionResponse(
                success=True,
                message=message,
                warnings=result.warnings,
                request_id=result.request_id,
                storage_type=storage_type,
                storage_label=storage_label,
                profiles_added=max(0, after_profiles - before_profiles),
                playbooks_added=max(0, after_playbooks - before_playbooks),
            )
        except Exception as e:
            return PublishUserInteractionResponse(success=False, message=str(e))

    def search_interactions(
        self,
        request: SearchInteractionRequest | dict,
    ) -> SearchInteractionResponse:
        """Search for user interactions.

        Args:
            request (SearchInteractionRequest): The search request

        Returns:
            SearchInteractionResponse: Response containing matching interactions
        """
        if not self._is_storage_configured():
            return SearchInteractionResponse(
                success=True, interactions=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = SearchInteractionRequest(**request)
        query_embedding = self._maybe_get_query_embedding(
            request.query, request.search_mode
        )
        interactions = self._get_storage().search_interaction(
            request, query_embedding=query_embedding
        )
        return SearchInteractionResponse(
            success=True,
            interactions=interactions,
            msg=f"Found {len(interactions)} matching interaction(s)",
        )

    @_require_storage(DeleteUserInteractionResponse)
    def delete_interaction(
        self,
        request: DeleteUserInteractionRequest | dict,
    ) -> DeleteUserInteractionResponse:
        """Delete user interactions.

        Args:
            request (DeleteUserInteractionRequest): The delete request

        Returns:
            DeleteUserInteractionResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = DeleteUserInteractionRequest(**request)
        self._get_storage().delete_user_interaction(request)
        return DeleteUserInteractionResponse(
            success=True, message="Deleted successfully"
        )

    @_require_storage(DeleteRequestResponse)
    def delete_request(
        self,
        request: DeleteRequestRequest | dict,
    ) -> DeleteRequestResponse:
        """Delete a request and all its associated interactions.

        Args:
            request (DeleteRequestRequest): The delete request containing request_id

        Returns:
            DeleteRequestResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = DeleteRequestRequest(**request)
        self._get_storage().delete_request(request.request_id)
        return DeleteRequestResponse(success=True, message="Deleted successfully")

    @_require_storage(DeleteSessionResponse)
    def delete_session(
        self,
        request: DeleteSessionRequest | dict,
    ) -> DeleteSessionResponse:
        """Delete all requests and interactions in a session.

        Args:
            request (DeleteSessionRequest): The delete request containing session_id

        Returns:
            DeleteSessionResponse: Response containing success status, message, and deleted count
        """
        if isinstance(request, dict):
            request = DeleteSessionRequest(**request)
        deleted_count = self._get_storage().delete_session(request.session_id)
        return DeleteSessionResponse(
            success=True,
            deleted_requests_count=deleted_count,
            message=f"Deleted {deleted_count} item(s)",
        )

    @_require_storage(BulkDeleteResponse)
    def delete_all_interactions_bulk(self) -> BulkDeleteResponse:
        """Delete all requests and their associated interactions.

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        self._get_storage().delete_all_requests()
        return BulkDeleteResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_requests_by_ids(
        self,
        request: DeleteRequestsByIdsRequest | dict,
    ) -> BulkDeleteResponse:
        """Delete requests by their IDs.

        Args:
            request (DeleteRequestsByIdsRequest): The delete request containing request_ids

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        if isinstance(request, dict):
            request = DeleteRequestsByIdsRequest(**request)
        deleted = self._get_storage().delete_requests_by_ids(request.request_ids)
        return BulkDeleteResponse(
            success=True, deleted_count=deleted, message=f"Deleted {deleted} item(s)"
        )

    def get_interactions(
        self,
        request: GetInteractionsRequest | dict,
    ) -> GetInteractionsResponse:
        """Get user interactions.

        Args:
            request (GetInteractionsRequest): The get request

        Returns:
            GetInteractionsResponse: Response containing user interactions
        """
        if not self._is_storage_configured():
            return GetInteractionsResponse(
                success=True, interactions=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetInteractionsRequest(**request)
        interactions = self._get_storage().get_user_interaction(request.user_id)
        interactions = sorted(interactions, key=lambda x: x.created_at, reverse=True)

        # Apply time filters
        if request.start_time:
            interactions = [
                i
                for i in interactions
                if i.created_at >= int(request.start_time.timestamp())
            ]
        if request.end_time:
            interactions = [
                i
                for i in interactions
                if i.created_at <= int(request.end_time.timestamp())
            ]

        # Apply top_k limit
        if request.top_k:
            interactions = interactions[: request.top_k]

        return GetInteractionsResponse(
            success=True,
            interactions=interactions,
            msg=f"Found {len(interactions)} interaction(s)",
        )

    def get_all_interactions(self, limit: int = 100) -> GetInteractionsResponse:
        """Get all user interactions across all users.

        Args:
            limit (int, optional): Maximum number of interactions to return. Defaults to 100.

        Returns:
            GetInteractionsResponse: Response containing all user interactions
        """
        if not self._is_storage_configured():
            return GetInteractionsResponse(
                success=True, interactions=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        interactions = self._get_storage().get_all_interactions(limit=limit)
        interactions = sorted(interactions, key=lambda x: x.created_at, reverse=True)
        return GetInteractionsResponse(
            success=True,
            interactions=interactions,
            msg=f"Found {len(interactions)} interaction(s)",
        )
