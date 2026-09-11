import logging
from typing import Any

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.lib._storage_labels import describe_storage
from reflexio.models.api_schema.common import sanitise_for_log
from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    GetInteractionsResponse,
    SearchInteractionRequest,
    SearchInteractionResponse,
)
from reflexio.models.api_schema.service_schemas import (
    BulkDeleteResponse,
    ClearUserDataRequest,
    ClearUserDataResponse,
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
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)


def _safe_coverage(
    storage: BaseStorage, user_id: str, request_id: str
) -> tuple[dict[str, Any] | None, dict[str, int] | None]:
    """Read extraction coverage and counts, or ``(None, None)`` on failure.

    Both reads happen after the publish has committed, so they describe the
    write rather than performing it. Never raises — surfacing a reporting
    failure as a failed publish would tell the caller to retry work that is
    already durable.

    Args:
        storage (BaseStorage): Storage the interactions were committed to.
        user_id (str): Owner of the extraction stream.
        request_id (str): Request whose coverage is being reported.

    Returns:
        tuple[dict[str, Any] | None, dict[str, int] | None]: The
            ``extraction_status`` and ``extraction_counts`` results, or
            ``(None, None)`` if either read failed.
    """
    try:
        return (
            storage.extraction_status(user_id, request_id),
            storage.extraction_counts(user_id, request_id),
        )
    except Exception:  # noqa: BLE001
        # `request_id` is caller-supplied (NonEmptyStr: no length cap, no
        # character restrictions), so a newline in it would forge a line in a
        # shared multi-tenant log stream. `user_id` is caller-supplied for the
        # same reason.
        logger.warning(
            "Publish for user %s committed, but reading extraction coverage for "
            "request %s failed; reporting the publish without coverage fields.",
            sanitise_for_log(user_id),
            sanitise_for_log(request_id),
            exc_info=True,
        )
        return None, None


class InteractionsMixin(ReflexioBase):
    def publish_interaction(
        self,
        request: PublishUserInteractionRequest | dict,
        *,
        use_publish_limiter: bool = True,
        publish_limiter_wait_forever: bool = True,
        defer_learning: bool = False,
    ) -> PublishUserInteractionResponse:
        """Publish user interactions.

        Args:
            request (Union[PublishUserInteractionRequest, dict]): The publish user interaction request
            use_publish_limiter: Retained for compatibility. Automatic extraction
                always uses the shared durable worker budget.
            publish_limiter_wait_forever: Retained for compatibility; waiting for
                coverage is bounded by the publish deadline.
            defer_learning: Return after durable admission instead of waiting
                for this request's extraction coverage.

        Returns:
            PublishUserInteractionResponse: Response containing success status and message
        """
        if not self._is_storage_configured():
            return PublishUserInteractionResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        storage = self._get_storage()
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
            result = generation_service.run(
                request,
                use_publish_limiter=use_publish_limiter,
                publish_limiter_wait_forever=publish_limiter_wait_forever,
                defer_learning=defer_learning,
            )
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
            # Reporting reads, made AFTER the interactions are durably
            # committed. They are inside the same ``try`` as the publish, so an
            # unguarded raise here would report ``success=False`` for work that
            # is already on disk -- the caller would retry a publish that
            # happened. Degrade to omitted coverage/count fields instead; this
            # is the invariant the pre-durable ``_safe_count`` helper carried
            # ("a storage hiccup during counting shouldn't block the publish
            # itself from returning a useful response").
            status, counts = _safe_coverage(
                storage, request.user_id, result.request_id or ""
            )
            return PublishUserInteractionResponse(
                success=True,
                message=message,
                warnings=result.warnings,
                request_id=result.request_id,
                storage_type=storage_type,
                storage_label=storage_label,
                profiles_added=counts["profile"] if counts else None,
                playbooks_added=counts["playbook"] if counts else None,
                learning_status=(
                    None
                    if status is None
                    else ("done" if status["status"] == "done" else "deferred")
                ),
                learning_reason=None if status is None else status["reason"],
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

    @_require_storage(ClearUserDataResponse)
    def clear_user_data(
        self,
        request: ClearUserDataRequest | dict,
    ) -> ClearUserDataResponse:
        """Delete all rows scoped to a single ``user_id``.

        Wipes the user's interactions, user playbooks, profiles, and
        requests. Intentionally does NOT touch ``agent_playbooks`` —
        those are the cross-project rollup of skills and have no
        ``user_id`` column. Used by paired-protocol harnesses (e.g.
        SWE-bench) to isolate per-task data on a shared backend without
        nuking sibling tasks' rows.

        Args:
            request (ClearUserDataRequest | dict): Request containing
                the ``user_id`` whose data should be cleared.

        Returns:
            ClearUserDataResponse: Per-entity deletion counts.
        """
        if isinstance(request, dict):
            request = ClearUserDataRequest(**request)
        deleted_counts = self._get_storage().clear_user_data(request.user_id)
        total = sum(deleted_counts.values())
        return ClearUserDataResponse(
            success=True,
            deleted_counts=deleted_counts,
            message=f"Cleared {total} row(s) for user {request.user_id!r}",
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
