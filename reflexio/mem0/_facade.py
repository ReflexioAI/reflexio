"""Scope-aware Reflexio lifecycle operations for the mem0 wrappers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.service_schemas import (
    BulkDeleteResponse,
    ClearUserDataRequest,
    ClearUserDataResponse,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybookResponse,
    DeleteAgentPlaybooksByIdsRequest,
    DeleteProfilesByIdsRequest,
    DeleteRequestRequest,
    DeleteRequestResponse,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DeleteUserInteractionRequest,
    DeleteUserInteractionResponse,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybookResponse,
    DeleteUserPlaybooksByIdsRequest,
    DeleteUserProfileRequest,
    DeleteUserProfileResponse,
)


class ReflexioNotConfiguredError(RuntimeError):
    """Raised when an explicit Reflexio operation has no configured client."""


class ReflexioOperationError(RuntimeError):
    """Raised when an explicit Reflexio lifecycle operation fails."""


def _require_success[ResponseT](operation: str, response: ResponseT) -> ResponseT:
    if not bool(getattr(response, "success", False)):
        raise ReflexioOperationError(f"Reflexio {operation} failed")
    return response


def _sync_operation[ResponseT](
    operation: str, call: Callable[[], ResponseT]
) -> ResponseT:
    try:
        return _require_success(operation, call())
    except ReflexioOperationError:
        raise
    except Exception as exc:
        raise ReflexioOperationError(f"Reflexio {operation} failed") from exc


async def _async_operation[ResponseT](
    operation: str, call: Callable[[], Awaitable[ResponseT]]
) -> ResponseT:
    try:
        return _require_success(operation, await call())
    except ReflexioOperationError:
        raise
    except Exception as exc:
        raise ReflexioOperationError(f"Reflexio {operation} failed") from exc


class _FacadeBase:
    def __init__(
        self,
        client: ReflexioClient | None,
        resolve_user: Callable[[str, str | None], str],
        resolve_session: Callable[[str, str | None, str | None, str | None], str],
    ) -> None:
        self._client = client
        self._resolve_user = resolve_user
        self._resolve_session = resolve_session

    @property
    def configured(self) -> bool:
        """Whether Reflexio was configured for this wrapper instance."""
        return self._client is not None

    def _require_client(self) -> ReflexioClient:
        if self._client is None:
            raise ReflexioNotConfiguredError(
                "Reflexio is not configured; set REFLEXIO_API_KEY or REFLEXIO_URL"
            )
        return self._client


class ReflexioFacade(_FacadeBase):
    """Blocking, scope-aware Reflexio cleanup for ``MemoryClient``."""

    def clear_user_data(
        self, *, user_id: str, app_id: str | None = None
    ) -> ClearUserDataResponse:
        """Clear user-owned Reflexio data; shared agent playbooks are retained."""
        client = self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        return _sync_operation(
            "clear_user_data", lambda: client.clear_user_data(scoped_user)
        )

    def delete_session_records(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
    ) -> DeleteSessionResponse:
        """Delete requests/interactions for a scoped mem0 run or fallback session."""
        client = self._require_client()
        session_id = self._resolve_session(user_id, app_id, agent_id, run_id)
        return _sync_operation(
            "delete_session_records",
            lambda: cast(
                DeleteSessionResponse,
                client.delete_session(session_id, wait_for_response=True),
            ),
        )

    def delete_profile(
        self, *, user_id: str, profile_id: str, app_id: str | None = None
    ) -> DeleteUserProfileResponse:
        client = self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        return _sync_operation(
            "delete_profile",
            lambda: cast(
                DeleteUserProfileResponse,
                client.delete_profile(
                    scoped_user, profile_id=profile_id, wait_for_response=True
                ),
            ),
        )

    def delete_interaction(
        self, *, user_id: str, interaction_id: int, app_id: str | None = None
    ) -> DeleteUserInteractionResponse:
        client = self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        return _sync_operation(
            "delete_interaction",
            lambda: cast(
                DeleteUserInteractionResponse,
                client.delete_interaction(
                    scoped_user, interaction_id, wait_for_response=True
                ),
            ),
        )

    def delete_request(self, *, request_id: str) -> DeleteRequestResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_request",
            lambda: cast(
                DeleteRequestResponse,
                client.delete_request(request_id, wait_for_response=True),
            ),
        )

    def delete_user_playbook(
        self, *, user_playbook_id: int
    ) -> DeleteUserPlaybookResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_user_playbook",
            lambda: cast(
                DeleteUserPlaybookResponse,
                client.delete_user_playbook(user_playbook_id, wait_for_response=True),
            ),
        )

    def delete_agent_playbook(
        self, *, agent_playbook_id: int
    ) -> DeleteAgentPlaybookResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_agent_playbook",
            lambda: cast(
                DeleteAgentPlaybookResponse,
                client.delete_agent_playbook(agent_playbook_id, wait_for_response=True),
            ),
        )

    def delete_requests_by_ids(self, request_ids: list[str]) -> BulkDeleteResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_requests_by_ids", lambda: client.delete_requests_by_ids(request_ids)
        )

    def delete_profiles_by_ids(self, profile_ids: list[str]) -> BulkDeleteResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_profiles_by_ids", lambda: client.delete_profiles_by_ids(profile_ids)
        )

    def delete_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_agent_playbooks_by_ids",
            lambda: client.delete_agent_playbooks_by_ids(agent_playbook_ids),
        )

    def delete_user_playbooks_by_ids(
        self, user_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        client = self._require_client()
        return _sync_operation(
            "delete_user_playbooks_by_ids",
            lambda: client.delete_user_playbooks_by_ids(user_playbook_ids),
        )


class AsyncReflexioFacade(_FacadeBase):
    """Native-async, scope-aware cleanup for ``AsyncMemoryClient``."""

    async def _request_model[ResponseT](
        self,
        operation: str,
        method: str,
        endpoint: str,
        request: Any,
        response_type: type[ResponseT],
    ) -> ResponseT:
        client = self._require_client()

        async def call() -> ResponseT:
            data = await client._make_async_request(  # noqa: SLF001 - package facade
                method, endpoint, json=request.model_dump(mode="json")
            )
            return response_type(**data)

        return await _async_operation(operation, call)

    async def clear_user_data(
        self, *, user_id: str, app_id: str | None = None
    ) -> ClearUserDataResponse:
        self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        response = await self._request_model(
            "clear_user_data",
            "POST",
            "/api/clear_user_data",
            ClearUserDataRequest(user_id=scoped_user),
            ClearUserDataResponse,
        )
        self._require_client()._cache.clear()  # noqa: SLF001 - package facade
        return response

    async def delete_session_records(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        app_id: str | None = None,
        agent_id: str | None = None,
    ) -> DeleteSessionResponse:
        self._require_client()
        session_id = self._resolve_session(user_id, app_id, agent_id, run_id)
        return await self._request_model(
            "delete_session_records",
            "DELETE",
            "/api/delete_session",
            DeleteSessionRequest(session_id=session_id),
            DeleteSessionResponse,
        )

    async def delete_profile(
        self, *, user_id: str, profile_id: str, app_id: str | None = None
    ) -> DeleteUserProfileResponse:
        self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        response = await self._request_model(
            "delete_profile",
            "DELETE",
            "/api/delete_profile",
            DeleteUserProfileRequest(user_id=scoped_user, profile_id=profile_id),
            DeleteUserProfileResponse,
        )
        self._require_client()._cache.invalidate(  # noqa: SLF001 - package facade
            "get_profiles"
        )
        return response

    async def delete_interaction(
        self, *, user_id: str, interaction_id: int, app_id: str | None = None
    ) -> DeleteUserInteractionResponse:
        self._require_client()
        scoped_user = self._resolve_user(user_id, app_id)
        return await self._request_model(
            "delete_interaction",
            "DELETE",
            "/api/delete_interaction",
            DeleteUserInteractionRequest(
                user_id=scoped_user, interaction_id=interaction_id
            ),
            DeleteUserInteractionResponse,
        )

    async def delete_request(self, *, request_id: str) -> DeleteRequestResponse:
        return await self._request_model(
            "delete_request",
            "DELETE",
            "/api/delete_request",
            DeleteRequestRequest(request_id=request_id),
            DeleteRequestResponse,
        )

    async def delete_user_playbook(
        self, *, user_playbook_id: int
    ) -> DeleteUserPlaybookResponse:
        return await self._request_model(
            "delete_user_playbook",
            "DELETE",
            "/api/delete_user_playbook",
            DeleteUserPlaybookRequest(user_playbook_id=user_playbook_id),
            DeleteUserPlaybookResponse,
        )

    async def delete_agent_playbook(
        self, *, agent_playbook_id: int
    ) -> DeleteAgentPlaybookResponse:
        response = await self._request_model(
            "delete_agent_playbook",
            "DELETE",
            "/api/delete_agent_playbook",
            DeleteAgentPlaybookRequest(agent_playbook_id=agent_playbook_id),
            DeleteAgentPlaybookResponse,
        )
        self._require_client()._cache.invalidate(  # noqa: SLF001 - package facade
            "get_agent_playbooks"
        )
        return response

    async def delete_requests_by_ids(
        self, request_ids: list[str]
    ) -> BulkDeleteResponse:
        return await self._request_model(
            "delete_requests_by_ids",
            "DELETE",
            "/api/delete_requests_by_ids",
            DeleteRequestsByIdsRequest(request_ids=request_ids),
            BulkDeleteResponse,
        )

    async def delete_profiles_by_ids(
        self, profile_ids: list[str]
    ) -> BulkDeleteResponse:
        response = await self._request_model(
            "delete_profiles_by_ids",
            "DELETE",
            "/api/delete_profiles_by_ids",
            DeleteProfilesByIdsRequest(profile_ids=profile_ids),
            BulkDeleteResponse,
        )
        self._require_client()._cache.invalidate(  # noqa: SLF001 - package facade
            "get_profiles"
        )
        return response

    async def delete_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        response = await self._request_model(
            "delete_agent_playbooks_by_ids",
            "DELETE",
            "/api/delete_agent_playbooks_by_ids",
            DeleteAgentPlaybooksByIdsRequest(agent_playbook_ids=agent_playbook_ids),
            BulkDeleteResponse,
        )
        self._require_client()._cache.invalidate(  # noqa: SLF001 - package facade
            "get_agent_playbooks"
        )
        return response

    async def delete_user_playbooks_by_ids(
        self, user_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        return await self._request_model(
            "delete_user_playbooks_by_ids",
            "DELETE",
            "/api/delete_user_playbooks_by_ids",
            DeleteUserPlaybooksByIdsRequest(user_playbook_ids=user_playbook_ids),
            BulkDeleteResponse,
        )
