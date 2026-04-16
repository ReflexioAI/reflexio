import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, TypeVar
from urllib.parse import urljoin

import aiohttp
import requests

from reflexio.defaults import DEFAULT_AGENT_VERSION
from reflexio.models.api_schema.retriever_schema import (
    ConversationTurn,
    GetAgentPlaybooksRequest,
    GetAgentPlaybooksViewResponse,
    GetAgentSuccessEvaluationResultsRequest,
    GetEvaluationResultsViewResponse,
    GetInteractionsRequest,
    GetInteractionsViewResponse,
    GetProfilesViewResponse,
    GetRequestsRequest,
    GetRequestsViewResponse,
    GetUserPlaybooksRequest,
    GetUserPlaybooksViewResponse,
    GetUserProfilesRequest,
    ProfileChangeLogViewResponse,
    SearchAgentPlaybookRequest,
    SearchAgentPlaybooksViewResponse,
    SearchInteractionRequest,
    SearchInteractionsViewResponse,
    SearchProfilesViewResponse,
    SearchUserPlaybookRequest,
    SearchUserPlaybooksViewResponse,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
    UnifiedSearchViewResponse,
    UpdateAgentPlaybookRequest,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusRequest,
    UpdatePlaybookStatusResponse,
    UpdateUserPlaybookRequest,
    UpdateUserPlaybookResponse,
)
from reflexio.models.config_schema import SearchMode

IS_TEST_ENV = os.environ.get("IS_TEST_ENV", "false").strip() == "true"

BACKEND_URL = "http://127.0.0.1:8000" if IS_TEST_ENV else "https://www.reflexio.ai/"

from reflexio.models.api_schema.domain.entities import (
    UpgradeProfilesRequest,
    UpgradeProfilesResponse,
    UpgradeUserPlaybooksRequest,
    UpgradeUserPlaybooksResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddAgentPlaybookResponse,
    AddUserPlaybookRequest,
    AddUserPlaybookResponse,
    AddUserProfileRequest,
    AddUserProfileResponse,
    AgentPlaybook,
    BulkDeleteResponse,
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
    GetOperationStatusResponse,
    InteractionData,
    ManualPlaybookGenerationRequest,
    ManualProfileGenerationRequest,
    MyConfigResponse,
    OperationStatus,
    PlaybookStatus,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    RerunProfileGenerationRequest,
    RerunProfileGenerationResponse,
    RunPlaybookAggregationRequest,
    RunPlaybookAggregationResponse,
    Status,
    UserPlaybook,
    UserProfile,
    WhoamiResponse,
)
from reflexio.models.config_schema import Config

from .cache import InMemoryCache

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ReflexioAPIError(Exception):
    """Raised when the API returns a successful HTTP status but the body
    cannot be decoded as JSON.

    This is the "your REFLEXIO_URL points at a non-API host / a proxy
    returned an HTML error page with status 200" class of failure. The
    error message includes the URL, status code, Content-Type, and a
    truncated body preview so the user can see at a glance what went
    wrong. The CLI's ``handle_errors`` decorator renders this as a
    structured ``CliError`` with ``error_type="api"``.
    """


class ReflexioClient:
    """Client for interacting with the Reflexio API."""

    # Shared thread pool for all instances to maximize efficiency
    _thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reflexio")

    def __init__(
        self, api_key: str = "", url_endpoint: str = "", timeout: int = 300
    ) -> None:
        """Initialize the Reflexio client.

        Args:
            api_key (str): API key for authentication. Falls back to REFLEXIO_API_KEY env var.
            url_endpoint (str): Base URL for the API. Falls back to REFLEXIO_API_URL env var,
                then to the default backend URL.
            timeout (int): Default request timeout in seconds (default 300)
        """
        self.base_url = (
            url_endpoint or os.environ.get("REFLEXIO_API_URL", "") or BACKEND_URL
        )
        self.api_key = api_key or os.environ.get("REFLEXIO_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        # Treat any API 3xx as an error instead of silently following.
        # ``requests`` default would demote POST→GET on 302, which has
        # historically turned misconfigured base URLs (e.g. pointing at a
        # marketing site that 302s to its www subdomain) into silent
        # publish losses: the POST became a GET to a marketing page that
        # returned HTML 200, which then crashed ``response.json()``.
        self.session.max_redirects = 0
        self._cache = InMemoryCache()

    def _get_auth_headers(self) -> dict:
        """Get authentication headers with Bearer token if api_key is configured.

        Returns:
            dict: Headers with Authorization bearer token, or empty dict if no api_key
        """
        if self.api_key:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        return {}

    def _get_headers(self) -> dict:
        """Get default headers for API requests.

        Returns:
            dict: Headers with content-type and optional authorization
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _convert_to_model(self, data: dict | object, model_class: type[T]) -> T:
        """Convert dict to model instance if needed.

        Args:
            data: Either a dict or already an instance of model_class
            model_class: The target class to convert to

        Returns:
            An instance of model_class
        """
        if isinstance(data, dict):
            return model_class(**data)
        return data  # type: ignore[reportReturnType]

    def _build_request(
        self,
        request: T | dict | None,
        model_class: type[T],
        **kwargs: Any,
    ) -> T:
        """Build request object from request param or kwargs.

        Args:
            request: Optional request object or dict
            model_class: The request class to instantiate
            **kwargs: Field values to use if request is None

        Returns:
            An instance of model_class
        """
        if request is not None:
            return self._convert_to_model(request, model_class)  # type: ignore[reportReturnType]
        # Filter out None values and build from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return model_class(**filtered_kwargs)

    def _fire_and_forget(
        self,
        async_func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Execute an async request in fire-and-forget mode.

        Args:
            async_func: Asynchronous function to call
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(async_func(*args, **kwargs))
        except RuntimeError:
            self._thread_pool.submit(lambda: asyncio.run(async_func(*args, **kwargs)))

    async def _make_async_request(
        self, method: str, endpoint: str, headers: dict | None = None, **kwargs: Any
    ) -> Any:
        """Make an async HTTP request to the API."""
        url = urljoin(self.base_url, endpoint)

        # Merge auth headers with any provided headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        async with aiohttp.ClientSession() as async_session:
            response = await async_session.request(
                method, url, headers=request_headers, **kwargs
            )
            response.raise_for_status()
            return await response.json()

    def _make_request(
        self, method: str, endpoint: str, headers: dict | None = None, **kwargs: Any
    ) -> Any:
        """Make an HTTP request to the API and decode the JSON response.

        Beyond ``requests.raise_for_status()`` this also guards against
        the "2xx with non-JSON body" failure mode — typically a
        misconfigured base URL pointing at a marketing/proxy host that
        returns an HTML error or redirect page with status 200. Those
        cases produce a ``ReflexioAPIError`` with the URL + body
        preview instead of an opaque ``JSONDecodeError`` deep in the
        stack.

        Args:
            method: HTTP method (GET, POST, DELETE).
            endpoint: API endpoint path.
            headers: Additional headers to include in the request.
            **kwargs: Extra kwargs passed through to ``requests``.

        Returns:
            dict: Decoded JSON response, or ``{}`` for empty 2xx
                bodies (e.g. 204 No Content).

        Raises:
            requests.HTTPError: For 4xx/5xx status codes.
            ReflexioAPIError: For 2xx responses whose body isn't JSON.
        """
        url = urljoin(self.base_url, endpoint)

        # Merge auth headers with any provided headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)

        self.session.headers.update(request_headers)
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()

        # Empty body on a successful response (e.g. 204 No Content).
        # Only treat genuine empty bytes as "no content" — MagicMock
        # tests leave ``content`` as a MagicMock and should fall through.
        content = response.content
        if isinstance(content, (bytes, bytearray)) and not content:
            return {}

        # Content-Type guard is only enforced when the header is a real
        # string. Test fixtures that mock ``Session.request`` without
        # wiring headers return MagicMock values here and should keep
        # going straight to ``.json()`` for backward compatibility.
        content_type_raw = response.headers.get("Content-Type", "")
        if isinstance(content_type_raw, str) and content_type_raw:
            content_type = content_type_raw.lower()
            if "json" not in content_type:
                body_preview = (
                    response.text[:200] if isinstance(response.text, str) else ""
                )
                raise ReflexioAPIError(
                    f"Expected JSON from {method} {url} but got "
                    f"Content-Type={content_type} "
                    f"(status {response.status_code}). "
                    f"Body preview: {body_preview!r}. "
                    "Is REFLEXIO_URL pointing at the API host?"
                )
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as exc:
            body_preview = response.text[:200] if isinstance(response.text, str) else ""
            raise ReflexioAPIError(
                f"{method} {url} returned status {response.status_code} but "
                f"the body is not valid JSON: {exc}. "
                f"Body preview: {body_preview!r}. "
                "Is REFLEXIO_URL pointing at the API host?"
            ) from exc

    def _publish_interaction_sync(
        self,
        request: PublishUserInteractionRequest,
        wait_for_response: bool = False,
    ) -> PublishUserInteractionResponse:
        """Internal sync method to publish interaction.

        Args:
            request (PublishUserInteractionRequest): The publish request
            wait_for_response (bool): If True, server processes synchronously and returns real result
        """
        params = {"wait_for_response": "true"} if wait_for_response else None
        response = self._make_request(
            "POST",
            "/api/publish_interaction",
            json=request.model_dump(),
            params=params,
        )
        return PublishUserInteractionResponse(**response)

    async def _publish_interaction_async(
        self,
        request: PublishUserInteractionRequest,
        wait_for_response: bool = False,
    ) -> PublishUserInteractionResponse:
        """Internal async method to publish interaction.

        Args:
            request (PublishUserInteractionRequest): The publish request
            wait_for_response (bool): If True, server processes synchronously and returns real result
        """
        params = {"wait_for_response": "true"} if wait_for_response else None
        response = await self._make_async_request(
            "POST",
            "/api/publish_interaction",
            json=request.model_dump(),
            params=params,
        )
        return PublishUserInteractionResponse(**response)

    def publish_interaction(
        self,
        user_id: str,
        interactions: Sequence[InteractionData | dict],
        source: str = "",
        agent_version: str = DEFAULT_AGENT_VERSION,
        session_id: str | None = None,
        wait_for_response: bool = False,
        skip_aggregation: bool = False,
        force_extraction: bool = False,
    ) -> PublishUserInteractionResponse:
        """Publish user interactions.

        Always blocks on the HTTP round-trip so the caller can see
        4xx/5xx, JSON-decode errors, and network errors. The
        ``wait_for_response`` parameter controls whether the **server**
        processes extraction synchronously; the client-side POST is
        always synchronous.

        In server-async mode (``wait_for_response=False``), the server
        returns 200 as soon as it has registered a BackgroundTask —
        typically ~100 ms. That's fine to block on from the CLI and
        eliminates a prior fragility where CLI fire-and-forget relied
        on a thread pool whose atexit handler wouldn't run under
        SIGTERM, so publishes from a Claude Code subagent could be
        silently lost.

        Library users who need a truly non-blocking call can submit
        the request through ``_fire_and_forget`` directly.

        Args:
            user_id: The user ID.
            interactions: List of interaction data.
            source: The source of the interaction.
            agent_version: The agent version.
            session_id: Optional session ID for grouping requests.
            wait_for_response: If True, the **server** waits for
                extraction to complete before returning (longer HTTP
                call, response includes real profile/playbook counts).
                If False, the server returns immediately after queuing
                extraction. The client blocks on the HTTP round-trip
                in both cases.
            skip_aggregation: If True, extract profiles/playbooks but
                skip aggregation to agent playbooks.
            force_extraction: If True, bypass batch_interval checks
                and always run extractors.

        Returns:
            PublishUserInteractionResponse: Server response. In
                ``wait_for_response=False`` mode this is a bare
                acknowledgement ("Interaction queued for processing")
                without extraction counts; in ``wait_for_response=True``
                mode it includes request_id, storage routing, and
                deltas.
        """
        interaction_data_list = [
            (
                InteractionData(**interaction_request)
                if isinstance(interaction_request, dict)
                else interaction_request
            )
            for interaction_request in interactions
        ]
        request = PublishUserInteractionRequest(
            session_id=session_id,
            user_id=user_id,
            interaction_data_list=interaction_data_list,
            source=source,
            agent_version=agent_version,
            skip_aggregation=skip_aggregation,
            force_extraction=force_extraction,
        )
        return self._publish_interaction_sync(
            request, wait_for_response=wait_for_response
        )

    def search_interactions(
        self,
        request: SearchInteractionRequest | dict | None = None,
        *,
        user_id: str | None = None,
        request_id: str | None = None,
        query: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        top_k: int | None = None,
        most_recent_k: int | None = None,
        search_mode: SearchMode | None = None,
    ) -> SearchInteractionsViewResponse:
        """Search for user interactions.

        Args:
            request (Optional[SearchInteractionRequest]): The search request object (alternative to kwargs)
            user_id (str): The user ID to search for
            request_id (Optional[str]): Filter by specific request ID
            query (Optional[str]): Search query string
            start_time (Optional[datetime]): Filter by start time
            end_time (Optional[datetime]): Filter by end time
            top_k (Optional[int]): Maximum number of results to return
            most_recent_k (Optional[int]): Return most recent k interactions

        Returns:
            SearchInteractionsViewResponse: Response containing matching interactions
        """
        req = self._build_request(
            request,
            SearchInteractionRequest,
            user_id=user_id,
            request_id=request_id,
            query=query,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
            most_recent_k=most_recent_k,
            search_mode=search_mode,
        )
        response = self._make_request(
            "POST",
            "/api/search_interactions",
            json=req.model_dump(),
        )
        return SearchInteractionsViewResponse(**response)

    def search_profiles(
        self,
        request: SearchUserProfileRequest | dict | None = None,
        *,
        user_id: str | None = None,
        generated_from_request_id: str | None = None,
        query: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        top_k: int | None = None,
        source: str | None = None,
        custom_feature: str | None = None,
        extractor_name: str | None = None,
        threshold: float | None = None,
        enable_reformulation: bool | None = None,
        search_mode: SearchMode | None = None,
    ) -> SearchProfilesViewResponse:
        """Search for user profiles.

        Args:
            request (Optional[SearchUserProfileRequest]): The search request object (alternative to kwargs)
            user_id (str): The user ID to search for
            generated_from_request_id (Optional[str]): Filter by request ID that generated the profile
            query (Optional[str]): Search query string
            start_time (Optional[datetime]): Filter by start time
            end_time (Optional[datetime]): Filter by end time
            top_k (Optional[int]): Maximum number of results to return (default: 10)
            source (Optional[str]): Filter by source
            custom_feature (Optional[str]): Filter by custom feature
            extractor_name (Optional[str]): Filter by extractor name
            threshold (Optional[float]): Similarity threshold (default: 0.7)
            enable_reformulation (Optional[bool]): Enable LLM query reformulation (default: False)

        Returns:
            SearchProfilesViewResponse: Response containing matching profiles
        """
        req = self._build_request(
            request,
            SearchUserProfileRequest,
            user_id=user_id,
            generated_from_request_id=generated_from_request_id,
            query=query,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
            source=source,
            custom_feature=custom_feature,
            extractor_name=extractor_name,
            threshold=threshold,
            enable_reformulation=enable_reformulation,
            search_mode=search_mode,
        )
        response = self._make_request(
            "POST", "/api/search_profiles", json=req.model_dump()
        )
        return SearchProfilesViewResponse(**response)

    def search_user_playbooks(
        self,
        request: SearchUserPlaybookRequest | dict | None = None,
        *,
        query: str | None = None,
        user_id: str | None = None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        status_filter: list[Status | None] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        enable_reformulation: bool | None = None,
        search_mode: SearchMode | None = None,
    ) -> SearchUserPlaybooksViewResponse:
        """Search for user playbooks with semantic/text search and filtering.

        Args:
            request (Optional[SearchUserPlaybookRequest]): The search request object (alternative to kwargs)
            query (Optional[str]): Query for semantic/text search
            user_id (Optional[str]): Filter by user (via request_id linkage to requests table)
            agent_version (Optional[str]): Filter by agent version
            playbook_name (Optional[str]): Filter by playbook name
            start_time (Optional[datetime]): Start time for created_at filter
            end_time (Optional[datetime]): End time for created_at filter
            status_filter (Optional[list[Optional[Status]]]): Filter by status (None for CURRENT, PENDING, ARCHIVED)
            top_k (Optional[int]): Maximum number of results to return (default: 10)
            threshold (Optional[float]): Similarity threshold for vector search (default: 0.4)
            enable_reformulation (Optional[bool]): Enable LLM query reformulation (default: False)

        Returns:
            SearchUserPlaybooksViewResponse: Response containing matching user playbooks
        """
        req = self._build_request(
            request,
            SearchUserPlaybookRequest,
            query=query,
            user_id=user_id,
            agent_version=agent_version,
            playbook_name=playbook_name,
            start_time=start_time,
            end_time=end_time,
            status_filter=status_filter,
            top_k=top_k,
            threshold=threshold,
            enable_reformulation=enable_reformulation,
            search_mode=search_mode,
        )
        response = self._make_request(
            "POST", "/api/search_user_playbooks", json=req.model_dump()
        )
        return SearchUserPlaybooksViewResponse(**response)

    def search_agent_playbooks(
        self,
        request: SearchAgentPlaybookRequest | dict | None = None,
        *,
        query: str | None = None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: PlaybookStatus | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        enable_reformulation: bool | None = None,
        search_mode: SearchMode | None = None,
    ) -> SearchAgentPlaybooksViewResponse:
        """Search for agent playbooks with semantic/text search and filtering.

        Args:
            request (Optional[SearchAgentPlaybookRequest]): The search request object (alternative to kwargs)
            query (Optional[str]): Query for semantic/text search
            agent_version (Optional[str]): Filter by agent version
            playbook_name (Optional[str]): Filter by playbook name
            start_time (Optional[datetime]): Start time for created_at filter
            end_time (Optional[datetime]): End time for created_at filter
            status_filter (Optional[list[Optional[Status]]]): Filter by status (None for CURRENT, PENDING, ARCHIVED)
            playbook_status_filter (Optional[PlaybookStatus]): Filter by playbook status (PENDING, APPROVED, REJECTED)
            top_k (Optional[int]): Maximum number of results to return (default: 10)
            threshold (Optional[float]): Similarity threshold for vector search (default: 0.4)
            enable_reformulation (Optional[bool]): Enable LLM query reformulation (default: False)

        Returns:
            SearchAgentPlaybooksViewResponse: Response containing matching agent playbooks
        """
        req = self._build_request(
            request,
            SearchAgentPlaybookRequest,
            query=query,
            agent_version=agent_version,
            playbook_name=playbook_name,
            start_time=start_time,
            end_time=end_time,
            status_filter=status_filter,
            playbook_status_filter=playbook_status_filter,
            top_k=top_k,
            threshold=threshold,
            enable_reformulation=enable_reformulation,
            search_mode=search_mode,
        )
        response = self._make_request(
            "POST", "/api/search_agent_playbooks", json=req.model_dump()
        )
        return SearchAgentPlaybooksViewResponse(**response)

    def _delete_profile_sync(
        self, request: DeleteUserProfileRequest
    ) -> DeleteUserProfileResponse:
        """Internal sync method to delete profile."""
        response = self._make_request(
            "DELETE",
            "/api/delete_profile",
            json=request.model_dump(),
        )
        return DeleteUserProfileResponse(**response)

    async def _delete_profile_async(
        self, request: DeleteUserProfileRequest
    ) -> DeleteUserProfileResponse:
        """Internal async method to delete profile."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_profile",
            json=request.model_dump(),
        )
        return DeleteUserProfileResponse(**response)

    def delete_profile(
        self,
        user_id: str,
        profile_id: str = "",
        search_query: str = "",
        wait_for_response: bool = False,
    ) -> DeleteUserProfileResponse | None:
        """Delete user profiles.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            user_id (str): The user ID
            profile_id (str, optional): Specific profile ID to delete
            search_query (str, optional): Query to match profiles for deletion
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteUserProfileResponse]: Response containing success status and message if wait_for_response=True, None otherwise
        """
        request = DeleteUserProfileRequest(
            user_id=user_id,
            profile_id=profile_id,
            search_query=search_query,
        )

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_profile_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_profile_async, request)
        return None

    def _delete_interaction_sync(
        self, request: DeleteUserInteractionRequest
    ) -> DeleteUserInteractionResponse:
        """Internal sync method to delete interaction."""
        response = self._make_request(
            "DELETE",
            "/api/delete_interaction",
            json=request.model_dump(),
        )
        return DeleteUserInteractionResponse(**response)

    async def _delete_interaction_async(
        self, request: DeleteUserInteractionRequest
    ) -> DeleteUserInteractionResponse:
        """Internal async method to delete interaction."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_interaction",
            json=request.model_dump(),
        )
        return DeleteUserInteractionResponse(**response)

    def delete_interaction(
        self, user_id: str, interaction_id: int, wait_for_response: bool = False
    ) -> DeleteUserInteractionResponse | None:
        """Delete a user interaction.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            user_id (str): The user ID
            interaction_id (int): The interaction ID to delete
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteUserInteractionResponse]: Response containing success status and message if wait_for_response=True, None otherwise
        """
        request = DeleteUserInteractionRequest(
            user_id=user_id,
            interaction_id=interaction_id,
        )

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_interaction_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_interaction_async, request)
        return None

    def _delete_request_sync(
        self, request: DeleteRequestRequest
    ) -> DeleteRequestResponse:
        """Internal sync method to delete request."""
        response = self._make_request(
            "DELETE",
            "/api/delete_request",
            json=request.model_dump(),
        )
        return DeleteRequestResponse(**response)

    async def _delete_request_async(
        self, request: DeleteRequestRequest
    ) -> DeleteRequestResponse:
        """Internal async method to delete request."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_request",
            json=request.model_dump(),
        )
        return DeleteRequestResponse(**response)

    def delete_request(
        self, request_id: str, wait_for_response: bool = False
    ) -> DeleteRequestResponse | None:
        """Delete a request and all its associated interactions.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            request_id (str): The request ID to delete
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteRequestResponse]: Response containing success status and message if wait_for_response=True, None otherwise
        """
        request = DeleteRequestRequest(request_id=request_id)

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_request_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_request_async, request)
        return None

    def _delete_session_sync(
        self, request: DeleteSessionRequest
    ) -> DeleteSessionResponse:
        """Internal sync method to delete session."""
        response = self._make_request(
            "DELETE",
            "/api/delete_session",
            json=request.model_dump(),
        )
        return DeleteSessionResponse(**response)

    async def _delete_session_async(
        self, request: DeleteSessionRequest
    ) -> DeleteSessionResponse:
        """Internal async method to delete session."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_session",
            json=request.model_dump(),
        )
        return DeleteSessionResponse(**response)

    def delete_session(
        self, session_id: str, wait_for_response: bool = False
    ) -> DeleteSessionResponse | None:
        """Delete all requests and interactions in a session.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            session_id (str): The session ID to delete
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteSessionResponse]: Response containing success status, message, and deleted count if wait_for_response=True, None otherwise
        """
        request = DeleteSessionRequest(session_id=session_id)

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_session_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_session_async, request)
        return None

    def _delete_agent_playbook_sync(
        self, request: DeleteAgentPlaybookRequest
    ) -> DeleteAgentPlaybookResponse:
        """Internal sync method to delete agent playbook."""
        response = self._make_request(
            "DELETE",
            "/api/delete_agent_playbook",
            json=request.model_dump(),
        )
        return DeleteAgentPlaybookResponse(**response)

    async def _delete_agent_playbook_async(
        self, request: DeleteAgentPlaybookRequest
    ) -> DeleteAgentPlaybookResponse:
        """Internal async method to delete agent playbook."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_agent_playbook",
            json=request.model_dump(),
        )
        return DeleteAgentPlaybookResponse(**response)

    def delete_agent_playbook(
        self, agent_playbook_id: int, wait_for_response: bool = False
    ) -> DeleteAgentPlaybookResponse | None:
        """Delete an agent playbook by ID.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            agent_playbook_id (int): The agent playbook ID to delete
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteAgentPlaybookResponse]: Response containing success status and message if wait_for_response=True, None otherwise
        """
        request = DeleteAgentPlaybookRequest(agent_playbook_id=agent_playbook_id)

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_agent_playbook_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_agent_playbook_async, request)
        return None

    def _delete_user_playbook_sync(
        self, request: DeleteUserPlaybookRequest
    ) -> DeleteUserPlaybookResponse:
        """Internal sync method to delete user playbook."""
        response = self._make_request(
            "DELETE",
            "/api/delete_user_playbook",
            json=request.model_dump(),
        )
        return DeleteUserPlaybookResponse(**response)

    async def _delete_user_playbook_async(
        self, request: DeleteUserPlaybookRequest
    ) -> DeleteUserPlaybookResponse:
        """Internal async method to delete user playbook."""
        response = await self._make_async_request(
            "DELETE",
            "/api/delete_user_playbook",
            json=request.model_dump(),
        )
        return DeleteUserPlaybookResponse(**response)

    def delete_user_playbook(
        self, user_playbook_id: int, wait_for_response: bool = False
    ) -> DeleteUserPlaybookResponse | None:
        """Delete a user playbook by ID.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            user_playbook_id (int): The user playbook ID to delete
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.

        Returns:
            Optional[DeleteUserPlaybookResponse]: Response containing success status and message if wait_for_response=True, None otherwise
        """
        request = DeleteUserPlaybookRequest(user_playbook_id=user_playbook_id)

        if wait_for_response:
            # Synchronous blocking call
            return self._delete_user_playbook_sync(request)
        # Non-blocking fire-and-forget
        self._fire_and_forget(self._delete_user_playbook_async, request)
        return None

    def get_profile_change_log(self) -> ProfileChangeLogViewResponse:
        """Get profile change log.

        Returns:
            ProfileChangeLogViewResponse: Response containing profile change log
        """
        response = self._make_request("GET", "/api/profile_change_log")
        return ProfileChangeLogViewResponse(**response)

    def get_interactions(
        self,
        request: GetInteractionsRequest | dict | None = None,
        *,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        top_k: int | None = None,
    ) -> GetInteractionsViewResponse:
        """Get user interactions.

        Args:
            request (Optional[GetInteractionsRequest]): The list request object (alternative to kwargs)
            user_id (str): The user ID to get interactions for
            start_time (Optional[datetime]): Filter by start time
            end_time (Optional[datetime]): Filter by end time
            top_k (Optional[int]): Maximum number of results to return (default: 30)

        Returns:
            GetInteractionsViewResponse: Response containing list of interactions
        """
        req = self._build_request(
            request,
            GetInteractionsRequest,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
        )
        response = self._make_request(
            "POST",
            "/api/get_interactions",
            json=req.model_dump(),
        )
        return GetInteractionsViewResponse(**response)

    def get_profiles(
        self,
        request: GetUserProfilesRequest | dict | None = None,
        force_refresh: bool = False,
        *,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        top_k: int | None = None,
        status_filter: list[Status | str | None] | None = None,
    ) -> GetProfilesViewResponse:
        """Get user profiles.

        Args:
            request (Optional[GetUserProfilesRequest]): The list request object (alternative to kwargs)
            force_refresh (bool, optional): If True, bypass cache and fetch fresh data. Defaults to False.
            user_id (str): The user ID to get profiles for
            start_time (Optional[datetime]): Filter by start time
            end_time (Optional[datetime]): Filter by end time
            top_k (Optional[int]): Maximum number of results to return (default: 30)
            status_filter (Optional[list[Optional[Union[Status, str]]]]): Filter by profile status. Accepts Status enum or string values (e.g., "archived", "pending").

        Returns:
            GetProfilesViewResponse: Response containing list of profiles
        """
        # Convert string status values to Status enum
        converted_status_filter = None
        if status_filter is not None:
            converted_status_filter = []
            for status in status_filter:
                if status is None:
                    converted_status_filter.append(None)
                elif isinstance(status, str):
                    converted_status_filter.append(Status(status))
                else:
                    converted_status_filter.append(status)

        req = self._build_request(
            request,
            GetUserProfilesRequest,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
            status_filter=converted_status_filter,
        )

        # Check cache if not forcing refresh
        if not force_refresh:
            cached_result = self._cache.get(
                "get_profiles",
                user_id=req.user_id,
                start_time=req.start_time,
                end_time=req.end_time,
                top_k=req.top_k,
                status_filter=req.status_filter,
            )
            if cached_result is not None:
                return cached_result

        # Make API call
        response = self._make_request(
            "POST",
            "/api/get_profiles",
            json=req.model_dump(),
        )
        result = GetProfilesViewResponse(**response)

        # Store in cache
        self._cache.set(
            "get_profiles",
            result,
            user_id=req.user_id,
            start_time=req.start_time,
            end_time=req.end_time,
            top_k=req.top_k,
            status_filter=req.status_filter,
        )

        return result

    def get_all_interactions(
        self,
        limit: int = 100,
    ) -> GetInteractionsViewResponse:
        """Get all user interactions across all users.

        Args:
            limit (int, optional): Maximum number of interactions to return. Defaults to 100.

        Returns:
            GetInteractionsViewResponse: Response containing all user interactions
        """
        response = self._make_request(
            "GET",
            f"/api/get_all_interactions?limit={limit}",
        )
        return GetInteractionsViewResponse(**response)

    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: str | None = None,
    ) -> GetProfilesViewResponse:
        """Get all user profiles across all users.

        Args:
            limit (int, optional): Maximum number of profiles to return. Defaults to 100.
            status_filter (str, optional): Filter by profile status. Accepts
                ``"current"``, ``"pending"``, or ``"archived"``. If ``None``
                (the default), profiles with any status are returned.

        Returns:
            GetProfilesViewResponse: Response containing all user profiles
        """
        from urllib.parse import urlencode

        params: dict[str, str | int] = {"limit": limit}
        if status_filter:
            params["status_filter"] = status_filter
        response = self._make_request(
            "GET",
            f"/api/get_all_profiles?{urlencode(params)}",
        )
        return GetProfilesViewResponse(**response)

    def set_config(self, config: Config | dict) -> dict:
        """Set configuration for the organization.

        Args:
            config (Union[Config, dict]): The configuration to set

        Returns:
            dict: Response containing success status and message
        """
        config = self._convert_to_model(config, Config)  # type: ignore[reportAssignmentType]
        return self._make_request(
            "POST",
            "/api/set_config",
            json=config.model_dump(),  # type: ignore[reportAttributeAccessIssue]
        )

    def get_config(self) -> Config:
        """Get configuration for the organization.

        Returns:
            Config: The current configuration
        """
        response = self._make_request(
            "GET",
            "/api/get_config",
        )
        return Config(**response)

    def get_user_playbooks(
        self,
        request: GetUserPlaybooksRequest | dict | None = None,
        *,
        limit: int | None = None,
        user_id: str | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
    ) -> GetUserPlaybooksViewResponse:
        """Get user playbooks.

        Args:
            request (Optional[GetUserPlaybooksRequest]): The get request object (alternative to kwargs)
            limit (Optional[int]): Maximum number of results to return (default: 100)
            user_id (Optional[str]): Filter by user ID
            playbook_name (Optional[str]): Filter by playbook name
            agent_version (Optional[str]): Filter by agent version
            status_filter (Optional[list[Optional[Status]]]): Filter by status

        Returns:
            GetUserPlaybooksViewResponse: Response containing user playbooks
        """
        req = self._build_request(
            request,
            GetUserPlaybooksRequest,
            limit=limit,
            user_id=user_id,
            playbook_name=playbook_name,
            agent_version=agent_version,
            status_filter=status_filter,
        )
        response = self._make_request(
            "POST",
            "/api/get_user_playbooks",
            json=req.model_dump(),
        )
        return GetUserPlaybooksViewResponse(**response)

    def add_user_playbook(
        self,
        user_playbooks: list[UserPlaybook | dict],
    ) -> AddUserPlaybookResponse:
        """Add user playbooks directly to storage.

        Args:
            user_playbooks (list[Union[UserPlaybook, dict]]): List of user playbooks to add.
                Each user playbook should contain:
                - agent_version (str): Required. The agent version.
                - request_id (str): Required. The request ID.
                - content (str): Required. The playbook content.
                - playbook_name (str): Optional. The playbook name/category.

        Returns:
            AddUserPlaybookResponse: Response containing success status, message, and added count.
        """
        # Convert dicts to UserPlaybook objects if needed
        user_playbook_list = [
            UserPlaybook(**rf) if isinstance(rf, dict) else rf for rf in user_playbooks
        ]
        request = AddUserPlaybookRequest(user_playbooks=user_playbook_list)
        response = self._make_request(
            "POST",
            "/api/add_user_playbook",
            json=request.model_dump(),
        )
        return AddUserPlaybookResponse(**response)

    def add_agent_playbooks(
        self,
        agent_playbooks: list[AgentPlaybook | dict],
    ) -> AddAgentPlaybookResponse:
        """Add agent playbooks directly to storage.

        Args:
            agent_playbooks (list[Union[AgentPlaybook, dict]]): List of agent playbooks to add.
                Each agent playbook should contain:
                - agent_version (str): Required. The agent version.
                - content (str): Required. The playbook content.
                - playbook_status (PlaybookStatus): Required. The playbook approval status.
                - playbook_metadata (str): Required. Metadata about the playbook.
                - playbook_name (str): Optional. The playbook name/category.

        Returns:
            AddAgentPlaybookResponse: Response containing success status, message, and added count.
        """
        # Convert dicts to AgentPlaybook objects if needed
        agent_playbook_list = [
            AgentPlaybook(**fb) if isinstance(fb, dict) else fb
            for fb in agent_playbooks
        ]
        request = AddAgentPlaybookRequest(agent_playbooks=agent_playbook_list)
        response = self._make_request(
            "POST",
            "/api/add_agent_playbook",
            json=request.model_dump(),
        )
        return AddAgentPlaybookResponse(**response)

    @staticmethod
    def _coerce_user_profile(profile: UserProfile | dict) -> UserProfile:
        """Coerce a dict into a ``UserProfile``, filling in missing required fields.

        Required fields on ``UserProfile`` that callers commonly omit
        (``profile_id``, ``last_modified_timestamp``, ``generated_from_request_id``)
        are filled in with sensible client-side defaults so that a minimal dict
        like ``{"user_id": "u", "content": "x"}`` validates successfully.

        Args:
            profile (Union[UserProfile, dict]): The profile to coerce. If it is
                already a ``UserProfile`` it is returned unchanged.

        Returns:
            UserProfile: A fully-populated ``UserProfile`` instance.
        """
        if isinstance(profile, UserProfile):
            return profile
        data = dict(profile)
        data.setdefault("profile_id", f"cli-{uuid.uuid4().hex[:12]}")
        data.setdefault("last_modified_timestamp", int(datetime.now(UTC).timestamp()))
        data.setdefault("generated_from_request_id", "cli-manual")
        data.setdefault("source", "cli-manual")
        return UserProfile(**data)

    def add_user_profile(
        self,
        user_profiles: list[UserProfile | dict],
    ) -> AddUserProfileResponse:
        """Add user profiles directly to storage, bypassing inference.

        Mirror of :meth:`add_user_playbook` for the profile resource.
        Useful for seeding a known fact about the user (testing,
        migration, manual fact injection) without going through the
        interaction-based generation pipeline. The server populates
        the embedding automatically.

        Args:
            user_profiles (list[Union[UserProfile, dict]]): List of user profiles to add.
                Each profile must contain at least:
                - user_id (str): The user the profile belongs to.
                - content (str): The profile content (used for embedding).
                When passing dicts, missing required fields are auto-populated
                client-side with sensible defaults: ``profile_id`` becomes
                ``f"cli-{uuid4.hex[:12]}"``, ``last_modified_timestamp`` is set to
                ``int(datetime.now(UTC).timestamp())``, and
                ``generated_from_request_id`` defaults to ``"cli-manual"``.

        Returns:
            AddUserProfileResponse: Response containing success status, message, and added count.
        """
        user_profile_list = [self._coerce_user_profile(p) for p in user_profiles]
        request = AddUserProfileRequest(user_profiles=user_profile_list)
        response = self._make_request(
            "POST",
            "/api/add_user_profile",
            json=request.model_dump(),
        )
        return AddUserProfileResponse(**response)

    def update_user_playbook(
        self,
        user_playbook_id: int,
        *,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
    ) -> UpdateUserPlaybookResponse:
        """Update editable fields of a user playbook in place.

        Pass only the fields you want to change. Fields left as
        ``None`` are not touched on the server side.

        Args:
            user_playbook_id (int): The ID of the user playbook to update.
            playbook_name (Optional[str]): New playbook category name.
            content (Optional[str]): New content text.
            trigger (Optional[str]): New trigger condition.
            rationale (Optional[str]): New rationale text.

        Returns:
            UpdateUserPlaybookResponse: Response containing success status and message.
        """
        request = UpdateUserPlaybookRequest(
            user_playbook_id=user_playbook_id,
            playbook_name=playbook_name,
            content=content,
            trigger=trigger,
            rationale=rationale,
        )
        response = self._make_request(
            "PUT",
            "/api/update_user_playbook",
            json=request.model_dump(),
        )
        return UpdateUserPlaybookResponse(**response)

    def update_agent_playbook(
        self,
        agent_playbook_id: int,
        *,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
        playbook_status: PlaybookStatus | None = None,
    ) -> UpdateAgentPlaybookResponse:
        """Update editable fields of an agent playbook in place.

        Pass only the fields you want to change. To change ONLY the
        approval status, prefer :meth:`update_agent_playbook_status` —
        it has tighter semantics and a single-purpose endpoint.

        Args:
            agent_playbook_id (int): The ID of the agent playbook to update.
            playbook_name (Optional[str]): New playbook category name.
            content (Optional[str]): New content text.
            trigger (Optional[str]): New trigger condition.
            rationale (Optional[str]): New rationale text.
            playbook_status (Optional[PlaybookStatus]): New approval status.

        Returns:
            UpdateAgentPlaybookResponse: Response containing success status and message.
        """
        request = UpdateAgentPlaybookRequest(
            agent_playbook_id=agent_playbook_id,
            playbook_name=playbook_name,
            content=content,
            trigger=trigger,
            rationale=rationale,
            playbook_status=playbook_status,
        )
        response = self._make_request(
            "PUT",
            "/api/update_agent_playbook",
            json=request.model_dump(),
        )
        return UpdateAgentPlaybookResponse(**response)

    def update_agent_playbook_status(
        self,
        agent_playbook_id: int,
        *,
        playbook_status: PlaybookStatus,
    ) -> UpdatePlaybookStatusResponse:
        """Update only the approval status of an agent playbook.

        This is the dedicated endpoint for the approval workflow
        (approve / pending / reject). Use it instead of
        :meth:`update_agent_playbook` when the only change is the
        ``playbook_status`` — the server enforces tighter validation
        and writes a smaller change log.

        Args:
            agent_playbook_id (int): The ID of the agent playbook.
            playbook_status (PlaybookStatus): New approval status
                (``PENDING``, ``APPROVED``, or ``REJECTED``).

        Returns:
            UpdatePlaybookStatusResponse: Response containing success status and message.
        """
        request = UpdatePlaybookStatusRequest(
            agent_playbook_id=agent_playbook_id,
            playbook_status=playbook_status,
        )
        response = self._make_request(
            "PUT",
            "/api/update_agent_playbook_status",
            json=request.model_dump(),
        )
        return UpdatePlaybookStatusResponse(**response)

    def get_agent_playbooks(
        self,
        request: GetAgentPlaybooksRequest | dict | None = None,
        force_refresh: bool = False,
        *,
        limit: int | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: PlaybookStatus | None = None,
    ) -> GetAgentPlaybooksViewResponse:
        """Get agent playbooks.

        Args:
            request (Optional[GetAgentPlaybooksRequest]): The get request object (alternative to kwargs)
            force_refresh (bool, optional): If True, bypass cache and fetch fresh data. Defaults to False.
            limit (Optional[int]): Maximum number of results to return (default: 100)
            playbook_name (Optional[str]): Filter by playbook name
            agent_version (Optional[str]): Filter by agent version
            status_filter (Optional[list[Optional[Status]]]): Filter by status
            playbook_status_filter (Optional[PlaybookStatus]): Filter by playbook status (default: APPROVED)

        Returns:
            GetAgentPlaybooksViewResponse: Response containing agent playbooks
        """
        req = self._build_request(
            request,
            GetAgentPlaybooksRequest,
            limit=limit,
            playbook_name=playbook_name,
            agent_version=agent_version,
            status_filter=status_filter,
            playbook_status_filter=playbook_status_filter,
        )

        # Check cache if not forcing refresh
        if not force_refresh:
            cached_result = self._cache.get(
                "get_agent_playbooks",
                limit=req.limit,
                playbook_name=req.playbook_name,
                status_filter=req.status_filter,
                playbook_status_filter=req.playbook_status_filter,
            )
            if cached_result is not None:
                return cached_result

        # Make API call
        response = self._make_request(
            "POST",
            "/api/get_agent_playbooks",
            json=req.model_dump(),
        )
        result = GetAgentPlaybooksViewResponse(**response)

        # Store in cache
        self._cache.set(
            "get_agent_playbooks",
            result,
            limit=req.limit,
            playbook_name=req.playbook_name,
            status_filter=req.status_filter,
            playbook_status_filter=req.playbook_status_filter,
        )

        return result

    def get_requests(
        self,
        request: GetRequestsRequest | dict | None = None,
        *,
        user_id: str | None = None,
        request_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        top_k: int | None = None,
    ) -> GetRequestsViewResponse:
        """Get requests with their associated interactions, grouped by session.

        Args:
            request (Optional[GetRequestsRequest]): The get request object (alternative to kwargs)
            user_id (Optional[str]): Filter by user ID
            request_id (Optional[str]): Filter by request ID
            start_time (Optional[datetime]): Filter by start time
            end_time (Optional[datetime]): Filter by end time
            top_k (Optional[int]): Maximum number of results to return (default: 30)

        Returns:
            GetRequestsViewResponse: Response containing requests grouped by session with their interactions
        """
        req = self._build_request(
            request,
            GetRequestsRequest,
            user_id=user_id,
            request_id=request_id,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
        )
        response = self._make_request(
            "POST",
            "/api/get_requests",
            json=req.model_dump(),
        )
        return GetRequestsViewResponse(**response)

    def get_agent_success_evaluation_results(
        self,
        request: GetAgentSuccessEvaluationResultsRequest | dict | None = None,
        *,
        limit: int | None = None,
        agent_version: str | None = None,
    ) -> GetEvaluationResultsViewResponse:
        """Get agent success evaluation results.

        Args:
            request (Optional[GetAgentSuccessEvaluationResultsRequest]): The get request object (alternative to kwargs)
            limit (Optional[int]): Maximum number of results to return (default: 100)
            agent_version (Optional[str]): Filter by agent version

        Returns:
            GetEvaluationResultsViewResponse: Response containing agent success evaluation results
        """
        req = self._build_request(
            request,
            GetAgentSuccessEvaluationResultsRequest,
            limit=limit,
            agent_version=agent_version,
        )
        response = self._make_request(
            "POST",
            "/api/get_agent_success_evaluation_results",
            json=req.model_dump(),
        )
        return GetEvaluationResultsViewResponse(**response)

    def _poll_operation_status(
        self, service_name: str, poll_interval: float = 3.0, max_wait: float = 600.0
    ) -> GetOperationStatusResponse:
        """
        Poll the operation status endpoint until the operation completes, fails, or is cancelled.

        Args:
            service_name: The service name to poll (e.g. "profile_generation", "playbook_generation")
            poll_interval: Seconds between polls
            max_wait: Maximum seconds to wait before raising TimeoutError

        Returns:
            GetOperationStatusResponse: Final operation status
        """
        start = time.monotonic()
        while True:
            try:
                response = self._make_request(
                    "GET",
                    "/api/get_operation_status",
                    params={"service_name": service_name},
                )
            except Exception as e:
                logger.warning("Failed to poll operation status: %s", e)
                elapsed = time.monotonic() - start
                if elapsed + poll_interval > max_wait:
                    raise TimeoutError(
                        f"Operation '{service_name}' did not complete within {max_wait}s"
                    ) from e
                time.sleep(poll_interval)
                continue
            status_response = GetOperationStatusResponse(**response)
            op = status_response.operation_status
            if op and op.status in (
                OperationStatus.COMPLETED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
            ):
                return status_response
            elapsed = time.monotonic() - start
            if elapsed + poll_interval > max_wait:
                raise TimeoutError(
                    f"Operation '{service_name}' did not complete within {max_wait}s"
                )
            time.sleep(poll_interval)

    def _rerun_profile_generation_sync(
        self, request: RerunProfileGenerationRequest
    ) -> RerunProfileGenerationResponse:
        """Internal sync method to rerun profile generation.

        Submits the request, then polls operation status until completion.
        """
        response = self._make_request(
            "POST",
            "/api/rerun_profile_generation",
            json=request.model_dump(),
        )
        initial = RerunProfileGenerationResponse(**response)
        if not initial.success:
            return initial

        # Poll until the background task completes
        try:
            status_response = self._poll_operation_status("profile_generation")
            op = status_response.operation_status
            if op and op.status == OperationStatus.COMPLETED:
                return RerunProfileGenerationResponse(
                    success=True,
                    msg="Profile generation completed",
                    profiles_generated=op.processed_users,
                )
            if op and op.status == OperationStatus.FAILED:
                return RerunProfileGenerationResponse(
                    success=False,
                    msg=op.error_message or "Profile generation failed",
                )
            return RerunProfileGenerationResponse(
                success=False,
                msg="Profile generation was cancelled",
            )
        except (TimeoutError, Exception) as e:
            logger.warning("Error while polling profile generation status: %s", e)
            return initial

    async def _rerun_profile_generation_async(
        self, request: RerunProfileGenerationRequest
    ) -> RerunProfileGenerationResponse:
        """Internal async method to rerun profile generation."""
        response = await self._make_async_request(
            "POST",
            "/api/rerun_profile_generation",
            json=request.model_dump(),
        )
        return RerunProfileGenerationResponse(**response)

    def rerun_profile_generation(
        self,
        request: RerunProfileGenerationRequest | dict | None = None,
        wait_for_response: bool = False,
        *,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        source: str | None = None,
        extractor_names: list[str] | None = None,
    ) -> RerunProfileGenerationResponse | None:
        """Rerun profile generation for users.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            request (Optional[RerunProfileGenerationRequest]): The rerun request object (alternative to kwargs)
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.
            user_id (Optional[str]): Specific user ID to rerun for. If None, runs for all users.
            start_time (Optional[datetime]): Filter interactions by start time.
            end_time (Optional[datetime]): Filter interactions by end time.
            source (Optional[str]): Filter interactions by source.
            extractor_names (Optional[list[str]]): List of specific extractor names to run. If None, runs all extractors.

        Returns:
            Optional[RerunProfileGenerationResponse]: Response containing success status, message, profiles_generated count, and operation_id if wait_for_response=True, None otherwise.
        """
        req = self._build_request(
            request,
            RerunProfileGenerationRequest,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            source=source,
            extractor_names=extractor_names,
        )

        if wait_for_response:
            return self._rerun_profile_generation_sync(req)
        self._fire_and_forget(self._rerun_profile_generation_async, req)
        return None

    def upgrade_profiles(
        self,
        *,
        user_id: str | None = None,
        only_affected_users: bool = True,
    ) -> UpgradeProfilesResponse:
        """Promote PENDING profiles to CURRENT, archive old CURRENT, delete old ARCHIVED.

        Args:
            user_id: Specific user ID to upgrade. If None, upgrades all users.
            only_affected_users: If True, only upgrade users who have pending profiles.

        Returns:
            UpgradeProfilesResponse: Counts of archived, promoted, and deleted profiles.
        """
        req = UpgradeProfilesRequest(
            user_id=user_id,
            only_affected_users=only_affected_users,
        )
        response = self._make_request(
            "POST",
            "/api/upgrade_all_profiles",
            json=req.model_dump(),
        )
        return UpgradeProfilesResponse(**response)

    def upgrade_user_playbooks(
        self,
        *,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> UpgradeUserPlaybooksResponse:
        """Promote PENDING user playbooks to CURRENT, archive old CURRENT, delete old ARCHIVED.

        Args:
            agent_version: Filter by agent version. If None, upgrades all versions.
            playbook_name: Filter by playbook name. If None, upgrades all playbooks.

        Returns:
            UpgradeUserPlaybooksResponse: Counts of archived, promoted, and deleted playbooks.
        """
        req = UpgradeUserPlaybooksRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
        response = self._make_request(
            "POST",
            "/api/upgrade_all_user_playbooks",
            json=req.model_dump(),
        )
        return UpgradeUserPlaybooksResponse(**response)

    async def _manual_profile_generation_async(
        self, request: ManualProfileGenerationRequest
    ) -> None:
        """Internal async method for manual profile generation."""
        await self._make_async_request(
            "POST",
            "/api/manual_profile_generation",
            json=request.model_dump(),
        )

    def manual_profile_generation(
        self,
        request: ManualProfileGenerationRequest | dict | None = None,
        *,
        user_id: str | None = None,
        source: str | None = None,
        extractor_names: list[str] | None = None,
    ) -> None:
        """Manually trigger profile generation with window-sized interactions (fire-and-forget).

        Unlike rerun_profile_generation which uses ALL interactions and outputs PENDING status,
        this method uses window-sized interactions (from batch_size config) and
        outputs profiles with CURRENT status.

        This is a fire-and-forget operation that runs asynchronously in the background.

        Args:
            request (Optional[ManualProfileGenerationRequest]): The request object (alternative to kwargs)
            user_id (Optional[str]): Specific user ID to generate for. If None, generates for all users.
            source (Optional[str]): Filter interactions by source.
            extractor_names (Optional[list[str]]): List of specific extractor names to run. If None, runs all extractors with allow_manual_trigger=True.

        Returns:
            None: This method always returns None (fire-and-forget).
        """
        req = self._build_request(
            request,
            ManualProfileGenerationRequest,
            user_id=user_id,
            source=source,
            extractor_names=extractor_names,
        )
        self._fire_and_forget(self._manual_profile_generation_async, req)

    def _rerun_playbook_generation_sync(
        self, request: RerunPlaybookGenerationRequest
    ) -> RerunPlaybookGenerationResponse:
        """Internal sync method to rerun playbook generation.

        Submits the request, then polls operation status until completion.
        """
        response = self._make_request(
            "POST",
            "/api/rerun_playbook_generation",
            json=request.model_dump(),
        )
        initial = RerunPlaybookGenerationResponse(**response)
        if not initial.success:
            return initial

        # Poll until the background task completes
        try:
            status_response = self._poll_operation_status("feedback_generation")
            op = status_response.operation_status
            if op and op.status == OperationStatus.COMPLETED:
                return RerunPlaybookGenerationResponse(
                    success=True,
                    msg="Playbook generation completed",
                    playbooks_generated=op.processed_users,
                )
            if op and op.status == OperationStatus.FAILED:
                return RerunPlaybookGenerationResponse(
                    success=False,
                    msg=op.error_message or "Playbook generation failed",
                )
            return RerunPlaybookGenerationResponse(
                success=False,
                msg="Playbook generation was cancelled",
            )
        except (TimeoutError, Exception) as e:
            logger.warning("Error while polling playbook generation status: %s", e)
            return initial

    async def _rerun_playbook_generation_async(
        self, request: RerunPlaybookGenerationRequest
    ) -> RerunPlaybookGenerationResponse:
        """Internal async method to rerun playbook generation."""
        response = await self._make_async_request(
            "POST",
            "/api/rerun_playbook_generation",
            json=request.model_dump(),
        )
        return RerunPlaybookGenerationResponse(**response)

    def rerun_playbook_generation(
        self,
        request: RerunPlaybookGenerationRequest | dict | None = None,
        wait_for_response: bool = False,
        *,
        agent_version: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        playbook_name: str | None = None,
    ) -> RerunPlaybookGenerationResponse | None:
        """Rerun playbook generation for an agent version.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            request (Optional[RerunPlaybookGenerationRequest]): The rerun request object (alternative to kwargs)
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.
            agent_version (str): Required. The agent version to evaluate.
            start_time (Optional[datetime]): Filter by start time.
            end_time (Optional[datetime]): Filter by end time.
            playbook_name (Optional[str]): Specific playbook type to generate.

        Returns:
            Optional[RerunPlaybookGenerationResponse]: Response containing success status, message, playbooks_generated count, and operation_id if wait_for_response=True, None otherwise.
        """
        req = self._build_request(
            request,
            RerunPlaybookGenerationRequest,
            agent_version=agent_version,
            start_time=start_time,
            end_time=end_time,
            playbook_name=playbook_name,
        )

        if wait_for_response:
            return self._rerun_playbook_generation_sync(req)
        self._fire_and_forget(self._rerun_playbook_generation_async, req)
        return None

    async def _manual_playbook_generation_async(
        self, request: ManualPlaybookGenerationRequest
    ) -> None:
        """Internal async method for manual playbook generation."""
        await self._make_async_request(
            "POST",
            "/api/manual_playbook_generation",
            json=request.model_dump(),
        )

    def manual_playbook_generation(
        self,
        request: ManualPlaybookGenerationRequest | dict | None = None,
        *,
        agent_version: str | None = None,
        source: str | None = None,
        playbook_name: str | None = None,
    ) -> None:
        """Manually trigger playbook generation with window-sized interactions (fire-and-forget).

        Unlike rerun_playbook_generation which uses ALL interactions and outputs PENDING status,
        this method uses window-sized interactions (from batch_size config) and
        outputs playbooks with CURRENT status.

        This is a fire-and-forget operation that runs asynchronously in the background.

        Args:
            request (Optional[ManualPlaybookGenerationRequest]): The request object (alternative to kwargs)
            agent_version (str): Required. The agent version to evaluate.
            source (Optional[str]): Filter interactions by source.
            playbook_name (Optional[str]): Specific playbook type to generate.

        Returns:
            None: This method always returns None (fire-and-forget).
        """
        req = self._build_request(
            request,
            ManualPlaybookGenerationRequest,
            agent_version=agent_version,
            source=source,
            playbook_name=playbook_name,
        )
        self._fire_and_forget(self._manual_playbook_generation_async, req)

    def _run_playbook_aggregation_sync(
        self, request: RunPlaybookAggregationRequest
    ) -> RunPlaybookAggregationResponse:
        """Internal sync method to run playbook aggregation."""
        response = self._make_request(
            "POST",
            "/api/run_playbook_aggregation",
            json=request.model_dump(),
        )
        return RunPlaybookAggregationResponse(**response)

    async def _run_playbook_aggregation_async(
        self, request: RunPlaybookAggregationRequest
    ) -> RunPlaybookAggregationResponse:
        """Internal async method to run playbook aggregation."""
        response = await self._make_async_request(
            "POST",
            "/api/run_playbook_aggregation",
            json=request.model_dump(),
        )
        return RunPlaybookAggregationResponse(**response)

    def run_playbook_aggregation(
        self,
        request: RunPlaybookAggregationRequest | dict | None = None,
        wait_for_response: bool = False,
        *,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> RunPlaybookAggregationResponse | None:
        """Run playbook aggregation to cluster similar user playbooks.

        This method is optimized for resource efficiency:
        - In async contexts (e.g., FastAPI): Uses existing event loop (most efficient)
        - In sync contexts: Uses shared thread pool (avoids thread creation overhead)

        Args:
            request (Optional[RunPlaybookAggregationRequest]): The aggregation request object (alternative to kwargs)
            wait_for_response (bool, optional): If True, wait for response. If False, send request without waiting. Defaults to False.
            agent_version (str): Required. The agent version.
            playbook_name (str): Required. The playbook type to aggregate.

        Returns:
            Optional[RunPlaybookAggregationResponse]: Response containing success status and message if wait_for_response=True, None otherwise.
        """
        req = self._build_request(
            request,
            RunPlaybookAggregationRequest,
            agent_version=agent_version,
            playbook_name=playbook_name,
        )

        if wait_for_response:
            return self._run_playbook_aggregation_sync(req)
        self._fire_and_forget(self._run_playbook_aggregation_async, req)
        return None

    def search(
        self,
        request: UnifiedSearchRequest | dict | None = None,
        *,
        query: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
        user_id: str | None = None,
        enable_reformulation: bool | None = None,
        conversation_history: list[ConversationTurn] | list[dict] | None = None,
        search_mode: SearchMode | str | None = None,
    ) -> UnifiedSearchViewResponse:
        """Search across all entity types (profiles, agent playbooks, user playbooks).

        Runs query reformulation and searches all entity types in parallel.
        Query reformulation is controlled per-request via the enable_reformulation parameter.

        Args:
            request (Optional[UnifiedSearchRequest]): The search request object (alternative to kwargs)
            query (str): Search query text
            top_k (Optional[int]): Maximum results per entity type (default: 5)
            threshold (Optional[float]): Similarity threshold for vector search (default: 0.3)
            agent_version (Optional[str]): Filter by agent version (agent_playbooks, user_playbooks)
            playbook_name (Optional[str]): Filter by playbook name (agent_playbooks, user_playbooks)
            user_id (Optional[str]): Filter by user ID (profiles, user_playbooks)
            enable_reformulation (Optional[bool]): Enable LLM query reformulation (default: False)
            conversation_history (Optional[list[ConversationTurn] | list[dict]]): Prior conversation turns for context-aware query reformulation. Accepts ConversationTurn objects or dicts with "role" and "content" keys.
            search_mode (Optional[SearchMode | str]): Search mode to use. Accepts SearchMode enum or string value ("vector", "fts", "hybrid").

        Returns:
            UnifiedSearchViewResponse: Combined search results from all entity types
        """
        req = self._build_request(
            request,
            UnifiedSearchRequest,
            query=query,
            top_k=top_k,
            threshold=threshold,
            agent_version=agent_version,
            playbook_name=playbook_name,
            user_id=user_id,
            enable_reformulation=enable_reformulation,
            conversation_history=conversation_history,
            search_mode=search_mode,
        )
        response = self._make_request("POST", "/api/search", json=req.model_dump())
        return UnifiedSearchViewResponse(**response)

    # =========================================================================
    # Bulk Delete Operations
    # =========================================================================

    def delete_requests_by_ids(self, request_ids: list[str]) -> BulkDeleteResponse:
        """Delete multiple requests by their IDs.

        Args:
            request_ids (list[str]): List of request IDs to delete

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        req = DeleteRequestsByIdsRequest(request_ids=request_ids)
        response = self._make_request(
            "DELETE", "/api/delete_requests_by_ids", json=req.model_dump()
        )
        return BulkDeleteResponse(**response)

    def delete_profiles_by_ids(self, profile_ids: list[str]) -> BulkDeleteResponse:
        """Delete multiple profiles by their IDs.

        Args:
            profile_ids (list[str]): List of profile IDs to delete

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        req = DeleteProfilesByIdsRequest(profile_ids=profile_ids)
        response = self._make_request(
            "DELETE", "/api/delete_profiles_by_ids", json=req.model_dump()
        )
        return BulkDeleteResponse(**response)

    def delete_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        """Delete multiple agent playbooks by their IDs.

        Args:
            agent_playbook_ids (list[int]): List of agent playbook IDs to delete

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        req = DeleteAgentPlaybooksByIdsRequest(agent_playbook_ids=agent_playbook_ids)
        response = self._make_request(
            "DELETE", "/api/delete_agent_playbooks_by_ids", json=req.model_dump()
        )
        return BulkDeleteResponse(**response)

    def delete_user_playbooks_by_ids(
        self, user_playbook_ids: list[int]
    ) -> BulkDeleteResponse:
        """Delete multiple user playbooks by their IDs.

        Args:
            user_playbook_ids (list[int]): List of user playbook IDs to delete

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        req = DeleteUserPlaybooksByIdsRequest(user_playbook_ids=user_playbook_ids)
        response = self._make_request(
            "DELETE", "/api/delete_user_playbooks_by_ids", json=req.model_dump()
        )
        return BulkDeleteResponse(**response)

    def whoami(self) -> WhoamiResponse:
        """Return the server's view of the caller's org and storage routing.

        Returns:
            WhoamiResponse: Masked summary of org ID and resolved storage.
                Never contains raw credentials — safe to print.
        """
        response = self._make_request("GET", "/api/whoami")
        return WhoamiResponse(**response)

    def get_my_config(self) -> MyConfigResponse:
        """Return raw storage credentials for the caller's org.

        Used by ``reflexio config pull`` / ``config storage`` to let users
        move their per-org server-side config to a fresh machine.

        Returns:
            MyConfigResponse: Unmasked storage config dict, or
                ``success=False`` when no storage is configured
                server-side or the endpoint is disabled.
        """
        response = self._make_request("GET", "/api/my_config")
        return MyConfigResponse(**response)

    def delete_all_interactions(self) -> BulkDeleteResponse:
        """Delete all requests and their associated interactions.

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        response = self._make_request("DELETE", "/api/delete_all_interactions")
        return BulkDeleteResponse(**response)

    def delete_all_profiles(self) -> BulkDeleteResponse:
        """Delete all profiles.

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        response = self._make_request("DELETE", "/api/delete_all_profiles")
        return BulkDeleteResponse(**response)

    def delete_all_playbooks(self) -> BulkDeleteResponse:
        """Delete all playbooks (both user and agent).

        Cascading variant — wipes both playbook stores. For per-entity
        semantics use :meth:`delete_all_user_playbooks` (user only) or
        :meth:`delete_all_agent_playbooks` (agent only).

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        response = self._make_request("DELETE", "/api/delete_all_playbooks")
        return BulkDeleteResponse(**response)

    def delete_all_user_playbooks(self) -> BulkDeleteResponse:
        """Delete all user playbooks (user only, not agent).

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        response = self._make_request("DELETE", "/api/delete_all_user_playbooks")
        return BulkDeleteResponse(**response)

    def delete_all_agent_playbooks(self) -> BulkDeleteResponse:
        """Delete all agent playbooks (agent only, not user).

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        response = self._make_request("DELETE", "/api/delete_all_agent_playbooks")
        return BulkDeleteResponse(**response)
