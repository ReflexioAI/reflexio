from abc import abstractmethod

from reflexio.models.api_schema.domain import Request
from reflexio.models.api_schema.internal_schema import (
    RequestInteractionDataModel,
    SessionDescriptor,
    SessionFirstRequest,
)


class RequestMixin:
    """Mixin for request CRUD methods."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to storage.

        Args:
            request: Request object to store
        """
        raise NotImplementedError

    @abstractmethod
    def get_request(self, request_id: str) -> Request | None:
        """Get a request by its ID.

        Args:
            request_id: The request ID to retrieve

        Returns:
            Request object if found, None otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def delete_request(self, request_id: str) -> None:
        """Delete a request by its ID.

        Args:
            request_id: The request ID to delete
        """
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Delete all requests and interactions in a session.

        Args:
            session_id: The session ID to delete

        Returns:
            int: Number of requests deleted
        """
        raise NotImplementedError

    @abstractmethod
    def delete_all_requests(self) -> None:
        """Delete all requests and their associated interactions."""
        raise NotImplementedError

    @abstractmethod
    def delete_requests_by_ids(self, request_ids: list[str]) -> int:
        """Delete requests and their associated interactions by request IDs.

        Args:
            request_ids (list[str]): List of request IDs to delete

        Returns:
            int: Number of requests deleted
        """
        raise NotImplementedError

    @abstractmethod
    def get_sessions(
        self,
        user_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        top_k: int | None = 30,
        offset: int = 0,
    ) -> dict[str, list[RequestInteractionDataModel]]:
        """Get requests with their associated interactions, grouped by session_id.

        Args:
            user_id (str, optional): User ID to filter requests.
            request_id (str, optional): Specific request ID to retrieve
            session_id (str, optional): Specific session ID to retrieve
            start_time (int, optional): Start timestamp for filtering
            end_time (int, optional): End timestamp for filtering
            top_k (int, optional): Maximum number of requests to return
            offset (int): Number of requests to skip for pagination. Defaults to 0.

        Returns:
            dict[str, list[RequestInteractionDataModel]]: Dictionary mapping session_id to list of RequestInteractionDataModel objects
        """
        raise NotImplementedError

    @abstractmethod
    def get_rerun_user_ids(
        self,
        user_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        source: str | None = None,
        agent_version: str | None = None,
    ) -> list[str]:
        """Get distinct user IDs that have matching requests for rerun workflows.

        Args:
            user_id (str, optional): Restrict to a specific user ID.
            start_time (int, optional): Start timestamp for request filtering.
            end_time (int, optional): End timestamp for request filtering.
            source (str, optional): Restrict to requests from a source.
            agent_version (str, optional): Restrict to requests with an agent version.

        Returns:
            list[str]: Distinct user IDs matching the filters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_requests_by_session(self, user_id: str, session_id: str) -> list[Request]:
        """Get all requests for a specific session.

        Args:
            user_id (str): User ID to filter requests
            session_id (str): Session ID to filter by

        Returns:
            list[Request]: List of Request objects in the session
        """
        raise NotImplementedError

    @abstractmethod
    def get_session_ids_in_window(
        self, from_ts: int, to_ts: int
    ) -> list[SessionDescriptor]:
        """Return one descriptor per distinct (user_id, session_id, agent_version, source) tuple with at least one request whose ``created_at`` falls inside ``[from_ts, to_ts]``.

        A session that records requests under multiple ``agent_version`` or
        ``source`` values within the window yields one descriptor per distinct
        combination, so regen workers can invoke ``run_group_evaluation`` once
        per (user, session, agent_version, source) tuple without conflating
        runs from different agent versions.

        Args:
            from_ts (int): Inclusive lower bound (Unix seconds).
            to_ts (int): Inclusive upper bound (Unix seconds).

        Returns:
            list[SessionDescriptor]: Deduped descriptors ordered by session_id.
        """
        raise NotImplementedError

    def get_first_requests_by_session_ids(
        self, session_ids: list[str]
    ) -> dict[str, SessionFirstRequest]:
        """Return earliest-request metadata for each requested session.

        Default implementation preserves the historical storage contract by
        calling ``get_sessions`` per session. SQL backends should override with
        a set-based query.
        """
        out: dict[str, SessionFirstRequest] = {}
        for session_id in set(session_ids):
            requests = []
            page_size = 1000
            offset = 0
            while True:
                grouped = self.get_sessions(
                    session_id=session_id,
                    top_k=page_size,
                    offset=offset,
                )
                rows = grouped.get(session_id) or []
                requests.extend(r.request for r in rows if r.request is not None)
                if len(rows) < page_size:
                    break
                offset += page_size
            if not requests:
                continue
            first = min(requests, key=lambda r: r.created_at)
            out[session_id] = SessionFirstRequest(
                session_id=session_id,
                user_id=first.user_id,
                source=first.source or "",
                created_at=first.created_at,
            )
        return out
