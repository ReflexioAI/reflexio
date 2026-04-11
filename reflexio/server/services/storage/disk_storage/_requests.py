import logging

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)

logger = logging.getLogger(__name__)


class RequestMixin:
    # ------------------------------------------------------------------
    # Request methods
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        with self._lock:
            path = self._entity_path(
                self._requests_dir(),
                str(request.request_id),
            )
            self._write_entity(path, request)

    def get_request(self, request_id: str) -> Request | None:
        with self._lock:
            path = self._entity_path(self._requests_dir(), request_id)
            if not path.exists():
                return None
            return self._read_entity(path, Request)

    def delete_request(self, request_id: str) -> None:
        with self._lock:
            # Delete all interactions associated with this request
            for p in self._scan_entities(self._interactions_dir(), recursive=True):
                interaction = self._read_entity(p, Interaction)
                if interaction.request_id == request_id:
                    self._delete_embedding(p)
                    p.unlink()

            # Delete the request
            path = self._entity_path(self._requests_dir(), request_id)
            if path.exists():
                path.unlink()

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            # Find all request IDs in this session
            request_ids: list[str] = []
            for p in self._scan_entities(self._requests_dir()):
                req = self._read_entity(p, Request)
                if req.session_id == session_id:
                    request_ids.append(req.request_id)

            if not request_ids:
                return 0

            request_id_set = set(request_ids)

            # Delete interactions for these requests
            for p in self._scan_entities(self._interactions_dir(), recursive=True):
                interaction = self._read_entity(p, Interaction)
                if interaction.request_id in request_id_set:
                    self._delete_embedding(p)
                    p.unlink()

            # Delete the requests
            for req_id in request_ids:
                path = self._entity_path(self._requests_dir(), req_id)
                if path.exists():
                    path.unlink()

            return len(request_ids)

    def delete_all_requests(self) -> None:
        """Delete all requests and their associated interactions."""
        with self._lock:
            self._clear_dir(self._interactions_dir())
            self._clear_dir(self._requests_dir())

    def delete_requests_by_ids(self, request_ids: list[str]) -> int:
        if not request_ids:
            return 0
        request_id_set = set(request_ids)
        with self._lock:
            # Delete interactions
            for p in self._scan_entities(self._interactions_dir(), recursive=True):
                interaction = self._read_entity(p, Interaction)
                if interaction.request_id in request_id_set:
                    self._delete_embedding(p)
                    p.unlink()

            # Delete requests
            deleted_count = 0
            for req_id in request_ids:
                path = self._entity_path(self._requests_dir(), req_id)
                if path.exists():
                    path.unlink()
                    deleted_count += 1

            return deleted_count

    def get_requests_by_session(self, user_id: str, session_id: str) -> list[Request]:
        with self._lock:
            all_requests = self._list_entities(self._requests_dir(), Request)
        return [
            r
            for r in all_requests
            if r.user_id == user_id and r.session_id == session_id
        ]

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
        with self._lock:
            all_requests = self._list_entities(self._requests_dir(), Request)

        # Filter requests
        requests: list[Request] = []
        for req in all_requests:
            if user_id and req.user_id != user_id:
                continue
            if request_id and req.request_id != request_id:
                continue
            if session_id and req.session_id != session_id:
                continue
            if start_time and req.created_at < start_time:
                continue
            if end_time and req.created_at > end_time:
                continue
            requests.append(req)

        # Sort by created_at descending
        requests.sort(key=lambda x: x.created_at, reverse=True)

        # Apply offset and limit
        effective_limit = top_k or 100
        requests = requests[offset : offset + effective_limit]

        # Group requests by session_id
        groups_dict: dict[str, list[Request]] = {}
        for req in requests:
            group_name = req.session_id or ""
            if group_name not in groups_dict:
                groups_dict[group_name] = []
            groups_dict[group_name].append(req)

        # Get interactions
        if user_id:
            user_interactions = self.get_user_interaction(user_id)
        else:
            user_interactions = self._list_entities_recursive(
                self._interactions_dir(), Interaction
            )

        # Group interactions by request_id
        interactions_by_request_id: dict[str, list[Interaction]] = {}
        for interaction in user_interactions:
            if interaction.request_id not in interactions_by_request_id:
                interactions_by_request_id[interaction.request_id] = []
            interactions_by_request_id[interaction.request_id].append(interaction)

        # Build grouped result
        grouped_results: dict[str, list[RequestInteractionDataModel]] = {}
        for group_name, group_requests in groups_dict.items():
            grouped_results[group_name] = []
            for req in group_requests:
                associated_interactions = interactions_by_request_id.get(
                    req.request_id, []
                )
                associated_interactions = sorted(
                    associated_interactions, key=lambda x: x.created_at
                )
                grouped_results[group_name].append(
                    RequestInteractionDataModel(
                        session_id=group_name,
                        request=req,
                        interactions=associated_interactions,
                    )
                )

        return grouped_results

    def get_rerun_user_ids(
        self,
        user_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        source: str | None = None,
        agent_version: str | None = None,
    ) -> list[str]:
        with self._lock:
            all_requests = self._list_entities(self._requests_dir(), Request)

        user_ids: set[str] = set()
        for req in all_requests:
            if user_id and req.user_id != user_id:
                continue
            if start_time and req.created_at < start_time:
                continue
            if end_time and req.created_at > end_time:
                continue
            if source and req.source != source:
                continue
            if agent_version and req.agent_version != agent_version:
                continue
            user_ids.add(req.user_id)

        return sorted(user_ids)
