import json
import logging
import time
from pathlib import Path

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.server.services.storage.error import StorageError

logger = logging.getLogger(__name__)


class OperationMixin:
    # ------------------------------------------------------------------
    # Operation State methods
    # ------------------------------------------------------------------

    def create_operation_state(self, service_name: str, operation_state: dict) -> None:
        with self._lock:
            safe_name = self._safe_filename(service_name)
            path = self._operation_states_dir() / f"{safe_name}.json"
            if path.exists():
                raise StorageError(
                    f"Operation state already exists for service '{service_name}'"
                )

            state_data = {
                "service_name": service_name,
                "operation_state": operation_state,
                "updated_at": self._current_timestamp(),
            }
            self._write_dict(path, state_data)

    def upsert_operation_state(self, service_name: str, operation_state: dict) -> None:
        with self._lock:
            safe_name = self._safe_filename(service_name)
            path = self._operation_states_dir() / f"{safe_name}.json"

            if path.exists():
                state_data = json.loads(path.read_text())
                state_data["operation_state"] = operation_state
                state_data["updated_at"] = self._current_timestamp()
            else:
                state_data = {
                    "service_name": service_name,
                    "operation_state": operation_state,
                    "updated_at": self._current_timestamp(),
                }
            self._write_dict(path, state_data)

    def get_operation_state(self, service_name: str) -> dict | None:
        safe_name = self._safe_filename(service_name)
        path = self._operation_states_dir() / f"{safe_name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _get_operation_state_entry(self, service_name: str) -> tuple[Path, dict | None]:
        """Get the path and data for an operation state entry."""
        safe_name = self._safe_filename(service_name)
        path = self._operation_states_dir() / f"{safe_name}.json"
        if not path.exists():
            return path, None
        return path, json.loads(path.read_text())

    def get_operation_state_with_new_request_interaction(
        self,
        service_name: str,
        user_id: str | None,
        sources: list[str] | None = None,
    ) -> tuple[dict, list[RequestInteractionDataModel]]:
        with self._lock:
            _, state_entry = self._get_operation_state_entry(service_name)
        operation_state: dict = state_entry if isinstance(state_entry, dict) else {}

        last_processed_ids = operation_state.get("last_processed_interaction_ids") or []
        if not isinstance(last_processed_ids, list):
            last_processed_ids = []
        processed_set = {str(item) for item in last_processed_ids}

        last_processed_timestamp = operation_state.get("last_processed_timestamp")
        if not isinstance(last_processed_timestamp, int):
            last_processed_timestamp = None

        # Get interactions
        if user_id is not None:
            all_interactions_list = self._list_entities(
                self._user_dir(self._interactions_dir(), user_id), Interaction
            )
        else:
            all_interactions_list = self._list_entities_recursive(
                self._interactions_dir(), Interaction
            )

        # Collect new interactions
        new_interactions: list[Interaction] = []
        for interaction in all_interactions_list:
            created_at = interaction.created_at
            if last_processed_timestamp is not None and created_at is not None:
                if created_at > last_processed_timestamp:
                    new_interactions.append(interaction)
                    continue
                if (
                    created_at == last_processed_timestamp
                    and str(interaction.interaction_id) not in processed_set
                ):
                    new_interactions.append(interaction)
                    continue
            elif str(interaction.interaction_id) not in processed_set:
                new_interactions.append(interaction)

        new_interactions.sort(key=lambda item: item.created_at or 0)

        # Group interactions by request_id
        interactions_by_request: dict[str, list[Interaction]] = {}
        for interaction in new_interactions:
            request_id = interaction.request_id
            if request_id not in interactions_by_request:
                interactions_by_request[request_id] = []
            interactions_by_request[request_id].append(interaction)

        # Build RequestInteractionDataModel objects
        sessions: list[RequestInteractionDataModel] = []
        for request_id, interactions in interactions_by_request.items():
            request = self.get_request(request_id)
            if request is None:
                request = Request(
                    request_id=request_id,
                    user_id=(
                        interactions[0].user_id if interactions else (user_id or "")
                    ),
                    created_at=interactions[0].created_at if interactions else 0,
                )

            if sources is not None and request.source not in sources:
                continue

            group_name = request.session_id or request_id

            sessions.append(
                RequestInteractionDataModel(
                    session_id=group_name,
                    request=request,
                    interactions=interactions,
                )
            )

        sessions.sort(
            key=lambda g: (
                min(i.created_at or 0 for i in g.interactions) if g.interactions else 0
            )
        )

        return operation_state, sessions

    def get_last_k_interactions_grouped(
        self,
        user_id: str | None,
        k: int,
        sources: list[str] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        agent_version: str | None = None,
    ) -> tuple[list[RequestInteractionDataModel], list[Interaction]]:
        with self._lock:
            if user_id is not None:
                all_interactions = self._list_entities(
                    self._user_dir(self._interactions_dir(), user_id), Interaction
                )
            else:
                all_interactions = self._list_entities_recursive(
                    self._interactions_dir(), Interaction
                )

        all_interactions.sort(key=lambda x: x.interaction_id or 0, reverse=True)

        # Batch-load all requests upfront to avoid N+1 file reads
        all_request_ids = {i.request_id for i in all_interactions}
        requests_cache: dict[str, Request | None] = {
            rid: self.get_request(rid) for rid in all_request_ids
        }

        # Filter and take first K
        flat_interactions: list[Interaction] = []
        for interaction in all_interactions:
            if len(flat_interactions) >= k:
                break
            if start_time is not None and (
                interaction.created_at is None or interaction.created_at < start_time
            ):
                continue
            if end_time is not None and (
                interaction.created_at is None or interaction.created_at > end_time
            ):
                continue
            if sources is not None or agent_version is not None:
                request = requests_cache.get(interaction.request_id)
                if sources is not None and (
                    request is None or request.source not in sources
                ):
                    continue
                if agent_version is not None and (
                    request is None or request.agent_version != agent_version
                ):
                    continue
            flat_interactions.append(interaction)

        # Group by request_id
        interactions_by_request: dict[str, list[Interaction]] = {}
        for interaction in flat_interactions:
            request_id = interaction.request_id
            if request_id not in interactions_by_request:
                interactions_by_request[request_id] = []
            interactions_by_request[request_id].append(interaction)

        # Build RequestInteractionDataModel objects
        sessions: list[RequestInteractionDataModel] = []
        for request_id, interactions in interactions_by_request.items():
            request = requests_cache.get(request_id)
            if request is None:
                request = Request(
                    request_id=request_id,
                    user_id=(
                        interactions[0].user_id if interactions else (user_id or "")
                    ),
                    created_at=interactions[0].created_at if interactions else 0,
                )

            group_name = request.session_id or request_id

            interactions_sorted = sorted(
                interactions, key=lambda x: x.interaction_id or 0
            )

            sessions.append(
                RequestInteractionDataModel(
                    session_id=group_name,
                    request=request,
                    interactions=interactions_sorted,
                )
            )

        sessions.sort(
            key=lambda g: (
                min(i.interaction_id or 0 for i in g.interactions)
                if g.interactions
                else 0
            )
        )

        return sessions, flat_interactions

    def update_operation_state(self, service_name: str, operation_state: dict) -> None:
        with self._lock:
            safe_name = self._safe_filename(service_name)
            path = self._operation_states_dir() / f"{safe_name}.json"
            if not path.exists():
                raise StorageError(
                    f"Operation state does not exist for service '{service_name}'"
                )

            state_data = json.loads(path.read_text())
            state_data["operation_state"] = operation_state
            state_data["updated_at"] = self._current_timestamp()
            self._write_dict(path, state_data)

    def get_all_operation_states(self) -> list[dict]:
        op_dir = self._operation_states_dir()
        if not op_dir.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(op_dir.glob("*.json"))]

    def delete_operation_state(self, service_name: str) -> None:
        with self._lock:
            safe_name = self._safe_filename(service_name)
            path = self._operation_states_dir() / f"{safe_name}.json"
            if path.exists():
                path.unlink()

    def delete_all_operation_states(self) -> None:
        with self._lock:
            self._clear_dir(self._operation_states_dir())

    def try_acquire_in_progress_lock(
        self, state_key: str, request_id: str, stale_lock_seconds: int = 300
    ) -> dict:
        current_time = int(time.time())

        with self._lock:
            safe_name = self._safe_filename(state_key)
            path = self._operation_states_dir() / f"{safe_name}.json"

            if path.exists():
                state_entry = json.loads(path.read_text())
                current_state = state_entry.get("operation_state", {})
            else:
                state_entry = None
                current_state = {}

            in_progress = current_state.get("in_progress", False)
            started_at = current_state.get("started_at", 0)

            # Case 1 & 2: No lock or stale lock - acquire it
            if not in_progress or (current_time - started_at >= stale_lock_seconds):
                new_state = {
                    "in_progress": True,
                    "started_at": current_time,
                    "current_request_id": request_id,
                    "pending_request_id": None,
                }
                state_data = {
                    "service_name": state_key,
                    "operation_state": new_state,
                    "updated_at": self._current_timestamp(),
                }
                self._write_dict(path, state_data)
                return {"acquired": True, "state": new_state}

            # Case 3: Active lock - update pending
            current_state["pending_request_id"] = request_id
            state_data = {
                "service_name": state_key,
                "operation_state": current_state,
                "updated_at": self._current_timestamp(),
            }
            self._write_dict(path, state_data)
            return {"acquired": False, "state": current_state}
