import logging
from pathlib import Path

from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    PlaybookStatus,
    Request,
    Status,
    StructuredData,
    UserPlaybook,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.services.storage.storage_base import matches_status_filter

logger = logging.getLogger(__name__)


class PlaybookMixin:
    # ------------------------------------------------------------------
    # User playbook methods
    # ------------------------------------------------------------------

    def save_user_playbooks(self, user_playbooks: list[UserPlaybook]) -> None:
        with self._lock:
            next_id = self._next_id(self._user_playbooks_dir())
            for i, up in enumerate(user_playbooks):
                if up.user_playbook_id == 0:
                    up.user_playbook_id = next_id + i
                path = self._entity_path(
                    self._user_playbooks_dir(),
                    str(up.user_playbook_id),
                )
                self._write_entity(path, up)
                self._write_embedding(path, up.embedding)
        self._trigger_qmd_update()

    def get_user_playbooks(
        self,
        limit: int = 100,
        user_id: str | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        include_embedding: bool = False,  # noqa: ARG002
    ) -> list[UserPlaybook]:
        with self._lock:
            all_playbooks = self._list_entities(
                self._user_playbooks_dir(), UserPlaybook
            )

        playbooks: list[UserPlaybook] = []
        for up in all_playbooks:
            if user_id is not None and up.user_id != user_id:
                continue
            if playbook_name and up.playbook_name != playbook_name:
                continue
            if agent_version is not None and up.agent_version != agent_version:
                continue
            if status_filter is not None and up.status not in status_filter:
                continue
            if start_time is not None and up.created_at < start_time:
                continue
            if end_time is not None and up.created_at > end_time:
                continue
            playbooks.append(up)
            if len(playbooks) >= limit:
                break
        return playbooks

    def count_user_playbooks(
        self,
        user_id: str | None = None,
        playbook_name: str | None = None,
        min_user_playbook_id: int | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
    ) -> int:
        with self._lock:
            all_playbooks = self._list_entities(
                self._user_playbooks_dir(), UserPlaybook
            )

        count = 0
        for up in all_playbooks:
            if user_id is not None and up.user_id != user_id:
                continue
            if playbook_name and up.playbook_name != playbook_name:
                continue
            if (
                min_user_playbook_id is not None
                and up.user_playbook_id <= min_user_playbook_id
            ):
                continue
            if agent_version is not None and up.agent_version != agent_version:
                continue
            if status_filter is not None and up.status not in status_filter:
                continue
            count += 1
        return count

    def count_user_playbooks_by_session(self, session_id: str) -> int:
        with self._lock:
            # Get all request_ids for this session
            all_requests = self._list_entities(self._requests_dir(), Request)
            request_ids = {
                r.request_id for r in all_requests if r.session_id == session_id
            }

            if not request_ids:
                return 0

            all_playbooks = self._list_entities(
                self._user_playbooks_dir(), UserPlaybook
            )
        return sum(1 for up in all_playbooks if up.request_id in request_ids)

    def delete_all_user_playbooks(self) -> None:
        with self._lock:
            self._clear_dir(self._user_playbooks_dir())

    def delete_all_user_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        with self._lock:
            for p in self._scan_entities(self._user_playbooks_dir()):
                up = self._read_entity(p, UserPlaybook)
                if self._should_delete_playbook(up, playbook_name, agent_version):
                    self._delete_embedding(p)
                    p.unlink()

    def delete_user_playbook(self, user_playbook_id: int) -> None:
        with self._lock:
            path = self._entity_path(
                self._user_playbooks_dir(),
                str(user_playbook_id),
            )
            if path.exists():
                self._delete_embedding(path)
                path.unlink()

    def update_all_user_playbooks_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        with self._lock:
            updated_count = 0
            up_dir = self._user_playbooks_dir()
            if not up_dir.exists():
                return 0

            for p in self._scan_entities(up_dir):
                up = self._read_entity(p, UserPlaybook)

                # Apply optional filters
                if agent_version is not None and up.agent_version != agent_version:
                    continue
                if playbook_name is not None and up.playbook_name != playbook_name:
                    continue

                # Check if playbook matches old_status
                status_matches = False
                if old_status is None or (
                    hasattr(old_status, "value") and old_status.value is None
                ):
                    if up.status is None:
                        status_matches = True
                elif isinstance(old_status, Status) and up.status == old_status:
                    status_matches = True

                if status_matches:
                    up.status = new_status
                    self._write_entity(p, up)
                    self._write_embedding(p, up.embedding)
                    updated_count += 1

            logger.info(
                "Updated %s user playbooks from %s to %s",
                updated_count,
                old_status,
                new_status,
            )
            return updated_count

    def delete_all_user_playbooks_by_status(
        self,
        status: Status,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        with self._lock:
            deleted_count = 0
            up_dir = self._user_playbooks_dir()
            if not up_dir.exists():
                return 0

            for p in self._scan_entities(up_dir):
                up = self._read_entity(p, UserPlaybook)

                if isinstance(status, Status) and up.status == status:
                    # Apply optional filters
                    if (
                        agent_version is not None and up.agent_version != agent_version
                    ) or (
                        playbook_name is not None and up.playbook_name != playbook_name
                    ):
                        continue
                    self._delete_embedding(p)
                    p.unlink()
                    deleted_count += 1

            logger.info(
                "Deleted %s user playbooks with status %s", deleted_count, status
            )
            return deleted_count

    def delete_user_playbooks_by_ids(self, user_playbook_ids: list[int]) -> int:
        """Delete user playbooks by their IDs.

        Args:
            user_playbook_ids (list[int]): List of user playbook IDs to delete.

        Returns:
            int: Number of user playbooks actually deleted.
        """
        up_dir = self._user_playbooks_dir()
        if not up_dir.exists():
            return 0

        deleted_count = 0
        with self._lock:
            for upid in user_playbook_ids:
                path = self._entity_path(up_dir, str(upid))
                if path.exists():
                    self._delete_embedding(path)
                    path.unlink()
                    deleted_count += 1

        logger.info(
            "Deleted %d of %d user playbooks by IDs",
            deleted_count,
            len(user_playbook_ids),
        )
        return deleted_count

    def has_user_playbooks_with_status(
        self,
        status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> bool:
        for p in self._scan_entities(self._user_playbooks_dir()):
            up = self._read_entity(p, UserPlaybook)

            if agent_version is not None and up.agent_version != agent_version:
                continue
            if playbook_name is not None and up.playbook_name != playbook_name:
                continue

            status_matches = False
            if status is None or (hasattr(status, "value") and status.value is None):
                if up.status is None:
                    status_matches = True
            elif isinstance(status, Status) and up.status == status:
                status_matches = True

            if status_matches:
                return True

        return False

    @staticmethod
    def _user_playbook_matches_filters(
        up: UserPlaybook,
        *,
        user_id: str | None,
        agent_version: str | None,
        playbook_name: str | None,
        start_time: int | None,
        end_time: int | None,
        status_filter: list[Status | None] | None,
        request_user_map: dict[str, str],
    ) -> bool:
        """Check if a UserPlaybook passes all search filters."""
        if user_id and request_user_map.get(up.request_id) != user_id:
            return False
        if agent_version and up.agent_version != agent_version:
            return False
        if playbook_name and up.playbook_name != playbook_name:
            return False
        if start_time and up.created_at < start_time:
            return False
        if end_time and up.created_at > end_time:
            return False
        return status_filter is None or matches_status_filter(up.status, status_filter)

    def search_user_playbooks(
        self,
        request: SearchUserPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[UserPlaybook]:
        query = request.query
        user_id = request.user_id
        agent_version = request.agent_version
        playbook_name = request.playbook_name
        start_time = int(request.start_time.timestamp()) if request.start_time else None
        end_time = int(request.end_time.timestamp()) if request.end_time else None
        status_filter = request.status_filter
        match_count = request.top_k or 10

        # Build request_id -> user_id map if user_id filter is provided
        request_user_map: dict[str, str] = {}
        if user_id:
            all_requests = self._list_entities(self._requests_dir(), Request)
            request_user_map = {r.request_id: r.user_id for r in all_requests}

        filter_kwargs = {
            "user_id": user_id,
            "agent_version": agent_version,
            "playbook_name": playbook_name,
            "start_time": start_time,
            "end_time": end_time,
            "status_filter": status_filter,
            "request_user_map": request_user_map,
        }

        # QMD-accelerated search when SearchOptions are provided with a query
        if options and query:
            qmd_results = self._qmd.search(query, options.search_mode, match_count)

            results: list[UserPlaybook] = []
            for qmd_r in qmd_results:
                p = Path(qmd_r.filepath).resolve()
                if not p.exists():
                    continue
                if not p.is_relative_to(self._user_playbooks_dir()):
                    continue
                up = self._read_entity(p, UserPlaybook)
                if self._user_playbook_matches_filters(up, **filter_kwargs):
                    results.append(up)
                    if len(results) >= match_count:
                        break
            return results

        # Fallback: Python substring matching
        all_playbooks = self._list_entities(self._user_playbooks_dir(), UserPlaybook)

        results_fallback: list[UserPlaybook] = []
        for up in all_playbooks:
            if query and query.lower() not in up.content.lower():
                continue
            if self._user_playbook_matches_filters(up, **filter_kwargs):
                results_fallback.append(up)
                if len(results_fallback) >= match_count:
                    break

        return results_fallback

    @staticmethod
    def _agent_playbook_matches_filters(
        ap: AgentPlaybook,
        *,
        agent_version: str | None,
        playbook_name: str | None,
        start_time: int | None,
        end_time: int | None,
        status_filter: list[Status | None] | None,
        playbook_status_filter: PlaybookStatus | None,
    ) -> bool:
        """Check if an AgentPlaybook passes all search filters."""
        if agent_version and ap.agent_version != agent_version:
            return False
        if playbook_name and ap.playbook_name != playbook_name:
            return False
        if start_time and ap.created_at < start_time:
            return False
        if end_time and ap.created_at > end_time:
            return False
        if (
            playbook_status_filter is not None
            and ap.playbook_status != playbook_status_filter
        ):
            return False
        return status_filter is None or matches_status_filter(ap.status, status_filter)

    def search_agent_playbooks(
        self,
        request: SearchAgentPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[AgentPlaybook]:
        query = request.query
        agent_version = request.agent_version
        playbook_name = request.playbook_name
        start_time = int(request.start_time.timestamp()) if request.start_time else None
        end_time = int(request.end_time.timestamp()) if request.end_time else None
        status_filter = request.status_filter
        playbook_status_filter = request.playbook_status_filter
        match_count = request.top_k or 10

        filter_kwargs = {
            "agent_version": agent_version,
            "playbook_name": playbook_name,
            "start_time": start_time,
            "end_time": end_time,
            "status_filter": status_filter,
            "playbook_status_filter": playbook_status_filter,
        }

        # QMD-accelerated search when SearchOptions are provided with a query
        if options and query:
            qmd_results = self._qmd.search(query, options.search_mode, match_count)

            results: list[AgentPlaybook] = []
            for qmd_r in qmd_results:
                p = Path(qmd_r.filepath).resolve()
                if not p.exists():
                    continue
                if not p.is_relative_to(self._agent_playbooks_dir()):
                    continue
                ap = self._read_entity(p, AgentPlaybook)
                if self._agent_playbook_matches_filters(ap, **filter_kwargs):
                    results.append(ap)
                    if len(results) >= match_count:
                        break
            return results

        # Fallback: Python substring matching
        all_playbooks = self._list_entities(self._agent_playbooks_dir(), AgentPlaybook)

        results_fallback: list[AgentPlaybook] = []
        for ap in all_playbooks:
            if query and query.lower() not in ap.content.lower():
                continue
            if self._agent_playbook_matches_filters(ap, **filter_kwargs):
                results_fallback.append(ap)
                if len(results_fallback) >= match_count:
                    break

        return results_fallback

    # ------------------------------------------------------------------
    # Agent Playbook methods
    # ------------------------------------------------------------------

    def save_agent_playbooks(
        self, agent_playbooks: list[AgentPlaybook]
    ) -> list[AgentPlaybook]:
        with self._lock:
            next_id = self._next_id(self._agent_playbooks_dir())
            for i, ap in enumerate(agent_playbooks):
                if not ap.agent_playbook_id:
                    ap.agent_playbook_id = next_id + i
                path = self._entity_path(
                    self._agent_playbooks_dir(),
                    str(ap.agent_playbook_id),
                )
                self._write_entity(path, ap)
                self._write_embedding(path, ap.embedding)
        self._trigger_qmd_update()
        return agent_playbooks

    def get_agent_playbooks(
        self,
        limit: int = 100,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
    ) -> list[AgentPlaybook]:
        with self._lock:
            all_playbooks = self._list_entities(
                self._agent_playbooks_dir(), AgentPlaybook
            )

        playbooks: list[AgentPlaybook] = []
        for ap in all_playbooks:
            if agent_version is not None and ap.agent_version != agent_version:
                continue
            # Apply status filter
            if status_filter is not None:
                if ap.status not in status_filter:
                    continue
            else:
                if ap.status == Status.ARCHIVED:
                    continue

            if (
                playbook_status_filter
                and ap.playbook_status not in playbook_status_filter
            ):
                continue

            if playbook_name and ap.playbook_name != playbook_name:
                continue

            playbooks.append(ap)
            if len(playbooks) >= limit:
                break
        return playbooks

    def update_agent_playbook_status(
        self, agent_playbook_id: int, playbook_status: PlaybookStatus
    ) -> None:
        with self._lock:
            path = self._entity_path(
                self._agent_playbooks_dir(),
                str(agent_playbook_id),
            )
            if not path.exists():
                raise ValueError(
                    f"Agent playbook with ID {agent_playbook_id} not found"
                )

            ap = self._read_entity(path, AgentPlaybook)
            ap.playbook_status = playbook_status
            self._write_entity(path, ap)
            self._write_embedding(path, ap.embedding)

    def update_agent_playbook(
        self,
        agent_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        structured_data: StructuredData | None = None,
        playbook_status: PlaybookStatus | None = None,
    ) -> None:
        with self._lock:
            path = self._entity_path(
                self._agent_playbooks_dir(),
                str(agent_playbook_id),
            )
            if not path.exists():
                raise ValueError(
                    f"Agent playbook with ID {agent_playbook_id} not found"
                )

            ap = self._read_entity(path, AgentPlaybook)
            if playbook_name is not None:
                ap.playbook_name = playbook_name
            if content is not None:
                ap.content = content
            if structured_data is not None:
                ap.structured_data = structured_data
            if playbook_status is not None:
                ap.playbook_status = playbook_status
            self._write_entity(path, ap)
            self._write_embedding(path, ap.embedding)

    def update_user_playbook(
        self,
        user_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        structured_data: StructuredData | None = None,
    ) -> None:
        with self._lock:
            path = self._entity_path(
                self._user_playbooks_dir(),
                str(user_playbook_id),
            )
            if not path.exists():
                raise ValueError(f"User playbook with ID {user_playbook_id} not found")

            up = self._read_entity(path, UserPlaybook)
            if playbook_name is not None:
                up.playbook_name = playbook_name
            if content is not None:
                up.content = content
            if structured_data is not None:
                up.structured_data = structured_data
            self._write_entity(path, up)
            self._write_embedding(path, up.embedding)

    def archive_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        with self._lock:
            ap_dir = self._agent_playbooks_dir()

            for p in self._scan_entities(ap_dir):
                ap = self._read_entity(p, AgentPlaybook)
                if (
                    self._should_delete_playbook(ap, playbook_name, agent_version)
                    and ap.playbook_status != PlaybookStatus.APPROVED
                ):
                    ap.status = Status.ARCHIVED
                    self._write_entity(p, ap)
                    self._write_embedding(p, ap.embedding)

    def restore_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        with self._lock:
            ap_dir = self._agent_playbooks_dir()

            for p in self._scan_entities(ap_dir):
                ap = self._read_entity(p, AgentPlaybook)
                if (
                    self._should_delete_playbook(ap, playbook_name, agent_version)
                    and ap.status == Status.ARCHIVED
                ):
                    ap.status = None
                    self._write_entity(p, ap)
                    self._write_embedding(p, ap.embedding)

    def delete_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        with self._lock:
            ap_dir = self._agent_playbooks_dir()

            for p in self._scan_entities(ap_dir):
                ap = self._read_entity(p, AgentPlaybook)
                if (
                    self._should_delete_playbook(ap, playbook_name, agent_version)
                    and ap.status == Status.ARCHIVED
                ):
                    self._delete_embedding(p)
                    p.unlink()

    def archive_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        if not agent_playbook_ids:
            return
        id_set = set(agent_playbook_ids)
        with self._lock:
            for p in self._scan_entities(self._agent_playbooks_dir()):
                ap = self._read_entity(p, AgentPlaybook)
                if (
                    ap.agent_playbook_id in id_set
                    and ap.playbook_status != PlaybookStatus.APPROVED
                ):
                    ap.status = Status.ARCHIVED
                    self._write_entity(p, ap)
                    self._write_embedding(p, ap.embedding)

    def restore_archived_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> None:
        if not agent_playbook_ids:
            return
        id_set = set(agent_playbook_ids)
        with self._lock:
            for p in self._scan_entities(self._agent_playbooks_dir()):
                ap = self._read_entity(p, AgentPlaybook)
                if ap.agent_playbook_id in id_set and ap.status == Status.ARCHIVED:
                    ap.status = None
                    self._write_entity(p, ap)
                    self._write_embedding(p, ap.embedding)

    def delete_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        if not agent_playbook_ids:
            return
        id_set = set(agent_playbook_ids)
        with self._lock:
            for apid in id_set:
                path = self._entity_path(self._agent_playbooks_dir(), str(apid))
                if path.exists():
                    self._delete_embedding(path)
                    path.unlink()

    def delete_all_agent_playbooks(self) -> None:
        with self._lock:
            self._clear_dir(self._agent_playbooks_dir())

    def delete_agent_playbook(self, agent_playbook_id: int) -> None:
        with self._lock:
            path = self._entity_path(
                self._agent_playbooks_dir(),
                str(agent_playbook_id),
            )
            if path.exists():
                self._delete_embedding(path)
                path.unlink()

    def delete_all_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        with self._lock:
            for p in self._scan_entities(self._agent_playbooks_dir()):
                ap = self._read_entity(p, AgentPlaybook)
                if self._should_delete_playbook(ap, playbook_name, agent_version):
                    self._delete_embedding(p)
                    p.unlink()

    # ------------------------------------------------------------------
    # Agent Success Evaluation methods
    # ------------------------------------------------------------------

    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        with self._lock:
            next_id = self._next_id(self._evaluations_dir())
            for i, result in enumerate(results):
                path = self._entity_path(
                    self._evaluations_dir(),
                    str(next_id + i),
                )
                self._write_entity(path, result)
                self._write_embedding(path, result.embedding)
        self._trigger_qmd_update()

    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        all_results = self._list_entities(
            self._evaluations_dir(), AgentSuccessEvaluationResult
        )

        results: list[AgentSuccessEvaluationResult] = []
        for result in all_results:
            if agent_version is not None and result.agent_version != agent_version:
                continue
            results.append(result)
            if len(results) >= limit:
                break
        return results

    def delete_all_agent_success_evaluation_results(self) -> None:
        with self._lock:
            self._clear_dir(self._evaluations_dir())
