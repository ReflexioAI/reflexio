from __future__ import annotations

from contextlib import suppress
from typing import Any

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperationTarget,
    UserEraseResult,
    UserExportResult,
)
from reflexio.server.services.governance.subject_refs import (
    request_ref,
    stable_id,
    subject_ref,
)

_DELETE_TARGET_NAME_TO_RESULT_KEY = {
    "interaction": "interactions",
    "user_playbook": "user_playbooks",
    "profile": "profiles",
    "request": "requests",
    "agent_success_evaluation_result": "agent_success_evaluation_results",
    "profile_purge": "purged_profiles",
    "user_playbook_purge": "purged_user_playbooks",
}
_REQUIRED_DELETE_TARGET_NAMES = tuple(_DELETE_TARGET_NAME_TO_RESULT_KEY)
_USER_PLAYBOOK_PAGE_SIZE = 1000


class GovernanceService:
    def __init__(self, *, storage: Any, org_id: str, ref_secret: str) -> None:
        self.storage = storage
        self.org_id = org_id
        self.ref_secret = ref_secret

    def export_user(self, *, user_id: str, request_id: str) -> UserExportResult:
        subref = subject_ref(user_id, self.ref_secret)
        reqref = request_ref(request_id, self.ref_secret)
        export_id = stable_id("export", f"{self.org_id}:export:{subref}:{reqref}")
        requests, sessions = self._load_user_requests_and_sessions(user_id)
        bundle: dict[str, Any] = {
            "profiles": [
                profile.model_dump()
                for profile in self.storage.get_user_profile(user_id)
            ],
            "interactions": [
                interaction.model_dump()
                for interaction in self.storage.get_user_interaction(user_id)
            ],
            "requests": [request.model_dump() for request in requests],
            "sessions": sessions,
            "user_playbooks": [
                playbook.model_dump() for playbook in self._iter_user_playbooks(user_id)
            ],
        }
        self.storage.append_audit_event(
            AuditEvent(
                org_id=self.org_id,
                operation="EXPORT",
                entity_type="request",
                subject_ref=subref,
                request_ref=reqref,
                idempotency_key=export_id,
                detail={"count": sum(len(items) for items in bundle.values())},
            )
        )
        return UserExportResult(subject_ref=subref, export_id=export_id, bundle=bundle)

    def erase_user(self, *, user_id: str, request_id: str) -> UserEraseResult:
        subref = subject_ref(user_id, self.ref_secret)
        reqref = request_ref(request_id, self.ref_secret)
        idempotency_key = stable_id(
            "idem",
            f"{self.org_id}:user_erasure:{subref}:{reqref}",
        )
        purge_id = stable_id("purge", idempotency_key)
        purge = self.storage.begin_purge_operation(
            purge_id=purge_id,
            idempotency_key=idempotency_key,
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=subref,
            request_ref=reqref,
        )
        if purge.status == "complete":
            return UserEraseResult(
                subject_ref=subref, purge_id=purge_id, status="complete"
            )

        try:
            if not self.storage.purge_targets_prepared(purge_id):
                self.storage.prepare_governance_erase_targets(
                    purge_id,
                    user_id,
                )

            self.storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

            if not self._delete_targets_complete(purge_id):
                self.storage.apply_governance_user_data_delete(purge_id, user_id)
            deleted_counts = self._deleted_counts_from_targets(purge_id)

            rebuilt_agent_playbook_ids = self._rebuild_agent_playbooks(purge_id)
            completed = self.storage.complete_purge_operation_with_audit(
                purge_id,
                AuditEvent(
                    org_id=self.org_id,
                    operation="ERASE",
                    entity_type="request",
                    subject_ref=subref,
                    request_ref=reqref,
                    idempotency_key=purge_id,
                    detail={
                        "deleted_counts": deleted_counts,
                        "rebuilt_agent_playbook_ids": rebuilt_agent_playbook_ids,
                    },
                ),
            )
        except Exception as exc:
            with suppress(Exception):
                self.storage.fail_purge_operation(
                    purge_id,
                    error_code="governance_erase_failed",
                    error_detail=type(exc).__name__,
                )
            raise
        return UserEraseResult(
            subject_ref=subref,
            purge_id=purge_id,
            status=completed.status,
            deleted_counts=deleted_counts,
            rebuilt_agent_playbook_ids=rebuilt_agent_playbook_ids,
        )

    def _load_user_requests_and_sessions(
        self, user_id: str
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        requests: list[Any] = []
        sessions_by_id: dict[str, list[str]] = {}
        offset = 0
        page_size = 1000

        while True:
            grouped_sessions = self.storage.get_sessions(
                user_id=user_id,
                top_k=page_size,
                offset=offset,
            )
            returned_rows = 0
            for session_id, rows in grouped_sessions.items():
                returned_rows += len(rows)
                request_ids = sessions_by_id.setdefault(session_id, [])
                for row in rows:
                    if row.request is None:
                        continue
                    requests.append(row.request)
                    request_ids.append(row.request.request_id)
            if returned_rows < page_size:
                break
            offset += page_size

        sessions = [
            {"session_id": session_id, "request_ids": request_ids}
            for session_id, request_ids in sessions_by_id.items()
        ]
        return requests, sessions

    def _iter_user_playbooks(self, user_id: str) -> list[Any]:
        playbooks: list[Any] = []
        offset = 0
        while True:
            page = self.storage.get_user_playbooks(
                user_id=user_id,
                limit=_USER_PLAYBOOK_PAGE_SIZE,
                offset=offset,
            )
            playbooks.extend(page)
            if len(page) < _USER_PLAYBOOK_PAGE_SIZE:
                break
            offset += _USER_PLAYBOOK_PAGE_SIZE
        return playbooks

    def _delete_targets_complete(self, purge_id: str) -> bool:
        delete_targets = {
            target.target_name: target
            for target in self.storage.list_purge_targets(purge_id, phase="delete")
        }
        return all(
            delete_targets.get(target_name) is not None
            and delete_targets[target_name].status == "complete"
            for target_name in _REQUIRED_DELETE_TARGET_NAMES
        )

    def _deleted_counts_from_targets(self, purge_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for target in self.storage.list_purge_targets(purge_id, phase="delete"):
            result_key = _DELETE_TARGET_NAME_TO_RESULT_KEY.get(target.target_name)
            if result_key is None:
                continue
            counts[result_key] = int(target.deleted_count)
        return counts

    def _rebuild_agent_playbooks(self, purge_id: str) -> list[int]:
        rebuilt_ids: list[int] = []
        for target in self.storage.list_purge_targets(
            purge_id,
            phase="rebuild_without_erased_sources",
        ):
            if target.target_name != "agent_playbook" or not target.target_ref:
                continue
            agent_playbook_id = int(target.target_ref)
            if target.status == "complete":
                rebuilt_ids.append(agent_playbook_id)
                continue
            remaining_source_windows = self._remaining_source_windows(target)
            rebuild_fields = self._build_rebuilt_agent_playbook_fields(
                remaining_source_windows
            )
            self.storage.apply_governance_agent_playbook_rebuild(
                purge_id=purge_id,
                agent_playbook_id=agent_playbook_id,
                remaining_source_windows=remaining_source_windows,
                content=rebuild_fields["content"],
                trigger=rebuild_fields["trigger"],
                rationale=rebuild_fields["rationale"],
                blocking_issue=rebuild_fields["blocking_issue"],
                expanded_terms=rebuild_fields["expanded_terms"],
                tags=rebuild_fields["tags"],
            )
            rebuilt_ids.append(agent_playbook_id)
        return rebuilt_ids

    def _remaining_source_windows(
        self,
        target: PurgeOperationTarget,
    ) -> list[dict[str, object]]:
        detail = target.detail or {}
        remaining = detail.get("remaining_source_windows", [])
        if not isinstance(remaining, list):
            raise ValueError("remaining_source_windows must be a list")
        return remaining

    def _build_rebuilt_agent_playbook_fields(
        self,
        remaining_source_windows: list[dict[str, object]],
    ) -> dict[str, Any]:
        user_playbook_ids: list[int] = []
        for window in remaining_source_windows:
            raw_user_playbook_id = window.get("user_playbook_id")
            if isinstance(raw_user_playbook_id, int):
                user_playbook_ids.append(raw_user_playbook_id)
        playbooks_by_id = {
            playbook.user_playbook_id: playbook
            for playbook in self.storage.get_user_playbooks_by_ids_any_user(
                user_playbook_ids
            )
            if playbook.user_playbook_id
        }
        remaining_playbooks = [
            playbooks_by_id[user_playbook_id]
            for user_playbook_id in user_playbook_ids
            if user_playbook_id in playbooks_by_id
        ]
        return {
            "content": self._join_non_empty_strings(
                playbook.content for playbook in remaining_playbooks
            ),
            "trigger": self._join_non_empty_strings(
                playbook.trigger for playbook in remaining_playbooks
            ),
            "rationale": self._join_non_empty_strings(
                playbook.rationale for playbook in remaining_playbooks
            ),
            "blocking_issue": next(
                (
                    playbook.blocking_issue.model_dump()
                    for playbook in remaining_playbooks
                    if playbook.blocking_issue is not None
                ),
                None,
            ),
            "expanded_terms": self._join_non_empty_strings(
                playbook.expanded_terms for playbook in remaining_playbooks
            ),
            "tags": self._merge_tags(remaining_playbooks),
        }

    def _join_non_empty_strings(self, values: Any) -> str | None:
        joined = "\n".join(value for value in values if value)
        return joined or None

    def _merge_tags(self, playbooks: list[Any]) -> list[str] | None:
        merged_tags: list[str] = []
        for playbook in playbooks:
            for tag in playbook.tags or []:
                if tag not in merged_tags:
                    merged_tags.append(tag)
        return merged_tags or None
