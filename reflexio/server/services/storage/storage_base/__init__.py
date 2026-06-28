from reflexio.models.api_schema.domain.enums import Status

from ._agent_run import (
    NOT_APPLICABLE_ANSWER,
    AgentBinding,
    AgentRunMixin,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    PendingToolCallUpsertResult,
    PriorAnswerMatch,
    RunToolDependencyKind,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    canonical_json,
    embedding_similarity,
    human_feedback_scope,
    is_not_applicable_tool_result,
    normalize_dedup_text,
    not_applicable_tool_result,
)
from ._base import BaseStorageCore, matches_status_filter
from ._extras import ExtrasMixin
from ._governance import GovernanceMixin
from ._lineage import EntityType, LineageEventMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin
from ._retrieval_log import RetrievalLogMixin
from ._shadow_verdicts import ShadowVerdictsMixin
from ._share_links import ShareLinkMixin
from ._stall_state import StallStateMixin


class BaseStorage(
    AgentRunMixin,
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    RetrievalLogMixin,
    GovernanceMixin,
    LineageEventMixin,
    OperationMixin,
    ExtrasMixin,
    ShareLinkMixin,
    StallStateMixin,
    ShadowVerdictsMixin,
    BaseStorageCore,
):
    """Base class for storage."""

    def _is_lineage_tombstone(self, entity_type: EntityType, entity_id: str) -> bool:
        """Return True when the entity row is a tombstone (merged_into or superseded_by set).

        Uses the portable ``include_tombstones=True`` getters so the check
        works for every backend without backend-specific SQL.

        Args:
            entity_type (EntityType): ``"profile"`` or ``"user_playbook"``.
            entity_id (str): The entity's primary key as a string.

        Returns:
            bool: True if the row has ``merged_into`` or ``superseded_by`` set.
        """
        if entity_type == "profile":
            row = self.get_profile_by_id(entity_id, include_tombstones=True)
            if row is None:
                return False
            return bool(row.merged_into or row.superseded_by)
        if entity_type == "user_playbook":
            row = self.get_user_playbook_by_id(int(entity_id), include_tombstones=True)
            if row is None:
                return False
            return bool(row.merged_into or row.superseded_by)
        return False

    def _partition_purge_vs_delete(
        self, entity_type: EntityType, ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split entity ids into purge-eligible and hard-delete sets.

        An entity is purge-eligible when it is a tombstone (``merged_into`` or
        ``superseded_by`` is set) **or** is pointed to by another row
        (``has_inbound_lineage_refs`` returns True). All other ids go to the
        hard-delete set.

        Uses only portable storage reads so this helper works for every
        backend (SQLite, Supabase, Postgres). Callers that need
        backend-specific efficiency may override ``clear_user_data`` instead.

        Args:
            entity_type (EntityType): ``"profile"`` or ``"user_playbook"``.
            ids (list[str]): Entity primary keys as strings.

        Returns:
            tuple[list[str], list[str]]: ``(purge_ids, delete_ids)`` where
                ``purge_ids`` are content-purge candidates and ``delete_ids``
                are safe to hard-delete.
        """
        purge_ids: list[str] = []
        delete_ids: list[str] = []
        for eid in ids:
            if self._is_lineage_tombstone(
                entity_type, eid
            ) or self.has_inbound_lineage_refs(entity_type=entity_type, entity_id=eid):
                purge_ids.append(eid)
            else:
                delete_ids.append(eid)
        return purge_ids, delete_ids

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        """Delete all rows scoped to a single ``user_id``.

        Removes the user's interactions, user playbooks, profiles, and
        requests. Intentionally does NOT touch ``agent_playbooks`` — those
        are the cross-project rollup of skills and have no ``user_id``
        column. This is the data-isolation primitive used by paired
        protocols (e.g. SWE-bench) that share a single backend across
        parallel tasks without one task's clear-all nuking another
        in-flight task's rows.

        **Lineage-aware erasure:** rows that are tombstones (``merged_into``
        or ``superseded_by`` is set) *or* are pointed to by another row
        (``has_inbound_lineage_refs`` returns True) are content-purged
        (skeleton kept, body blanked) rather than hard-deleted. Standalone
        rows with no lineage involvement are hard-deleted.

        The default implementation composes existing per-user / by-ids
        primitives so any backend that implements those (sqlite, supabase,
        postgres, ...) gets correct behaviour for free.
        Subclasses MAY override for atomic / transactional efficiency.

        **Atomicity note (default path only):** this default implementation is
        NOT wrapped in a single transaction. Each ``purge_content`` call commits
        independently, so a crash mid-erasure may leave some purge-eligible rows
        with PII intact until ``clear_user_data`` is re-invoked. The call is
        idempotent — re-invoking it is safe and will finish the erasure.

        Args:
            user_id (str): The user id whose rows should be deleted.

        Returns:
            dict[str, int]: Per-entity counts with keys ``interactions``,
                ``user_playbooks``, ``profiles``, ``requests``,
                ``purged_profiles``, and ``purged_user_playbooks``.
                ``profiles`` and ``user_playbooks`` reflect hard-deleted
                counts; purged rows are counted separately.
        """
        interaction_count = len(self.get_user_interaction(user_id))

        # All statuses a user's row can have — including tombstones (SUPERSEDED,
        # MERGED). Erasure MUST reach every row the user owns regardless of
        # status; the old filter excluded tombstones, leaving them in the DB
        # after clear_user_data (GDPR regression).
        _all_statuses: list[Status | None] = [
            None,  # CURRENT
            Status.ARCHIVED,
            Status.PENDING,
            Status.ARCHIVE_IN_PROGRESS,
            Status.SUPERSEDED,
            Status.MERGED,
        ]

        # Snapshot user_playbook ids for the user and partition into
        # purge vs delete sets.
        raw_upb_ids = [
            str(up.user_playbook_id)
            for up in self.get_user_playbooks(
                user_id=user_id,
                limit=1_000_000,
                status_filter=_all_statuses,
            )
            if up.user_playbook_id is not None
        ]
        purge_upb_ids, delete_upb_ids = self._partition_purge_vs_delete(
            "user_playbook", raw_upb_ids
        )

        # Snapshot profile ids for the user and partition into
        # purge vs delete sets.
        raw_profile_ids = [
            p.profile_id
            for p in self.get_user_profile(user_id, status_filter=_all_statuses)
            if p.profile_id is not None
        ]
        purge_profile_ids, delete_profile_ids = self._partition_purge_vs_delete(
            "profile", raw_profile_ids
        )

        # Snapshot request ids for the user so we can both count and
        # delete via delete_requests_by_ids — there is no
        # delete_all_requests_for_user primitive.
        request_ids = [
            session_item.request.request_id
            for session_items in self.get_sessions(
                user_id=user_id, top_k=1_000_000
            ).values()
            for session_item in session_items
        ]

        # Delete in dependency-safe order: interactions first (they
        # reference requests), then user playbooks, then profiles, then
        # requests. Only the delete-sets are hard-deleted; purge-sets are
        # content-purged below.
        self.delete_all_interactions_for_user(user_id)
        deleted_user_playbooks = (
            self.delete_user_playbooks_by_ids([int(i) for i in delete_upb_ids])
            if delete_upb_ids
            else 0
        )
        deleted_profiles = (
            self.delete_profiles_by_ids(delete_profile_ids) if delete_profile_ids else 0
        )
        deleted_requests = (
            self.delete_requests_by_ids(request_ids) if request_ids else 0
        )

        # Content-purge the purge-eligible rows after the hard-deletes.
        for pid in purge_profile_ids:
            self.purge_content(entity_type="profile", entity_id=pid)
        for upid in purge_upb_ids:
            self.purge_content(entity_type="user_playbook", entity_id=upid)

        return {
            "interactions": interaction_count,
            "user_playbooks": deleted_user_playbooks,
            "profiles": deleted_profiles,
            "requests": deleted_requests,
            "purged_profiles": len(purge_profile_ids),
            "purged_user_playbooks": len(purge_upb_ids),
        }


__all__ = [
    "AgentBinding",
    "AgentRunMixin",
    "EntityType",
    "LineageEventMixin",
    "AgentRunRecord",
    "AgentRunStatus",
    "BaseStorage",
    "NOT_APPLICABLE_ANSWER",
    "PendingToolCallRecord",
    "PendingToolCallStatus",
    "PendingToolCallUpsertResult",
    "PlaybookMixin",
    "RetrievalLogMixin",
    "PriorAnswerMatch",
    "RunToolDependencyKind",
    "RunToolDependencyRecord",
    "ShadowVerdictsMixin",
    "ShareLinkMixin",
    "StallStateMixin",
    "build_pending_tool_call_dedup_key",
    "build_scope_hash",
    "canonical_json",
    "embedding_similarity",
    "human_feedback_scope",
    "is_not_applicable_tool_result",
    "matches_status_filter",
    "not_applicable_tool_result",
    "normalize_dedup_text",
]
