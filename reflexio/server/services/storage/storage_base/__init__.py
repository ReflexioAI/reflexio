from reflexio.models.api_schema.domain.enums import Status

from ._agent_run import (
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
    normalize_dedup_text,
)
from ._base import BaseStorageCore, matches_status_filter
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin
from ._shadow_verdicts import ShadowVerdictsMixin
from ._share_links import ShareLinkMixin
from ._stall_state import StallStateMixin


class BaseStorage(
    AgentRunMixin,
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
    ShareLinkMixin,
    StallStateMixin,
    ShadowVerdictsMixin,
    BaseStorageCore,
):
    """Base class for storage."""

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        """Delete all rows scoped to a single ``user_id``.

        Removes the user's interactions, user playbooks, profiles, and
        requests. Intentionally does NOT touch ``agent_playbooks`` — those
        are the cross-project rollup of skills and have no ``user_id``
        column. This is the data-isolation primitive used by paired
        protocols (e.g. SWE-bench) that share a single backend across
        parallel tasks without one task's clear-all nuking another
        in-flight task's rows.

        The default implementation composes existing per-user / by-ids
        primitives so any backend that implements those (sqlite, supabase,
        postgres, ...) gets correct behaviour for free.
        Subclasses MAY override for atomic / transactional efficiency.

        Args:
            user_id (str): The user id whose rows should be deleted.

        Returns:
            dict[str, int]: Per-entity deletion counts with keys
                ``interactions``, ``user_playbooks``, ``profiles``, and
                ``requests``. Counts reflect pre-deletion totals for the
                user (i.e. how many rows existed for that user).
        """
        interaction_count = len(self.get_user_interaction(user_id))

        # Snapshot user_playbook ids for the user so we can both count
        # and delete via the existing bulk-by-ids primitive (no per-user
        # playbook delete primitive exists at the mixin level).
        user_playbook_ids = [
            up.user_playbook_id
            for up in self.get_user_playbooks(
                user_id=user_id,
                limit=1_000_000,
                status_filter=[None, Status.ARCHIVED, Status.PENDING],
            )
            if up.user_playbook_id is not None
        ]
        profile_count = len(
            self.get_user_profile(
                user_id, status_filter=[None, Status.ARCHIVED, Status.PENDING]
            )
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
        # reference requests), then user playbooks (also reference
        # requests), then profiles, then the requests themselves.
        self.delete_all_interactions_for_user(user_id)
        deleted_user_playbooks = (
            self.delete_user_playbooks_by_ids(user_playbook_ids)
            if user_playbook_ids
            else 0
        )
        self.delete_all_profiles_for_user(user_id)
        deleted_requests = (
            self.delete_requests_by_ids(request_ids) if request_ids else 0
        )

        return {
            "interactions": interaction_count,
            "user_playbooks": deleted_user_playbooks,
            "profiles": profile_count,
            "requests": deleted_requests,
        }


__all__ = [
    "AgentBinding",
    "AgentRunMixin",
    "AgentRunRecord",
    "AgentRunStatus",
    "BaseStorage",
    "PendingToolCallRecord",
    "PendingToolCallStatus",
    "PendingToolCallUpsertResult",
    "PlaybookMixin",
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
    "matches_status_filter",
    "normalize_dedup_text",
]
