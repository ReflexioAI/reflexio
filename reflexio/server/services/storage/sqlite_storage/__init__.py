from ._base import (
    SQLiteStorageBase,
    _cosine_similarity,
    _effective_search_mode,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
    parse_status,
)
from ._extras import ExtrasMixin
from ._governance import SQLiteGovernanceMixin
from ._learning_jobs import SQLiteLearningJobStoreMixin
from ._lineage import SQLiteLineageMixin
from ._operations import OperationMixin
from ._requests import RequestMixin
from ._session_outcomes import SessionOutcomeStoreMixin
from ._shadow_verdicts import ShadowVerdictsMixin as SQLiteShadowVerdictsMixin
from ._share_links import SQLiteShareLinkMixin
from ._stall_state import (
    SQLiteStallStateMixin,
    StallReason,
    StallState,
    clear_stall_state,
    get_stall_state,
    init_stall_state_table,
    mark_stall_notified,
    upsert_stall_state,
)
from .agent_run import (
    SQLiteAgentRunStoreMixin,
    SQLitePendingToolCallStoreMixin,
    SQLiteRunToolDependencyStoreMixin,
)
from .base import SQLiteDeletionMixin, SQLiteFtsVecMixin
from .governance import (
    AuditEventStoreMixin,
    GovernanceEraseExecutionMixin,
    PurgeOperationStoreMixin,
    RebuildHideMixin,
    SubjectBarrierMixin,
)
from .playbook import (
    AgentEvaluationResultStoreMixin,
    AgentPlaybookStoreMixin,
    OptimizationJobStoreMixin,
    PlaybookAggregationStoreMixin,
    PlaybookSourceLinkageMixin,
    UserPlaybookStoreMixin,
)
from .profiles import InteractionStoreMixin, ProfileSearchMixin, ProfileStoreMixin


class SQLiteStorage(
    SQLiteLearningJobStoreMixin,
    SQLiteAgentRunStoreMixin,
    SQLitePendingToolCallStoreMixin,
    SQLiteRunToolDependencyStoreMixin,
    ProfileStoreMixin,
    InteractionStoreMixin,
    ProfileSearchMixin,
    RequestMixin,
    SessionOutcomeStoreMixin,
    PlaybookAggregationStoreMixin,
    AgentPlaybookStoreMixin,
    UserPlaybookStoreMixin,
    PlaybookSourceLinkageMixin,
    OptimizationJobStoreMixin,
    AgentEvaluationResultStoreMixin,
    AuditEventStoreMixin,
    PurgeOperationStoreMixin,
    SubjectBarrierMixin,
    GovernanceEraseExecutionMixin,
    RebuildHideMixin,
    SQLiteGovernanceMixin,
    SQLiteLineageMixin,
    OperationMixin,
    ExtrasMixin,
    SQLiteShareLinkMixin,
    SQLiteStallStateMixin,
    SQLiteShadowVerdictsMixin,
    SQLiteDeletionMixin,
    SQLiteFtsVecMixin,
    SQLiteStorageBase,
):
    """SQLite-based storage with FTS5 and hybrid search."""

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        # Hold the re-entrant storage lock across the composed clear so a marker
        # cannot land between outcome cleanup and request deletion.
        with self._lock:
            return super().clear_user_data(user_id)


__all__ = [
    "SQLiteStorage",
    "_cosine_similarity",
    "_effective_search_mode",
    "_sanitize_fts_query",
    "_true_rrf_merge",
    "_vector_rank_rows",
    "parse_status",
    "StallReason",
    "StallState",
    "clear_stall_state",
    "get_stall_state",
    "mark_stall_notified",
    "upsert_stall_state",
]
