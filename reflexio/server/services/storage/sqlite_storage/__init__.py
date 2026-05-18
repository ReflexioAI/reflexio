from ._base import (
    SQLiteStorageBase,
    _cosine_similarity,
    _effective_search_mode,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
)
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin
from ._share_links import SQLiteShareLinkMixin
from ._stall_state import (
    StallReason,
    StallState,
    SQLiteStallStateMixin,
    clear_stall_state,
    get_stall_state,
    init_stall_state_table,
    mark_stall_notified,
    upsert_stall_state,
)


class SQLiteStorage(
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
    SQLiteShareLinkMixin,
    SQLiteStallStateMixin,
    SQLiteStorageBase,
):
    """SQLite-based storage with FTS5 and hybrid search."""

    pass


__all__ = [
    "SQLiteStorage",
    "_cosine_similarity",
    "_effective_search_mode",
    "_sanitize_fts_query",
    "_true_rrf_merge",
    "_vector_rank_rows",
    "StallReason",
    "StallState",
    "clear_stall_state",
    "get_stall_state",
    "mark_stall_notified",
    "upsert_stall_state",
]
