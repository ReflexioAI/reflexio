from ._base import (
    SQLiteStorageBase,
    _cosine_similarity,
    _effective_search_mode,
    _sanitize_fts_query,
    _true_rrf_merge,
)
from ._extras import ExtrasMixin
from ._operations import OperationMixin
from ._playbook import PlaybookMixin
from ._profiles import ProfileMixin
from ._requests import RequestMixin


class SQLiteStorage(
    ProfileMixin,
    RequestMixin,
    PlaybookMixin,
    OperationMixin,
    ExtrasMixin,
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
]
