"""In-memory record of search results already served to a session.

When a unified search request carries a ``session_id``, results returned for
that ``(org_id, session_id)`` are remembered here so later searches in the
same session can skip items the session has already received and surface the
next-best matches instead. The session is the dedup scope: concurrent
sessions never affect each other, and a request without a ``session_id``
neither reads nor writes this cache.

The cache is deliberately process-local and best-effort. Losing it (restart,
another replica) only means an item may be served to a session twice — the
pre-dedup behavior. The time- and size-based limits below are memory bounds,
not dedup semantics: within a live session an item stays suppressed for the
session's lifetime.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

# Entity key: (entity_kind, entity_id) with entity_kind one of
# "profile" | "user_playbook" | "agent_playbook" and entity_id stringified.
EntityKey = tuple[str, str]

_MAX_SESSIONS = 1024
_MAX_ENTRIES_PER_SESSION = 2000
_IDLE_EVICTION_SECONDS = 12 * 60 * 60


@dataclass
class _SessionEntry:
    last_access: float
    # OrderedDict used as an insertion-ordered set for FIFO eviction.
    entries: OrderedDict[EntityKey, None] = field(default_factory=OrderedDict)


class SessionSeenCache:
    """Bounded per-(org, session) set of entity keys already served.

    Bounds: least-recently-used session eviction beyond ``max_sessions``,
    FIFO entry eviction beyond ``max_entries_per_session``, and lazy removal
    of sessions idle longer than ``idle_eviction_seconds``.
    """

    def __init__(
        self,
        max_sessions: int = _MAX_SESSIONS,
        max_entries_per_session: int = _MAX_ENTRIES_PER_SESSION,
        idle_eviction_seconds: float = _IDLE_EVICTION_SECONDS,
    ) -> None:
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._max_entries_per_session = max_entries_per_session
        self._idle_eviction_seconds = idle_eviction_seconds
        self._sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()

    def seen(self, org_id: str, session_id: str) -> frozenset[EntityKey]:
        """Return the entity keys already served to this org's session.

        Args:
            org_id (str): Organization the session belongs to
            session_id (str): Session identifier from the search request

        Returns:
            frozenset[EntityKey]: Keys previously recorded for the session
        """
        now = time.monotonic()
        key = (org_id, session_id)
        with self._lock:
            self._evict_idle(now)
            entry = self._sessions.get(key)
            if entry is None:
                return frozenset()
            entry.last_access = now
            self._sessions.move_to_end(key)
            return frozenset(entry.entries)

    def record(self, org_id: str, session_id: str, keys: Iterable[EntityKey]) -> None:
        """Record entity keys as served to this org's session.

        Args:
            org_id (str): Organization the session belongs to
            session_id (str): Session identifier from the search request
            keys (Iterable[EntityKey]): Entity keys returned to the caller
        """
        now = time.monotonic()
        session_key = (org_id, session_id)
        entity_keys = list(keys)
        with self._lock:
            self._evict_idle(now)
            entry = self._sessions.get(session_key)
            if entry is None:
                if not entity_keys:
                    # Nothing to remember: don't let empty results consume a
                    # session slot and evict a live session's seen-state.
                    return
                entry = _SessionEntry(last_access=now)
                self._sessions[session_key] = entry
            entry.last_access = now
            self._sessions.move_to_end(session_key)
            for entity_key in entity_keys:
                entry.entries[entity_key] = None
            while len(entry.entries) > self._max_entries_per_session:
                entry.entries.popitem(last=False)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self) -> None:
        """Drop all sessions (test hook)."""
        with self._lock:
            self._sessions.clear()

    def _evict_idle(self, now: float) -> None:
        # Sessions are LRU-ordered, so idle ones cluster at the front.
        while self._sessions:
            _, entry = next(iter(self._sessions.items()))
            if now - entry.last_access <= self._idle_eviction_seconds:
                break
            self._sessions.popitem(last=False)


# Process-wide instance used by the unified search service.
session_seen_cache = SessionSeenCache()
