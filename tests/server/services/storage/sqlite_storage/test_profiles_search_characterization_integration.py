"""Characterization tests: SQLite interaction-search + profile status lookup (Tier-1 Task 1).

Phase-A characterization net established BEFORE the ``ProfileMixin`` decomposition
(``_profiles.py`` -> ProfileStore / InteractionStore / Search). These pin the CURRENT
behavior of two methods that had ZERO storage-level coverage in the OSS suite, so a
later VERBATIM code move into the Search / ProfileStore buckets is provably safe.

Methods characterized here (SQLite side):
  - ``search_interaction`` — the keystone of the Search bucket. Only ever exercised via
    the service layer before this file. Pins the main filter/ranking paths: FTS query,
    user-id isolation, ``request_id`` filter, ``most_recent_k`` recency fetch, the
    no-query/no-``most_recent_k`` empty result, and the HYBRID-without-embedding
    fallback to FTS.
  - ``get_user_ids_with_status`` — no coverage anywhere in the OSS suite. Pins the
    DISTINCT user-id-by-status effect (NULL status vs a concrete status).

Behavior was captured against today's code and these tests assert exactly that; they
must pass on the current tree.
"""

from __future__ import annotations

import time

import pytest

from reflexio.models.api_schema.retriever_schema import SearchInteractionRequest
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "search-char-org") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_interaction(
    interaction_id: int,
    user_id: str,
    content: str,
    request_id: str,
    created_at: int,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        content=content,
        created_at=created_at,
    )


def _make_profile(
    user_id: str,
    profile_id: str,
    status: Status | None = None,
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content="content",
        last_modified_timestamp=int(time.time()),
        generated_from_request_id=f"req_{profile_id}",
        status=status,
    )


def _ids(interactions: list[Interaction]) -> list[int]:
    return [i.interaction_id for i in interactions]


# ---------------------------------------------------------------------------
# search_interaction — filter + ranking paths (Search bucket keystone)
# ---------------------------------------------------------------------------


class TestSearchInteractionFts:
    """FTS-mode search: term match, ordering-agnostic membership, user isolation."""

    def _seed(self, s: SQLiteStorage) -> int:
        now = int(time.time())
        s.add_user_interaction(
            "u1", _make_interaction(1, "u1", "apple pie recipe", "rA", now - 50)
        )
        s.add_user_interaction(
            "u1", _make_interaction(2, "u1", "banana bread baking", "rB", now - 40)
        )
        s.add_user_interaction(
            "u1", _make_interaction(3, "u1", "apple orchard visit", "rA", now - 30)
        )
        s.add_user_interaction(
            "u2", _make_interaction(4, "u2", "apple something else", "rC", now - 20)
        )
        return now

    def test_fts_query_matches_only_term_rows(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        res = s.search_interaction(
            SearchInteractionRequest(
                user_id="u1", query="apple", search_mode=SearchMode.FTS
            )
        )
        # Only the two "apple" rows for u1 match; the "banana" row is excluded.
        assert sorted(_ids(res)) == [1, 3]

    def test_fts_query_is_user_scoped(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        res = s.search_interaction(
            SearchInteractionRequest(
                user_id="u2", query="apple", search_mode=SearchMode.FTS
            )
        )
        # u2 only sees its own interaction, never u1's rows.
        assert sorted(_ids(res)) == [4]

    def test_fts_no_match_returns_empty(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        res = s.search_interaction(
            SearchInteractionRequest(
                user_id="u1", query="zzznomatch", search_mode=SearchMode.FTS
            )
        )
        assert res == []


class TestSearchInteractionHybridFallback:
    """HYBRID (default) with a query but no ``query_embedding`` falls back to FTS."""

    def test_hybrid_without_embedding_falls_back_to_fts(self, tmp_path) -> None:
        s = _store(tmp_path)
        now = int(time.time())
        s.add_user_interaction(
            "u1", _make_interaction(1, "u1", "apple pie recipe", "rA", now - 50)
        )
        s.add_user_interaction(
            "u1", _make_interaction(2, "u1", "banana bread baking", "rB", now - 40)
        )
        s.add_user_interaction(
            "u1", _make_interaction(3, "u1", "apple orchard visit", "rA", now - 30)
        )
        # Default search_mode is HYBRID; no query_embedding is passed, so the method
        # falls back to FTS ranking over the "apple" term.
        res = s.search_interaction(
            SearchInteractionRequest(user_id="u1", query="apple", most_recent_k=5)
        )
        assert sorted(_ids(res)) == [1, 3]


class TestSearchInteractionNoQuery:
    """No-query paths: request_id filter, most_recent_k recency fetch, and empty result."""

    def _seed(self, s: SQLiteStorage) -> int:
        now = int(time.time())
        s.add_user_interaction(
            "u1", _make_interaction(1, "u1", "apple pie recipe", "rA", now - 50)
        )
        s.add_user_interaction(
            "u1", _make_interaction(2, "u1", "banana bread baking", "rB", now - 40)
        )
        s.add_user_interaction(
            "u1", _make_interaction(3, "u1", "apple orchard visit", "rA", now - 30)
        )
        return now

    def test_request_id_filter_with_most_recent_k(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        res = s.search_interaction(
            SearchInteractionRequest(user_id="u1", request_id="rA", most_recent_k=10)
        )
        # Only interactions tagged request_id "rA" are returned (1 and 3).
        assert sorted(_ids(res)) == [1, 3]

    def test_most_recent_k_returns_newest_oldest_first(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        res = s.search_interaction(
            SearchInteractionRequest(user_id="u1", most_recent_k=2)
        )
        # The two most-recent interactions (3 @ now-30, 2 @ now-40) returned
        # oldest-first (the method reverses the DESC fetch).
        assert _ids(res) == [2, 3]

    def test_no_query_no_most_recent_k_returns_empty(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        # No query AND no most_recent_k -> the method short-circuits to [].
        res = s.search_interaction(SearchInteractionRequest(user_id="u1"))
        assert res == []


# ---------------------------------------------------------------------------
# get_user_ids_with_status — DISTINCT user ids by profile status
# ---------------------------------------------------------------------------


class TestGetUserIdsWithStatus:
    def _seed(self, s: SQLiteStorage) -> None:
        s.add_user_profile("ua", [_make_profile("ua", "pa", status=None)])
        s.add_user_profile("ub", [_make_profile("ub", "pb", status=Status.ARCHIVED)])
        s.add_user_profile("uc", [_make_profile("uc", "pc", status=None)])

    def test_null_status_returns_current_users(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        # status=None matches the NULL-status (CURRENT) profiles only.
        assert sorted(s.get_user_ids_with_status(None)) == ["ua", "uc"]

    def test_concrete_status_returns_matching_users(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        assert sorted(s.get_user_ids_with_status(Status.ARCHIVED)) == ["ub"]

    def test_unused_status_returns_empty(self, tmp_path) -> None:
        s = _store(tmp_path)
        self._seed(s)
        assert s.get_user_ids_with_status(Status.PENDING) == []

    def test_distinct_user_ids_not_duplicated(self, tmp_path) -> None:
        s = _store(tmp_path)
        # Two NULL-status profiles for the SAME user -> user id appears once.
        s.add_user_profile(
            "udup",
            [
                _make_profile("udup", "d1", status=None),
                _make_profile("udup", "d2", status=None),
            ],
        )
        assert s.get_user_ids_with_status(None) == ["udup"]
