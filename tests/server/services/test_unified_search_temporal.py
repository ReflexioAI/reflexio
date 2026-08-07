"""Integration of reformulation temporal signals into run_unified_search.

Mocks the QueryReformulator seam (returning ``ReformulationResult`` with
temporal signals) and asserts the pipeline: threads query-derived time
windows into the per-arm storage requests, skips the combined single-RPC
path when a window is present, and applies wants_current /
recency_dominant post-processing to the final lists.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    UnifiedSearchRequest,
)
from reflexio.server.services.unified_search_service import run_unified_search

_NOW = int(datetime.now(UTC).timestamp())
_DAY = 86_400


def _profile(
    pid: str,
    age_days: int,
    *,
    content: str | None = None,
) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id="u1",
        content=content or f"content-{pid}",
        last_modified_timestamp=_NOW - age_days * _DAY,
        generated_from_request_id="r1",
    )


def _mock_storage(profiles: list[UserProfile] | None = None) -> MagicMock:
    storage = MagicMock()
    storage._get_embedding.return_value = [0.1] * 8
    storage.search_user_profile.return_value = profiles or []
    storage.search_agent_playbooks.return_value = []
    storage.search_user_playbooks.return_value = []
    storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    return storage


def _run(storage: MagicMock, reformulation: ReformulationResult, **request_overrides):
    with patch(
        "reflexio.server.services.unified_search_service.QueryReformulator"
    ) as reformulator_cls:
        reformulator_cls.return_value.rewrite.return_value = reformulation
        return run_unified_search(
            request=UnifiedSearchRequest(
                query="original query",
                user_id="u1",
                enable_reformulation=True,
                **request_overrides,
            ),
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )


def test_time_window_reaches_per_arm_requests():
    storage = _mock_storage([_profile("in-window", 3)])
    result = _run(
        storage,
        ReformulationResult(
            standalone_query="rules this week", start_days_ago=7, end_days_ago=0
        ),
    )

    assert result.success
    profile_request = storage.search_user_profile.call_args[0][0]
    expected_start = datetime.now(UTC) - timedelta(days=7)
    assert profile_request.start_time is not None
    assert abs((profile_request.start_time - expected_start).total_seconds()) < 60
    playbook_request = storage.search_user_playbooks.call_args[0][0]
    assert playbook_request.start_time is not None


def test_time_window_skips_single_rpc_path():
    storage = _mock_storage([_profile("p", 1)])
    storage.supports_unified_hybrid_search = True
    storage.unified_hybrid_search = MagicMock()

    _run(
        storage,
        ReformulationResult(standalone_query="q", start_days_ago=7, end_days_ago=0),
    )

    storage.unified_hybrid_search.assert_not_called()
    storage.search_user_profile.assert_called_once()


def test_no_window_keeps_per_arm_requests_unbounded():
    storage = _mock_storage([_profile("p", 1)])
    _run(storage, ReformulationResult(standalone_query="q"))
    profile_request = storage.search_user_profile.call_args[0][0]
    assert profile_request.start_time is None
    assert profile_request.end_time is None


def test_wants_current_collapses_stale_near_duplicate():
    stale = _profile(
        "stale", 90, content="User uses pip for Python package management."
    )
    fresh = _profile(
        "fresh", 2, content="User switched to uv for Python package management."
    )
    storage = _mock_storage([stale, fresh])  # stale ranked first by retrieval

    result = _run(
        storage,
        ReformulationResult(standalone_query="package manager", wants_current=True),
    )

    assert [p.profile_id for p in result.profiles] == ["fresh", "stale"]


def test_recency_dominant_orders_by_timestamp():
    old = _profile("old", 120, content="deploys to Heroku production")
    new = _profile("new", 1, content="deploys to AWS ECS production")
    storage = _mock_storage([old, new])

    result = _run(
        storage,
        ReformulationResult(
            standalone_query="current deploy target", recency_dominant=True
        ),
    )

    assert [p.profile_id for p in result.profiles] == ["new", "old"]


def test_no_signals_leaves_ordering_untouched():
    a = _profile("a", 90)
    b = _profile("b", 1)
    storage = _mock_storage([a, b])

    result = _run(storage, ReformulationResult(standalone_query="q"))

    assert [p.profile_id for p in result.profiles] == ["a", "b"]


def test_window_and_wants_current_combined():
    # A single reformulation can carry both a time window and wants_current
    # (e.g. "what package manager did we adopt this week?"): the window must
    # reach the per-arm requests AND the freshness collapse must still apply.
    stale = _profile("stale", 6, content="User uses pip for Python package management.")
    fresh = _profile(
        "fresh", 2, content="User switched to uv for Python package management."
    )
    storage = _mock_storage([stale, fresh])

    result = _run(
        storage,
        ReformulationResult(
            standalone_query="package manager this week",
            start_days_ago=7,
            end_days_ago=0,
            wants_current=True,
        ),
    )

    profile_request = storage.search_user_profile.call_args[0][0]
    assert profile_request.start_time is not None
    assert [p.profile_id for p in result.profiles] == ["fresh", "stale"]


def test_wants_current_composes_with_relevance_floor(monkeypatch):
    # The floor path returns unwrapped entities; near-duplicates must still
    # collapse freshest-first in its output. (Superseded/expired rows never
    # reach these pools — storage search excludes them at SQL level; the
    # storage contract test pins that invariant.)
    from reflexio.models.config_schema import RetrievalFloorConfig
    from reflexio.server.services import unified_search_service as uss

    stale = _profile(
        "stale", 90, content="User uses pip for Python package management."
    )
    fresh = _profile(
        "fresh", 2, content="User switched to uv for Python package management."
    )

    monkeypatch.setattr(
        uss,
        "_run_phase_a",
        lambda **_kw: (
            ReformulationResult(standalone_query="q", wants_current=True),
            None,
            False,
        ),
    )
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: ([stale, fresh], [], []))

    def fake_score(query, docs):  # noqa: ARG001
        return "cross-encoder/ms-marco-MiniLM-L-6-v2", [1.0] * len(docs)

    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs_with_model",
        side_effect=fake_score,
    ):
        result = uss.run_unified_search(
            request=UnifiedSearchRequest(query="q", user_id="u1", top_k=5),
            org_id="test-org",
            storage=_mock_storage(),
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
            retrieval_floor=RetrievalFloorConfig(enabled=True),
        )

    assert [p.profile_id for p in result.profiles] == ["fresh", "stale"]


def test_temporal_signals_widen_fetch_pool_and_cut_after_reordering():
    """Temporal reordering runs after relevance ranking, so signal-carrying
    queries must (a) request a wider per-arm pool from storage and (b) cut to
    top_k only AFTER the temporal pass — otherwise the fresh/newest item that
    ranks below top_k on text relevance can never be promoted."""
    # Newest profile ranks LAST on relevance; with top_k=2 it would be sliced
    # away before the recency sort without the wider working pool.
    profiles = [_profile(f"old-{i}", 100 + i) for i in range(5)]
    profiles.append(_profile("newest", 1))
    storage = _mock_storage(profiles)

    result = _run(
        storage,
        ReformulationResult(standalone_query="q", recency_dominant=True),
        top_k=2,
    )

    fetch_request = storage.search_user_profile.call_args[0][0]
    assert fetch_request.top_k >= 20  # temporal pool floor reached storage
    assert len(result.profiles) == 2  # final cut still honors top_k
    assert result.profiles[0].profile_id == "newest"


def test_no_signals_keeps_fetch_pool_untouched():
    storage = _mock_storage([_profile("a", 1)])
    _run(storage, ReformulationResult(standalone_query="q"), top_k=2)
    fetch_request = storage.search_user_profile.call_args[0][0]
    assert fetch_request.top_k == 2
