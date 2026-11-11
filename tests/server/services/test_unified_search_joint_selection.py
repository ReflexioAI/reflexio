"""Joint selection sees widened native pools before lineage and session acceptance."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
from reflexio.server import search_runtime
from reflexio.server.extensions import register_service
from reflexio.server.services.retrieval.agent_selection import AGENT_PLAYBOOK_SELECTOR
from reflexio.server.services.retrieval.playbook_selection import (
    JOINT_PLAYBOOK_SELECTOR,
)
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.retrieval.user_selection import USER_PLAYBOOK_SELECTOR
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.unified_search_service import _run_phase_b as run_phase_b
from tests.server.services.test_unified_search_agent_selection import (  # noqa: F401
    native as native_fixture,
)
from tests.server.services.test_unified_search_agent_selection import (
    run,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    yield from cast(Any, native_fixture).__wrapped__(monkeypatch)


def test_joint_selection_precedes_lineage_and_skips_retired_selectors(native):
    provider = Mock()
    provider.select.return_value = (native.agents[1:], native.users)
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    old_agent, old_user = Mock(), Mock()
    register_service(AGENT_PLAYBOOK_SELECTOR, old_agent)
    register_service(USER_PLAYBOOK_SELECTOR, old_user)
    result = run(native, session_id="joint")
    # Rejected agent 1 cannot suppress its source user 11.
    assert result.user_playbooks == native.users
    assert result.agent_playbooks == native.agents[1:]
    assert result.profiles == native.profiles
    provider.select.assert_called_once()
    args = provider.select.call_args.kwargs
    assert args["agent_playbooks"] == native.agents
    assert args["user_playbooks"] == native.users
    assert args["request"].user_id == "u" and args["org_id"] == "org"
    assert args["storage"] is native.storage
    assert native.seen["top_k"] == 3 and native.seen["playbook_top_k"] == 20
    old_agent.select.assert_not_called()
    old_user.select.assert_not_called()
    seen = session_seen_cache.seen("org", "joint")
    assert ("user_playbook", "11") in seen
    assert ("agent_playbook", "1") not in seen


def test_joint_sees_twenty_and_can_promote_last_native_result(native):
    native.agents[:] = [
        AgentPlaybook(agent_playbook_id=i, agent_version="v1", content=str(i))
        for i in range(1, 26)
    ]
    provider = Mock()
    provider.select.side_effect = lambda **kw: ([kw["agent_playbooks"][-1]], [])
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = run(native)
    assert result.agent_playbooks == [native.agents[19]]
    assert len(provider.select.call_args.kwargs["agent_playbooks"]) == 20
    assert result.profiles == native.profiles


def test_joint_caps_agents_before_lineage_and_backfills_users(native):
    native.agents[:] = [
        AgentPlaybook(agent_playbook_id=i, agent_version="v1", content=str(i))
        for i in range(1, 8)
    ]
    native.users[:] = [
        UserPlaybook(
            user_playbook_id=i,
            user_id="u",
            agent_version="v1",
            request_id="r",
            content=str(i),
        )
        for i in range(11, 18)
    ]
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.side_effect = (
        lambda ids: {1: [11, 12], **({6: [13]} if 6 in ids else {})}
    )
    provider = Mock()
    provider.select.return_value = (native.agents, native.users)
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    # run() defaults to top_k=3; the joint method respects a smaller caller cap.
    result = run(native)
    assert result.agent_playbooks == native.agents[:3]
    assert result.user_playbooks == native.users[2:5]
    assert (
        native.storage.get_source_user_playbook_ids_for_agent_playbooks.call_args.args
        == ([1, 2, 3],)
    )


@pytest.mark.parametrize("invalid", ["copy", "duplicate", "cross_type", "shape"])
def test_joint_rejects_invalid_identity_without_committing_session(native, invalid):
    def select(**kw):
        agents, users = kw["agent_playbooks"], kw["user_playbooks"]
        if invalid == "copy":
            return [agents[0].model_copy()], users
        if invalid == "duplicate":
            return [agents[0], agents[0]], users
        if invalid == "cross_type":
            return users, agents
        return [agents, users]

    register_service(JOINT_PLAYBOOK_SELECTOR, SimpleNamespace(select=select))
    with pytest.raises(ReadinessUnavailableError):
        run(native, session_id="bad")
    assert session_seen_cache.seen("org", "bad") == frozenset()


def test_successful_empty_is_success_and_records_only_profiles(native):
    provider = Mock()
    provider.select.return_value = ([], [])
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = run(native, session_id="empty")
    assert result.success
    assert result.agent_playbooks == result.user_playbooks == []
    assert session_seen_cache.seen("org", "empty") == frozenset({("profile", "p")})


def test_joint_refusal_propagates_without_legacy_or_session_commit(native):
    provider = Mock()
    provider.select.side_effect = ReadinessUnavailableError("judge unavailable")
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    old = Mock()
    register_service(AGENT_PLAYBOOK_SELECTOR, old)
    with pytest.raises(ReadinessUnavailableError, match="judge unavailable"):
        run(native, session_id="failed")
    old.select.assert_not_called()
    assert not session_seen_cache.seen("org", "failed")


def test_profile_only_does_not_enter_or_widen_joint_selection(native):
    provider = Mock()
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = run(native, entity_types=["profiles"])
    assert result.profiles == native.profiles
    assert native.seen["top_k"] == native.seen["playbook_top_k"] == 3
    provider.select.assert_not_called()


def test_joint_expiry_after_selection_cannot_commit_session(native):
    import time

    scope = search_runtime.SearchScope(deadline=time.monotonic() + 1)
    token = search_runtime._scope.set(scope)
    try:

        def select(**kw):
            scope.deadline = time.monotonic() - 1
            return kw["agent_playbooks"], kw["user_playbooks"]

        register_service(JOINT_PLAYBOOK_SELECTOR, SimpleNamespace(select=select))
        with pytest.raises(search_runtime.SearchDeadlineError):
            run(native, session_id="expired")
    finally:
        search_runtime._scope.reset(token)
    assert not session_seen_cache.seen("org", "expired")


def test_joint_result_limit_is_five_even_when_caller_requests_more(native):
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.server.services import unified_search_service as service

    native.agents[:] = [
        AgentPlaybook(agent_playbook_id=i, agent_version="v1", content=str(i))
        for i in range(1, 9)
    ]
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    provider = Mock()
    provider.select.return_value = (native.agents, native.users)
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = service.run_unified_search(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=8),
        org_id="org",
        storage=native.storage,
        llm_client=Mock(),
        prompt_manager=Mock(),
    )
    assert result.agent_playbooks == native.agents[:5]
    assert native.seen["top_k"] == 8
    assert result.profiles == native.profiles


@pytest.mark.parametrize("has_window", [False, True])
def test_widened_fanout_preserves_each_native_scope(native, monkeypatch, has_window):
    from datetime import UTC, datetime

    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.models.config_schema import SearchMode
    from reflexio.server.services import unified_search_service as service

    storage = native.storage
    storage.supports_unified_hybrid_search = True
    combined = Mock(
        side_effect=AssertionError("Different per-type limits require fanout")
    )
    monkeypatch.setattr(service, "_run_phase_b_single_rpc", combined)
    monkeypatch.setattr(service, "_unified_single_rpc_enabled", lambda: True)
    profiles, agents, users = (
        Mock(return_value=[]),
        Mock(return_value=[]),
        Mock(return_value=[]),
    )
    monkeypatch.setattr(service, "_search_profiles_via_storage", profiles)
    monkeypatch.setattr(service, "_search_agent_playbooks_via_storage", agents)
    monkeypatch.setattr(service, "_search_user_playbooks_via_storage", users)
    request = UnifiedSearchRequest(
        query="q",
        user_id="u",
        agent_version="v1",
        tags=["billing"],
        source="unit",
        threshold=0.7,
        search_mode=SearchMode.FTS,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC) if has_window else None
    result = run_phase_b(
        request=request,
        org_id="org",
        storage=storage,
        embedding=None,
        query="q",
        top_k=5,
        playbook_top_k=20,
        threshold=0.7,
        search_mode=SearchMode.FTS,
        start_time=start,
    )
    assert result == ([], [], [])
    assert profiles.call_args.args[2] == 5
    assert agents.call_args.args[2:4] == (20, 0.7)
    assert agents.call_args.args[7] is not None  # Native SearchOptions.
    user_request = users.call_args.args[1]
    assert user_request.top_k == 20 and user_request.threshold == 0.7
    assert user_request.user_id == "u" and user_request.agent_version == "v1"
    assert user_request.tags == ["billing"] and user_request.source == "unit"
    assert (
        user_request.start_time == start and user_request.search_mode == SearchMode.FTS
    )
    combined.assert_not_called()


def test_empty_entity_types_use_default_joint_playbook_scope(native):
    provider = Mock()
    provider.select.return_value = ([], [])
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = run(native, entity_types=[])
    assert result.agent_playbooks == result.user_playbooks == []
    provider.select.assert_called_once()


@pytest.mark.parametrize("failure", ["exception", "missing", "malformed"])
def test_joint_lineage_unavailability_prevents_session_commit(native, failure):
    provider = Mock()
    provider.select.return_value = (native.agents, native.users)
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    if failure == "exception":
        native.storage.get_source_user_playbook_ids_for_agent_playbooks.side_effect = (
            RuntimeError("offline")
        )
    elif failure == "missing":
        native.storage.get_source_user_playbook_ids_for_agent_playbooks = None
    else:
        native.storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = (
            None
        )
    with pytest.raises(ReadinessUnavailableError, match="source linkage"):
        run(native, session_id="lineage")
    assert not session_seen_cache.seen("org", "lineage")


def test_absent_joint_provider_retains_existing_lineage_failure_behavior(native):
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.side_effect = (
        RuntimeError("offline")
    )
    result = run(native)
    assert result.success and result.user_playbooks == native.users


@pytest.mark.parametrize("branch", ["plain", "floor", "recency", "floor_recency"])
@pytest.mark.parametrize("signal", ["recency_dominant", "wants_current"])
def test_joint_temporal_selection_preserves_full_working_pool(
    native, monkeypatch, branch, signal
):
    from reflexio.models.api_schema.retriever_schema import (
        ReformulationResult,
        UnifiedSearchRequest,
    )
    from reflexio.models.config_schema import RetrievalFloorConfig
    from reflexio.server.llm.rerank.common import MULTILINGUAL_RERANK_MODEL
    from reflexio.server.services import unified_search_service as service
    from reflexio.server.services.retrieval import relevance_floor
    from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem

    native.agents[:] = [
        AgentPlaybook(
            agent_playbook_id=i,
            agent_version="v1",
            content="Acme rollout policy",
            created_at=1000 + i,
        )
        for i in range(1, 26)
    ]
    native.users[:] = [
        UserPlaybook(
            user_playbook_id=i,
            user_id="u",
            agent_version="v1",
            request_id="r",
            content="Acme rollout policy",
            created_at=1000 + i,
        )
        for i in range(1, 26)
    ]
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    monkeypatch.setattr(
        service,
        "_run_phase_a",
        lambda **_kw: (
            ReformulationResult(standalone_query="q", **{signal: True}),
            native.vector,
            False,
        ),
    )
    recency = (
        RecencyConfig(pool_size=25, max_penalty_frac=0, max_penalty_logit=0)
        if "recency" in branch
        else None
    )

    def retrieve(**kw):
        rows = (native.profiles, native.agents, native.users)
        if recency is not None:
            return tuple([ScoredItem(item, 1.0) for item in arm] for arm in rows)
        return rows

    monkeypatch.setattr(service, "_run_phase_b", retrieve)
    floor = RetrievalFloorConfig(candidate_pool_size=25 if "floor" in branch else 0)
    monkeypatch.setattr(service, "reranker_enabled", lambda: True)
    monkeypatch.setattr(
        relevance_floor,
        "score_pairs_with_model",
        lambda _query, documents: (
            MULTILINGUAL_RERANK_MODEL,
            [100.0] * len(documents),
        ),
    )
    provider = Mock()
    provider.select.side_effect = lambda **kw: (
        kw["agent_playbooks"],
        kw["user_playbooks"],
    )
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    result = service.run_unified_search(
        request=UnifiedSearchRequest(query="latest policy", user_id="u", top_k=25),
        org_id="org",
        storage=native.storage,
        llm_client=Mock(),
        prompt_manager=Mock(),
        retrieval_floor=floor,
        recency=recency,
    )
    selected_pool = provider.select.call_args.kwargs
    assert [row.agent_playbook_id for row in selected_pool["agent_playbooks"]] == list(
        range(25, 5, -1)
    )
    assert [row.user_playbook_id for row in selected_pool["user_playbooks"]] == list(
        range(25, 5, -1)
    )
    assert [row.agent_playbook_id for row in result.agent_playbooks] == [
        25,
        24,
        23,
        22,
        21,
    ]
    assert [row.user_playbook_id for row in result.user_playbooks] == [
        25,
        24,
        23,
        22,
        21,
    ]


def test_joint_decides_the_resolved_task_used_for_native_retrieval(native, monkeypatch):
    from reflexio.models.api_schema.retriever_schema import (
        ReformulationResult,
        UnifiedSearchRequest,
    )
    from reflexio.server.services import unified_search_service as service

    resolved = "Deploy the primary EU Acme Redis cluster now; on-call approval is explicitly granted."
    monkeypatch.setattr(
        service,
        "_run_phase_a",
        lambda **_kw: (
            ReformulationResult(standalone_query=resolved),
            native.vector,
            False,
        ),
    )
    provider = Mock()
    provider.select.return_value = ([], [])
    register_service(JOINT_PLAYBOOK_SELECTOR, provider)
    original = UnifiedSearchRequest(
        query="deploy it now", user_id="u", agent_version="v1", tags=["release"]
    )
    service.run_unified_search(
        request=original,
        org_id="org",
        storage=native.storage,
        llm_client=Mock(),
        prompt_manager=Mock(),
    )
    judged = provider.select.call_args.kwargs["request"]
    assert judged.query == native.seen["query"] == resolved
    assert judged.user_id == "u" and judged.agent_version == "v1"
    assert judged.tags == ["release"]
    assert original.query == "deploy it now"
