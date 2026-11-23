"""Native integration: extra retrieval retains scope and records only served results."""

import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, Mock

import pytest

from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    UnifiedSearchRequest,
)
from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
from reflexio.models.config_schema import SearchMode
from reflexio.server import search_runtime
from reflexio.server.services import unified_search_service as service
from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.storage.storage_base import BaseStorage
from tests.server.services.test_profile_selection import profile

pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    primary = [profile(k) for k in "ABCDE"]
    secondary = [profile(k) for k in "AFDGBCE"]
    agents = [AgentPlaybook(agent_playbook_id=11, agent_version="v", content="a")]
    users = [
        UserPlaybook(
            user_playbook_id=17,
            user_id="u",
            agent_version="v",
            request_id="r",
            content="b",
        )
    ]
    storage = SimpleNamespace(
        embedding_model_name="local/multilingual-e5-small",
        supports_embedding=True,
        record_served_profile_access=Mock(),
        get_source_user_playbook_ids_for_agent_playbooks=Mock(return_value={}),
    )
    phase_a = Mock(
        return_value=(
            ReformulationResult(standalone_query="Python payments"),
            [0.1] * 512,
            False,
        )
    )
    monkeypatch.setattr(service, "_run_phase_a", phase_a)
    calls = []
    state = SimpleNamespace(
        primary=primary,
        secondary=secondary,
        agents=agents,
        users=users,
        storage=storage,
        calls=calls,
        phase_a=phase_a,
        fail=False,
        expire=None,
        scored=False,
    )

    def retrieve(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return (state.primary, agents, users)
        if state.expire is not None:
            state.expire.deadline = time.monotonic() - 1
        if state.fail:
            return (None, None, None)
        items = state.secondary
        if state.scored:
            items = [ScoredItem(p, 1.0 - i * 0.01) for i, p in enumerate(items)]
        return (items, [], [])

    monkeypatch.setattr(service, "_run_phase_b", retrieve)
    session_seen_cache.clear()
    yield state
    session_seen_cache.clear()


def run(native, **changes):
    request = UnifiedSearchRequest(
        query="Python payments",
        user_id="u",
        session_id="s",
        tags=["billing"],
        agent_version="v",
        **changes,
    )
    return service.run_unified_search(
        request=request,
        org_id="org",
        storage=cast(BaseStorage, native.storage),
        llm_client=Mock(),
        prompt_manager=Mock(),
    )


def test_native_secondary_uses_same_scope_and_records_final_only(native):
    result = run(native)
    assert [p.profile_id for p in result.profiles or []] == list("FDABC")
    native.storage.record_served_profile_access.assert_called_once_with(
        user_id="u", profile_ids=list("ABCDF"), accessed_at=ANY
    )
    assert isinstance(
        native.storage.record_served_profile_access.call_args.kwargs["accessed_at"],
        int,
    )
    assert (
        result.agent_playbooks == native.agents
        and result.user_playbooks == native.users
    )
    assert len(native.calls) == 2 and native.phase_a.call_count == 1
    second = native.calls[1]
    assert second["request"].entity_types == ["profiles"]
    assert (
        second["request"].user_id == "u"
        and second["request"].tags == ["billing"]
        and (second["request"].agent_version == "v")
    )
    assert second["org_id"] == "org" and second["storage"] is native.storage
    assert second["embedding"] is native.phase_a.return_value[1]
    assert (
        second["query"] == "python OR payments"
        and second["threshold"] == 0
        and (second["search_mode"] == SearchMode.FTS)
    )
    assert session_seen_cache.seen("org", "s") == frozenset(
        [("profile", x) for x in "FDABC"]
        + [("agent_playbook", "11"), ("user_playbook", "17")]
    )


def test_sparse_and_empty_paths(native):
    native.primary = native.primary[:2]
    result = run(native)
    assert [p.profile_id for p in result.profiles or []] == list("ABFDG")
    assert native.calls[1]["search_mode"] == SearchMode.HYBRID
    assert native.calls[1]["query"] == "Python payments"
    native.calls.clear()
    native.primary = []
    result = run(native, reset_session_seen=True)
    assert result.profiles == [] and len(native.calls) == 1


def test_recency_and_session_exclusions_stay_with_native_owner(native):
    native.secondary = [profile(k) for k in "GAFDHBCE"]
    native.scored = True
    session_seen_cache.record(
        "org", "s", frozenset([("profile", "G"), ("user_playbook", "99")])
    )
    request = UnifiedSearchRequest(query="Python payments", user_id="u", session_id="s")
    result = service.run_unified_search(
        request=request,
        org_id="org",
        storage=cast(BaseStorage, native.storage),
        llm_client=Mock(),
        prompt_manager=Mock(),
        recency=RecencyConfig(),
    )
    assert [p.profile_id for p in result.profiles or []] == list("FDABC")
    assert native.calls[1]["top_k"] == 22 and native.calls[1]["recency_on"]
    assert ("profile", "G") in session_seen_cache.seen("org", "s")


@pytest.mark.parametrize("expiry", [False, True])
def test_failed_or_expired_secondary_never_commits_served_state(native, expiry):
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 30)
    if expiry:
        native.expire = scope
    else:
        native.fail = True
    token = search_runtime._scope.set(scope)
    try:
        with pytest.raises(
            search_runtime.SearchDeadlineError if expiry else ReadinessUnavailableError
        ):
            run(native)
        assert not scope.success_callbacks and (not session_seen_cache.seen("org", "s"))
    finally:
        search_runtime._scope.reset(token)


def test_explicit_caller_threshold_preserves_single_native_retrieval(native):
    result = run(native, threshold=0.2)
    assert result.profiles == native.primary and len(native.calls) == 1


def test_other_embedding_policy_keeps_existing_profile_search(native):
    native.storage.embedding_model_name = "local/nomic-embed-text-v2-moe"
    result = run(native)
    assert result.profiles == native.primary and len(native.calls) == 1


def test_e5_sparse_secondary_preserves_resolved_time_window(native):
    native.primary = native.primary[:2]
    native.phase_a.return_value = (
        ReformulationResult(
            standalone_query="Python payments", start_days_ago=7, end_days_ago=0
        ),
        [0.1] * 512,
        False,
    )
    run(native)
    first, second = native.calls
    assert first["start_time"] is not None and first["end_time"] is not None
    assert second["start_time"] == first["start_time"]
    assert second["end_time"] == first["end_time"]
    assert second["query"] == first["query"]


@pytest.fixture
def empty_joint(native, monkeypatch):
    native.primary = []
    recovered = profile("native-location")
    recovered.content = "The user lives in San Francisco."
    native.secondary = [recovered]
    native.scored = True
    native.agents[0].trigger = "Dinner recommendations"
    native.users[0].trigger = "User asks for dinner recommendations"
    native.phase_a.return_value = (
        ReformulationResult(standalone_query="Dinner recommendations"),
        [0.1] * 512,
        False,
    )
    native.mode = "jev"
    monkeypatch.setattr(service, "judge_mode", lambda: native.mode)
    original = service._run_phase_b

    def scoped(**kwargs):
        result = original(**kwargs)
        wanted = set(
            kwargs["request"].entity_types
            or ("profiles", "agent_playbooks", "user_playbooks")
        )
        return tuple(
            values if kind in wanted else []
            for kind, values in zip(
                ("profiles", "agent_playbooks", "user_playbooks"), result, strict=True
            )
        )

    monkeypatch.setattr(service, "_run_phase_b", scoped)
    native.prune_profiles = False

    def judge(wire):
        answers = {}
        for key, question in wire["questions"].items():
            if key.endswith("conflict"):
                answers[key] = {"type": "noul", "noul": 0.0}
            else:
                rejected = (
                    native.prune_profiles
                    and question["instructions"]["candidate"]["type"] == "profile"
                )
                answers[key] = {
                    "type": "score",
                    "probabilities": {
                        "0": float(rejected),
                        "1": 0.0,
                        "2": 0.0,
                        "3": float(not rejected),
                    },
                }
        return {"model": wire["model"], "answers": answers}

    native.judge = Mock(side_effect=judge)
    monkeypatch.setattr(service, "call_jev", native.judge)
    return native


def run_empty_joint(native, **changes):
    request = UnifiedSearchRequest(
        **{
            "query": "Dinner recommendations",
            "user_id": "u",
            "entity_types": ["profiles", "user_playbooks"],
            "agent_version": "v",
            "tags": ["billing"],
            "source": "authored-source",
            **changes,
        }
    )
    return service.run_unified_search(
        request,
        "org",
        cast(BaseStorage, native.storage),
        Mock(),
        Mock(),
        recency=RecencyConfig(),
    )


def test_empty_profile_with_native_user_reaches_actual_joint_judge(empty_joint):
    result = run_empty_joint(empty_joint)
    assert result.profiles == empty_joint.secondary
    assert result.profiles[0] is empty_joint.secondary[0]
    assert len(empty_joint.calls) == 2 and empty_joint.phase_a.call_count == 1
    second = empty_joint.calls[1]
    assert second["request"].entity_types == ["profiles"]
    assert second["request"].user_id == "u" and second["request"].tags == ["billing"]
    assert (
        second["request"].source == "authored-source"
        and second["request"].agent_version == "v"
    )
    assert second["storage"] is empty_joint.storage and second["org_id"] == "org"
    assert second["embedding"] is empty_joint.phase_a.return_value[1]
    assert second["threshold"] == 0 and second["search_mode"] == SearchMode.HYBRID
    assert second["top_k"] == 20
    empty_joint.judge.assert_called_once()
    evidence = [
        question["instructions"]["candidate"]
        for key, question in empty_joint.judge.call_args.args[0]["questions"].items()
        if key.endswith("usefulness")
    ]
    assert {"type": "profile", "text": empty_joint.secondary[0].content} in evidence


def test_empty_recovery_is_candidate_recall_not_forced_final_membership(empty_joint):
    empty_joint.prune_profiles = True
    result = run_empty_joint(empty_joint)
    assert result.profiles == [] and len(empty_joint.calls) == 2
    empty_joint.judge.assert_called_once()
    empty_joint.storage.record_served_profile_access.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "off",
        "no-user-anchor",
        "profiles-only",
        "agent-mix",
        "all-arms",
        "threshold",
        "top-k",
        "session",
        "embedding-failure",
        "other-model",
    ],
)
def test_empty_recall_does_not_expand_other_policy_shapes(empty_joint, case):
    changes = {}
    if case == "off":
        empty_joint.mode = "off"
    elif case == "no-user-anchor":
        empty_joint.users.clear()
    elif case == "profiles-only":
        changes["entity_types"] = ["profiles"]
    elif case == "agent-mix":
        changes["entity_types"] = ["profiles", "agent_playbooks"]
    elif case == "all-arms":
        changes["entity_types"] = None
    elif case == "threshold":
        changes["threshold"] = 0.0
    elif case == "top-k":
        changes["top_k"] = 10
    elif case == "session":
        changes["session_id"] = "s"
    elif case == "embedding-failure":
        empty_joint.phase_a.return_value = (
            empty_joint.phase_a.return_value[0],
            None,
            True,
        )
    else:
        empty_joint.storage.embedding_model_name = "local/nomic-embed-text-v2-moe"
    assert run_empty_joint(empty_joint, **changes).profiles == []
    assert len(empty_joint.calls) == 1


@pytest.mark.parametrize("case", ["primary-failure", "secondary-failure", "deadline"])
def test_empty_recall_failure_never_accepts_results(empty_joint, case):
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 30)
    if case == "primary-failure":
        empty_joint.primary = None
    elif case == "secondary-failure":
        empty_joint.fail = True
    else:
        empty_joint.expire = scope
    token = search_runtime._scope.set(scope)
    try:
        with pytest.raises(
            search_runtime.SearchDeadlineError
            if case == "deadline"
            else ReadinessUnavailableError
        ):
            run_empty_joint(empty_joint)
        assert not scope.success_callbacks
        empty_joint.storage.record_served_profile_access.assert_not_called()
        empty_joint.judge.assert_not_called()
    finally:
        search_runtime._scope.reset(token)
