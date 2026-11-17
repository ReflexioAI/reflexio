"""Native-plan orchestration preserves each source-owned acceptance contract."""

import time
from collections.abc import Iterator
from contextvars import ContextVar
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import ANY, Mock

import pytest

from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    UnifiedSearchRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import RetrievalFloorConfig, SearchMode
from reflexio.profile_authoring_protocol import (
    ESTIMATED_LIFETIME_POLICY,
    PROFILE_AUTHORING_POLICY,
    SOURCE_RETENTION_POLICY,
)
from reflexio.server import search_runtime
from reflexio.server.services import unified_search_service as service
from reflexio.server.services.retrieval.judge_payload import MODEL
from reflexio.server.services.retrieval.recency import RecencyConfig, ScoredItem
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.storage.error import ReadinessUnavailableError
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.unit
KINDS = ("profiles", "agent_playbooks", "user_playbooks")
HAS_ATTENTION = PROFILE_AUTHORING_POLICY == SOURCE_RETENTION_POLICY
attention = (
    import_module("reflexio.server.services.profile.attention")
    if HAS_ATTENTION
    else None
)


def test_compiled_policy_owns_profile_acceptance():
    assert PROFILE_AUTHORING_POLICY in (
        ESTIMATED_LIFETIME_POLICY,
        SOURCE_RETENTION_POLICY,
    )
    assert hasattr(service, "record_accepted_profiles") is HAS_ATTENTION
    assert (
        find_spec("reflexio.server.services.profile.attention") is not None
    ) is HAS_ATTENTION


def assert_final_access(native, profile_ids):
    if HAS_ATTENTION:
        native.storage.record_served_profile_access.assert_called_once_with(
            user_id="u", profile_ids=profile_ids, accessed_at=ANY
        )
    else:
        native.storage.record_served_profile_access.assert_not_called()


def profile(number: int) -> UserProfile:
    return UserProfile(
        profile_id=f"p{number}",
        user_id="u",
        content=f"profile evidence {number}",
        last_modified_timestamp=1,
        generated_from_request_id="fixture",
    )


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    banks: dict[str, list[Any]] = {
        "profiles": [profile(i) for i in range(1, 121)],
        "agent_playbooks": [
            AgentPlaybook(
                agent_playbook_id=i,
                agent_version="v",
                content=f"PRIVATE AGENT CONTENT {i}",
                trigger=f"agent trigger {i}",
            )
            for i in range(1, 121)
        ],
        "user_playbooks": [
            UserPlaybook(
                user_playbook_id=100 + i,
                user_id="u",
                agent_version="v",
                request_id="fixture",
                content=f"PRIVATE USER CONTENT {i}",
                trigger=f"user trigger {i}",
            )
            for i in range(1, 121)
        ],
    }
    state = SimpleNamespace(
        banks=banks,
        links={},
        mode="jev",
        events=[],
        extra_profile=None,
        agent_indices=None,
        user_reverse=False,
        return_empty=False,
        reformulation=ReformulationResult(standalone_query="Canonical task"),
    )
    storage = SimpleNamespace(
        org_id="org",
        supports_embedding=True,
        embedding_model_name="local/multilingual-e5-small",
        get_profile_access_epochs=Mock(return_value={}),
        record_served_profile_access=Mock(),
    )
    storage.get_source_user_playbook_ids_for_agent_playbooks = Mock(
        side_effect=lambda ids: {i: state.links.get(i, []) for i in ids}
    )
    state.storage = storage

    def phase_a(**kwargs):
        state.events.append("phase_a")
        return state.reformulation, [0.1] * 512, False

    def phase_b(**kwargs):
        state.events.append("phase_b")
        requested = set(kwargs["request"].entity_types or KINDS)
        return tuple(
            [
                ScoredItem(item, float(1000 - index))
                for index, item in enumerate(banks[kind][: kwargs["top_k"]])
            ]
            if kind in requested and not state.return_empty
            else []
            for kind in KINDS
        )

    def agent_select(*, items, **kwargs):
        state.events.append("agent")
        return (
            items
            if state.agent_indices is None
            else [items[i] for i in state.agent_indices]
        )

    def user_select(*, items, **kwargs):
        state.events.append("user")
        return list(reversed(items)) if state.user_reverse else items

    def profile_select(*, items, **kwargs):
        state.events.append("profile")
        return (
            ([state.extra_profile] + items[:4])
            if state.extra_profile is not None
            else items
        )

    def judge(wire):
        state.events.append("judge")
        assert state.dto.call_count == 0
        if HAS_ATTENTION:
            assert attention is not None
            snapshot = attention._SNAPSHOT.get()
            assert snapshot is not None and snapshot.storage is storage
        else:
            assert not hasattr(service, "record_accepted_profiles")
        return reply(wire)

    state.phase_a = Mock(side_effect=phase_a)
    state.phase_b = Mock(side_effect=phase_b)
    state.agent = Mock(side_effect=agent_select)
    state.user = Mock(side_effect=user_select)
    state.profile = Mock(side_effect=profile_select)
    state.judge = Mock(side_effect=judge)
    state.dto = Mock(wraps=service.UnifiedSearchResponse)
    state.plan = Mock(wraps=service.NativePlan)
    state.rank = Mock(wraps=service._apply_combined_score_recency)
    monkeypatch.setattr(service, "_run_phase_a", state.phase_a)
    monkeypatch.setattr(service, "_run_phase_b", state.phase_b)
    monkeypatch.setattr(service, "select_agent_playbooks", state.agent)
    monkeypatch.setattr(service, "select_user_playbooks", state.user)
    monkeypatch.setattr(service, "select_profiles", state.profile)
    monkeypatch.setattr(service, "call_jev", state.judge)
    monkeypatch.setattr(service, "judge_mode", lambda: state.mode)
    monkeypatch.setattr(service, "UnifiedSearchResponse", state.dto)
    monkeypatch.setattr(service, "NativePlan", state.plan)
    monkeypatch.setattr(service, "_apply_combined_score_recency", state.rank)
    session_seen_cache.clear()
    yield state
    session_seen_cache.clear()


def reply(
    wire: dict[str, Any],
    *,
    promote: set[str] | None = None,
    prune: set[str] | None = None,
) -> dict[str, Any]:
    answers = {}
    for key, question in wire["questions"].items():
        candidate = question["instructions"]["candidate"]
        evidence = candidate.get("text", candidate.get("trigger"))
        if key.endswith("conflict"):
            answers[key] = {"type": "noul", "noul": 0.1}
        else:
            values = (
                [1, 0, 0, 0]
                if evidence in (prune or set())
                else [0, 0, 0, 1]
                if evidence in (promote or set())
                else [0.3, 0.24, 0.46, 0]
            )
            answers[key] = {
                "type": "score",
                "probabilities": dict(zip("0123", values, strict=True)),
            }
    return {"model": MODEL, "answers": answers}


def run(
    native: SimpleNamespace,
    *,
    recency: RecencyConfig | None = None,
    floor: RetrievalFloorConfig | None = None,
    **changes,
):
    request = UnifiedSearchRequest(query="  Canonical task  ", user_id="u", **changes)
    return service.run_unified_search(
        request,
        "org",
        cast(BaseStorage, native.storage),
        Mock(),
        Mock(),
        retrieval_floor=floor or RetrievalFloorConfig(),
        recency=recency or RecencyConfig(),
    )


def assert_no_publication(native: SimpleNamespace) -> None:
    native.dto.assert_not_called()
    native.storage.record_served_profile_access.assert_not_called()
    assert not session_seen_cache.seen("org", "s")


def test_default_recency_one_native_snapshot_selectors_before_judge_and_one_final_dto(
    native,
):
    native.agent_indices = [0, 2, 4]
    native.user_reverse = True
    native.extra_profile = profile(999)
    result = run(native)
    native.phase_a.assert_called_once()
    native.phase_b.assert_called_once()
    native.agent.assert_called_once()
    native.user.assert_called_once()
    native.profile.assert_called_once()
    native.dto.assert_called_once()
    assert native.rank.call_count == 3
    assert [c.kwargs["top_k"] for c in native.rank.call_args_list] == [20, 20, 20]
    assert native.events == ["phase_a", "phase_b", "agent", "user", "profile", "judge"]
    assert native.phase_b.call_args.kwargs["top_k"] == 20
    assert native.plan.call_args.args == ("Canonical task",)
    pools = native.plan.call_args.kwargs["pools"]
    assert [p["id"] for p in pools["profiles"]] == ["p999"] + [
        f"p{i}" for i in range(1, 20)
    ]
    assert [p["id"] for p in pools["agent_playbooks"]][:5] == [1, 3, 5, 2, 4]
    assert [p["id"] for p in pools["user_playbooks"]][:5] == [105, 104, 103, 102, 101]
    assert result.profiles[0] is native.extra_profile
    assert result.agent_playbooks[0] is native.banks["agent_playbooks"][0]
    assert result.user_playbooks[0] is native.banks["user_playbooks"][4]
    wire = native.judge.call_args.args[0]
    assert wire["state"]["request"] == "Canonical task"
    assert "PRIVATE" not in str(wire)
    assert "profile evidence 999" in str(wire)
    assert_final_access(native, ["p1", "p2", "p3", "p4", "p999"])


@pytest.mark.parametrize("case", ["off", "profile_only", "empty"])
def test_off_profile_only_and_empty_never_dispatch_external_judge(native, case):
    changes = {}
    if case == "off":
        native.mode = "off"
    elif case == "profile_only":
        changes["entity_types"] = ["profiles"]
    else:
        native.return_empty = True
    result = run(native, **changes)
    assert result.success
    native.judge.assert_not_called()
    native.dto.assert_called_once()


@pytest.mark.parametrize("top_k", [1, 10, 100])
def test_unsupported_result_caps_preserve_native_count_no_top5_truncation(
    native, top_k
):
    result = run(native, top_k=top_k)
    assert all(len(getattr(result, kind)) == top_k for kind in KINDS)
    native.judge.assert_not_called()
    native.plan.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"threshold": 0.0},
        {"session_id": "s"},
        {"session_id": "s", "reset_session_seen": True},
        {"search_mode": SearchMode.FTS},
        {"enable_reformulation": True},
        {"conversation_history": [{"role": "user", "content": "previous"}]},
    ],
)
def test_unsupported_caller_shape_keeps_native_path_without_judge(native, change):
    result = run(native, **change)
    assert result.success
    native.judge.assert_not_called()
    native.plan.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        "model",
        "recency",
        "recency_disabled",
        "recency_penalty",
        "floor",
        "temporal",
        "window",
    ],
)
def test_native_model_floor_recency_and_time_capabilities_preserved(
    native, monkeypatch, case
):
    options: dict[str, Any] = {}
    if case == "model":
        native.storage.embedding_model_name = "text-embedding-3-small"
    elif case == "recency":
        options["recency"] = RecencyConfig(pool_size=30)
    elif case == "recency_disabled":
        options["recency"] = RecencyConfig(pool_size=0)
    elif case == "recency_penalty":
        options["recency"] = RecencyConfig(max_penalty_frac=0.1)
    elif case == "floor":
        options["floor"] = RetrievalFloorConfig(
            candidate_pool_size=30, profile_floor=0.0
        )
        floors = Mock(
            side_effect=lambda **kw: tuple(
                service._unwrap_items(kw[kind])[: kw["top_k"]] for kind in KINDS
            )
        )
        monkeypatch.setattr(service, "_apply_floors", floors)
    elif case == "temporal":
        native.reformulation = ReformulationResult(
            standalone_query="Canonical task", wants_current=True
        )
    else:
        native.reformulation = ReformulationResult(
            standalone_query="Canonical task", start_days_ago=1
        )
        options["enable_reformulation"] = True
    assert run(native, **options).success
    native.judge.assert_not_called()
    native.plan.assert_not_called()
    if case == "floor":
        assert floors.call_args.kwargs["cfg"].profile_floor == 0.0
    if case == "window":
        assert native.phase_b.call_args.kwargs["start_time"] is not None


def test_session_seen_items_backfill_and_reset_remain_native(native):
    session_seen_cache.record(
        "org",
        "s",
        [("profile", "p1"), ("agent_playbook", "1"), ("user_playbook", "101")],
    )
    result = run(native, session_id="s")
    assert result.profiles[0].profile_id == "p2"
    assert result.agent_playbooks[0].agent_playbook_id == 2
    assert result.user_playbooks[0].user_playbook_id == 102
    assert native.phase_b.call_args.kwargs["top_k"] == 23
    native.storage.record_served_profile_access.reset_mock()
    result = run(native, session_id="s", reset_session_seen=True)
    assert result.profiles[0].profile_id == "p1"
    native.judge.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["readiness", "malformed", "deadline", "expire_after_reply"]
)
def test_judge_failure_or_expiry_never_constructs_response_or_accepts_any_ids(
    native, failure
):
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 30)
    token = search_runtime._scope.set(scope)

    def judge(wire):
        if failure == "readiness":
            raise ReadinessUnavailableError("synthetic")
        if failure == "deadline":
            raise search_runtime.SearchDeadlineError()
        if failure == "malformed":
            return {"model": "wrong", "answers": {}}
        scope.deadline = time.monotonic() - 1
        return reply(wire)

    native.judge.side_effect = judge
    expected = (
        search_runtime.SearchDeadlineError
        if failure in ("deadline", "expire_after_reply")
        else ReadinessUnavailableError
    )
    try:
        with pytest.raises(expected):
            run(native)
        assert_no_publication(native)
        assert not scope.success_callbacks
    finally:
        search_runtime._scope.reset(token)


def test_source_user_pool_suppression_precedes_native_anchors_without_backfill(native):
    native.links = {1: [101]}
    native.agent_indices = [1, 2, 3, 4]
    result = run(native)
    anchors = native.plan.call_args.kwargs["anchors"]
    assert anchors["user_playbooks"] == [102, 103, 104, 105]
    pools = native.plan.call_args.kwargs["pools"]
    assert [x["id"] for x in pools["user_playbooks"]] == list(range(102, 121))
    assert [x.user_playbook_id for x in result.user_playbooks] == [102, 103, 104, 105]


def test_novel_agent_source_user_suppressed_after_judgment_before_publication(native):
    native.links = {6: [106]}
    native.judge.side_effect = lambda wire: reply(
        wire, promote={"agent trigger 6", "user trigger 6"}
    )
    result = run(native)
    assert [a.agent_playbook_id for a in result.agent_playbooks] == [1, 2, 3, 4, 6]
    assert [u.user_playbook_id for u in result.user_playbooks] == [101, 102, 103, 104]
    assert result.agent_playbooks[-1] is native.banks["agent_playbooks"][5]
    assert (
        native.storage.get_source_user_playbook_ids_for_agent_playbooks.call_args.args[
            0
        ]
        == [1, 2, 3, 4, 6]
    )


def test_unknown_source_linkage_refuses_before_judgment_or_acceptance(native):
    native.storage.get_source_user_playbook_ids_for_agent_playbooks = None
    with pytest.raises(ReadinessUnavailableError):
        run(native)
    native.judge.assert_not_called()
    assert_no_publication(native)


@pytest.mark.parametrize(
    "mutation", ["duplicate_raw", "changed_anchor", "during_judge"]
)
def test_native_source_fingerprint_violation_never_publishes(native, mutation):
    if mutation == "duplicate_raw":
        native.banks["agent_playbooks"][1] = native.banks["agent_playbooks"][0]
    elif mutation == "changed_anchor":
        native.agent.side_effect = lambda *, items, **_kwargs: (
            [items[0].model_copy(update={"content": "changed"})] + items[1:]
        )
    else:

        def judge(wire):
            native.banks["agent_playbooks"][0].content = "changed during judgment"
            return reply(wire)

        native.judge.side_effect = judge
    with pytest.raises(ReadinessUnavailableError):
        run(native)
    assert_no_publication(native)


@pytest.mark.parametrize("kind", ["agent_playbooks", "user_playbooks"])
def test_default_single_pb_type_uses_judge_and_retains_only_requested_original_objects(
    native, kind
):
    result = run(native, entity_types=[kind])
    native.judge.assert_called_once()
    assert len(getattr(result, kind)) == 5
    assert getattr(result, kind)[0] is native.banks[kind][0]
    assert all(not getattr(result, other) for other in KINDS if other != kind)
    assert len(native.judge.call_args.args[0]["questions"]) == 40


def test_final_profile_selection_only_is_recorded_after_judge(native):
    def judge(wire):
        native.storage.record_served_profile_access.assert_not_called()
        native.dto.assert_not_called()
        return reply(wire, prune={"profile evidence 1"}, promote={"profile evidence 6"})

    native.judge.side_effect = judge
    result = run(native)
    assert [p.profile_id for p in result.profiles] == ["p2", "p3", "p4", "p5", "p6"]
    assert_final_access(native, ["p2", "p3", "p4", "p5", "p6"])


def test_http_owned_scope_defers_final_access_and_retains_registering_context(native):
    marker = ContextVar("native_test_context", default="outside")
    marker_token = marker.set("origin")
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 30)
    scope_token = search_runtime._scope.set(scope)
    observed = []
    native.storage.record_served_profile_access.side_effect = lambda **_kwargs: (
        observed.append((marker.get(), _kwargs["profile_ids"]))
    )
    try:
        result = run(native)
        native.storage.record_served_profile_access.assert_not_called()
        if HAS_ATTENTION:
            assert attention is not None
            assert [entry[1] for entry in scope.success_callbacks] == [
                "profile_attention"
            ]
            _order, _name, context, callback = scope.success_callbacks[0]
            assert context.get(marker) == "origin"
            snapshot = context.get(attention._SNAPSHOT)
            assert snapshot is not None and snapshot.storage is native.storage
            marker.set("different")
            context.run(callback)
            assert observed == [
                ("origin", sorted(p.profile_id for p in result.profiles))
            ]
        else:
            assert not scope.success_callbacks
            assert not hasattr(service, "record_accepted_profiles")
    finally:
        search_runtime._scope.reset(scope_token)
        marker.reset(marker_token)


def test_partial_requested_native_arm_refuses_before_judge_or_publication(native):
    native.phase_b.side_effect = lambda **_kwargs: (
        None,
        native.banks["agent_playbooks"][:5],
        native.banks["user_playbooks"][:5],
    )
    with pytest.raises(ReadinessUnavailableError):
        run(native)
    native.judge.assert_not_called()
    assert_no_publication(native)
