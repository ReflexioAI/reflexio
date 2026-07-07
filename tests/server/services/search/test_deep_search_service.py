"""Unit tests for AgenticUnifiedSearchService (mocked LLM + storage).

The LLM seam is ``llm_client.generate_chat_response`` returning real
``SearchPlan`` / ``ReflectVerdict`` instances in call order (plan first,
then one reflect per round). Storage is a MagicMock whose per-arm search
methods return real entity models, so the real executor fan-out, fusion,
ordering, and suppression code paths run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import DeepSearchConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.search.deep_search_schemas import (
    PlannedSubquery,
    ReflectVerdict,
    RerankOutput,
    SearchPlan,
)
from reflexio.server.services.search.deep_search_service import (
    AgenticUnifiedSearchService,
)

_NOW = int(datetime.now(UTC).timestamp())
_DAY = 86_400


def _profile(pid: str, age_days: int) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id="u1",
        content=f"content-{pid}",
        last_modified_timestamp=_NOW - age_days * _DAY,
        generated_from_request_id="r1",
    )


def _user_playbook(upid: int, age_days: int) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=upid,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        content=f"rule-{upid}",
        created_at=_NOW - age_days * _DAY,
    )


def _agent_playbook(apid: int, age_days: int) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=apid,
        agent_version="v1",
        content=f"team-rule-{apid}",
        created_at=_NOW - age_days * _DAY,
    )


def _storage(
    *,
    profiles: list[UserProfile] | None = None,
    user_playbooks: list[UserPlaybook] | None = None,
    agent_playbooks: list[AgentPlaybook] | None = None,
) -> MagicMock:
    storage = MagicMock()
    storage.supports_embedding = False
    storage.search_user_profile.return_value = profiles or []
    storage.search_user_playbooks.return_value = user_playbooks or []
    storage.search_agent_playbooks.return_value = agent_playbooks or []
    storage.get_source_user_playbook_ids_for_agent_playbooks.return_value = {}
    return storage


def _service(
    storage: MagicMock, llm_responses: list[object]
) -> tuple[AgenticUnifiedSearchService, MagicMock]:
    llm = MagicMock()
    llm.generate_chat_response.side_effect = llm_responses
    ctx = MagicMock()
    ctx.storage = storage
    ctx.prompt_manager = PromptManager()
    service = AgenticUnifiedSearchService(
        llm_client=llm, request_context=ctx, config=DeepSearchConfig()
    )
    return service, llm


def _request(**kwargs: object) -> UnifiedSearchRequest:
    defaults: dict[str, object] = {
        "query": "what do I use?",
        "user_id": "u1",
        "search_depth": "deep",
    }
    defaults.update(kwargs)
    return UnifiedSearchRequest(**defaults)  # type: ignore[arg-type]


_PROFILE_PLAN = SearchPlan(
    subqueries=[PlannedSubquery(arm="profiles", query="what do I use?")]
)


def _sufficient(ranked: list[str]) -> ReflectVerdict:
    return ReflectVerdict(sufficiency="sufficient", ranked_candidate_ids=ranked)


# ---------------------------------------------------------------------------
# Happy path: plan → execute → reflect, ranked order respected.
# ---------------------------------------------------------------------------


def test_reflect_ranking_orders_response():
    storage = _storage(profiles=[_profile("a", 9), _profile("b", 1)])
    service, llm = _service(storage, [_PROFILE_PLAN, _sufficient(["P:b", "P:a"])])

    response = service.search(_request(), org_id="org")

    assert [p.profile_id for p in response.profiles] == ["b", "a"]
    assert llm.generate_chat_response.call_count == 2  # plan + 1 reflect
    assert response.agent_answer is None
    assert response.agent_trace is not None and "sufficient" in response.agent_trace


def test_hallucinated_ranked_keys_dropped_and_unranked_appended():
    storage = _storage(profiles=[_profile("a", 9), _profile("b", 1)])
    service, _ = _service(storage, [_PROFILE_PLAN, _sufficient(["P:ghost", "P:b"])])

    response = service.search(_request(), org_id="org")

    # ghost dropped; ranked b first; unranked a appended in retrieval order.
    assert [p.profile_id for p in response.profiles] == ["b", "a"]


def test_top_k_caps_each_arm():
    storage = _storage(profiles=[_profile(f"p{i}", i) for i in range(8)])
    service, _ = _service(
        storage, [_PROFILE_PLAN, _sufficient([f"P:p{i}" for i in range(8)])]
    )

    response = service.search(_request(top_k=3), org_id="org")

    assert len(response.profiles) == 3


# ---------------------------------------------------------------------------
# Corrective round: at most 2 reflect calls, structurally.
# ---------------------------------------------------------------------------


def test_corrective_round_runs_once_then_reranks():
    storage = _storage(
        profiles=[_profile("a", 5)], user_playbooks=[_user_playbook(1, 5)]
    )
    insufficient = ReflectVerdict(
        sufficiency="insufficient",
        ranked_candidate_ids=[],
        corrective_subqueries=[
            PlannedSubquery(arm="user_playbooks", query="rules about X")
        ],
    )
    # After the corrective round the final ordering comes from the dedicated
    # listwise reranker (NOT a second reflect) — total LLM calls stay at 3
    # and no further corrective round is possible structurally.
    rerank = RerankOutput(ranked_candidate_ids=["UP:1", "P:a"])
    service, llm = _service(storage, [_PROFILE_PLAN, insufficient, rerank])

    response = service.search(_request(), org_id="org")

    assert llm.generate_chat_response.call_count == 3  # plan + reflect + rerank
    assert storage.search_user_playbooks.call_count == 1  # corrective executed
    assert [p.profile_id for p in response.profiles] == ["a"]
    assert [p.user_playbook_id for p in response.user_playbooks] == [1]


def test_no_corrective_when_sufficient():
    storage = _storage(profiles=[_profile("a", 5)])
    sufficient_with_stray_corrective = ReflectVerdict(
        sufficiency="sufficient",
        ranked_candidate_ids=["P:a"],
        corrective_subqueries=[PlannedSubquery(arm="profiles", query="stray")],
    )
    service, llm = _service(storage, [_PROFILE_PLAN, sufficient_with_stray_corrective])

    service.search(_request(), org_id="org")

    assert llm.generate_chat_response.call_count == 2
    assert storage.search_user_profile.call_count == 1  # no second execution


# ---------------------------------------------------------------------------
# Time windows: planner day offsets become absolute datetimes on the request.
# ---------------------------------------------------------------------------


def test_time_window_conversion_reaches_storage_request():
    storage = _storage(user_playbooks=[_user_playbook(1, 3)])
    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(
                arm="user_playbooks",
                query="rules this week",
                start_days_ago=7,
                end_days_ago=0,
            )
        ]
    )
    service, _ = _service(storage, [plan, _sufficient(["UP:1"])])

    service.search(_request(), org_id="org")

    request_arg = storage.search_user_playbooks.call_args[0][0]
    expected_start = datetime.now(UTC) - timedelta(days=7)
    assert request_arg.start_time is not None
    assert abs((request_arg.start_time - expected_start).total_seconds()) < 60
    assert request_arg.end_time is not None
    assert abs((request_arg.end_time - datetime.now(UTC)).total_seconds()) < 60


# ---------------------------------------------------------------------------
# Arm scoping.
# ---------------------------------------------------------------------------


def test_profiles_arm_dropped_without_user_id():
    storage = _storage(user_playbooks=[_user_playbook(1, 5)])
    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(arm="profiles", query="ignored"),
            PlannedSubquery(arm="user_playbooks", query="rules"),
        ]
    )
    service, _ = _service(storage, [plan, _sufficient(["UP:1"])])

    response = service.search(_request(user_id=None), org_id="org")

    storage.search_user_profile.assert_not_called()
    assert response.profiles == []
    assert [p.user_playbook_id for p in response.user_playbooks] == [1]


def test_entity_types_filter_respected():
    storage = _storage(agent_playbooks=[_agent_playbook(1, 5)])
    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(arm="agent_playbooks", query="team rules"),
            PlannedSubquery(arm="profiles", query="dropped"),
        ]
    )
    service, _ = _service(storage, [plan, _sufficient(["AP:1"])])

    response = service.search(_request(entity_types=["agent_playbooks"]), org_id="org")

    storage.search_user_profile.assert_not_called()
    storage.search_user_playbooks.assert_not_called()
    assert [p.agent_playbook_id for p in response.agent_playbooks] == [1]


def test_no_allowed_arms_returns_empty_success():
    service, llm = _service(_storage(), [])

    response = service.search(
        _request(user_id=None, entity_types=["profiles"]), org_id="org"
    )

    assert response.success is True
    assert response.profiles == []
    llm.generate_chat_response.assert_not_called()


# ---------------------------------------------------------------------------
# Degradation paths.
# ---------------------------------------------------------------------------


def test_planner_failure_degrades_to_verbatim_fan_out():
    storage = _storage(
        profiles=[_profile("a", 5)],
        user_playbooks=[_user_playbook(1, 5)],
        agent_playbooks=[_agent_playbook(2, 5)],
    )
    service, _ = _service(
        storage,
        [RuntimeError("planner down"), _sufficient(["P:a", "UP:1", "AP:2"])],
    )

    response = service.search(_request(), org_id="org")

    assert storage.search_user_profile.call_count == 1
    assert storage.search_user_playbooks.call_count == 1
    assert storage.search_agent_playbooks.call_count == 1
    assert response.profiles and response.user_playbooks and response.agent_playbooks


def test_reflect_failure_keeps_retrieval_order():
    storage = _storage(profiles=[_profile("a", 9), _profile("b", 1)])
    service, llm = _service(storage, [_PROFILE_PLAN, RuntimeError("reflect down")])

    response = service.search(_request(), org_id="org")

    assert [p.profile_id for p in response.profiles] == ["a", "b"]
    assert llm.generate_chat_response.call_count == 2  # no corrective after fallback


def test_missing_storage_raises_for_dispatch_fallback():
    ctx = MagicMock()
    ctx.storage = None
    service = AgenticUnifiedSearchService(
        llm_client=MagicMock(), request_context=ctx, config=DeepSearchConfig()
    )
    with pytest.raises(RuntimeError):
        service.search(_request(), org_id="org")


# ---------------------------------------------------------------------------
# Recency-dominant ordering.
# ---------------------------------------------------------------------------


def test_recency_dominant_overrides_reflect_order():
    storage = _storage(profiles=[_profile("old", 120), _profile("new", 1)])
    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(
                arm="profiles", query="current target", recency_dominant=True
            )
        ]
    )
    # Reflect (wrongly) ranks the stale profile first; recency must win.
    service, _ = _service(storage, [plan, _sufficient(["P:old", "P:new"])])

    response = service.search(_request(), org_id="org")

    assert [p.profile_id for p in response.profiles] == ["new", "old"]
