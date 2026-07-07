"""Unit tests for the deep-search PLAN stage (mocked LLM, real prompts)."""

from __future__ import annotations

from unittest.mock import MagicMock

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.search.deep_search_schemas import (
    PlannedSubquery,
    SearchPlan,
)
from reflexio.server.services.search.planner import fallback_plan, plan_subqueries

_ARMS = ["profiles", "user_playbooks", "agent_playbooks"]


def _plan(**kwargs) -> SearchPlan:
    llm = MagicMock()
    llm.generate_chat_response.return_value = kwargs.pop("llm_response")
    side_effect = kwargs.pop("llm_side_effect", None)
    if side_effect is not None:
        llm.generate_chat_response.side_effect = side_effect
    defaults = {
        "query": "what do I prefer?",
        "conversation_history": None,
        "allowed_arms": _ARMS,
        "max_subqueries": 4,
        "llm_client": llm,
        "prompt_manager": PromptManager(),
    }
    defaults.update(kwargs)
    return plan_subqueries(**defaults)


def test_valid_plan_passes_through():
    plan = SearchPlan(
        subqueries=[
            PlannedSubquery(arm="profiles", query="preference"),
            PlannedSubquery(arm="user_playbooks", query="preference rule"),
        ]
    )
    result = _plan(llm_response=plan)
    assert [sq.arm for sq in result.subqueries] == ["profiles", "user_playbooks"]


def test_disallowed_arms_dropped_and_clamped():
    plan = SearchPlan(
        subqueries=[PlannedSubquery(arm="profiles", query=f"q{i}") for i in range(6)]
        + [PlannedSubquery(arm="agent_playbooks", query="disallowed")]
    )
    result = _plan(
        llm_response=plan,
        allowed_arms=["profiles"],
        max_subqueries=3,
    )
    assert len(result.subqueries) == 3
    assert all(sq.arm == "profiles" for sq in result.subqueries)


def test_llm_exception_falls_back_to_verbatim_fan_out():
    result = _plan(llm_response=None, llm_side_effect=RuntimeError("planner down"))
    assert [sq.arm for sq in result.subqueries] == _ARMS
    assert all(sq.query == "what do I prefer?" for sq in result.subqueries)


def test_non_plan_response_falls_back():
    result = _plan(llm_response="not a plan")
    assert [sq.arm for sq in result.subqueries] == _ARMS


def test_all_subqueries_filtered_falls_back():
    plan = SearchPlan(
        subqueries=[PlannedSubquery(arm="agent_playbooks", query="only disallowed")]
    )
    result = _plan(llm_response=plan, allowed_arms=["profiles"])
    assert [sq.arm for sq in result.subqueries] == ["profiles"]


def test_fallback_plan_shape():
    plan = fallback_plan("q", ["user_playbooks"])
    assert len(plan.subqueries) == 1
    assert plan.subqueries[0].arm == "user_playbooks"
    assert plan.subqueries[0].query == "q"
