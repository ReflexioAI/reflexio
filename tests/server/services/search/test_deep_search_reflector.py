"""Unit tests for the deep-search REFLECT stage (mocked LLM, real prompts)."""

from __future__ import annotations

from unittest.mock import MagicMock

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.search.deep_search_schemas import (
    PlannedSubquery,
    ReflectVerdict,
)
from reflexio.server.services.search.executor import Candidate
from reflexio.server.services.search.reflector import reflect_and_rank


def _candidate(key: str, arm: str = "profiles") -> Candidate:
    entity = MagicMock()
    entity.content = f"content-{key}"
    entity.last_modified_timestamp = 1_000
    return Candidate(key=key, arm=arm, entity=entity, subquery_indices=[0])


def _reflect(*, llm_response=None, llm_side_effect=None, **kwargs) -> ReflectVerdict:
    llm = MagicMock()
    llm.generate_chat_response.return_value = llm_response
    if llm_side_effect is not None:
        llm.generate_chat_response.side_effect = llm_side_effect
    defaults = {
        "query": "what is current?",
        "plan_notes": "",
        "candidates": [_candidate("P:a"), _candidate("P:b")],
        "allow_corrective": True,
        "llm_client": llm,
        "prompt_manager": PromptManager(),
    }
    defaults.update(kwargs)
    return reflect_and_rank(**defaults)


def test_hallucinated_keys_dropped():
    verdict = _reflect(
        llm_response=ReflectVerdict(
            sufficiency="sufficient",
            ranked_candidate_ids=["P:b", "P:ghost", "P:a"],
        )
    )
    assert verdict.ranked_candidate_ids == ["P:b", "P:a"]


def test_corrective_emptied_when_not_allowed():
    verdict = _reflect(
        llm_response=ReflectVerdict(
            sufficiency="insufficient",
            ranked_candidate_ids=[],
            corrective_subqueries=[
                PlannedSubquery(arm="profiles", query="retry differently")
            ],
        ),
        allow_corrective=False,
    )
    assert verdict.corrective_subqueries == []
    assert verdict.sufficiency == "insufficient"


def test_corrective_kept_when_allowed():
    verdict = _reflect(
        llm_response=ReflectVerdict(
            sufficiency="partial",
            ranked_candidate_ids=["P:a"],
            corrective_subqueries=[
                PlannedSubquery(arm="user_playbooks", query="rules about X")
            ],
        )
    )
    assert len(verdict.corrective_subqueries) == 1


def test_llm_exception_degrades_to_retrieval_order():
    verdict = _reflect(llm_side_effect=RuntimeError("reflect down"))
    assert verdict.sufficiency == "sufficient"
    assert verdict.ranked_candidate_ids == ["P:a", "P:b"]
    assert verdict.corrective_subqueries == []


def test_non_verdict_response_degrades_to_retrieval_order():
    verdict = _reflect(llm_response="not a verdict")
    assert verdict.sufficiency == "sufficient"
    assert verdict.ranked_candidate_ids == ["P:a", "P:b"]
