"""Unit tests for the deep-search listwise reranker (mocked LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.search.deep_search_schemas import RerankOutput
from reflexio.server.services.search.executor import Candidate
from reflexio.server.services.search.reranker import (
    _WINDOW_SIZE,
    listwise_rerank,
)


def _candidate(key: str, arm: str = "profiles") -> Candidate:
    entity = MagicMock()
    entity.content = f"content-{key}"
    entity.last_modified_timestamp = 1_000
    return Candidate(key=key, arm=arm, entity=entity, subquery_indices=[0])


def _rerank(candidates, llm_responses, **kwargs):
    llm = MagicMock()
    llm.generate_chat_response.side_effect = llm_responses
    defaults = {
        "query": "which tool do we use?",
        "candidates": candidates,
        "llm_client": llm,
        "prompt_manager": PromptManager(),
    }
    defaults.update(kwargs)
    return listwise_rerank(**defaults), llm


def test_single_window_reorders():
    pool = [_candidate("P:a"), _candidate("P:b"), _candidate("P:c")]
    ordered, llm = _rerank(
        pool, [RerankOutput(ranked_candidate_ids=["P:c", "P:a", "P:b"])]
    )
    assert [c.key for c in ordered] == ["P:c", "P:a", "P:b"]
    assert llm.generate_chat_response.call_count == 1


def test_hallucinated_and_missing_keys_handled():
    pool = [_candidate("P:a"), _candidate("P:b"), _candidate("P:c")]
    # ghost dropped; omitted P:b appended after ranked ones in prior order.
    ordered, _ = _rerank(
        pool, [RerankOutput(ranked_candidate_ids=["P:ghost", "P:c", "P:a"])]
    )
    assert [c.key for c in ordered] == ["P:c", "P:a", "P:b"]


def test_recency_dominant_arms_skipped():
    pool = [
        _candidate("P:a", arm="profiles"),
        _candidate("UP:1", arm="user_playbooks"),
        _candidate("UP:2", arm="user_playbooks"),
    ]
    ordered, llm = _rerank(
        pool,
        [RerankOutput(ranked_candidate_ids=["UP:2", "UP:1"])],
        skip_arms={"profiles"},
    )
    # Only the playbook arm was reranked; the skipped profile is appended.
    assert [c.key for c in ordered] == ["UP:2", "UP:1", "P:a"]
    prompt = llm.generate_chat_response.call_args[1]["messages"][0]["content"]
    assert "P:a" not in prompt


def test_fewer_than_two_candidates_makes_no_call():
    pool = [_candidate("P:a")]
    ordered, llm = _rerank(pool, [])
    assert [c.key for c in ordered] == ["P:a"]
    llm.generate_chat_response.assert_not_called()


def test_listwise_failure_falls_back_to_pointwise_scores():
    pool = [_candidate("P:a"), _candidate("P:b")]
    with patch(
        "reflexio.server.services.search.reranker.score_pairs_llm",
        return_value=[1.0, 9.0],
    ) as mock_pointwise:
        ordered, _ = _rerank(pool, [RuntimeError("listwise down")])
    mock_pointwise.assert_called_once()
    assert [c.key for c in ordered] == ["P:b", "P:a"]


def test_double_failure_falls_back_to_identity():
    pool = [_candidate("P:a"), _candidate("P:b")]
    with patch(
        "reflexio.server.services.search.reranker.score_pairs_llm",
        return_value=None,
    ):
        ordered, _ = _rerank(pool, [RuntimeError("listwise down")])
    assert [c.key for c in ordered] == ["P:a", "P:b"]


def test_sliding_window_bubbles_late_candidates_forward():
    pool = [_candidate(f"P:{i}") for i in range(_WINDOW_SIZE + 5)]
    best_key = pool[-1].key  # the true best sits at the very end of the pool

    def promote_best(**kwargs):
        # A consistent judge: whenever the best candidate is visible in the
        # window, rank it first; keep everything else in order. The
        # tail-to-head pass must bubble it across windows to the front.
        prompt = kwargs["messages"][0]["content"]
        keys = [
            line.split(" | ")[0]
            for line in prompt.splitlines()
            if line.startswith("P:")
        ]
        if best_key in keys:
            keys = [best_key, *[k for k in keys if k != best_key]]
        return RerankOutput(ranked_candidate_ids=keys)

    llm = MagicMock()
    llm.generate_chat_response.side_effect = promote_best
    ordered = listwise_rerank(
        query="q",
        candidates=pool,
        llm_client=llm,
        prompt_manager=PromptManager(),
    )
    assert ordered[0].key == best_key
    assert llm.generate_chat_response.call_count >= 2  # multiple windows ran
