"""Unit tests for the LLM-as-reranker helper.

Verifies the success path, the parse-failure / empty-response / count-mismatch
fallback paths, and the no-LLM-infra short-circuit. The helper is meant to
be a soft dependency: any failure must return ``None`` so callers can fall
back to the cross-encoder or hybrid order without a 500.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from reflexio.server.llm.rerank.llm_reranker import (
    _format_docs_block,
    _parse_scores,
    score_pairs_llm,
)


class _PromptManagerStub:
    """Minimal PromptManager-shape stub returning a canned rendered prompt."""

    def __init__(self, rendered: str = "<rendered>") -> None:
        self.rendered = rendered
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def render_prompt(self, prompt_id: str, variables: dict[str, Any]) -> str:
        self.calls.append((prompt_id, variables))
        return self.rendered


class _LLMClientStub:
    """Minimal LiteLLMClient-shape stub returning canned responses."""

    def __init__(self, response: Any = "[5, 7, 3]", raise_on_call: Exception | None = None) -> None:
        self.response = response
        self.raise_on_call = raise_on_call
        self.calls: list[dict[str, Any]] = []

    def generate_response(self, prompt: str, **kwargs: Any) -> Any:
        self.calls.append({"prompt": prompt, **kwargs})
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.response


def test_parse_scores_success() -> None:
    assert _parse_scores("[5, 7, 3]", 3) == [5.0, 7.0, 3.0]


def test_parse_scores_with_surrounding_prose() -> None:
    assert _parse_scores("Here are the scores: [5, 7, 3] (done)", 3) == [5.0, 7.0, 3.0]


def test_parse_scores_count_mismatch_returns_none() -> None:
    assert _parse_scores("[5, 7]", 3) is None


def test_parse_scores_invalid_json_returns_none() -> None:
    assert _parse_scores("not json at all", 3) is None


def test_parse_scores_non_numeric_returns_none() -> None:
    assert _parse_scores('[5, "x", 3]', 3) is None


def test_parse_scores_empty_array_with_zero_expected() -> None:
    assert _parse_scores("[]", 0) == []


def test_format_docs_block_collapses_internal_whitespace() -> None:
    got = _format_docs_block(["a\nb", "c d", "  e  "])
    assert got == "1. a b\n2. c d\n3. e"


def test_score_pairs_llm_empty_docs_returns_empty_list() -> None:
    assert score_pairs_llm("q", [], None, None) == []


def test_score_pairs_llm_no_llm_client_returns_none() -> None:
    pm = _PromptManagerStub()
    assert score_pairs_llm("q", ["a"], None, pm) is None


def test_score_pairs_llm_no_prompt_manager_returns_none() -> None:
    client = _LLMClientStub()
    assert score_pairs_llm("q", ["a"], client, None) is None


def test_score_pairs_llm_success_path() -> None:
    pm = _PromptManagerStub()
    client = _LLMClientStub(response="[10, 5, 2]")

    scores = score_pairs_llm("grocery", ["Walmart", "Thrive Market", "Mexico"], client, pm)

    assert scores == [10.0, 5.0, 2.0]
    # Prompt rendered with the expected variables.
    assert pm.calls == [
        (
            "rerank_relevance",
            {
                "query": "grocery",
                "docs_block": "1. Walmart\n2. Thrive Market\n3. Mexico",
                "num_docs": "3",
            },
        )
    ]
    # LLM was called exactly once with the rendered prompt.
    assert len(client.calls) == 1
    assert client.calls[0]["prompt"] == "<rendered>"


def test_score_pairs_llm_llm_error_returns_none() -> None:
    pm = _PromptManagerStub()
    client = _LLMClientStub(raise_on_call=RuntimeError("LLM exploded"))
    assert score_pairs_llm("q", ["a", "b"], client, pm) is None


def test_score_pairs_llm_empty_response_returns_none() -> None:
    pm = _PromptManagerStub()
    client = _LLMClientStub(response="")
    assert score_pairs_llm("q", ["a"], client, pm) is None


def test_score_pairs_llm_non_string_response_returns_none() -> None:
    pm = _PromptManagerStub()
    client = _LLMClientStub(response=MagicMock())  # not a str
    assert score_pairs_llm("q", ["a"], client, pm) is None


def test_score_pairs_llm_count_mismatch_returns_none() -> None:
    pm = _PromptManagerStub()
    # 3 docs but only 2 scores returned
    client = _LLMClientStub(response="[5, 7]")
    assert score_pairs_llm("q", ["a", "b", "c"], client, pm) is None


def test_score_pairs_llm_prompt_render_failure_returns_none() -> None:
    pm = _PromptManagerStub()
    pm.render_prompt = lambda *_a, **_kw: (_ for _ in ()).throw(  # type: ignore[assignment]
        ValueError("missing variable")
    )
    client = _LLMClientStub()
    assert score_pairs_llm("q", ["a"], client, pm) is None
    # LLM was NOT called when prompt render failed.
    assert client.calls == []
