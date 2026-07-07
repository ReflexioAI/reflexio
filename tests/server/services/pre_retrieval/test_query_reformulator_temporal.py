"""Unit tests for the reformulator's structured output + temporal signals."""

from __future__ import annotations

from unittest.mock import MagicMock

from reflexio.models.api_schema.retriever_schema import ReformulationResult
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval import QueryReformulator


def _reformulator(llm_response: object = None, side_effect: object = None):
    llm = MagicMock()
    llm.generate_chat_response.return_value = llm_response
    if side_effect is not None:
        llm.generate_chat_response.side_effect = side_effect
    return QueryReformulator(llm_client=llm, prompt_manager=PromptManager()), llm


def test_structured_result_with_temporal_signals_passes_through():
    reformulator, llm = _reformulator(
        ReformulationResult(
            standalone_query="rules added this week",
            start_days_ago=7,
            end_days_ago=0,
        )
    )
    result = reformulator.rewrite("what rules did we add this week?")
    assert result.standalone_query == "rules added this week"
    assert result.start_days_ago == 7
    assert result.end_days_ago == 0
    assert result.has_temporal_signals
    # Structured call: response_format is the ReformulationResult schema.
    call_kwargs = llm.generate_chat_response.call_args[1]
    assert call_kwargs["response_format"] is ReformulationResult


def test_wants_current_and_recency_dominant_signals():
    reformulator, _ = _reformulator(
        ReformulationResult(
            standalone_query="current deployment target",
            recency_dominant=True,
            wants_current=True,
        )
    )
    result = reformulator.rewrite("what is my current deploy target?")
    assert result.recency_dominant is True
    assert result.wants_current is True


def test_no_signals_when_query_is_atemporal():
    reformulator, _ = _reformulator(
        ReformulationResult(standalone_query="agent failed to process refund")
    )
    result = reformulator.rewrite("agent failed to refund")
    assert not result.has_temporal_signals


def test_llm_failure_falls_back_to_original_query_no_signals():
    reformulator, _ = _reformulator(side_effect=RuntimeError("llm down"))
    result = reformulator.rewrite("what rules did we add this week?")
    assert result.standalone_query == "what rules did we add this week?"
    assert not result.has_temporal_signals


def test_non_structured_response_falls_back():
    reformulator, _ = _reformulator("just a string")
    result = reformulator.rewrite("original query")
    assert result.standalone_query == "original query"
    assert not result.has_temporal_signals


def test_suspect_rewritten_query_distrusts_signals_too():
    reformulator, _ = _reformulator(
        ReformulationResult(
            standalone_query="```json {bad}",
            wants_current=True,
        )
    )
    result = reformulator.rewrite("what package manager do I use?")
    assert result.standalone_query == "what package manager do I use?"
    assert result.wants_current is False


def test_rewritten_query_is_sanitized():
    reformulator, _ = _reformulator(
        ReformulationResult(
            standalone_query="  'reformulated   query'  ",
            wants_current=True,
        )
    )
    result = reformulator.rewrite("q")
    assert result.standalone_query == "reformulated query"
    assert result.wants_current is True
