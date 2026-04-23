"""Unit tests for LLMJudge + JudgeScore."""

from unittest.mock import MagicMock

import pytest

from tests.eval.judge import JudgeScore, LLMJudge


def test_judge_score_parses_llm_output():
    """When the client returns a JudgeScore directly, the judge passes it through."""
    client = MagicMock()
    client.generate_chat_response.return_value = JudgeScore(
        signal_f1=0.8,
        answer_correctness=0.0,
        grounded_rate=1.0,
        rationale="fine",
    )
    j = LLMJudge(
        client=client,
        rubric={
            "judge_model": "claude-sonnet-4-6",
            "prompt": "score: {expected} vs {actual}",
        },
    )
    s = j.score(expected={"x": 1}, actual={"x": 1})
    assert s.signal_f1 == 0.8
    assert s.grounded_rate == 1.0
    client.generate_chat_response.assert_called_once()


def test_judge_prompt_is_rendered_with_expected_and_actual():
    """The rubric placeholders are substituted before the LLM is called."""
    client = MagicMock()
    client.generate_chat_response.return_value = JudgeScore(
        signal_f1=0.5, answer_correctness=0.0, grounded_rate=1.0, rationale="ok"
    )
    j = LLMJudge(
        client=client,
        rubric={"judge_model": "m", "prompt": "E={expected} A={actual}"},
    )
    j.score(expected="EXP", actual="ACT")

    call_msgs = client.generate_chat_response.call_args.kwargs["messages"]
    assert call_msgs[0]["content"] == "E=EXP A=ACT"


def test_judge_passes_judge_model_as_override():
    client = MagicMock()
    client.generate_chat_response.return_value = JudgeScore(
        signal_f1=0.0, answer_correctness=0.0, grounded_rate=0.0, rationale=""
    )
    j = LLMJudge(
        client=client, rubric={"judge_model": "claude-haiku-4-5", "prompt": "p"}
    )
    j.score(expected={}, actual={})

    assert client.generate_chat_response.call_args.kwargs["model"] == "claude-haiku-4-5"


def test_judge_raises_typeerror_on_plain_string_response():
    """Misconfigured response_format could yield a str — we fail loudly."""
    client = MagicMock()
    client.generate_chat_response.return_value = "not a JudgeScore"
    j = LLMJudge(client=client, rubric={"judge_model": "m", "prompt": "p"})
    with pytest.raises(TypeError):
        j.score(expected={}, actual={})
