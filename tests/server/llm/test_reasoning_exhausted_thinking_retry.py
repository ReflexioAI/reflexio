"""A reasoning-exhausted answer is resent once with thinking disabled.

MiniMax counts thinking tokens toward ``max_tokens``; a body cut off by the
limit after reasoning spent the budget (blank, or a barely started answer) is
resent ONCE with ``thinking: disabled`` instead of the same-cap repair turn,
and only on routes that accept disabling thinking.
"""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.server.llm._model_compat import (
    MINIMAX_PREVIEW_MODEL,
    reasoning_exhausted_retry_extra_body,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    StructuredOutputRepairError,
)
from reflexio.server.services.extraction.profile_package_preparation import (
    _digest,
    _is_thinking_disabled_retry,
)

M3 = "minimax/MiniMax-M3"
VALID = '{"answer": "ok", "score": 42}'
EXHAUSTED = ("\n\n", "length")
# The production shape that parked windows: reasoning took nearly the whole
# 8,192-token budget and the JSON answer was cut off after a few characters.
CUT_AFTER_REASONING = ('{"answer": "o', "length")
DISABLED = {"thinking": {"type": "disabled"}}


class Answer(BaseModel):
    answer: str
    score: int


def _validator(output: BaseModel) -> list[str]:
    assert isinstance(output, Answer)
    return [] if output.score == 42 else [f"score must be 42, got {output.score}"]


def _response(
    content: str, finish_reason: str, reasoning_tokens: int = 8191
) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    choice.message.refusal = None
    choice.message.stop_reason = None
    choice.stop_reason = None
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "served-model"
    resp.stop_reason = None
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=8192, total_tokens=1)
    resp.usage.completion_tokens_details = MagicMock(reasoning_tokens=reasoning_tokens)
    resp.usage.prompt_tokens_details = None
    resp.usage.cache_creation_input_tokens = None
    resp.usage.cache_read_input_tokens = None
    return resp


def _run(
    model: str,
    responses: list[tuple[str, str] | tuple[str, str, int]],
    caplog: pytest.LogCaptureFixture,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], Any]:
    """Serve ``responses`` in order; return the sent params and the result/error."""
    calls: list[dict[str, Any]] = []

    def fake_completion(**params: Any) -> MagicMock:
        calls.append(params)
        content, finish_reason, *reasoning_tokens = responses[len(calls) - 1]
        return _response(content, finish_reason, *reasoning_tokens)

    client = LiteLLMClient(LiteLLMConfig(model=model, fallback_models=[]))
    with (
        patch("litellm.completion", side_effect=fake_completion),
        caplog.at_level(logging.INFO),
    ):
        try:
            outcome: Any = client.generate_chat_response_with_provenance(
                [{"role": "user", "content": "extract"}],
                response_format=Answer,
                structured_output_validator=_validator,
                max_tokens=8192,
                fallback_models=[],
                **kwargs,
            )
        except StructuredOutputRepairError as exc:
            outcome = exc
    return calls, outcome


def _events(caplog: pytest.LogCaptureFixture, event: str) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if f"event={event} " in r.getMessage()
    ]


def test_exhausted_answer_is_resent_once_without_thinking(caplog):
    calls, result = _run(M3, [EXHAUSTED, (VALID, "stop")], caplog)

    assert len(calls) == 2
    assert calls[0].get("extra_body") is None
    assert calls[1]["extra_body"] == DISABLED
    # The ORIGINAL prompt, not a repair conversation, at the same cap.
    assert calls[1]["messages"] == calls[0]["messages"]
    assert calls[1]["max_tokens"] == calls[0]["max_tokens"] == 8192
    assert result.value == Answer(answer="ok", score=42)
    assert result.thinking_disabled is True
    assert not _events(caplog, "llm_structured_repair_attempted")
    (line,) = _events(caplog, "llm_reasoning_exhausted_retry")
    assert "thinking=disabled outcome=succeeded" in line
    assert "reasoning_tokens=8191" in line
    assert any(
        r.levelno == logging.WARNING
        and "event=llm_reasoning_exhausted_retry " in r.getMessage()
        for r in caplog.records
    )


def test_answer_cut_off_after_reasoning_is_resent_without_thinking(caplog):
    """A partial answer behind a spent reasoning budget is not repaired at the cap.

    Before the fix this was classified ``parse`` and sent to the same-cap
    repair turn with thinking on, which truncated again; four such
    dispatches of the same request parked the extraction window.
    """
    calls, result = _run(M3, [CUT_AFTER_REASONING, (VALID, "stop")], caplog)

    assert len(calls) == 2
    assert calls[0].get("extra_body") is None
    assert calls[1]["extra_body"] == DISABLED
    assert calls[1]["messages"] == calls[0]["messages"]
    assert result.value == Answer(answer="ok", score=42)
    assert result.thinking_disabled is True
    assert not _events(caplog, "llm_structured_repair_attempted")
    (line,) = _events(caplog, "llm_reasoning_exhausted_retry")
    assert "thinking=disabled outcome=succeeded" in line


@pytest.mark.parametrize("reasoning_tokens", [0, 4096], ids=["none", "half"])
def test_answer_cut_off_by_its_own_length_still_repairs(caplog, reasoning_tokens):
    """Disabling thinking frees nothing when the answer itself used the budget."""
    calls, result = _run(
        M3,
        [('{"answer": "o', "length", reasoning_tokens), (VALID, "stop")],
        caplog,
    )

    assert len(calls) == 2
    assert all(call.get("extra_body") is None for call in calls)
    assert result.thinking_disabled is False
    (line,) = _events(caplog, "llm_structured_repair_attempted")
    assert "failure_kind=parse" in line
    assert not _events(caplog, "llm_reasoning_exhausted_retry")


def test_caller_extra_body_is_kept_beside_disabled_thinking(caplog):
    calls, result = _run(
        M3, [EXHAUSTED, (VALID, "stop")], caplog, extra_body={"reasoning_split": True}
    )

    assert calls[1]["extra_body"] == {"reasoning_split": True, **DISABLED}
    assert result.thinking_disabled is True


@pytest.mark.parametrize(
    ("retry", "outcome"),
    [
        (EXHAUSTED, "reasoning_exhausted"),
        (("not json", "stop"), "parse"),
        (('{"answer": "bad", "score": 1}', "stop"), "semantic"),
    ],
)
def test_failed_resend_falls_through_without_another_call(caplog, retry, outcome):
    calls, error = _run(M3, [EXHAUSTED, retry], caplog)

    assert len(calls) == 2
    assert isinstance(error, StructuredOutputRepairError)
    assert error.failure_kind == ("semantic" if outcome == "semantic" else "parse")
    assert not _events(caplog, "llm_structured_repair_attempted")
    (line,) = _events(caplog, "llm_reasoning_exhausted_retry")
    assert f"thinking=disabled outcome={outcome}" in line


@pytest.mark.parametrize(
    "first",
    [("not json", "stop"), ('{"answer": "bad", "score": 1}', "stop")],
    ids=["parse", "semantic"],
)
def test_parse_and_semantic_failures_still_repair(caplog, first):
    calls, result = _run(M3, [first, (VALID, "stop")], caplog)

    assert len(calls) == 2
    assert all(call.get("extra_body") is None for call in calls)
    # The repair turn echoes the answer and names the errors, as before.
    assert len(calls[1]["messages"]) == len(calls[0]["messages"]) + 2
    assert result.thinking_disabled is False
    assert len(_events(caplog, "llm_structured_repair_attempted")) == 1
    assert not _events(caplog, "llm_reasoning_exhausted_retry")


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (MINIMAX_PREVIEW_MODEL, {}),
        ("gpt-4o", {}),
        (M3, {"extra_body": {"thinking": {"type": "adaptive"}}}),
    ],
    ids=["flash-rejects-disabled", "other-provider", "caller-chose-thinking"],
)
def test_routes_that_cannot_disable_thinking_keep_the_repair(caplog, model, kwargs):
    calls, result = _run(model, [EXHAUSTED, (VALID, "stop")], caplog, **kwargs)

    assert len(calls) == 2
    assert all(
        (call.get("extra_body") or {}).get("thinking") != {"type": "disabled"}
        for call in calls
    )
    assert result.thinking_disabled is False
    (line,) = _events(caplog, "llm_structured_repair_attempted")
    assert "failure_kind=reasoning_exhausted" in line
    assert not _events(caplog, "llm_reasoning_exhausted_retry")


def test_retry_body_is_offered_only_where_documented():
    assert reasoning_exhausted_retry_extra_body({"model": M3}) == DISABLED
    assert (
        reasoning_exhausted_retry_extra_body({"model": MINIMAX_PREVIEW_MODEL}) is None
    )
    assert (
        reasoning_exhausted_retry_extra_body({"model": "minimax/MiniMax-M2.7"}) is None
    )
    assert (
        reasoning_exhausted_retry_extra_body(
            {"model": M3, "thinking": {"type": "adaptive"}}
        )
        is None
    )


def _contract(params: dict[str, Any]) -> str:
    return _digest(LiteLLMClient._completion_request_contract_from_params(params))


def test_pinned_profile_contract_admits_only_the_derived_resend():
    expected = {"model": M3, "max_tokens": 8192, "temperature": 0, "messages": []}
    resend = {**expected, "extra_body": DISABLED}

    assert _is_thinking_disabled_retry(LiteLLMClient, expected, _contract(resend))
    for drifted in (
        {**expected, "extra_body": {"thinking": {"type": "adaptive"}}},
        {**resend, "temperature": 1},
        {**resend, "max_tokens": 4096},
    ):
        assert not _is_thinking_disabled_retry(
            LiteLLMClient, expected, _contract(drifted)
        )
    flash = {**expected, "model": MINIMAX_PREVIEW_MODEL}
    assert not _is_thinking_disabled_retry(
        LiteLLMClient, flash, _contract({**flash, "extra_body": DISABLED})
    )
