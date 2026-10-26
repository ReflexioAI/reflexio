"""The ladder walk enforces ONE wall-clock budget across all its rungs.

Before, the walk summed per-rung hard timeouts on every call and warned when the
sum exceeded the budget -- 376 warnings in 14 h in production -- while enforcing
nothing: each rung and repair turn got a fresh full timeout. These tests drive
the walk on a fake clock (each provider call advances it) so the deadline
arithmetic is exercised without waiting.
"""

from __future__ import annotations

import logging
import time as real_time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from reflexio.server.llm import _litellm_text_generation as ttg
from reflexio.server.llm import call_deadline
from reflexio.server.llm.call_deadline import model_deadline
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)

_MESSAGES = [{"role": "user", "content": "hi"}]


def _response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    resp.usage.prompt_tokens_details = None
    resp.usage.cache_creation_input_tokens = None
    resp.usage.cache_read_input_tokens = None
    return resp


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


def _walk(
    monkeypatch, *, primary_burns: float, timeout: int = 30
) -> tuple[list[dict[str, Any]], Any]:
    """Two rungs; the primary fails after burning `primary_burns` fake seconds.

    Returns every call's (model, provider timeout, hard timeout) and the outcome.
    """
    clock = _Clock()
    # Only the module under test sees the fake clock.
    monkeypatch.setattr(
        ttg,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic,
            perf_counter=real_time.perf_counter,
            sleep=real_time.sleep,
            time=real_time.time,
        ),
    )
    monkeypatch.setattr(ttg, "_LADDER_WALL_CLOCK_BUDGET_SECONDS", 100.0)
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    client = LiteLLMClient(
        LiteLLMConfig(
            model="primary/a", timeout=timeout, fallback_models=["fallback/b"]
        )
    )
    calls: list[dict[str, Any]] = []

    def _fake(params, hard_timeout):
        calls.append(
            {
                "model": params["model"],
                "timeout": params.get("timeout"),
                "hard": hard_timeout,
            }
        )
        if params["model"] == "primary/a":
            clock.now += primary_burns
            raise TimeoutError("primary stalled")
        return _response("from-fallback")

    monkeypatch.setattr(client, "_completion_with_hard_timeout", _fake)
    try:
        outcome: Any = client.generate_chat_response(_MESSAGES)
    except LiteLLMClientError as exc:
        outcome = exc
    return calls, outcome


def test_a_fallback_with_ample_budget_gets_its_full_timeout(monkeypatch) -> None:
    calls, outcome = _walk(monkeypatch, primary_burns=35)
    assert outcome == "from-fallback"
    assert calls[1]["model"] == "fallback/b"
    assert calls[1]["hard"] == pytest.approx(35)  # 30 + 5 grace, unclamped


def test_a_fallback_is_clamped_to_what_the_budget_has_left(monkeypatch) -> None:
    calls, outcome = _walk(monkeypatch, primary_burns=80)
    assert outcome == "from-fallback"
    # 20 s left: the kill bound is the remainder, and the provider timeout sits
    # a grace below it so the provider gives up before the process is killed.
    assert calls[1]["hard"] == pytest.approx(20)
    assert calls[1]["timeout"] == pytest.approx(15)


def test_no_fallback_is_attempted_once_the_budget_is_spent(monkeypatch, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        calls, outcome = _walk(monkeypatch, primary_burns=100)
    assert [c["model"] for c in calls] == ["primary/a"]
    assert isinstance(outcome, LiteLLMClientError)
    assert "event=llm_ladder_budget_exhausted" in caplog.text


def test_a_long_configuration_is_not_re_warned_on_every_call(
    monkeypatch, caplog
) -> None:
    """The old per-call `llm_ladder_budget_exceeded` warning is gone."""
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    client = LiteLLMClient(
        LiteLLMConfig(model="primary/a", timeout=300, fallback_models=["fallback/b"])
    )
    monkeypatch.setattr(
        client,
        "_completion_with_hard_timeout",
        lambda *_args, **_kwargs: _response("ok"),
    )
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            client.generate_chat_response(_MESSAGES)
    assert "llm_ladder_budget" not in caplog.text


@pytest.mark.parametrize(
    ("generation_timeout", "warns"),
    [(300, False), (585, False), (590, False), (595, True)],
)
def test_boot_check_flags_a_timeout_that_leaves_no_fallback_time(
    monkeypatch, caplog, generation_timeout: int, warns: bool
) -> None:
    """Checked ONCE at boot, not per call: a primary that runs to its timeout
    must leave the fallback positive remaining time."""
    import reflexio.lib._base as lib_base
    from reflexio.server.llm.model_defaults import _warn_if_fallback_cannot_run

    monkeypatch.setattr(lib_base, "GENERATION_TIMEOUT_SECONDS", generation_timeout)
    # A primary plus grace equal to the entire budget leaves no fallback time.
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    with caplog.at_level(logging.WARNING):
        _warn_if_fallback_cannot_run(["zai/glm-5.2"])
    assert ("event=llm_fallback_unreachable" in caplog.text) is warns


def test_boot_check_is_silent_without_fallbacks(monkeypatch, caplog) -> None:
    import reflexio.lib._base as lib_base
    from reflexio.server.llm.model_defaults import _warn_if_fallback_cannot_run

    monkeypatch.setattr(lib_base, "GENERATION_TIMEOUT_SECONDS", 595)
    with caplog.at_level(logging.WARNING):
        _warn_if_fallback_cannot_run([])
    assert "llm_fallback_unreachable" not in caplog.text


@pytest.mark.parametrize(
    ("slot_wait", "dispatched_hard", "provider_timeout"),
    [(70, 30.0, 30), (95, 5.0, 30), (99.9, 0.1, 30), (100, None, 30), (97, 3.0, 1)],
)
def test_a_provider_slot_wait_is_charged_to_the_budget(
    monkeypatch,
    slot_wait: float,
    dispatched_hard: float | None,
    provider_timeout: int,
) -> None:
    """Codex on reflexio#568: a saturated provider can hold a turn in the slot
    queue after `_prepare_turn` sized it. The turn is re-clamped once the slot is
    held, and not dispatched at all if the wait spent the budget."""
    from contextlib import contextmanager

    clock = _Clock()
    monkeypatch.setattr(
        ttg,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic,
            perf_counter=real_time.perf_counter,
            sleep=real_time.sleep,
            time=real_time.time,
        ),
    )
    monkeypatch.setattr(ttg, "_LADDER_WALL_CLOCK_BUDGET_SECONDS", 100.0)
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    waits: list[float | None] = []

    @contextmanager
    def _slow_slot(model: str, *, max_wait: float | None = None):
        del model
        waits.append(max_wait)
        clock.now += slot_wait
        yield

    monkeypatch.setattr(ttg, "provider_slot", _slow_slot)
    client = LiteLLMClient(LiteLLMConfig(model="primary/a", timeout=provider_timeout))
    sent: list[float] = []

    def _fake(params, hard_timeout):
        assert params["timeout"] <= provider_timeout
        sent.append(hard_timeout)
        return _response("ok")

    monkeypatch.setattr(client, "_completion_with_hard_timeout", _fake)
    if dispatched_hard is None:
        with pytest.raises(LiteLLMClientError):
            client.generate_chat_response(_MESSAGES)
        assert sent == []
    else:
        assert client.generate_chat_response(_MESSAGES) == "ok"
        assert sent == [pytest.approx(dispatched_hard)]
    # The slot wait itself is bounded by what the walk had left.
    assert waits == [pytest.approx(100.0)]


class _Answer(BaseModel):
    value: str


def _repair_walk(
    monkeypatch, *, first_burns: float, caller_seconds: float | None
) -> tuple[list[dict[str, float]], Any, float | None]:
    """One rung, a validator that rejects the first answer, so a repair turn follows.

    Returns each dispatched call's start time and kill bound, the outcome, and
    the caller's absolute deadline (None when no caller deadline is installed).
    """
    clock = _Clock()
    # Admission now checks this same caller deadline; use one clock domain.
    monkeypatch.setattr(call_deadline, "monotonic", clock.monotonic)
    monkeypatch.setattr(
        ttg,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic,
            perf_counter=real_time.perf_counter,
            sleep=real_time.sleep,
            time=real_time.time,
        ),
    )
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    client = LiteLLMClient(LiteLLMConfig(model="primary/a", timeout=120))
    calls: list[dict[str, float]] = []

    def _fake(params, hard_timeout):
        calls.append({"start": clock.now, "hard": hard_timeout})
        if len(calls) == 1:
            clock.now += first_burns
            return _response('{"value": "bad"}')
        clock.now += 1
        return _response('{"value": "good"}')

    monkeypatch.setattr(client, "_completion_with_hard_timeout", _fake)

    def _validator(output: BaseModel) -> list[str]:
        return [] if getattr(output, "value", None) == "good" else ["not good"]

    deadline = None if caller_seconds is None else clock.now + caller_seconds
    try:
        with model_deadline(deadline) if deadline is not None else nullcontext():
            outcome: Any = client.generate_chat_response(
                _MESSAGES,
                response_format=_Answer,
                structured_output_validator=_validator,
            )
    except LiteLLMClientError as exc:
        outcome = exc
    return calls, outcome, deadline


def test_without_a_caller_deadline_the_repair_turn_gets_its_full_timeout(
    monkeypatch,
) -> None:
    # A successful first turn must finish before its own 125-second kill bound.
    calls, outcome, _ = _repair_walk(monkeypatch, first_burns=124, caller_seconds=None)
    assert isinstance(outcome, _Answer)
    assert [c["hard"] for c in calls] == [pytest.approx(125), pytest.approx(125)]


def test_a_repair_turn_is_clamped_to_end_by_the_caller_deadline(monkeypatch) -> None:
    calls, outcome, deadline = _repair_walk(
        monkeypatch, first_burns=124, caller_seconds=169
    )
    assert isinstance(outcome, _Answer)
    assert len(calls) == 2
    assert calls[1]["hard"] == pytest.approx(45)
    assert deadline is not None
    assert all(c["start"] + c["hard"] <= deadline + 1e-9 for c in calls)


def test_no_repair_turn_is_started_once_the_caller_deadline_is_spent(
    monkeypatch,
) -> None:
    """A first attempt consuming the deadline cannot fund another provider call."""
    calls, outcome, deadline = _repair_walk(
        monkeypatch, first_burns=129, caller_seconds=129
    )
    assert isinstance(outcome, LiteLLMClientError)
    assert len(calls) == 1
    assert deadline is not None
    assert calls[0]["start"] + calls[0]["hard"] <= deadline + 1e-9


def test_nested_deadlines_only_tighten() -> None:
    from reflexio.server.llm.call_deadline import model_call_deadline

    with model_deadline(100.0):
        with model_deadline(200.0):
            assert model_call_deadline.get() == 100.0
        with model_deadline(50.0):
            assert model_call_deadline.get() == 50.0
    assert model_call_deadline.get() is None


@pytest.mark.parametrize("remaining", [5.0, 1.0, 0.25])
def test_fallback_can_use_every_positive_remaining_budget(monkeypatch, remaining):
    calls, outcome = _walk(monkeypatch, primary_burns=100.0 - remaining)
    assert outcome == "from-fallback"
    assert len(calls) == 2
    assert calls[1]["hard"] == pytest.approx(remaining)
    assert 0 < calls[1]["timeout"] <= calls[1]["hard"]


def test_repair_turn_can_finish_inside_a_short_positive_caller_budget(monkeypatch):
    calls, outcome, deadline = _repair_walk(
        monkeypatch, first_burns=124, caller_seconds=129
    )
    assert outcome == _Answer(value="good")
    assert len(calls) == 2
    assert calls[1]["hard"] == pytest.approx(5)
    assert deadline is not None
    assert calls[1]["start"] + calls[1]["hard"] <= deadline


def test_preparation_preserves_shorter_provider_timeout(monkeypatch) -> None:
    calls, outcome = _walk(monkeypatch, primary_burns=97, timeout=1)
    assert outcome == "from-fallback"
    assert calls[-1]["timeout"] == 1
    assert calls[-1]["hard"] == 3
