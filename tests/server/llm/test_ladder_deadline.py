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
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm import _litellm_text_generation as ttg
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


def _walk(monkeypatch, *, primary_burns: float) -> tuple[list[dict[str, Any]], Any]:
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
    monkeypatch.setattr(ttg, "_LADDER_MIN_TURN_SECONDS", 10.0)
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", "5")
    client = LiteLLMClient(
        LiteLLMConfig(model="primary/a", timeout=30, fallback_models=["fallback/b"])
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
        calls, outcome = _walk(monkeypatch, primary_burns=95)
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


@pytest.mark.parametrize(("generation_timeout", "warns"), [(300, False), (595, True)])
def test_boot_check_flags_a_timeout_that_leaves_no_fallback_time(
    monkeypatch, caplog, generation_timeout: int, warns: bool
) -> None:
    """Checked ONCE at boot, not per call: a primary that runs to its timeout
    must leave the fallback at least the minimum turn."""
    import reflexio.lib._base as lib_base
    from reflexio.server.llm.model_defaults import _warn_if_fallback_cannot_run

    monkeypatch.setattr(lib_base, "GENERATION_TIMEOUT_SECONDS", generation_timeout)
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
