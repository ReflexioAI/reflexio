"""Actual client repairs remain visible without recording source/credential data."""

import json
import logging
from unittest.mock import MagicMock

import litellm
import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.test_support.reviewer_metrics import ReviewCallMetrics


class Result(BaseModel):
    value: int


def test_real_client_repair_is_counted_and_safe_projection_is_private(monkeypatch):
    client = LiteLLMClient(
        LiteLLMConfig(model="zai/glm-5.2", fallback_models=[], max_retries=0)
    )
    responses = iter(
        litellm.ModelResponse(
            model="glm-5.2",
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )
        for content in ("not JSON", '{"value": 5}')
    )
    transport = MagicMock(side_effect=lambda *_: next(responses))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    log = logging.getLogger("reflexio.server.llm.litellm_client")
    previous = log.level
    metrics = ReviewCallMetrics("zai/glm-5.2", 8192, "high")
    with metrics:
        result = client.generate_chat_response(
            [{"role": "user", "content": "PRIVATE_SOURCE_CANARY"}],
            max_tokens=8192,
            extra_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}},
            extra_headers={"private": "PRIVATE_CREDENTIAL_CANARY"},
            response_format=Result,
            parse_structured_output=True,
            structured_output_validator=lambda _: [],
            provider_request_guard=metrics.guard,
        )
    assert result == Result(value=5)
    assert transport.call_count == metrics.data["request_ends"] == 2
    assert len(metrics.data["provider_requests"]) == 2
    assert metrics.data["repair_attempts"] == metrics.data["repair_successes"] == 1
    assert metrics.data["repair_kinds"] == ["parse"]
    assert len(metrics.data["usage"]) == len(metrics.data["attempt_seconds"]) == 2
    assert metrics.data["wall_seconds"] > 0
    assert metrics.data["observation_complete"] is True
    assert "CANARY" not in json.dumps(metrics.data)
    assert log.level == previous and metrics not in log.handlers


@pytest.mark.parametrize(
    "change",
    [
        {"model": "another-model"},
        {"temperature": 0},
        {"num_retries": 1},
        {"max_tokens": 4096},
        {"extra_body": {}},
    ],
)
def test_guard_refuses_settings_drift_before_attempt(change):
    metrics = ReviewCallMetrics("zai/glm-5.2", 8192, "high")
    params = {
        "model": "zai/glm-5.2",
        "temperature": 0.7,
        "num_retries": 0,
        "max_tokens": 8192,
        "extra_body": {"reasoning_effort": "high", "thinking": {"type": "enabled"}},
        **change,
    }
    with pytest.raises(ValueError, match="drifted"):
        metrics.guard(params, 125, ("zai/glm-5.2",), True)
    assert metrics.data["provider_requests"] == []


def test_failed_call_restores_observer_and_does_not_claim_complete():
    metrics = ReviewCallMetrics("model", None, None)
    log = metrics.log
    previous = log.level
    with pytest.raises(TimeoutError), metrics:
        metrics.guard(
            {"model": "model", "temperature": 0.7, "num_retries": 0},
            125,
            ("model",),
            True,
        )
        raise TimeoutError("PRIVATE_FAILURE_CANARY")
    assert metrics.data["observation_complete"] is False
    assert "CANARY" not in json.dumps(metrics.data)
    assert log.level == previous and metrics not in log.handlers


def test_default_token_limit_is_frozen_across_repair():
    metrics = ReviewCallMetrics("model", None, None)
    params = {
        "model": "model",
        "temperature": 0.7,
        "num_retries": 0,
        "max_tokens": 8192,
    }
    metrics.guard(params, 125, ("model",), True)
    with pytest.raises(ValueError, match="drifted"):
        metrics.guard({**params, "max_tokens": 4096}, 125, ("model",), True)
    assert len(metrics.data["provider_requests"]) == 1


@pytest.mark.parametrize(
    "timing,usage", [(False, False), (True, False), (False, True), (True, True)]
)
def test_measurement_coverage_requires_timing_and_usage(timing, usage):
    metrics = ReviewCallMetrics("model", None, None)
    with metrics:
        metrics.guard(
            {"model": "model", "temperature": 0.7, "num_retries": 0},
            125,
            ("model",),
            True,
        )
        metrics.log.info(
            "event=llm_request_end %s",
            "elapsed_seconds=0.1" if timing else "status=ok",
        )
        if usage:
            metrics.log.info("Token usage - input: 2, output: 3, total: 5")
    assert metrics.data["request_lifecycle_complete"] is True
    assert metrics.data["timing_complete"] is timing
    assert metrics.data["usage_complete"] is usage
    assert metrics.data["observation_complete"] is (timing and usage)
    assert metrics.data["estimated_cost_complete"] is False
    assert all("estimated_cost_usd" not in row for row in metrics.data["usage"])
