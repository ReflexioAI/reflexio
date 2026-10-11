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
    with metrics, metrics.observe_responses(client):
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
    client = MagicMock()
    client._completion_with_hard_timeout.return_value = litellm.ModelResponse(
        model="model"
    )
    with metrics, metrics.observe_responses(client):
        metrics.guard(
            {"model": "model", "temperature": 0.7, "num_retries": 0},
            125,
            ("model",),
            True,
        )
        client._completion_with_hard_timeout({}, 125)
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


@pytest.mark.parametrize(
    "models,expected,complete",
    [
        (["glm-5.2", "glm-5.2"], None, True),
        (["glm-5.3", "glm-5.2"], None, False),
        (["glm-5.2", "glm-5.3"], None, False),
        ([None, "glm-5.2"], None, False),
        (["glm-5.2", None], None, False),
        (["glm-5.3", "glm-5.3"], "zai/glm-5.3", True),
    ],
)
def test_every_initial_and_repair_response_must_match_frozen_identity(
    monkeypatch, models, expected, complete
):
    client = LiteLLMClient(
        LiteLLMConfig(model="zai/glm-5.2", fallback_models=[], max_retries=0)
    )
    responses = []
    for model, content in zip(models, ["not JSON", '{"value": 5}'], strict=True):
        response = litellm.ModelResponse(
            model="placeholder",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "PRIVATE_REASONING_CANARY",
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )
        response.model = model
        responses.append(response)
    transport = MagicMock(side_effect=responses)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    metrics = ReviewCallMetrics(
        "zai/glm-5.2", 8192, "high", expected_response_model=expected
    )
    with metrics, metrics.observe_responses(client):
        result = client.generate_chat_response(
            [{"role": "user", "content": "PRIVATE_SOURCE_CANARY"}],
            max_tokens=8192,
            extra_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}},
            response_format=Result,
            parse_structured_output=True,
            structured_output_validator=lambda _: [],
            provider_request_guard=metrics.guard,
        )
    assert result == Result(value=5)
    assert transport.call_count == 2
    assert client._completion_with_hard_timeout is transport
    assert metrics.data["model_identity_complete"] is complete
    assert metrics.data["observation_complete"] is complete
    assert [row["model"] for row in metrics.data["provider_responses"]] == models
    assert all(
        row["choices"][0]["reasoning_chars"] == len("PRIVATE_REASONING_CANARY")
        for row in metrics.data["provider_responses"]
    )
    assert "CANARY" not in json.dumps(metrics.data)


def test_metadata_failure_preserves_original_response_and_restores_class_method():
    class Response:
        model = "model"

        @property
        def choices(self):
            raise ValueError("PRIVATE_METADATA_CANARY")

    response = Response()

    class Client:
        def _completion_with_hard_timeout(self, *_, **__):
            return response

    client = Client()
    metrics = ReviewCallMetrics("model", None, None)
    with metrics, metrics.observe_responses(client):
        assert client._completion_with_hard_timeout({}, 1) is response
    assert "_completion_with_hard_timeout" not in vars(client)
    assert metrics.data["response_observation_errors"] == ["ValueError"]
    assert metrics.data["model_identity_complete"] is False
    assert "CANARY" not in json.dumps(metrics.data)


def test_transport_exception_is_unchanged_and_restores_instance_override():
    error = TimeoutError("PRIVATE_TRANSPORT_CANARY")
    transport = MagicMock(side_effect=error)
    client = MagicMock()
    client._completion_with_hard_timeout = transport
    metrics = ReviewCallMetrics("model", None, None)
    with (
        pytest.raises(TimeoutError) as caught,
        metrics,
        metrics.observe_responses(client),
    ):
        client._completion_with_hard_timeout({}, 1)
    assert caught.value is error
    assert transport.call_count == 1
    assert client._completion_with_hard_timeout is transport
    assert metrics.data["provider_responses"] == []
    assert metrics.data["model_identity_complete"] is False
    assert "CANARY" not in json.dumps(metrics.data)


@pytest.mark.parametrize("failure", ["excess_choices", "metadata_append"])
def test_malformed_or_unrecordable_metadata_never_replaces_received_response(failure):
    response = litellm.ModelResponse(model="model")
    if failure == "excess_choices":
        response.choices *= 10000

    class BrokenList(list):
        def append(self, _):
            raise RuntimeError("PRIVATE_APPEND_CANARY")

    client = MagicMock()
    original = MagicMock(return_value=response)
    client._completion_with_hard_timeout = original
    metrics = ReviewCallMetrics("model", None, None)
    if failure == "metadata_append":
        metrics.data["provider_responses"] = BrokenList()
        metrics.data["response_observation_errors"] = BrokenList()
    with metrics, metrics.observe_responses(client):
        metrics.guard(
            {"model": "model", "temperature": 0.7, "num_retries": 0},
            125,
            ("model",),
            True,
        )
        assert client._completion_with_hard_timeout({}, 125) is response
        metrics.log.info("event=llm_request_end elapsed_seconds=0.1")
        metrics.log.info("Token usage - input: 2, output: 3, total: 5")
    assert original.call_count == 1
    assert client._completion_with_hard_timeout is original
    assert metrics.response_observation_failed is True
    assert metrics.data["model_identity_complete"] is False
    assert metrics.data["observation_complete"] is False
    assert "CANARY" not in json.dumps(metrics.data)


@pytest.mark.parametrize(
    "requested,public,hidden,expected,valid",
    [
        (
            "claude-code/default",
            "claude-code/default",
            {},
            "claude-code/default",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {"reflexio_provider": "claude-code"},
            "claude-code/default",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {"reflexio_provider": "claude-code", "reflexio_served_provider": "zai"},
            "claude-code/default",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {"reflexio_served_model": "glm-5.2"},
            "glm-5.2",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {
                "reflexio_served_provider": "private provider",
                "reflexio_served_model": "glm-5.2",
            },
            "glm-5.2",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {
                "reflexio_provider": "claude-code",
                "reflexio_served_provider": "zai",
                "reflexio_served_model": "glm-5.3",
            },
            "claude-code/default",
            False,
        ),
        (
            "claude-code/default",
            "claude-code/default",
            {
                "reflexio_provider": "claude-code",
                "reflexio_served_provider": "zai",
                "reflexio_served_model": "glm-5.3",
            },
            "glm-5.3",
            True,
        ),
        (
            "zai/glm-5.2",
            "zai/glm-5.2",
            {"reflexio_served_model": "glm-5.3"},
            "zai/glm-5.2",
            False,
        ),
        (
            "zai/glm-5.2",
            "zai/glm-5.2",
            {"reflexio_served_model": "glm-5.3"},
            "glm-5.3",
            True,
        ),
        ("zai/glm-5.2", "zai/glm-5.2", {"model": "glm-5.3"}, "zai/glm-5.2", True),
    ],
)
def test_authoritative_served_stamp_prevents_bridge_route_laundering(
    requested, public, hidden, expected, valid
):
    response = litellm.ModelResponse(model=public)
    response._hidden_params.update(hidden)
    client = MagicMock()
    original = MagicMock(return_value=response)
    client._completion_with_hard_timeout = original
    metrics = ReviewCallMetrics(requested, None, None, expected_response_model=expected)
    with metrics, metrics.observe_responses(client):
        metrics.guard(
            {"model": requested, "temperature": 0.7, "num_retries": 0},
            125,
            (requested,),
            True,
        )
        assert client._completion_with_hard_timeout({}, 125) is response
        metrics.log.info("event=llm_request_end elapsed_seconds=0.1")
        metrics.log.info("Token usage - input: 2, output: 3, total: 5")
    assert original.call_count == 1
    assert client._completion_with_hard_timeout is original
    assert metrics.data["model_identity_complete"] is valid
    assert metrics.data["observation_complete"] is valid
    assert "_hidden_params" not in json.dumps(metrics.data)
