"""Mandatory admission preserves native routes without inventing output policy."""

from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import (
    extraction_provider_guard,
    extraction_request_guard,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig


@pytest.mark.parametrize(
    "model",
    ["xai/grok-3-mini", "moonshot/moonshot-v1-8k", "zai/glm-5.2", "zai/glm-5.3-flashx"],
)
def test_native_default_route_admits_positive_input_without_operator_override(
    monkeypatch, model
):
    context = 32768
    monkeypatch.setattr(
        "litellm.get_model_info",
        lambda *_a, **_k: {"max_input_tokens": context, "max_output_tokens": context},
    )
    monkeypatch.setattr("litellm.token_counter", lambda **_k: 100)
    reserve = Mock()
    guard = extraction_request_guard(context_limits={}, reserve_attempt=reserve)
    client = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=None, fallback_models=[])
    )
    token = extraction_provider_guard.set(guard)
    try:
        params = client._build_completion_params(
            [{"role": "user", "content": "evidence"}]
        )[0]
        guard(params, 120, (), False)
    finally:
        extraction_provider_guard.reset(token)
    assert 0 < params["max_tokens"] <= context - 120
    reserve.assert_called_once()


@pytest.mark.parametrize("cap", [None, 32768])
def test_physical_headroom_does_not_bypass_oversized_input_or_explicit_caps(
    monkeypatch, cap
):
    from reflexio.server.llm.input_budget import InputBudgetExceededError

    monkeypatch.setattr(
        "litellm.get_model_info",
        lambda *_a, **_k: {"max_input_tokens": 32768, "max_output_tokens": 32768},
    )
    monkeypatch.setattr("litellm.token_counter", lambda **_k: 32768)
    reserve = Mock()
    guard = extraction_request_guard(context_limits={}, reserve_attempt=reserve)
    client = LiteLLMClient(
        LiteLLMConfig(model="xai/grok-3-mini", max_tokens=cap, fallback_models=[])
    )
    token = extraction_provider_guard.set(guard)
    try:
        with pytest.raises(InputBudgetExceededError):
            params = client._build_completion_params(
                [{"role": "user", "content": "oversized"}]
            )[0]
            guard(params, 120, (), False)
    finally:
        extraction_provider_guard.reset(token)
    reserve.assert_not_called()


@pytest.mark.parametrize(
    "route",
    [
        "https://api.z.ai.evil.invalid/api/coding/paas/v4",
        "https://api.z.ai/api/coding/paas/v4?alternate=1",
    ],
)
def test_similar_custom_route_cannot_claim_native_capacity(route):
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    client = LiteLLMClient(
        LiteLLMConfig(model="zai/glm-5.2", max_tokens=None, fallback_models=[])
    )
    token = extraction_provider_guard.set(Mock())
    try:
        with pytest.raises(
            InputBudgetConfigurationError, match="explicit output reservation"
        ):
            client._build_completion_params([], api_base=route)
    finally:
        extraction_provider_guard.reset(token)


def test_conflicting_endpoint_aliases_cannot_borrow_native_declaration():
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"zai/glm-5.2@https://api.z.ai/api/coding/paas/v4": 32768},
        reserve_attempt=reserve,
    )
    with pytest.raises(
        InputBudgetConfigurationError, match="Conflicting provider endpoint"
    ):
        guard(
            {
                "model": "zai/glm-5.2",
                "api_base": "https://api.z.ai/api/coding/paas/v4",
                "base_url": "https://different.invalid/v1",
                "max_tokens": 1024,
                "messages": [],
            },
            120,
            (),
            False,
        )
    reserve.assert_not_called()
