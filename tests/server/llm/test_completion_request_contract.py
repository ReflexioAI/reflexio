"""The public identity seam matches actual builder policy, never provider I/O."""

import pytest

from reflexio.models.config_schema import (
    APIKeyConfig,
    CustomEndpointConfig,
    OpenAIConfig,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole

KEYS = (
    "model",
    "api_base",
    "base_url",
    "api_version",
    "custom_llm_provider",
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "max_completion_tokens",
)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.4-mini", "minimax/MiniMax-M3"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"max_tokens": 1024, "temperature": 0},
        {"model_role": ModelRole.EXTRACTION_AGENT},
        {"max_tokens": None},
        {"fallback_models": None},
        {"fallback_models": []},
    ],
)
def test_contract_matches_actual_builder_and_preserves_none_keys(
    monkeypatch, model, overrides
):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("Identity resolution cannot dispatch"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    monkeypatch.setenv("REFLEXIO_LLM_SEED", "19")
    client = LiteLLMClient(
        LiteLLMConfig(
            model=model,
            max_tokens=2048,
            top_p=0.8,
            api_key_config=APIKeyConfig(
                openai=OpenAIConfig(api_key="contract-test-key")
            ),
        )
    )
    forwarded = Mock(return_value=object())
    monkeypatch.setattr(client, "_make_request", forwarded)
    client.generate_chat_response_with_provenance([], **overrides)
    params = client._build_completion_params(messages=[], **forwarded.call_args.kwargs)[
        0
    ]
    identity = client.get_completion_request_contract(**overrides)
    assert identity == {key: params.get(key) for key in KEYS}
    assert tuple(identity) == KEYS
    assert identity["seed"] == 19
    assert identity["base_url"] is None
    assert identity["custom_llm_provider"] is None
    assert "api_key" not in identity and "messages" not in identity
    assert "contract-test-key" not in identity.values()
    if overrides.get("max_tokens", 2048) is not None:
        cap = identity["max_completion_tokens"] or identity["max_tokens"]
        assert cap == overrides.get("max_tokens", 2048)
    dispatch.assert_not_called()


def test_custom_route_wins_role_and_descriptor_is_for_hashing_not_logging(monkeypatch):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("Identity resolution cannot dispatch"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-5.4-mini",
            max_tokens=2048,
            api_key_config=APIKeyConfig(
                custom_endpoint=CustomEndpointConfig(
                    model="openai/custom-contract",
                    api_key="custom-key",
                    api_base="https://example.invalid/v1?access_token=route-credential",
                )
            ),
        )
    )
    identity = client.get_completion_request_contract(
        model_role=ModelRole.EXTRACTION_AGENT, max_tokens=777, temperature=0
    )
    params = client._build_completion_params(
        messages=[],
        model_role=ModelRole.EXTRACTION_AGENT,
        max_tokens=777,
        temperature=0,
    )[0]
    assert identity == {key: params.get(key) for key in KEYS}
    assert identity["model"] == "openai/custom-contract"
    assert "route-credential" in identity["api_base"]
    assert identity["max_tokens"] == 777
    assert "api_key" not in identity and "custom-key" not in identity.values()
    dispatch.assert_not_called()


def test_unset_seed_and_explicit_temperature_none_keep_builder_semantics(monkeypatch):
    monkeypatch.delenv("REFLEXIO_LLM_SEED", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini", temperature=0.7))
    assert client.get_completion_request_contract()["temperature"] == 0.7
    changed = client.get_completion_request_contract(temperature=0.2)
    assert changed["temperature"] == 0.2 and changed["seed"] == 42
    assert (
        client.get_completion_request_contract(temperature=None)["temperature"] is None
    )
    assert client.config.temperature == 0.7


@pytest.mark.parametrize(
    "override", [None, [], ["gpt-4o-mini", "local/nomic", "gpt-4o", "gpt-4o"]]
)
def test_public_ladder_uses_actual_deduped_resolver_without_dispatch(
    monkeypatch, override
):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("No provider for ladder resolution"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-4o-mini",
            fallback_models=["gpt-4o-mini", "gpt-4o", "gpt-4o", "local/nomic"],
        )
    )
    assert client.get_completion_model_ladder() == ("gpt-4o-mini", "gpt-4o")
    options = {"fallback_models": override}
    forwarded = Mock(return_value=object())
    monkeypatch.setattr(client, "_make_request", forwarded)
    client.generate_chat_response_with_provenance([], **options)
    effective = forwarded.call_args.kwargs
    assert ("fallback_models" in effective) is (override is not None)
    assert client.get_completion_model_ladder(**options) == tuple(
        client._resolve_ladder(**effective)
    )
    assert client.get_completion_model_ladder(**options) == (
        ("gpt-4o-mini",) if override == [] else ("gpt-4o-mini", "gpt-4o")
    )
    dispatch.assert_not_called()


def test_public_ladder_uses_role_precedence_and_custom_endpoint_pin(monkeypatch):
    from unittest.mock import Mock

    import litellm

    dispatch = Mock(side_effect=AssertionError("No provider for ladder resolution"))
    monkeypatch.setattr(litellm, "completion", dispatch)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", fallback_models=["gpt-4o", "gpt-4o"])
    )
    overrides = {"model": "ignored-model", "model_role": ModelRole.EXTRACTION_AGENT}
    primary = client.get_completion_request_contract(**overrides)["model"]
    assert client.get_completion_model_ladder(**overrides) == tuple(
        dict.fromkeys([primary, "gpt-4o"])
    )
    assert primary != "ignored-model"
    client.config.api_key_config = APIKeyConfig(
        custom_endpoint=CustomEndpointConfig(
            model="openai/pinned",
            api_key="test-private",
            api_base="https://example.invalid/v1",
        )
    )
    assert client.get_completion_model_ladder(**overrides) == ("openai/pinned",)
    assert (
        client.get_completion_request_contract(**overrides)["model"] == "openai/pinned"
    )
    dispatch.assert_not_called()
