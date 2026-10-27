"""Gateway credentials cannot replace a caller's selected completion model."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import APIKeyConfig, Config, LLMConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

pytestmark = pytest.mark.unit


def endpoint():
    return {"api_key": "test-gateway-key", "api_base": "https://gateway.invalid/v1"}


def test_current_schema_rejects_endpoint_model_instead_of_ignoring_it():
    with pytest.raises(ValidationError, match="Obsolete configuration fields: model"):
        APIKeyConfig.model_validate(
            {"custom_endpoint": {**endpoint(), "model": "openai/legacy"}}
        )


@pytest.mark.parametrize("model", [None, "", "  "])
def test_gateway_requires_a_selected_generation_model(model):
    with pytest.raises(ValidationError, match="generation_model_name"):
        Config.model_validate(
            {
                "storage_config": None,
                "api_key_config": {"custom_endpoint": endpoint()},
                "llm_config": {"generation_model_name": model},
            }
        )


def test_explicit_conversion_preserves_effective_pin_and_opaque_metadata():
    source = {
        "storage_config": None,
        "api_key_config": {
            "custom_endpoint": {**endpoint(), "model": "openai/legacy", "opaque": 7},
            "openai": {"api_key": "different-embedding-key"},
        },
        "llm_config": {
            "generation_model_name": "ignored-generation",
            "should_run_model_name": "ignored-should-run",
            "pre_retrieval_model_name": "ignored-reformulator",
            "embedding_model_name": "openai/text-embedding-3-small",
            "opaque": {"keep": True},
        },
        "opaque": {"keep": True},
    }
    before = deepcopy(source)
    result = canonicalize_config_document(source)
    assert source == before
    assert result["api_key_config"] == {
        **source["api_key_config"],
        "custom_endpoint": {**endpoint(), "opaque": 7},
    }
    assert result["llm_config"] == {
        **source["llm_config"],
        "generation_model_name": "openai/legacy",
        "should_run_model_name": "openai/legacy",
        "pre_retrieval_model_name": "openai/legacy",
    }
    assert result["opaque"] == source["opaque"]
    assert canonicalize_config_document(result) == result


@pytest.mark.parametrize("model", [None, "", " ", 17, {}])
def test_conversion_rejects_invalid_legacy_model_without_leaking_values(model):
    source = {
        "storage_config": None,
        "api_key_config": {"custom_endpoint": {**endpoint(), "model": model}},
    }
    before = deepcopy(source)
    with pytest.raises(ValueError):
        canonicalize_config_document(source)
    assert source == before


@pytest.mark.parametrize("role", [r for r in ModelRole if r != ModelRole.EMBEDDING])
def test_gateway_role_inherits_existing_generation_choice(role):
    keys = APIKeyConfig.model_validate({"custom_endpoint": endpoint()})
    selected = resolve_model_name(
        role,
        site_var_value="ignored-provider-default",
        api_key_config=keys,
        llm_config=LLMConfig(generation_model_name="openai/gateway-model"),
    )
    assert selected == "openai/gateway-model"


@pytest.mark.parametrize(
    "role,field",
    [
        (ModelRole.SHOULD_RUN, "should_run_model_name"),
        (ModelRole.PRE_RETRIEVAL, "pre_retrieval_model_name"),
    ],
)
def test_gateway_retains_deliberate_role_choice(role, field):
    keys = APIKeyConfig.model_validate({"custom_endpoint": endpoint()})
    assert (
        resolve_model_name(
            role,
            api_key_config=keys,
            llm_config=LLMConfig.model_validate(
                {"generation_model_name": "openai/general", field: "openai/special"}
            ),
        )
        == "openai/special"
    )


@pytest.mark.parametrize("role", [None, ModelRole.GENERATION, ModelRole.SHOULD_RUN])
def test_gateway_dispatch_respects_selected_model_without_enabling_fallback(role):
    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/configured",
            api_key_config=APIKeyConfig.model_validate({"custom_endpoint": endpoint()}),
            fallback_models=["openai/not-qualified-on-this-gateway"],
        )
    )
    assert client._resolve_primary_model(None, role) == "openai/configured"
    assert client._resolve_ladder(model="openai/requested", model_role=role) == [
        "openai/requested"
    ]
    assert client._resolve_api_key("openai/requested") == (
        "test-gateway-key",
        "https://gateway.invalid/v1",
        None,
    )
    assert client._resolve_api_key(for_embedding=True) == (None, None, None)


@pytest.mark.parametrize("role", [ModelRole.GENERATION, ModelRole.EXTRACTION_AGENT])
def test_tool_capability_checks_use_the_same_gateway_model_as_dispatch(
    monkeypatch, role
):
    from reflexio.server.llm.tools import ToolRegistry, run_tool_loop

    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/gateway-selected",
            api_key_config=APIKeyConfig.model_validate({"custom_endpoint": endpoint()}),
        )
    )
    checked = []

    def no_tools(model):
        checked.append(model)
        return False

    monkeypatch.setattr("reflexio.server.llm.tools.supports_tool_calling", no_tools)
    monkeypatch.setattr(
        "litellm.get_model_info", lambda _model: {"supports_function_calling": False}
    )
    with pytest.raises(RuntimeError, match="gateway-selected lacks tool-calling"):
        run_tool_loop(
            client=client,
            messages=[],
            registry=ToolRegistry(),
            model_role=role,
            max_steps=1,
            ctx=None,
        )
    assert checked == [client.get_completion_request_contract(model_role=role)["model"]]


@pytest.mark.parametrize("enclosing", [False, True])
def test_retired_gateway_rejection_does_not_echo_credentials(enclosing):
    from reflexio.models.config_schema import CustomEndpointConfig

    private = {
        "api_key": "sensitive-gateway-credential",
        "api_base": "https://gateway.invalid/v1?token=private-token",
        "model": "openai/old",
    }
    model = APIKeyConfig if enclosing else CustomEndpointConfig
    with pytest.raises(ValidationError) as error:
        model.model_validate({"custom_endpoint": private} if enclosing else private)
    for rendered in (str(error.value), repr(error.value)):
        assert "sensitive-gateway-credential" not in rendered
        assert "private-token" not in rendered


@pytest.mark.parametrize("empty_registry", [True, False])
def test_opaque_gateway_tool_loop_attempts_native_protocol(monkeypatch, empty_registry):
    from pydantic import BaseModel

    from reflexio.server.llm._litellm_types import (
        CompletionResult,
        ModelProvenance,
        ToolCallingChatResponse,
    )
    from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop

    class Finish(BaseModel):
        answer: str

    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return CompletionResult(
            value=ToolCallingChatResponse(
                content="done", tool_calls=None, finish_reason="stop"
            ),
            provenance=ModelProvenance(model_name="openai/orcarouter/auto"),
        )

    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/orcarouter/auto",
            api_key_config=APIKeyConfig.model_validate({"custom_endpoint": endpoint()}),
        )
    )
    monkeypatch.setattr(client, "generate_chat_response_with_provenance", complete)
    registry = ToolRegistry(
        []
        if empty_registry
        else [Tool(name="finish", args_model=Finish, handler=lambda _args, _ctx: {})]
    )
    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "Run"}],
        registry=registry,
        model_role=ModelRole.EXTRACTION_AGENT,
        max_steps=1,
        ctx=None,
    )
    assert len(calls) == 1
    assert result.finished_reason != "error"
    assert (
        client.get_completion_request_contract(model_role=ModelRole.EXTRACTION_AGENT)[
            "model"
        ]
        == "openai/orcarouter/auto"
    )
