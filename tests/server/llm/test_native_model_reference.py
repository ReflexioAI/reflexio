"""Native request identity cannot grant access to the operator's CLI account."""

import asyncio
import json
import multiprocessing
import pickle
import socket
import time
from unittest.mock import Mock, patch

import litellm
import pytest
from litellm.llms.custom_llm import CustomLLM

from reflexio.server.llm import _litellm_subprocess as worker
from reflexio.server.llm import _native_model_ref as authority
from reflexio.server.llm._litellm_types import ProviderRequestGuardError
from reflexio.test_support.llm_mock import unpatched_litellm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize(
    "model", [None, "default", "owned/provider/model#high", "λ model"]
)
def test_reference_preserves_host_default_and_opaque_model(host, model):
    value = f"local-cli/{host}" + ("" if model is None else "/" + model)
    parsed = authority.parse_native_model_ref(value)
    assert parsed.host == host and parsed.model == model and parsed.reference == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "local-cli/",
        "local-cli/foreign/private-canary",
        "local-cli/codex/",
        "local-cli/codex/  ",
        "local-cli/codex/a\x00b",
        "claude-code/private-canary",
    ],
)
def test_invalid_reference_refuses_without_values(value):
    with pytest.raises(authority.NativeCLISelectionError) as error:
        authority.parse_native_model_ref(value)
    assert "private-canary" not in str(error.value)


@pytest.mark.parametrize("setting", [None, "", "  "])
def test_unconfigured_operator_grants_no_native_execution(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv(authority.ENV_NATIVE_MODEL, raising=False)
    else:
        monkeypatch.setenv(authority.ENV_NATIVE_MODEL, setting)
    assert authority.configured_native_model() is None
    with pytest.raises(authority.NativeCLISelectionError):
        authority.authorize_native_model("local-cli/codex")
    assert authority.authorize_native_model("openai/owned-http") is None


@pytest.mark.parametrize(
    "requested,provider",
    [
        ("local-cli/codex/owned", None),
        ("codex/owned", "local-cli"),
        ("local-cli/codex/owned", "local-cli"),
    ],
)
def test_exact_operator_reference_is_authorized(monkeypatch, requested, provider):
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    selected = authority.authorize_native_model(requested, provider)
    assert selected == authority.NativeModelRef("codex", "owned")


@pytest.mark.parametrize(
    "requested,provider",
    [
        ("local-cli/codex/other", None),
        ("local-cli/opencode/owned", None),
        ("local-cli/codex/owned", "openai"),
        ("claude-code/default", None),
        ("default", "claude-code"),
    ],
)
def test_other_native_intent_or_transport_cannot_bypass_grant(
    monkeypatch, requested, provider
):
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    with pytest.raises(authority.NativeCLISelectionError):
        authority.authorize_native_model(requested, provider)


def test_http_gateway_opaque_legacy_label_does_not_become_native(monkeypatch):
    monkeypatch.delenv(authority.ENV_NATIVE_MODEL, raising=False)
    assert (
        authority.authorize_native_model("claude-code/opaque-http-label", "openai")
        is None
    )


def test_thread_rejection_is_request_local_and_not_lost_by_wrapping(monkeypatch):
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/allowed")

    async def invoke():
        try:
            await asyncio.to_thread(
                authority.authorize_native_model, "local-cli/codex/denied"
            )
        except authority.NativeCLISelectionError:
            raise RuntimeError("third-party exception wrapper") from None

    with authority.native_request_authority() as state:
        with pytest.raises(RuntimeError, match="third-party"):
            asyncio.run(invoke())
        with pytest.raises(authority.NativeCLISelectionError):
            state.raise_if_rejected()
    with authority.native_request_authority() as state:
        assert authority.authorize_native_model("local-cli/codex/allowed") is not None
        state.raise_if_rejected()


class _OwnedHandler(CustomLLM):
    def completion(self, *args, model=None, **kwargs):
        assert isinstance(model, str)
        authority.authorize_native_model("local-cli/codex/denied")
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "owned reply"},
                    "finish_reason": "stop",
                }
            ]
        )


def _captured_worker(params, result_queue, deadline):
    # Use the actual pinned custom-provider wrapper, never a model/network.
    litellm.custom_provider_map = [
        {"provider": "local-cli", "custom_handler": _OwnedHandler()}
    ]
    with (
        unpatched_litellm(),
        patch.object(
            socket.socket, "connect", side_effect=AssertionError("network forbidden")
        ),
    ):
        worker._litellm_completion_worker(params, result_queue, deadline)


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_actual_litellm_wrapper_and_process_snapshot_preserve_terminal_rejection(
    monkeypatch, method
):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} unavailable")
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/allowed")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.delenv("OPENCLAW_SMART_USE_LOCAL_CLI", raising=False)
    context = multiprocessing.get_context(method)
    result_queue = context.Queue()
    process = context.Process(
        target=_captured_worker,
        args=(
            {
                "model": "local-cli/codex/allowed",
                "messages": [{"role": "user", "content": "owned synthetic"}],
                "num_retries": 0,
            },
            result_queue,
            time.monotonic() + 20,
        ),
    )
    process.start()
    try:
        status, snapshot = result_queue.get(timeout=20)
        assert status == "error" and snapshot.request_guard_error
        restored = pickle.loads(pickle.dumps(snapshot))  # noqa: S301 — owned child snapshot round-trip only.
        with pytest.raises(ProviderRequestGuardError, match="operator-configured"):
            worker._raise_completion_error(restored)
    finally:
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join(3)
        result_queue.close()
        result_queue.join_thread()


def test_provider_error_text_cannot_forge_a_terminal_rejection():
    error = RuntimeError("NativeCLISelectionError operator-configured target")
    snapshot = worker._snapshot_completion_error(error, {"model": "openai/owned-http"})
    assert not snapshot.request_guard_error
    assert json.loads(json.dumps(snapshot.__dict__))["request_guard_error"] is False


def test_typed_guard_survives_snapshot_and_registration_failure(monkeypatch):
    monkeypatch.setattr(
        worker,
        "_register_local_cli_providers",
        lambda: (_ for _ in ()).throw(
            ProviderRequestGuardError("owned registration rejected")
        ),
    )
    result_queue = Mock()
    worker._litellm_completion_worker(
        {"model": "openai/owned-http"}, result_queue, time.monotonic() + 2
    )
    status, snapshot = result_queue.put.call_args.args[0]
    assert status == "error" and snapshot.request_guard_error
    with pytest.raises(ProviderRequestGuardError, match="registration rejected"):
        worker._raise_completion_error(snapshot)


@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize(
    "physical_model", [None, "default", "owned/provider/model#high"]
)
def test_configured_profile_supplies_every_text_role_below_role_choices(
    monkeypatch, host, physical_model
):
    from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

    reference = "local-cli/" + host + ("/" + physical_model if physical_model else "")
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, reference)
    for role in ModelRole:
        if role == ModelRole.EMBEDDING:
            assert resolve_model_name(role) == "local/minilm-l6-v2"
            continue
        assert resolve_model_name(role) == reference
        assert (
            resolve_model_name(role, site_var_value="openai/site-role")
            == "openai/site-role"
        )
        assert (
            resolve_model_name(
                role,
                config_override="openai/org-role",
                site_var_value="openai/site-role",
            )
            == "openai/org-role"
        )


@pytest.mark.parametrize("old_flag", ["1", "true"])
def test_unadopted_enabled_native_intent_cannot_select_paid_api(monkeypatch, old_flag):
    from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

    monkeypatch.delenv(authority.ENV_NATIVE_MODEL, raising=False)
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", old_flag)
    monkeypatch.setenv("OPENAI_API_KEY", "owned-fake-key")
    with pytest.raises(authority.NativeCLISelectionError, match="adoption"):
        resolve_model_name(ModelRole.GENERATION)
    assert (
        resolve_model_name(ModelRole.GENERATION, config_override="openai/explicit-role")
        == "openai/explicit-role"
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("CLAUDE_SMART_USE_LOCAL_CLI", "0"),
        ("CLAUDE_SMART_USE_LOCAL_CLI", "1"),
        ("CLAUDE_SMART_CLI_MODEL", "unadopted-physical-model"),
    ],
)
def test_new_profile_with_retired_selectors_requires_deliberate_adoption(
    monkeypatch, setting, value
):
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    monkeypatch.setenv(setting, value)
    with pytest.raises(authority.NativeCLISelectionError, match="adoption"):
        authority.authorize_native_model("local-cli/codex/owned")
    assert authority.authorize_native_model("openai/explicit-http") is None


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("role_override", [False, True])
def test_native_intent_is_guarded_before_gateway_or_role_rewriting(
    monkeypatch, gateway, role_override
):
    from reflexio.models.config_schema import APIKeyConfig, CustomEndpointConfig
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.llm.model_defaults import ModelRole

    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/allowed")
    keys = (
        APIKeyConfig(
            custom_endpoint=CustomEndpointConfig.model_validate(
                {
                    "api_key": "owned-fake-key",
                    "api_base": "https://gateway.example/v1",
                }
            )
        )
        if gateway
        else None
    )
    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/owned-http",
            api_key_config=keys,
            max_tokens=32,
            fallback_models=["openai/other-http"],
        )
    )
    dispatch = patch.object(client, "_completion_with_hard_timeout")
    with dispatch as physical, pytest.raises(authority.NativeCLISelectionError):
        client.generate_response(
            "synthetic owned input",
            model="local-cli/codex/denied",
            model_role=ModelRole.EVALUATION if role_override else None,
        )
    physical.assert_not_called()
    # An authorized explicit native target retains its identity under both paths.
    assert (
        client._resolve_primary_model(
            "local-cli/codex/allowed", ModelRole.EVALUATION if role_override else None
        )
        == "local-cli/codex/allowed"
    )


def test_independent_openclaw_grant_remains_usable(monkeypatch):
    from reflexio.server.llm import model_defaults
    from reflexio.server.llm.providers import openclaw_provider

    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    monkeypatch.setenv("OPENCLAW_SMART_USE_LOCAL_CLI", "1")
    monkeypatch.setattr(openclaw_provider, "is_openclaw_available", lambda: True)
    providers = model_defaults.detect_available_providers()
    assert providers[:2] == ["local-cli", "openclaw"]
    assert authority.authorize_native_model("openclaw/default") is None


def test_backend_profile_does_not_read_client_coding_host(monkeypatch):
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    target = authority.authorize_native_model("local-cli/codex/owned")
    assert target is not None and target.host == "codex"


def test_model_remainder_is_not_stripped_by_operator_reader(monkeypatch):
    reference = "local-cli/codex/ owned/model "
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, reference)
    configured = authority.configured_native_model()
    assert configured is not None and configured.reference == reference
    authorized = authority.authorize_native_model(reference)
    assert authorized is not None and authorized.model == " owned/model "


@pytest.mark.parametrize("requested", ["codex/owned", "local-cli/codex/owned"])
def test_explicit_native_provider_normalizes_before_capability_and_credentials(
    monkeypatch, requested
):
    from reflexio.models.config_schema import APIKeyConfig, CustomEndpointConfig
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.llm.model_defaults import ModelRole

    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    client = LiteLLMClient(
        LiteLLMConfig(
            model="openai/owned-http",
            max_tokens=32,
            api_key_config=APIKeyConfig(
                custom_endpoint=CustomEndpointConfig.model_validate(
                    {
                        "api_key": "owned-gateway-key",
                        "api_base": "https://gateway.example/v1",
                    }
                )
            ),
        )
    )
    params, *_ = client._build_completion_params(
        [],
        model=requested,
        custom_llm_provider="local-cli",
        model_role=ModelRole.EVALUATION,
    )
    assert params["model"] == "local-cli/codex/owned"
    assert params["custom_llm_provider"] == "local-cli"
    assert not params.get("api_key") and not params.get("api_base")


@pytest.mark.parametrize(
    "profile", [None, "local-cli/opencode/owned", "local-cli/codex/other"]
)
def test_actual_native_handler_cannot_bypass_operator_grant(monkeypatch, profile):
    import sys

    from reflexio.server.llm.providers import native_cli_provider

    if profile is None:
        monkeypatch.delenv(authority.ENV_NATIVE_MODEL, raising=False)
    else:
        monkeypatch.setenv(authority.ENV_NATIVE_MODEL, profile)
    dispatch = Mock(side_effect=AssertionError("unauthorized physical dispatch"))
    monkeypatch.setattr(native_cli_provider, "_run_cli_stream", dispatch)
    with pytest.raises(authority.NativeCLISelectionError):
        native_cli_provider.NativeCLILLM(cli_path=sys.executable).completion(
            model="codex/owned",
            messages=[{"role": "user", "content": "owned synthetic"}],
        )
    dispatch.assert_not_called()


@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
def test_actual_handler_uses_operator_host_and_explicit_model(monkeypatch, host):
    import sys

    from reflexio.server.llm.providers import native_cli_provider
    from reflexio.server.llm.providers.claude_code_stream_parser import ParseResult

    reference = f"local-cli/{host}/owned/provider/model#high"
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, reference)
    # Coding-session identity is independent of the backend runtime.
    monkeypatch.setenv("CLAUDE_SMART_HOST", "different-client-host")
    dispatch = Mock(return_value=ParseResult(True, "owned reply"))
    monkeypatch.setattr(native_cli_provider, "_run_cli_stream", dispatch)
    response = native_cli_provider.NativeCLILLM(cli_path=sys.executable).completion(
        model=reference,
        messages=[{"role": "user", "content": "owned synthetic"}],
        timeout=1,
    )
    assert dispatch.call_args.kwargs["host"] == host
    assert dispatch.call_args.kwargs["model"] == "owned/provider/model#high"
    assert response.model == reference
    assert response._hidden_params.get("reflexio_served_model") is None


def test_native_only_startup_does_not_open_an_api_key_wizard(monkeypatch, tmp_path):
    from reflexio.cli.commands.services import _ensure_llm_configured
    from reflexio.server.llm.model_defaults import provider_env_vars

    for name in provider_env_vars():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    # No installed native program or paid request is needed to preserve intent.
    with patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as wizard:
        _ensure_llm_configured(tmp_path / "owned.env")
    wizard.assert_not_called()


@pytest.mark.parametrize("host", ["claude-code", "codex"])
@pytest.mark.parametrize("suffix", ["", ".cmd", ".js"])
def test_known_bridge_cannot_override_operator_runtime(monkeypatch, host, suffix):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.llm.providers import native_cli_provider

    reference = "local-cli/" + host
    bridge = "/owned/opencode-claude-compat" + suffix
    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, reference)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", bridge)
    client = LiteLLMClient(
        LiteLLMConfig(
            model=reference,
            fallback_models=["openai/owned-http"],
            max_tokens=32,
        )
    )
    with (
        patch.object(client, "_completion_with_hard_timeout") as dispatch,
        pytest.raises(authority.NativeCLISelectionError, match="bridge runtime"),
    ):
        client.generate_response("owned synthetic")
    dispatch.assert_not_called()
    with (
        patch.object(native_cli_provider, "_run_cli_stream") as dispatch,
        pytest.raises(authority.NativeCLISelectionError, match="bridge runtime"),
    ):
        native_cli_provider.NativeCLILLM(cli_path=bridge).completion(model=reference)
    dispatch.assert_not_called()


def test_explicit_http_legacy_label_preserves_opaque_message_wire_shape(monkeypatch):
    from reflexio.models.config_schema import AnthropicConfig, APIKeyConfig
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    monkeypatch.setenv(authority.ENV_NATIVE_MODEL, "local-cli/codex/owned")
    client = LiteLLMClient(
        LiteLLMConfig(
            model="claude-code/opaque-http-label",
            api_key_config=APIKeyConfig(
                anthropic=AnthropicConfig(api_key="owned-foreign-key")
            ),
        )
    )
    messages = [
        {"role": "system", "content": "owned synthetic system"},
        {"role": "user", "content": "owned synthetic input"},
    ]
    params, *_ = client._build_completion_params(
        messages,
        custom_llm_provider="openai",
        api_base="https://gateway.example/v1",
    )
    assert params["model"] == "claude-code/opaque-http-label"
    assert not params.get("api_key")
    assert "owned-foreign-key" not in repr(params)
    assert params["custom_llm_provider"] == "openai"
    assert params["messages"] == messages
    # Preserving an explicit HTTP label does not retain a native execution alias.
    with pytest.raises(authority.NativeCLISelectionError, match="Legacy native"):
        client._build_completion_params(messages)
