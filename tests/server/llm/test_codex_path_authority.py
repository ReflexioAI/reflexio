"""Obsolete native selection cannot silently change a route or executable."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)
from reflexio.server.llm.model_defaults import (
    ModelRole,
    detect_available_providers,
    resolve_model_name,
)
from reflexio.server.llm.providers import claude_code_provider as provider
from reflexio.server.llm.providers.claude_code_stream_parser import ParseResult

LEGACY = "CLAUDE_SMART_CODEX_PATH"
TARGET = "claude-code/default"


class Answer(BaseModel):
    answer: str


@pytest.fixture
def client_transport(monkeypatch):
    monkeypatch.setenv("CLAUDE_SMART_HOST", "codex")
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(provider.shutil, "which", lambda _name: None)
    monkeypatch.setattr(provider.litellm, "custom_provider_map", [])
    client = LiteLLMClient(
        LiteLLMConfig(model=TARGET, timeout=2, max_tokens=32, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"ok"}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="synthetic-response",
        _hidden_params={},
    )
    transport = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    return client, transport


@pytest.mark.parametrize("mode", ["plain", "structured", "validated"])
def test_retired_path_refuses_before_dispatch_or_paid_fallback(
    client_transport, monkeypatch, caplog, mode
):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-retired-target")
    client.config.fallback_models = ["gpt-4o-mini"]
    options: dict[str, Any] = {} if mode == "plain" else {"response_format": Answer}
    if mode == "validated":
        options["structured_output_validator"] = lambda _value: []
    with pytest.raises(ValueError, match=LEGACY) as failure:
        client.generate_chat_response(
            [{"role": "user", "content": "synthetic evidence"}], **options
        )
    transport.assert_not_called()
    assert "private-retired-target" not in str(failure.value) + caplog.text


@pytest.mark.parametrize("reached", [False, True])
def test_only_reached_native_fallback_checks_legacy_selection(
    client_transport, monkeypatch, reached
):
    client, transport = client_transport
    client.config.model = "gpt-4o-mini"
    client.config.fallback_models = [TARGET, "gpt-4o"]
    monkeypatch.setenv(LEGACY, "private-retired-target")
    if reached:
        transport.side_effect = LiteLLMClientError("synthetic primary outage")
        with pytest.raises(ValueError, match=LEGACY):
            client.generate_response("synthetic evidence")
    else:
        client.generate_response("synthetic evidence")
    assert transport.call_count == 1


@pytest.mark.parametrize("legacy", [None, "", " \t"])
def test_unset_or_blank_old_key_has_no_migration(client_transport, monkeypatch, legacy):
    client, transport = client_transport
    if legacy is None:
        monkeypatch.delenv(LEGACY, raising=False)
    else:
        monkeypatch.setenv(LEGACY, legacy)
    assert not provider.has_pending_codex_migration()
    client.generate_response("synthetic evidence")
    transport.assert_called_once()


@pytest.mark.parametrize("host", ["claude-code", "opencode"])
def test_other_hosts_ignore_unused_old_codex_key(client_transport, monkeypatch, host):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("CLAUDE_SMART_HOST", host)
    assert not provider.has_pending_codex_migration()
    client.generate_response("synthetic evidence")
    transport.assert_called_once()


def test_explicit_http_transport_ignores_native_settings(client_transport, monkeypatch):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/missing/codex-claude-compat")
    client.generate_response(
        "synthetic evidence",
        custom_llm_provider="openai",
        api_base="http://127.0.0.1:1/v1",
    )
    transport.assert_called_once()


@pytest.mark.parametrize("selection", ["old-key", "missing-bridge"])
@pytest.mark.parametrize(
    "role", [role for role in ModelRole if role != ModelRole.EMBEDDING]
)
def test_unavailable_obsolete_native_intent_never_auto_selects_http(
    client_transport, monkeypatch, selection, role
):
    client, transport = client_transport
    if selection == "old-key":
        monkeypatch.setenv(LEGACY, "private-retired-target")
    else:
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/missing/codex-claude-compat")
    assert not provider.is_claude_code_available()
    assert provider.has_pending_codex_migration()
    assert detect_available_providers()[0] == "claude-code"
    assert resolve_model_name(
        role=role, site_var_value=None, config_override=None, api_key_config=None
    ).startswith("claude-code/")
    with pytest.raises(ValueError, match="CLAUDE_SMART"):
        client.generate_response("synthetic evidence", model_role=role)
    transport.assert_not_called()
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "0")
    assert "claude-code" not in detect_available_providers()


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize("suffix", ["", ".cmd", ".js"])
def test_retired_bridge_refuses_across_hosts_without_dispatch(
    client_transport, monkeypatch, tmp_path, exists, host, suffix
):
    client, transport = client_transport
    bridge = tmp_path / f"codex-claude-compat{suffix}"
    if exists:
        bridge.write_text("must never execute")
        bridge.chmod(0o755)
    monkeypatch.setenv("CLAUDE_SMART_HOST", host)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(bridge))
    with pytest.raises(ValueError, match="bridge.*retired"):
        client.generate_response("synthetic evidence")
    transport.assert_not_called()
    dispatch = Mock(side_effect=AssertionError("retired bridge executed"))
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    with pytest.raises(ValueError, match="bridge.*retired"):
        provider.ClaudeCodeLLM().completion()
    dispatch.assert_not_called()


@pytest.mark.parametrize("valid", [False, True])
def test_only_valid_canonical_env_override_dominates_old_key(
    client_transport, monkeypatch, tmp_path, valid
):
    client, transport = client_transport
    canonical = tmp_path / "native codex"
    if valid:
        canonical.write_text("owned executable stand-in")
        canonical.chmod(0o755)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(canonical))
    monkeypatch.setenv(LEGACY, "private-unused-target")
    if valid:
        assert not provider.has_pending_codex_migration()
        assert provider._resolve_cli_path() == str(canonical)
        client.generate_response("synthetic evidence")
        transport.assert_called_once()
    else:
        with pytest.raises(ValueError, match=LEGACY):
            client.generate_response("synthetic evidence")
        transport.assert_not_called()


def test_explicit_constructor_preserves_its_existing_path(monkeypatch):
    monkeypatch.setenv("CLAUDE_SMART_HOST", "codex")
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/missing/codex-claude-compat")
    dispatch = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    provider.ClaudeCodeLLM(cli_path="/explicit/native-codex").completion()
    assert dispatch.call_args.kwargs["cli_path"] == "/explicit/native-codex"


def test_registration_reserves_pending_intent_without_resolving_old_value(
    client_transport, monkeypatch
):
    monkeypatch.setenv(LEGACY, "private-retired-target")
    monkeypatch.setattr(provider, "_REGISTERED", False)
    monkeypatch.setattr(provider, "_HANDLER", None)
    monkeypatch.setattr(provider.litellm, "custom_provider_map", [])
    assert provider._resolve_cli_path() is None
    assert provider.register_if_enabled()
    assert provider._HANDLER is not None
    with pytest.raises(ValueError, match=LEGACY):
        provider._HANDLER.completion()
