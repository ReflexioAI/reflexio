"""Obsolete native selection cannot silently change a route or executable."""

import importlib
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reflexio.server.llm._native_model_ref import NativeCLISelectionError
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
from reflexio.server.llm.providers import native_cli_provider as provider
from reflexio.server.llm.providers.claude_code_stream_parser import ParseResult

LEGACY = "CLAUDE_SMART_CODEX_PATH"
TARGET = "local-cli/codex"


class Answer(BaseModel):
    answer: str


@pytest.fixture
def client_transport(monkeypatch):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", TARGET)
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(provider.shutil, "which", lambda _name, **_kwargs: None)
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
    with pytest.raises(NativeCLISelectionError, match=LEGACY) as failure:
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
        with pytest.raises(NativeCLISelectionError, match=LEGACY):
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
    assert provider._codex_migration_error("codex") is None
    client.generate_response("synthetic evidence")
    transport.assert_called_once()


@pytest.mark.parametrize("host", ["claude-code", "opencode"])
def test_other_hosts_ignore_unused_old_codex_key(client_transport, monkeypatch, host):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + host)
    client.config.model = "local-cli/" + host
    assert provider._codex_migration_error(host) is None
    client.generate_response("synthetic evidence")
    transport.assert_called_once()


def test_explicit_http_transport_ignores_native_settings(client_transport, monkeypatch):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/missing/codex-claude-compat")
    client.config.model = "claude-code/opaque-http-label"
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
    assert detect_available_providers()[0] == "local-cli"
    assert resolve_model_name(
        role=role, site_var_value=None, config_override=None, api_key_config=None
    ).startswith("local-cli/")
    with pytest.raises(NativeCLISelectionError, match="CLAUDE_SMART"):
        client.generate_response("synthetic evidence", model_role=role)
    transport.assert_not_called()
    monkeypatch.delenv("REFLEXIO_NATIVE_CLI_MODEL")
    assert "local-cli" not in detect_available_providers()


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
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + host)
    client.config.model = "local-cli/" + host
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(bridge))
    with pytest.raises(NativeCLISelectionError, match="bridge.*retired"):
        client.generate_response("synthetic evidence")
    transport.assert_not_called()
    dispatch = Mock(side_effect=AssertionError("retired bridge executed"))
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    with pytest.raises(NativeCLISelectionError, match="bridge.*retired"):
        provider.NativeCLILLM().completion(model=client.config.model)
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
        assert provider._codex_migration_error("codex") is None
        assert provider._resolve_cli_path("codex") == str(canonical)
        client.generate_response("synthetic evidence")
        transport.assert_called_once()
    else:
        with pytest.raises(NativeCLISelectionError, match=LEGACY):
            client.generate_response("synthetic evidence")
        transport.assert_not_called()


def test_explicit_constructor_preserves_its_existing_path(monkeypatch):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", TARGET)
    monkeypatch.setenv(LEGACY, "private-unused-target")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/missing/codex-claude-compat")
    dispatch = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    provider.NativeCLILLM(cli_path="/explicit/native-codex").completion(model=TARGET)
    assert dispatch.call_args.kwargs["cli_path"] == "/explicit/native-codex"


def test_registration_reserves_pending_intent_without_resolving_old_value(
    client_transport, monkeypatch
):
    monkeypatch.setenv(LEGACY, "private-retired-target")
    monkeypatch.setattr(provider, "_HANDLER", None)
    monkeypatch.setattr(provider.litellm, "custom_provider_map", [])
    assert provider._resolve_cli_path("codex") is None
    assert provider.register_native_provider()
    assert provider._HANDLER is not None
    with pytest.raises(NativeCLISelectionError, match=LEGACY):
        provider._HANDLER.completion(model=TARGET)


@pytest.mark.parametrize(
    "module",
    [
        "reflexio.server.services.offline_tuner.open_world.shared_compaction",
        "reflexio.server.services.offline_tuner.open_world.analyst",
        "reflexio.server.services.offline_tuner.open_world.rule_evidence_v2",
    ],
)
def test_native_transport_consumers_remain_importable(module):
    importlib.import_module(module)


@pytest.fixture
def owned_path(monkeypatch, tmp_path):
    """A codex target with an owned PATH directory and working directory."""
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", TARGET)
    monkeypatch.setenv(LEGACY, "private-unused-target")
    (tmp_path / "bin").mkdir()
    (tmp_path / "work").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.chdir(tmp_path / "work")
    return tmp_path


def _executable(path):
    path.write_text("owned executable stand-in")
    path.chmod(0o755)
    return path


def _native_name(name):
    """The file a bare ``name`` resolves to on this platform's PATH."""
    return f"{name}.exe" if os.name == "nt" else name


def test_a_bare_canonical_override_on_path_dominates_the_old_key(
    owned_path, monkeypatch
):
    on_path = _executable(owned_path / "bin" / _native_name("native-codex"))
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "native-codex")
    assert provider._codex_migration_error("codex") is None
    assert provider._resolve_cli_path("codex") == str(on_path)


def test_a_working_directory_only_override_fails_closed(owned_path, monkeypatch):
    # Launch would never run this file, so it cannot dominate the old key.
    _executable(owned_path / "work" / _native_name("native-codex"))
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "native-codex")
    assert provider._resolve_cli_path("codex") is None
    assert LEGACY in (provider._codex_migration_error("codex") or "")


def test_windows_bare_names_resolve_only_through_path(owned_path, monkeypatch):
    monkeypatch.setattr(provider, "_is_windows", lambda: True)
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    monkeypatch.delenv(LEGACY)
    # Shims in the working directory, which Windows `shutil.which` searches first.
    for name in ("claude.cmd", "claude.exe", "codex.cmd", "native-codex.cmd"):
        _executable(owned_path / "work" / name)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "native-codex")
    assert provider._resolve_cli_override_path("native-codex") == "native-codex"
    assert provider._resolve_cli_path("codex") is None
    assert provider._resolve_cli_path("claude-code") is None
    # The same names on PATH are found there, extensionless name included.
    on_path = _executable(owned_path / "bin" / "native-codex.cmd")
    assert provider._resolve_cli_path("codex") == str(on_path)
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH")
    claude = _executable(owned_path / "bin" / "claude.exe")
    assert provider._resolve_cli_path("claude-code") == str(claude)
    # An empty PATH entry is an implicit working-directory search.
    monkeypatch.setenv("PATH", os.pathsep)
    assert provider._resolve_cli_path("claude-code") is None


def test_windows_explicit_relative_override_stays_in_its_directory(
    owned_path, monkeypatch
):
    # `.\\native-codex` names the working directory; suffix probing must not
    # turn it into the bare `native-codex.exe` and search PATH instead.
    monkeypatch.setattr(provider, "_is_windows", lambda: True)
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    monkeypatch.delenv(LEGACY)
    local = _executable(owned_path / "work" / "native-codex.exe")
    _executable(owned_path / "bin" / "native-codex.exe")
    override = "." + os.sep + "native-codex"
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", override)
    assert provider._resolve_cli_override_path(override) == str(local.absolute())
    assert provider._resolve_cli_path("codex") == str(local.absolute())


@pytest.mark.skipif(os.name == "nt", reason="POSIX PATH semantics")
@pytest.mark.parametrize("path", ["{bin}:", ":{bin}", "{bin}::", ":"])
def test_an_empty_posix_path_entry_never_searches_the_working_directory(
    owned_path, monkeypatch, path
):
    monkeypatch.delenv(LEGACY)
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
    _executable(owned_path / "work" / "codex")
    monkeypatch.setenv("PATH", path.format(bin=owned_path / "bin"))
    assert provider._resolve_cli_path("codex") is None
    on_path = _executable(owned_path / "bin" / "codex")
    if path != ":":
        assert provider._resolve_cli_path("codex") == str(on_path)
