"""Native transport contracts; physical probes run in the owner test module."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from reflexio.server.llm.providers import _opencode_cli as native
from reflexio.server.llm.providers import claude_code_provider as provider
from reflexio.server.llm.providers.claude_code_stream_parser import (
    ParseResult,
    classify_stall,
)


@pytest.mark.unit
@pytest.mark.parametrize("setting", ["path", "timeout"])
def test_invalid_native_configuration_logs_names_without_values(
    monkeypatch, caplog, setting
):
    if setting == "path":
        monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "private-canary-not-an-executable")
        monkeypatch.setattr(provider.shutil, "which", lambda _name: None)
        assert provider._resolve_cli_path() is None
    else:
        monkeypatch.setenv(
            "CLAUDE_SMART_CLI_TIMEOUT", "private-canary-invalid-duration"
        )
        provider.ClaudeCodeLLM()._timeout()
    assert "private-canary" not in caplog.text
    assert "CLAUDE_SMART_CLI_" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("shape", ["part", "text", "result"])
def test_native_text_shapes_and_fenced_unicode(shape):
    text = '```json\n{"answer":"λ"}\n```'
    event = (
        {"type": "text", "part": {"type": "text", "text": text}}
        if shape == "part"
        else {"type": shape, shape: text}
    )
    result = native.parse_output(json.dumps(event), 0, "private-stderr")
    assert result.success and result.terminal_text == '{"answer":"λ"}'
    assert result.cli_binary == "opencode"
    assert result.served_model is None and result.served_provider is None
    assert result.stderr_text == ""


@pytest.mark.unit
@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize(
    "error",
    [
        None,
        "private-canary",
        {"data": {"message": "not logged in private-canary resets 9:00am"}},
    ],
)
def test_error_event_defeats_partial_output_and_redacts_every_sink(
    monkeypatch, caplog, code, error
):
    raw = (
        json.dumps({"type": "text", "text": "private-canary"})
        + "\n"
        + json.dumps({"type": "error", "error": error})
    )
    result = native.parse_output(raw, code, "private-canary")
    assert not result.success and result.terminal_text == ""
    storage = Mock()
    handler = provider.ClaudeCodeLLM(cli_path=sys.executable, storage=storage)
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    monkeypatch.setattr(provider, "_run_cli_stream", Mock(return_value=result))
    with pytest.raises(provider.ClaudeCodeCLIError) as failure:
        handler.completion(
            messages=[{"role": "user", "content": "synthetic"}], timeout=1
        )
    sinks = str(failure.value) + caplog.text + repr(storage.mock_calls)
    assert "private-canary" not in sinks
    if isinstance(error, dict):
        assert classify_stall(result) == "auth_error"
        saved = storage.upsert_stall_state.call_args.kwargs
        assert saved["reason"] == "auth_error" and saved["reset_estimate"] is not None
    else:
        storage.upsert_stall_state.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "signal,reason",
    [("credit balance is too low", "billing_error"), ("not logged in", "auth_error")],
)
def test_stderr_classification_is_metadata_only(caplog, signal, reason):
    result = native.parse_output("", 1, signal + " private-canary resets 9:00am")
    assert classify_stall(result) == reason
    assert result.reset_estimate is not None
    assert "private-canary" not in caplog.text + repr(result)


@pytest.mark.unit
def test_generated_billing_words_are_not_failure_evidence():
    result = native.parse_output(
        json.dumps({"type": "text", "text": "credit balance is too low"}), 1, ""
    )
    assert not result.success and classify_stall(result) is None
    assert result.terminal_text == ""


@pytest.mark.unit
@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize(
    "bridge",
    [
        "opencode-claude-compat",
        "opencode-claude-compat.cmd",
        "opencode-claude-compat.js",
    ],
)
def test_opencode_bridges_keep_claude_protocol_and_legacy_cap(
    monkeypatch, host, bridge
):
    monkeypatch.setenv("CLAUDE_SMART_HOST", host)
    monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", "7")
    legacy = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_claude_stream", legacy)
    monkeypatch.setattr(
        native, "run_opencode", Mock(side_effect=AssertionError("bridge misrouted"))
    )
    handler = provider.ClaudeCodeLLM(cli_path=f"/plugin/{bridge}")
    response = handler.completion(timeout=0.25)
    assert response.choices[0].message.content == "ok"
    assert legacy.call_args.kwargs["timeout_seconds"] == 7
    assert response._hidden_params["reflexio_cli_binary"] == bridge.split("-")[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "budget,cap,constructor,expected",
    [(0.25, None, None, 0.25), (5, "2", None, 2), (5, "2", 1, 1), (0.25, "2", 1, 0.25)],
)
def test_native_allowance_has_no_implicit_default_and_preserves_explicit_caps(
    monkeypatch, budget, cap, constructor, expected
):
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    if cap is None:
        monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", cap)
    handler = provider.ClaudeCodeLLM(
        cli_path=sys.executable, timeout_seconds=constructor
    )
    monkeypatch.setattr(
        handler, "_timeout", Mock(side_effect=AssertionError("provider default used"))
    )
    dispatch = Mock(return_value=ParseResult(True, "ok", cli_binary="opencode"))
    monkeypatch.setattr(native, "run_opencode", dispatch)
    handler.completion(timeout=budget)
    assert dispatch.call_args.kwargs["timeout_seconds"] == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "timeout", [None, True, "3", 0, -1, float("inf"), float("nan"), 2**1024]
)
def test_native_invalid_budget_cannot_launch(monkeypatch, timeout):
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    dispatch = Mock()
    monkeypatch.setattr(native, "run_opencode", dispatch)
    with pytest.raises(ValueError, match="finite positive timeout"):
        provider.ClaudeCodeLLM(cli_path=sys.executable).completion(timeout=timeout)
    dispatch.assert_not_called()


@pytest.mark.unit
def test_native_async_handler_keeps_the_selected_budget(monkeypatch):
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    dispatch = Mock(return_value=ParseResult(True, "ok", cli_binary="opencode"))
    monkeypatch.setattr(native, "run_opencode", dispatch)
    response = asyncio.run(
        provider.ClaudeCodeLLM(cli_path=sys.executable).acompletion(timeout=0.25)
    )
    assert response.choices[0].message.content == "ok"
    assert dispatch.call_args.kwargs["timeout_seconds"] == 0.25


@pytest.mark.unit
@pytest.mark.parametrize("through_path", [False, True])
def test_native_relative_executable_is_anchored_before_changing_cwd(
    monkeypatch, tmp_path, through_path
):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bin with spaces" / "opencode.exe"
    executable.parent.mkdir()
    executable.touch()
    selected = []

    def launch(argv, **kwargs):
        selected.append((argv, kwargs))
        assert Path(kwargs["cwd"]) != tmp_path
        return subprocess.CompletedProcess(argv, 0, '{"type":"text","text":"ok"}', "")

    monkeypatch.setattr(native, "run_native_cli", launch)
    relative = str(executable.relative_to(tmp_path))
    if through_path:
        monkeypatch.setattr(native.shutil, "which", lambda _name: relative)
    native.run_opencode(
        cli_path="opencode" if through_path else relative,
        system_prompt="s",
        dialogue="p",
        model=None,
        timeout_seconds=1,
    )
    assert selected[0][0][0] == str(executable)
    assert 0 < selected[0][1]["timeout"] <= 1
    assert not Path(selected[0][1]["cwd"]).exists()


@pytest.mark.unit
def test_bridge_symlink_is_discriminated_before_native_path_canonicalization(
    monkeypatch, tmp_path
):
    if os.name == "nt":
        pytest.skip("POSIX symlink name contract; native Windows exe tested separately")
    bridge = tmp_path / "opencode-claude-compat"
    bridge.symlink_to(sys.executable)
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    legacy = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_claude_stream", legacy)
    provider.ClaudeCodeLLM(cli_path=str(bridge)).completion(timeout=0.25)
    assert legacy.call_args.kwargs["cli_path"] == str(bridge)


@pytest.mark.unit
def test_setup_failure_cannot_dispatch_or_expose_raw_error(monkeypatch):
    monkeypatch.setattr(
        native.tempfile, "mkdtemp", Mock(side_effect=OSError("private-canary"))
    )
    dispatch = Mock()
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    with pytest.raises(native.OpenCodeCLIError) as failure:
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            timeout_seconds=1,
        )
    assert "private-canary" not in str(failure.value)
    dispatch.assert_not_called()


@pytest.mark.unit
def test_decode_failure_is_metadata_only(monkeypatch):
    monkeypatch.setattr(
        native,
        "run_native_cli",
        Mock(
            side_effect=UnicodeDecodeError(
                "utf-8", b"private-canary", 0, 1, "bad output"
            )
        ),
    )
    with pytest.raises(native.OpenCodeCLIError, match="decoded") as failure:
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            timeout_seconds=1,
        )
    assert "private-canary" not in str(failure.value)


@pytest.mark.unit
def test_setup_expiry_cannot_launch(monkeypatch, tmp_path):
    dispatch = Mock()
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    times = iter((1.0, 3.0))
    monkeypatch.setattr(native.time, "monotonic", lambda: next(times))
    with pytest.raises(native.OpenCodeCLIError, match="before launch"):
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            timeout_seconds=1,
        )
    dispatch.assert_not_called()
