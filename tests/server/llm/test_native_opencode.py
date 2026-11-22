"""Native transport contracts; physical probes run in the owner test module."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from reflexio import opencode_executable
from reflexio.server.llm._litellm_types import (
    LLMHardTimeoutError,
    ProviderRequestGuardError,
)
from reflexio.server.llm.providers import _opencode_cli as native
from reflexio.server.llm.providers import native_cli_provider as provider
from reflexio.server.llm.providers.claude_code_stream_parser import (
    ParseResult,
    classify_stall,
)


@pytest.mark.unit
def test_executable_discovery_does_not_import_the_server_or_llm_providers():
    probe = """
import importlib.abc
import sys
import tempfile
from pathlib import Path

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "litellm" or fullname.startswith("reflexio.server"):
            raise AssertionError(f"Unexpected runtime import: {fullname}")

sys.meta_path.insert(0, BlockRuntimeImports())
sys.path.insert(0, sys.argv[1])
from reflexio.opencode_executable import resolve_executable

assert resolve_executable(sys.executable) == Path(sys.executable).absolute()
"""
    result = subprocess.run(  # noqa: S603 - runs the fixed probe above
        [sys.executable, "-I", "-c", probe, str(Path(__file__).resolve().parents[3])],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("setting", ["path", "timeout"])
def test_invalid_native_configuration_logs_names_without_values(
    monkeypatch, caplog, tmp_path, setting
):
    if setting == "path":
        monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "private-canary-not-an-executable")
        monkeypatch.setenv("PATH", str(tmp_path))
        assert provider._resolve_cli_path("opencode") is None
    else:
        monkeypatch.setenv(
            "CLAUDE_SMART_CLI_TIMEOUT", "private-canary-invalid-duration"
        )
        provider.NativeCLILLM()._timeout()
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
    handler = provider.NativeCLILLM(cli_path=sys.executable, storage=storage)
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    monkeypatch.setattr(provider, "_run_cli_stream", Mock(return_value=result))
    with pytest.raises(provider.ClaudeCodeCLIError) as failure:
        handler.completion(
            model="local-cli/" + "opencode",
            messages=[{"role": "user", "content": "synthetic"}],
            timeout=1,
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
def test_opencode_bridges_keep_claude_protocol_and_caller_bound(
    monkeypatch, host, bridge
):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/opencode")
    monkeypatch.setenv("CLAUDE_SMART_HOST", host)
    monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", "7")
    legacy = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_claude_stream", legacy)
    monkeypatch.setattr(
        native, "run_opencode", Mock(side_effect=AssertionError("bridge misrouted"))
    )
    handler = provider.NativeCLILLM(cli_path=f"/plugin/{bridge}")
    response = handler.completion(model="local-cli/opencode", timeout=0.25)
    assert response.choices[0].message.content == "ok"
    assert legacy.call_args.kwargs["timeout_seconds"] == 0.25
    assert response._hidden_params["reflexio_cli_binary"] == bridge.split("-")[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "budget,cap,constructor,expected",
    [(0.25, None, None, 0.25), (5, "2", None, 2), (5, "2", 1, 1), (0.25, "2", 1, 0.25)],
)
def test_native_allowance_has_no_implicit_default_and_preserves_explicit_caps(
    monkeypatch, budget, cap, constructor, expected
):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    if cap is None:
        monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", cap)
    handler = provider.NativeCLILLM(
        cli_path=sys.executable, timeout_seconds=constructor
    )
    monkeypatch.setattr(
        handler, "_timeout", Mock(side_effect=AssertionError("provider default used"))
    )
    dispatch = Mock(return_value=ParseResult(True, "ok", cli_binary="opencode"))
    monkeypatch.setattr(native, "run_opencode", dispatch)
    started = time.monotonic()
    handler.completion(model="local-cli/" + "opencode", timeout=budget)
    assert (
        started + expected
        <= dispatch.call_args.kwargs["deadline"]
        <= time.monotonic() + expected
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "timeout", [None, True, "3", 0, -1, float("inf"), float("nan"), 2**1024]
)
def test_native_invalid_budget_cannot_launch(monkeypatch, timeout):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    dispatch = Mock()
    monkeypatch.setattr(native, "run_opencode", dispatch)
    with pytest.raises(ValueError, match="finite positive timeout"):
        provider.NativeCLILLM(cli_path=sys.executable).completion(
            model="local-cli/" + "opencode", timeout=timeout
        )
    dispatch.assert_not_called()


@pytest.mark.unit
def test_native_async_handler_keeps_the_selected_budget(monkeypatch):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    dispatch = Mock(return_value=ParseResult(True, "ok", cli_binary="opencode"))
    monkeypatch.setattr(native, "run_opencode", dispatch)
    started = time.monotonic()
    response = asyncio.run(
        provider.NativeCLILLM(cli_path=sys.executable).acompletion(
            model="local-cli/" + "opencode", timeout=0.25
        )
    )
    assert response.choices[0].message.content == "ok"
    assert (
        started + 0.25
        <= dispatch.call_args.kwargs["deadline"]
        <= time.monotonic() + 0.25
    )


@pytest.mark.unit
@pytest.mark.parametrize("through_path", [False, True])
def test_native_relative_executable_is_anchored_before_changing_cwd(
    monkeypatch, tmp_path, through_path
):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "bin with spaces" / "opencode.exe"
    executable.parent.mkdir()
    executable.touch()
    executable.chmod(0o755)
    selected = []

    def launch(argv, **kwargs):
        selected.append((argv, kwargs))
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(
                argv,
                0,
                "  --model\n  --agent\n  --format\n  --title\n  --standalone\n",
                "",
            )
        assert Path(kwargs["cwd"]) != tmp_path
        return subprocess.CompletedProcess(argv, 0, '{"type":"text","text":"ok"}', "")

    monkeypatch.setattr(native, "run_native_cli", launch)
    relative = str(executable.relative_to(tmp_path))
    if through_path:
        monkeypatch.setattr(
            opencode_executable.shutil, "which", lambda _name, **_kwargs: relative
        )
    native.run_opencode(
        cli_path="opencode" if through_path else relative,
        system_prompt="s",
        dialogue="p",
        model=None,
        deadline=time.monotonic() + 1,
    )
    assert selected[0][0][0] == str(executable)
    assert 0 < selected[0][1]["timeout"] <= 1
    assert not Path(selected[0][1]["cwd"]).exists()


def _npm_install(tmp_path, local):
    package = tmp_path / "node_modules" / "opencode-ai"
    shim_dir = package.parent / ".bin" if local else tmp_path
    shim_dir.mkdir(parents=True, exist_ok=True)
    executable = package / "bin" / "opencode.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"MZ")
    target = (
        r"..\opencode-ai\bin\opencode.exe"
        if local
        else r"node_modules\opencode-ai\bin\opencode.exe"
    )
    shim = shim_dir / "opencode.cmd"
    shim.write_bytes(
        (
            "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\n"
            "EXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n"
            f'"%dp0%\\{target}"   %*\r\n'
        ).encode("ascii")
    )
    return shim, executable


@pytest.mark.unit
@pytest.mark.parametrize("local", [False, True])
def test_npm_opencode_shim_selects_its_native_package_executable(tmp_path, local):
    shim, executable = _npm_install(tmp_path, local)
    assert opencode_executable._npm_opencode_executable(shim) == executable


@pytest.mark.unit
@pytest.mark.parametrize(
    "damage", ["custom", "extra-command", "missing", "placeholder"]
)
def test_refuses_nonstandard_or_incomplete_npm_shims(tmp_path, damage):
    shim, executable = _npm_install(tmp_path, False)
    if damage == "custom":
        shim.write_text("@exit /b 99")
    elif damage == "extra-command":
        shim.write_bytes(shim.read_bytes() + b"@echo extra\r\n")
    elif damage == "missing":
        executable.unlink()
    else:
        executable.write_text("postinstall required")
    assert opencode_executable._npm_opencode_executable(shim) is None


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows PATH/PATHEXT executable selection")
@pytest.mark.parametrize("local", [False, True])
def test_windows_opencode_uses_package_binary_for_both_launches(
    monkeypatch, tmp_path, local
):
    shim, executable = _npm_install(tmp_path, local)
    monkeypatch.setenv("PATH", str(shim.parent))
    monkeypatch.setenv("PATHEXT", ".CMD")
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/opencode")
    launch = Mock(
        side_effect=[
            subprocess.CompletedProcess(
                [], 0, "--model\n--agent\n--format\n--title\n--standalone", ""
            ),
            subprocess.CompletedProcess([], 0, '{"type":"text","text":"owned"}', ""),
        ]
    )
    monkeypatch.setattr(native, "run_native_cli", launch)
    result = provider.NativeCLILLM().completion(
        model="local-cli/opencode",
        messages=[{"role": "user", "content": "owned"}],
        timeout=1,
    )
    assert result.choices[0].message.content == "owned"
    assert launch.call_count == 2
    assert all(Path(call.args[0][0]) == executable for call in launch.call_args_list)


@pytest.mark.unit
def test_bridge_symlink_is_discriminated_before_native_path_canonicalization(
    monkeypatch, tmp_path
):
    if os.name == "nt":
        pytest.skip("POSIX symlink name contract; native Windows exe tested separately")
    bridge = tmp_path / "opencode-claude-compat"
    bridge.symlink_to(sys.executable)
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    legacy = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_claude_stream", legacy)
    provider.NativeCLILLM(cli_path=str(bridge)).completion(
        model="local-cli/" + "opencode", timeout=0.25
    )
    assert legacy.call_args.kwargs["cli_path"] == str(bridge)


@pytest.mark.unit
def test_setup_failure_cannot_dispatch_or_expose_raw_error(monkeypatch):
    monkeypatch.setattr(
        tempfile, "mkdtemp", Mock(side_effect=OSError("private-canary"))
    )
    dispatch = Mock()
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    with pytest.raises(native.OpenCodeCLIError) as failure:
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            deadline=time.monotonic() + 1,
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
            deadline=time.monotonic() + 1,
        )
    assert "private-canary" not in str(failure.value)


@pytest.mark.unit
def test_setup_expiry_cannot_launch(monkeypatch, tmp_path):
    dispatch = Mock()
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    times = iter((1.0, 3.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(times))
    with pytest.raises(LLMHardTimeoutError, match="deadline expired"):
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            deadline=time.monotonic() + 1,
        )
    dispatch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize("bridge", [False, True])
@pytest.mark.parametrize(
    "budget,cap,constructor,expected",
    [
        (0.25, None, None, 0.25),
        (180, None, None, 180),
        (5, "2", None, 2),
        (5, "2", 1, 1),
        (0.25, "2", 1, 0.25),
    ],
)
def test_every_cli_protocol_uses_caller_budget(
    monkeypatch, host, bridge, budget, cap, constructor, expected
):
    monkeypatch.setenv("CLAUDE_SMART_HOST", host)
    target = "local-cli/" + ("opencode" if bridge else host)
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", target)
    if cap is None:
        monkeypatch.delenv("CLAUDE_SMART_CLI_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", cap)
    handler = provider.NativeCLILLM(
        cli_path="opencode-claude-compat" if bridge else sys.executable,
        timeout_seconds=constructor,
    )
    monkeypatch.setattr(
        handler, "_timeout", Mock(side_effect=AssertionError("implicit clock used"))
    )
    dispatch = Mock(return_value=ParseResult(True, "ok"))
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    handler.completion(model=target, timeout=budget)
    assert dispatch.call_args.kwargs["timeout_seconds"] == expected


@pytest.mark.unit
@pytest.mark.parametrize("host", ["claude-code", "codex", "opencode"])
@pytest.mark.parametrize(
    "budget", [None, True, "private-duration", 0, -1, float("inf"), float("nan")]
)
def test_explicit_invalid_cli_budget_refuses_before_launch(monkeypatch, host, budget):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + host)
    dispatch = Mock()
    monkeypatch.setattr(provider, "_run_cli_stream", dispatch)
    with pytest.raises(ValueError, match="finite positive timeout") as failure:
        provider.NativeCLILLM(cli_path=sys.executable).completion(
            model="local-cli/" + host, timeout=budget
        )
    assert "private-duration" not in str(failure.value)
    dispatch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("transport", [None, "local-cli", "openai"])
def test_native_parent_rejects_invalid_budget_without_retargeting_http(
    monkeypatch, transport
):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "claude-code")
    monkeypatch.delenv("CLAUDE_SMART_CODEX_PATH", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="local-cli/" + "claude-code", timeout=2))
    options = {"timeout": "private-duration"}
    if transport:
        options["custom_llm_provider"] = transport
    if transport == "openai":
        from reflexio.server.llm._litellm_types import ProviderRequestGuardError

        with pytest.raises(ProviderRequestGuardError, match="transport"):
            client._build_completion_params([], **options)
    else:
        with pytest.raises(ValueError, match="finite positive timeout") as failure:
            client._build_completion_params([], **options)
        assert "private-duration" not in str(failure.value)


@pytest.mark.unit
def test_invalid_portable_cli_cap_refuses_in_parent_before_fallback(monkeypatch):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "codex")
    monkeypatch.delenv("CLAUDE_SMART_CODEX_PATH", raising=False)
    monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", "private-duration")
    client = LiteLLMClient(
        LiteLLMConfig(
            model="local-cli/" + "codex", timeout=2, fallback_models=["gpt-4o-mini"]
        )
    )
    dispatch = Mock()
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(ValueError, match="CLAUDE_SMART_CLI_TIMEOUT") as failure:
        client.generate_response("synthetic evidence")
    assert "private-duration" not in str(failure.value)
    dispatch.assert_not_called()


COMMON = "  -m, --model string\n  --agent string\n  --format string\n  --title string\n"
LEGACY = COMMON + "  --attach string\n  --variant string\n"
MODERN = COMMON + "GLOBAL FLAGS\n  --standalone   Run with a private server\n"


@pytest.mark.unit
@pytest.mark.parametrize("help_text,private", [(LEGACY, False), (MODERN, True)])
def test_known_option_declarations_select_native_contract(help_text, private):
    assert ("--standalone" in native._run_options(help_text)) is private


@pytest.mark.unit
@pytest.mark.parametrize(
    "help_text",
    [
        "",
        "private-error --standalone",
        COMMON,
        COMMON + "  --standalone-extra\n",
        COMMON + "Error: --standalone is unsupported\n",
        json.dumps({"error": MODERN}),
    ],
)
def test_unknown_help_cannot_become_a_protocol_fallback(help_text):
    with pytest.raises(native.OpenCodeCLIError, match="capabilities"):
        native._run_options(help_text)


@pytest.mark.unit
@pytest.mark.parametrize("owned", [False, True])
def test_plugin_owned_generation_uses_compatible_cli_inside_request(monkeypatch, owned):
    requests = []
    if owned:
        monkeypatch.setenv(provider.PLUGIN_OPENCODE_EXECUTABLE_ENV, sys.executable)
    else:
        monkeypatch.setenv(provider.PLUGIN_OPENCODE_EXECUTABLE_ENV, "/other/opencode")
    monkeypatch.setenv("CLAUDE_SMART_CLI_TIMEOUT", "1.5")
    monkeypatch.setattr(time, "monotonic", lambda: 20.0)

    def run(argv, **kwargs):
        requests.append((argv, kwargs))
        output = MODERN if argv[-1] == "--help" else '{"type":"text","text":"done"}'
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(native, "run_native_cli", run)
    result = provider._run_cli_stream(
        cli_path=sys.executable,
        system_prompt="system",
        dialogue="task",
        timeout_seconds=8,
        host="opencode",
        model="provider/model",
    )
    assert result.success
    assert len(requests) == (1 if owned else 2)
    argv, capture = requests[-1]
    assert capture["input"] == "system\n\n## Task\ntask"
    assert capture["timeout"] == 8
    assert argv[-2:] == (
        ["--model", "provider/model"]
        if owned
        else ["--standalone", "--model=provider/model"]
    )
    assert ("--title" in argv) is not owned


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("provider/model#low", ["--model", "provider/model", "--variant=low"]),
        ("#low", ["--variant=low"]),
    ],
)
def test_plugin_bridge_keeps_variant_inline(monkeypatch, model, expected):
    def run(argv, **kwargs):
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(argv, 0, LEGACY, "")
        assert argv == [
            sys.executable,
            "run",
            "--format",
            "json",
            "--agent",
            native._AGENT,
            *expected,
        ]
        return subprocess.CompletedProcess(
            argv, 0, '{"type":"text","text":"owned reply"}', ""
        )

    monkeypatch.setattr(native, "run_native_cli", run)
    monkeypatch.setenv(provider.PLUGIN_OPENCODE_EXECUTABLE_ENV, sys.executable)
    result = provider._run_cli_stream(
        cli_path=sys.executable,
        system_prompt="owned context",
        dialogue="owned task",
        model=model,
        timeout_seconds=8,
        host="opencode",
    )
    assert result.success


@pytest.mark.unit
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("model", [None, "opaque/nested/model", "opaque/model#variant"])
def test_discovery_and_execution_spend_one_original_deadline(
    monkeypatch, private, model
):
    clock = [10.0]
    calls = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[-1] == "--help":
            clock[0] += 0.6
            return subprocess.CompletedProcess(
                argv, 0, MODERN if private else LEGACY, ""
            )
        clock[0] += 0.2
        return subprocess.CompletedProcess(
            argv, 0, '{"type":"text","text":"owned reply"}', ""
        )

    monkeypatch.setattr(native, "run_native_cli", run)
    result = native.run_opencode(
        cli_path=sys.executable,
        system_prompt="owned context",
        dialogue="owned task",
        model=model,
        deadline=time.monotonic() + 1,
    )
    assert result.success and result.terminal_text == "owned reply"
    assert len(calls) == 2 and calls[0][0][-2:] == ["run", "--help"]
    assert calls[0][1]["input"] == "" and calls[0][1]["timeout"] == 1
    argv, kwargs = calls[1]
    assert kwargs["timeout"] == pytest.approx(0.4)
    assert kwargs["input"] == "owned context\n\n## Task\nowned task"
    if model is None:
        assert not any(arg.startswith(("--model", "--variant")) for arg in argv)
    elif model == "opaque/nested/model":
        assert argv[-1] == "--model=opaque/nested/model"
        assert not any(arg.startswith("--variant") for arg in argv)
    elif private:
        assert argv[-1] == "--model=opaque/model#variant"
    else:
        assert argv[-2:] == ["--model=opaque/model", "--variant=variant"]
    assert ("--standalone" in argv) is private
    assert argv[argv.index("--title") + 1] == native._AGENT
    assert calls[0][1]["cwd"] == kwargs["cwd"]
    assert not Path(kwargs["cwd"]).exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "model", ["private-canary#", "#", "model# \t", " #private-canary"]
)
@pytest.mark.parametrize("legacy_wire", [False, True])
def test_empty_variant_components_refuse_before_any_launch(
    monkeypatch, model, legacy_wire
):
    dispatch = Mock(side_effect=OSError("must not probe"))
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    with pytest.raises(ProviderRequestGuardError, match="variant reference") as failure:
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="owned context",
            dialogue="owned task",
            model=model,
            deadline=time.monotonic() + 1,
            wire="plugin" if legacy_wire else "native",
        )
    assert "private-canary" not in str(failure.value)
    dispatch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("capabilities", [LEGACY, MODERN + "\n--variant"])
def test_variant_can_leave_model_selection_to_opencode(monkeypatch, capabilities):
    dispatch = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, capabilities, ""),
            subprocess.CompletedProcess(
                [], 0, '{"type":"text","text":"owned reply"}', ""
            ),
        ]
    )
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    result = native.run_opencode(
        cli_path=sys.executable,
        system_prompt="",
        dialogue="owned task",
        model="#high",
        deadline=time.monotonic() + 1,
    )
    assert result.success and dispatch.call_count == 2
    argv = dispatch.call_args.args[0]
    assert "--variant=high" in argv
    assert not any(arg.startswith("--model") for arg in argv)


@pytest.mark.unit
def test_variant_only_requires_an_advertised_variant_option(monkeypatch):
    dispatch = Mock(return_value=subprocess.CompletedProcess([], 0, MODERN, ""))
    monkeypatch.setattr(native, "run_native_cli", dispatch)
    with pytest.raises(ProviderRequestGuardError, match="variant without a model"):
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="owned task",
            model="#high",
            deadline=time.monotonic() + 1,
        )
    assert dispatch.call_count == 1


@pytest.mark.unit
def test_empty_legacy_variant_stays_terminal_through_worker_and_http_fallback(
    monkeypatch,
):
    import pickle
    import socket
    import time
    from unittest.mock import patch

    import litellm

    from reflexio.server.llm import _litellm_subprocess as worker
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.test_support.llm_mock import unpatched_litellm

    reference = "local-cli/opencode/provider/model#"
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", reference)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", sys.executable)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_CLI_MODEL", raising=False)
    monkeypatch.setattr(litellm, "custom_provider_map", [])
    monkeypatch.setattr(provider, "_HANDLER", None)
    discovery = Mock(return_value=subprocess.CompletedProcess([], 0, "", LEGACY))
    monkeypatch.setattr(native, "run_native_cli", discovery)
    models = []
    snapshots = []

    def physical(params, hard_timeout):
        models.append(params["model"])
        if params["model"] != reference:
            # A real HTTP rung is forbidden; the pre-fix walk reaches this
            # inert response and thereby fails the terminal-refusal assertion.
            return litellm.ModelResponse(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "unexpected fallback",
                        },
                        "finish_reason": "stop",
                    }
                ]
            )
        queue = Mock()
        with (
            unpatched_litellm(),
            patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network forbidden"),
            ),
        ):
            worker._litellm_completion_worker(
                params, queue, time.monotonic() + hard_timeout
            )
        status, snapshot = queue.put.call_args.args[0]
        assert status == "error"
        snapshots.append(snapshot)
        restored = pickle.loads(pickle.dumps(snapshot))  # noqa: S301 -- Owned in-process worker error snapshot only.
        return worker._raise_completion_error(restored)

    client = LiteLLMClient(
        LiteLLMConfig(
            model=reference,
            timeout=5,
            max_retries=0,
            max_tokens=32,
            fallback_models=["openai/owned-http"],
        )
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", physical)
    try:
        with pytest.raises(ProviderRequestGuardError, match="variant reference"):
            client.generate_response("owned synthetic input")
    finally:
        assert models == [reference]
    assert len(snapshots) == 1 and snapshots[0].request_guard_error
    discovery.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("component", ["model", "variant"])
def test_model_text_cannot_become_a_native_routing_option(
    monkeypatch, private, component
):
    # The recorded release-declared Yargs18 parser interprets paired option-
    # shaped values as --attach. Inline binding preserves their literal value.
    provenance = json.loads(
        (
            Path(__file__).parents[2]
            / "fixtures/opencode_v1_19_6_run_help.provenance.json"
        ).read_text()
    )["option_binding_probe"]["results"]
    value = provenance["modelBound"]["model"]
    assert provenance["modelPair"]["attach"] == "https://example.invalid"
    assert provenance["variantPair"]["attach"] == "https://example.invalid"
    assert provenance["variantBound"]["variant"] == value
    model = value if component == "model" else "provider/model#" + value
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(
                argv, 0, MODERN if private else LEGACY, ""
            )
        return subprocess.CompletedProcess(
            argv, 0, '{"type":"text","text":"owned reply"}', ""
        )

    monkeypatch.setattr(native, "run_native_cli", run)
    assert native.run_opencode(
        cli_path=sys.executable,
        system_prompt="owned context",
        dialogue="owned task",
        model=model,
        deadline=time.monotonic() + 1,
    ).success
    expected = (
        ["--model=provider/model", "--variant=" + value]
        if not private and component == "variant"
        else ["--model=" + model]
    )
    assert calls[-1][-len(expected) :] == expected
    assert not any(arg.startswith("--attach=") for arg in calls[-1])


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["help", "output"])
def test_expiry_after_parsing_never_returns_or_replays(monkeypatch, phase):
    clock = [10.0]
    calls = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            MODERN if argv[-1] == "--help" else '{"type":"text","text":"late"}',
            "",
        )

    target = "_run_options" if phase == "help" else "parse_output"
    original = getattr(native, target)

    def slow_parse(*args):
        result = original(*args)
        clock[0] = 12.0
        return result

    monkeypatch.setattr(native, target, slow_parse)
    monkeypatch.setattr(native, "run_native_cli", run)
    with pytest.raises(LLMHardTimeoutError):
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            deadline=time.monotonic() + 1,
        )
    assert len(calls) == (1 if phase == "help" else 2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "code,help_text", [(1, MODERN), (0, ""), (0, COMMON), (0, "private-canary")]
)
def test_failed_discovery_never_dispatches_model_or_leaks_output(
    monkeypatch, code, help_text
):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, code, help_text, "private-canary")

    monkeypatch.setattr(native, "run_native_cli", run)
    with pytest.raises(native.OpenCodeCLIError) as failure:
        native.run_opencode(
            cli_path=sys.executable,
            system_prompt="",
            dialogue="",
            model=None,
            deadline=time.monotonic() + 1,
        )
    assert len(calls) == 1 and calls[0][-1] == "--help"
    assert "private-canary" not in str(failure.value)


@pytest.mark.unit
@pytest.mark.parametrize("selection", ["native", "override", "automatic"])
def test_bare_selection_honors_path_instead_of_working_directory(
    monkeypatch, tmp_path, selection
):
    name = "opencode.exe" if os.name == "nt" else "opencode"
    cwd = tmp_path / "cwd"
    bins = tmp_path / "path"
    cwd.mkdir()
    bins.mkdir()
    for path in (cwd / name, bins / name):
        path.write_text("owned fixture; never executed")
        path.chmod(0o755)
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", str(bins))
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", name)
    if selection == "automatic":
        monkeypatch.delenv("CLAUDE_SMART_CLI_PATH")
    result = (
        provider._resolve_cli_path("opencode")
        if selection != "native"
        else native.resolve_executable(name)
    )
    assert result is not None
    assert Path(result) == bins / name
    (bins / name).unlink()
    if selection != "native":
        assert provider._resolve_cli_path("opencode") is None
    else:
        with pytest.raises(native.OpenCodeCLIError):
            native.resolve_executable(name)


@pytest.mark.unit
def test_invalid_explicit_override_cannot_retarget_to_default_path(
    monkeypatch, tmp_path, caplog
):
    fallback = tmp_path / ("opencode.exe" if os.name == "nt" else "opencode")
    fallback.write_text("owned fallback; never executed")
    fallback.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/" + "opencode")
    monkeypatch.setenv(
        "CLAUDE_SMART_CLI_PATH", str(tmp_path / "private-canary-missing")
    )
    assert provider._resolve_cli_path("opencode") is None
    assert (
        "CLAUDE_SMART_CLI_PATH" in caplog.text and "private-canary" not in caplog.text
    )


@pytest.mark.unit
@pytest.mark.parametrize("suffix", ["", ".cmd", ".js"])
@pytest.mark.parametrize("model", [None, "default", "opaque", "provider/model#high"])
def test_known_bridge_child_model_is_owned_by_native_reference(
    monkeypatch, suffix, model
):
    reference = "local-cli/opencode" + ("/" + model if model is not None else "")
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", reference)
    hints = {
        "CLAUDE_SMART_LEARNING_MODEL": "provider/stale",
        "CLAUDE_SMART_OPENCODE_MODEL": "provider/other",
        "CLAUDE_SMART_OPENCODE_VARIANT": "low",
    }
    for name, value in hints.items():
        monkeypatch.setenv(name, value)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"type": "result", "subtype": "success", "result": "owned"}),
            "",
        )
    )
    monkeypatch.setattr(provider, "run_native_cli", run)
    response = provider.NativeCLILLM(
        cli_path="/owned/opencode-claude-compat" + suffix
    ).completion(model=reference, timeout=1)
    assert response.choices[0].message.content == "owned"
    argv = run.call_args.args[0]
    if model is None:
        assert "--model" not in argv
    else:
        assert argv[argv.index("--model") + 1] == model
    child = run.call_args.kwargs["env"]
    assert child["CLAUDE_SMART_LEARNING_MODEL"] == (model or "")
    assert child["CLAUDE_SMART_OPENCODE_MODEL"] == ""
    assert child["CLAUDE_SMART_OPENCODE_VARIANT"] == ""
    assert run.call_args.kwargs["timeout"] == 1
    assert {name: os.environ.get(name) for name in hints} == hints


@pytest.mark.unit
@pytest.mark.parametrize("executable", ["claude", "custom-claude-wrapper"])
def test_nonbridge_claude_transport_preserves_independent_learning_environment(
    monkeypatch, executable
):
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", "local-cli/claude-code/owned")
    hints = {
        "CLAUDE_SMART_LEARNING_MODEL": "provider/optimizer",
        "CLAUDE_SMART_OPENCODE_MODEL": "provider/independent",
        "CLAUDE_SMART_OPENCODE_VARIANT": "low",
    }
    for name, value in hints.items():
        monkeypatch.setenv(name, value)
    run = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"type": "result", "subtype": "success", "result": "owned"}),
            "",
        )
    )
    monkeypatch.setattr(provider, "run_native_cli", run)
    provider.NativeCLILLM(cli_path="/owned/" + executable).completion(
        model="local-cli/claude-code/owned", timeout=1
    )
    child = run.call_args.kwargs["env"]
    assert {name: child.get(name) for name in hints} == hints
    assert {name: os.environ.get(name) for name in hints} == hints


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt", reason="POSIX PATH semantics")
@pytest.mark.parametrize("path", ["{bin}:", ":{bin}", ":"])
def test_bare_opencode_selection_skips_empty_path_entries(monkeypatch, tmp_path, path):
    # An empty PATH entry would anchor the bare name to the working directory.
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "bin").mkdir()
    for directory in (work, tmp_path / "bin"):
        if directory == tmp_path / "bin" and path == ":":
            continue
        stub = directory / "opencode"
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    monkeypatch.chdir(work)
    monkeypatch.setenv("PATH", path.format(bin=tmp_path / "bin"))
    if path == ":":
        with pytest.raises(opencode_executable.OpenCodeCLIError):
            opencode_executable.resolve_executable("opencode")
    else:
        assert opencode_executable.resolve_executable("opencode") == (
            tmp_path / "bin" / "opencode"
        )
