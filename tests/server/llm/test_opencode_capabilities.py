"""Discovery and selection cannot retarget or renew a native request."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm.providers import _opencode_cli as native
from reflexio.server.llm.providers import claude_code_provider as provider

COMMON = "  -m, --model string\n  --agent string\n  --format string\n  --title string\n"
LEGACY = COMMON + "  --attach string\n  --variant string\n"
MODERN = COMMON + "GLOBAL FLAGS\n  --standalone   Run with a private server\n"


@pytest.mark.parametrize("help_text,private", [(LEGACY, False), (MODERN, True)])
def test_known_option_declarations_select_native_contract(help_text, private):
    assert native._uses_private_server(help_text) is private


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
        native._uses_private_server(help_text)


@pytest.mark.parametrize("private", [False, True])
def test_discovery_and_execution_spend_one_original_deadline(monkeypatch, private):
    clock = [10.0]
    calls = []
    monkeypatch.setattr(native.time, "monotonic", lambda: clock[0])

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
        model="opaque/model#variant",
        timeout_seconds=1,
    )
    assert result.success and result.terminal_text == "owned reply"
    assert len(calls) == 2 and calls[0][0][-2:] == ["run", "--help"]
    assert calls[0][1]["input"] == "" and calls[0][1]["timeout"] == 1
    argv, kwargs = calls[1]
    assert kwargs["timeout"] == pytest.approx(0.4)
    assert kwargs["input"] == "owned context\n\n## Task\nowned task"
    assert argv[-2:] == ["--model", "opaque/model#variant"]
    assert ("--standalone" in argv) is private
    assert argv[argv.index("--title") + 1] == native._AGENT
    assert calls[0][1]["cwd"] == kwargs["cwd"]
    assert not Path(kwargs["cwd"]).exists()


@pytest.mark.parametrize("phase", ["help", "output"])
def test_expiry_after_parsing_never_returns_or_replays(monkeypatch, phase):
    clock = [10.0]
    calls = []
    monkeypatch.setattr(native.time, "monotonic", lambda: clock[0])

    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            MODERN if argv[-1] == "--help" else '{"type":"text","text":"late"}',
            "",
        )

    target = "_uses_private_server" if phase == "help" else "parse_output"
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
            timeout_seconds=1,
        )
    assert len(calls) == (1 if phase == "help" else 2)


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
            timeout_seconds=1,
        )
    assert len(calls) == 1 and calls[0][-1] == "--help"
    assert "private-canary" not in str(failure.value)


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
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", name)
    if selection == "automatic":
        monkeypatch.delenv("CLAUDE_SMART_CLI_PATH")
    result = (
        provider._resolve_cli_path()
        if selection != "native"
        else native.resolve_executable(name)
    )
    assert result is not None
    assert Path(result) == bins / name
    (bins / name).unlink()
    if selection != "native":
        assert provider._resolve_cli_path() is None
    else:
        with pytest.raises(native.OpenCodeCLIError):
            native.resolve_executable(name)


def test_invalid_explicit_override_cannot_retarget_to_default_path(
    monkeypatch, tmp_path, caplog
):
    fallback = tmp_path / ("opencode.exe" if os.name == "nt" else "opencode")
    fallback.write_text("owned fallback; never executed")
    fallback.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("CLAUDE_SMART_HOST", "opencode")
    monkeypatch.setenv(
        "CLAUDE_SMART_CLI_PATH", str(tmp_path / "private-canary-missing")
    )
    assert provider._resolve_cli_path() is None
    assert (
        "CLAUDE_SMART_CLI_PATH" in caplog.text and "private-canary" not in caplog.text
    )
