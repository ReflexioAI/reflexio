"""Claude Code and Codex generation runs from a private, empty directory.

Both CLIs load project configuration from their working directory (Codex
`.codex/config.toml`, Claude Code `.claude/settings.json`), which could pick
another model or endpoint than the launch identity binds. A generation call
therefore never inherits the daemon's directory.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from reflexio.server.llm import _native_cli
from reflexio.server.llm.providers import native_cli_provider as provider

_SETTINGS = {
    ".claude/settings.json": '{"model": "synthetic-project-model"}',
    ".codex/config.toml": 'model = "synthetic-project-model"\n',
}


@pytest.fixture
def daemon_dir(monkeypatch, tmp_path):
    """A daemon working directory carrying project settings for both CLIs."""
    monkeypatch.delenv("CLAUDE_SMART_CODEX_PATH", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
    daemon = tmp_path / "project"
    for relative, content in _SETTINGS.items():
        target = daemon / relative
        target.parent.mkdir(parents=True)
        target.write_text(content)
    monkeypatch.chdir(daemon)
    return daemon


def _launch_recorder(monkeypatch):
    seen: list[dict] = []

    def launch(argv, **kwargs):
        cwd = kwargs.get("cwd")
        seen.append(
            {
                "cwd": cwd,
                "pwd": kwargs["env"].get("PWD"),
                "contents": sorted(os.listdir(cwd)) if cwd else None,
            }
        )
        if "--output-last-message" in argv:
            output = argv[argv.index("--output-last-message") + 1]
            Path(output).write_text("codex reply", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")
        stream = (
            '{"type":"result","subtype":"success","is_error":false,'
            '"result":"claude reply"}\n'
        )
        return subprocess.CompletedProcess(argv, 0, stream, "")

    monkeypatch.setattr(provider, "run_native_cli", launch)
    return seen


@pytest.mark.parametrize(
    ("host", "cli"), [("claude-code", "claude"), ("codex", "codex")]
)
def test_generation_runs_from_a_fresh_private_directory(
    daemon_dir, monkeypatch, host, cli
):
    seen = _launch_recorder(monkeypatch)
    result = provider._run_cli_stream(
        f"/synthetic/{cli}", "Be terse.", "User: hi", 30, host, None
    )
    assert result.success
    (call,) = seen
    workdir = Path(call["cwd"])
    assert workdir != Path.cwd()
    assert daemon_dir not in workdir.parents
    assert call["pwd"] == call["cwd"]
    # Empty: neither the daemon's project settings nor anything else.
    assert call["contents"] == []
    for parent in workdir.parents:
        assert not (parent / ".codex" / "config.toml").exists()
        assert not (parent / ".claude" / "settings.json").exists()
    # Not delegated, so the call removes its own directory.
    assert not workdir.exists()


@pytest.mark.parametrize(
    ("host", "cli"), [("claude-code", "claude"), ("codex", "codex")]
)
def test_each_call_gets_its_own_directory(daemon_dir, monkeypatch, host, cli):
    seen = _launch_recorder(monkeypatch)
    for _ in range(2):
        provider._run_cli_stream(f"/synthetic/{cli}", "", "User: hi", 30, host, None)
    assert seen[0]["cwd"] != seen[1]["cwd"]


@pytest.mark.parametrize(
    ("host", "cli"), [("claude-code", "claude"), ("codex", "codex")]
)
def test_a_delegated_directory_lives_with_its_parent_owner(
    daemon_dir, monkeypatch, tmp_path, host, cli
):
    owner = tmp_path / "owner"
    owner.mkdir()
    monkeypatch.setattr(_native_cli, "native_cli_directory", lambda: str(owner))
    seen = _launch_recorder(monkeypatch)
    provider._run_cli_stream(f"/synthetic/{cli}", "", "User: hi", 30, host, None)
    workdir = Path(seen[0]["cwd"])
    assert workdir.parent == owner.resolve()
    # The parent owner retires it after the native process is reaped.
    assert workdir.is_dir()


def test_the_optimizer_codex_assistant_keeps_its_working_directory(
    daemon_dir, monkeypatch
):
    seen = _launch_recorder(monkeypatch)
    provider._run_codex_stream(
        codex_path="/synthetic/codex",
        system_prompt="",
        dialogue="User: hi",
        deadline=time.monotonic() + 30,
        optimizer=True,
    )
    assert seen[0]["cwd"] is None
