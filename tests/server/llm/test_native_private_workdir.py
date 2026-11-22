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
from reflexio.server.llm._native_model_ref import (
    ENV_NATIVE_MODEL,
    native_launch_identity_digest,
)
from reflexio.server.llm.providers import _opencode_cli
from reflexio.server.llm.providers import native_cli_provider as provider
from tests.server.llm.test_native_launch_identity import (
    executable_name,
    own_launch_environment,
    set_home,
    stub_executable,
)

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
                "argv": list(argv),
                "env": dict(kwargs["env"]),
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


# Per host: the path variables whose relative value must reach the child as
# the absolute path the identity binds (with the switch that makes the host
# read a cloud credential file, where needed).
_PATH_CASES = [
    ("claude-code", "CLAUDE_CONFIG_DIR", {}),
    ("claude-code", "AWS_SHARED_CREDENTIALS_FILE", {"CLAUDE_CODE_USE_BEDROCK": "1"}),
    ("claude-code", "GOOGLE_APPLICATION_CREDENTIALS", {"CLAUDE_CODE_USE_VERTEX": "1"}),
    ("codex", "CODEX_HOME", {}),
    ("opencode", "XDG_CONFIG_HOME", {}),
    ("opencode", "XDG_DATA_HOME", {}),
    ("opencode", "GOOGLE_APPLICATION_CREDENTIALS", {}),
    ("opencode", "AWS_WEB_IDENTITY_TOKEN_FILE", {}),
]


def _dispatch_env(monkeypatch, daemon: Path, host: str) -> dict[str, str]:
    """The environment the host's child receives for one generation call."""
    if host != "opencode":
        seen = _launch_recorder(monkeypatch)
        cli = "claude" if host == "claude-code" else "codex"
        provider._run_cli_stream(f"/synthetic/{cli}", "", "User: hi", 30, host, None)
        return seen[0]["env"]
    executable = stub_executable(daemon / "opencode-bin", "opencode")
    envs: list[dict[str, str]] = []

    def launch(argv, **kwargs):
        envs.append(dict(kwargs["env"]))
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(
                argv,
                0,
                "  --model\n  --agent\n  --format\n  --title\n  --standalone\n",
                "",
            )
        return subprocess.CompletedProcess(argv, 0, '{"type":"text","text":"ok"}', "")

    monkeypatch.setattr(_opencode_cli, "run_native_cli", launch)
    _opencode_cli.run_opencode(
        cli_path=str(executable),
        system_prompt="",
        dialogue="User: hi",
        model=None,
        deadline=time.monotonic() + 30,
    )
    return envs[-1]


@pytest.mark.parametrize(("host", "name", "switches"), _PATH_CASES)
@pytest.mark.parametrize("spelling", ["relative", "home"])
def test_a_path_variable_reaches_the_child_as_the_path_the_identity_binds(
    daemon_dir, monkeypatch, host, name, switches, spelling
):
    own_launch_environment(monkeypatch)
    set_home(monkeypatch, daemon_dir.parent / "home")
    monkeypatch.setenv(ENV_NATIVE_MODEL, f"local-cli/{host}")
    for switch, value in switches.items():
        monkeypatch.setenv(switch, value)
    if spelling == "relative":
        written = str(Path("account") / "file")
        expected = daemon_dir / "account" / "file"
    else:
        written = str(Path("~") / "account" / "file")
        expected = daemon_dir.parent / "home" / "account" / "file"
    monkeypatch.setenv(name, written)
    bound = native_launch_identity_digest()
    # The child gets the anchored absolute value, not the raw spelling it
    # would resolve from its private directory ...
    assert _dispatch_env(monkeypatch, daemon_dir, host)[name] == str(expected)
    # ... and that absolute value is exactly what the identity binds.
    monkeypatch.setenv(name, str(expected))
    assert native_launch_identity_digest() == bound


@pytest.mark.parametrize(
    ("host", "cli"), [("claude-code", "claude"), ("codex", "codex")]
)
def test_an_explicit_relative_constructor_path_is_anchored_to_the_callers_cwd(
    daemon_dir, monkeypatch, host, cli
):
    monkeypatch.setenv(ENV_NATIVE_MODEL, f"local-cli/{host}")
    executable = stub_executable(daemon_dir / "bin", cli)
    seen = _launch_recorder(monkeypatch)
    relative = str(Path(".") / "bin" / executable_name(cli))
    provider.NativeCLILLM(cli_path="." + os.sep + relative).completion(
        model=f"local-cli/{host}",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert seen[0]["argv"][0] == str(executable.absolute())
    assert seen[0]["cwd"] != str(daemon_dir)


def test_a_bare_constructor_name_stays_a_path_lookup(daemon_dir, monkeypatch):
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    seen = _launch_recorder(monkeypatch)
    provider.NativeCLILLM(cli_path="codex").completion(
        model="local-cli/codex", messages=[{"role": "user", "content": "hi"}]
    )
    assert seen[0]["argv"][0] == "codex"
