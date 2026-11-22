"""The durable native launch identity binds binary and account, not environment."""

import os
from pathlib import Path

import pytest

from reflexio.server.llm._native_model_ref import (
    CLI_TIMEOUT_ENV,
    ENV_NATIVE_MODEL,
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
    native_launch_identity_digest,
)

_OWNED = (
    "CLAUDE_SMART_USE_LOCAL_CLI",
    "CLAUDE_SMART_CLI_MODEL",
    "CLAUDE_SMART_CLI_PATH",
    "CLAUDE_SMART_CODEX_PATH",
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
    CLI_TIMEOUT_ENV,
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
)


def _executable(directory: Path, name: str = "claude") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """A claude-code target whose binary resolves from an owned PATH."""
    for name in _OWNED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/claude-code")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _executable(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_digest_is_stable_for_an_unchanged_launch(launch):
    assert native_launch_identity_digest() == native_launch_identity_digest()


@pytest.mark.parametrize(
    "change",
    [
        "cwd",
        "path-same-binary",
        "pathext",
        "timeout",
    ],
)
def test_environment_that_does_not_decide_who_answers_is_excluded(
    launch, monkeypatch, change
):
    before = native_launch_identity_digest()
    if change == "cwd":
        other = launch / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
    elif change == "path-same-binary":
        empty = launch / "empty"
        empty.mkdir()
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(empty), str(launch / "bin"), "/usr/bin"])
        )
    elif change == "pathext":
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    else:
        monkeypatch.setenv(CLI_TIMEOUT_ENV, "30")
    assert native_launch_identity_digest() == before


@pytest.mark.parametrize(
    "change",
    [
        "path-other-binary",
        "cli-path-override",
        "codex-path-override",
        "opencode-executable-override",
        "home",
        "claude-config-dir",
        "codex-home",
        "target-model",
        "target-host",
    ],
)
def test_what_decides_who_answers_is_bound(launch, monkeypatch, change):
    before = native_launch_identity_digest()
    if change == "path-other-binary":
        _executable(launch / "other-bin")
        monkeypatch.setenv("PATH", str(launch / "other-bin"))
    elif change == "cli-path-override":
        monkeypatch.setenv(
            "CLAUDE_SMART_CLI_PATH", str(_executable(launch / "override"))
        )
    elif change == "codex-path-override":
        monkeypatch.setenv("CLAUDE_SMART_CODEX_PATH", "/synthetic/codex")
    elif change == "opencode-executable-override":
        monkeypatch.setenv(PLUGIN_OPENCODE_EXECUTABLE_ENV, "/synthetic/opencode")
    elif change == "home":
        monkeypatch.setenv("HOME", str(launch / "other-home"))
    elif change == "claude-config-dir":
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(launch / "account-b"))
    elif change == "codex-home":
        monkeypatch.setenv("CODEX_HOME", str(launch / "codex-b"))
    elif change == "target-model":
        monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/claude-code/claude-x")
    else:
        _executable(launch / "bin", "codex")
        monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    assert native_launch_identity_digest() != before


def test_a_path_lookup_binds_its_absolute_result_not_symlink_targets(
    launch, monkeypatch
):
    # The launch path is the identity: an in-place upgrade that re-points a
    # symlink at the same path does not move it.
    real_a = _executable(launch / "versions" / "a")
    real_b = _executable(launch / "versions" / "b")
    link_dir = launch / "links"
    link_dir.mkdir()
    link = link_dir / "claude"
    link.symlink_to(real_a)
    monkeypatch.setenv("PATH", str(link_dir))
    before = native_launch_identity_digest()
    link.unlink()
    link.symlink_to(real_b)
    assert native_launch_identity_digest() == before
