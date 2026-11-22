"""The durable native launch identity binds binary and account, not environment.

The fixtures are portable by construction (not exercised on Windows here):
stubs carry ``.exe`` where ``shutil.which`` needs a PATHEXT match, the home
directory is set through both ``HOME`` and ``USERPROFILE`` (``Path.home()``
reads the latter on Windows), and only the symlink case skips when the
platform cannot create symlinks.
"""

import os
from pathlib import Path

import pytest

from reflexio.server.llm._native_model_ref import (
    CLI_TIMEOUT_ENV,
    ENV_NATIVE_MODEL,
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
    native_launch_identity_digest,
)
from reflexio.server.llm.providers.native_cli_provider import _resolve_cli_path

_OWNED = (
    "CLAUDE_SMART_USE_LOCAL_CLI",
    "CLAUDE_SMART_CLI_MODEL",
    "CLAUDE_SMART_CLI_PATH",
    "CLAUDE_SMART_CODEX_PATH",
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
    CLI_TIMEOUT_ENV,
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
)
# Namespaces the digest reads; cleared so the developer's own setup is inert.
_OWNED_PREFIXES = (
    "ANTHROPIC_",
    "CLAUDE_CODE_USE_",
    "CODEX_",
    "OPENAI_",
    "OPENCODE_",
    "AWS_",
    "GOOGLE_",
    "CLOUD_ML_",
)


def executable_name(name: str) -> str:
    """The file name ``shutil.which(name)`` finds on this platform."""
    return f"{name}.exe" if os.name == "nt" else name


def stub_executable(directory: Path, name: str = "claude") -> Path:
    """Create an executable stub that ``shutil.which`` resolves on PATH."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / executable_name(name)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def set_home(monkeypatch, path: Path) -> None:
    """Point ``Path.home()`` at ``path`` on POSIX and on Windows."""
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


def own_launch_environment(monkeypatch) -> None:
    for name in _OWNED:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.upper().startswith(_OWNED_PREFIXES):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """A claude-code target whose binary resolves from an owned PATH."""
    own_launch_environment(monkeypatch)
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/claude-code")
    set_home(monkeypatch, tmp_path / "home")
    stub_executable(tmp_path / "bin")
    stub_executable(tmp_path / "bin", "codex")
    stub_executable(tmp_path / "bin", "opencode")
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
        "another-hosts-namespace",
        "another-hosts-account-dir",
        "credential-rotation",
    ],
)
def test_environment_that_does_not_decide_who_answers_is_excluded(
    launch, monkeypatch, change
):
    if change == "credential-rotation":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-synthetic-one")
    before = native_launch_identity_digest()
    if change == "cwd":
        other = launch / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)
    elif change == "path-same-binary":
        empty = launch / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", os.pathsep.join([str(empty), str(launch / "bin")]))
    elif change == "pathext":
        monkeypatch.setenv("PATHEXT", ".EXE;.CMD")
    elif change == "timeout":
        monkeypatch.setenv(CLI_TIMEOUT_ENV, "30")
    elif change == "another-hosts-namespace":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic.example")
    elif change == "another-hosts-account-dir":
        monkeypatch.setenv("CODEX_HOME", str(launch / "codex-b"))
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-synthetic-two")
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
        "target-model",
        "target-host",
        "base-url",
        "bedrock-switch",
        "alias-remap",
        "credential-source",
    ],
)
def test_what_decides_who_answers_is_bound(launch, monkeypatch, change):
    before = native_launch_identity_digest()
    if change == "path-other-binary":
        stub_executable(launch / "other-bin")
        monkeypatch.setenv("PATH", str(launch / "other-bin"))
    elif change == "cli-path-override":
        monkeypatch.setenv(
            "CLAUDE_SMART_CLI_PATH", str(stub_executable(launch / "override"))
        )
    elif change == "codex-path-override":
        monkeypatch.setenv("CLAUDE_SMART_CODEX_PATH", "/synthetic/codex")
    elif change == "opencode-executable-override":
        monkeypatch.setenv(PLUGIN_OPENCODE_EXECUTABLE_ENV, "/synthetic/opencode")
    elif change == "home":
        set_home(monkeypatch, launch / "other-home")
    elif change == "claude-config-dir":
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(launch / "account-b"))
    elif change == "target-model":
        monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/claude-code/claude-x")
    elif change == "target-host":
        monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    elif change == "base-url":
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://synthetic.example")
    elif change == "bedrock-switch":
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    elif change == "alias-remap":
        monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "synthetic-model")
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-synthetic")
    assert native_launch_identity_digest() != before


def test_a_bare_name_override_is_the_path_lookup_dispatch_performs(launch, monkeypatch):
    # A same-named file in the working directory is NOT what subprocess runs.
    stub_executable(launch)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", executable_name("claude"))
    on_path = launch / "bin" / executable_name("claude")
    assert _resolve_cli_path("claude-code") == str(on_path.absolute())
    before = native_launch_identity_digest()
    stub_executable(launch / "other-bin")
    monkeypatch.setenv("PATH", str(launch / "other-bin"))
    assert _resolve_cli_path("claude-code") == str(
        (launch / "other-bin" / executable_name("claude")).absolute()
    )
    assert native_launch_identity_digest() != before


def test_a_relative_override_is_anchored_once_for_dispatch_and_identity(
    launch, monkeypatch
):
    stub_executable(launch / "tools")
    relative = str(Path("tools") / executable_name("claude"))
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", relative)
    assert _resolve_cli_path("claude-code") == str(
        (launch / "tools" / executable_name("claude")).absolute()
    )


@pytest.mark.parametrize(
    ("host", "name"),
    [("claude-code", "CLAUDE_CONFIG_DIR"), ("codex", "CODEX_HOME")],
)
def test_a_relative_account_dir_binds_its_effective_location(
    launch, monkeypatch, host, name
):
    monkeypatch.setenv(ENV_NATIVE_MODEL, f"local-cli/{host}")
    monkeypatch.setenv(name, "account")
    before = native_launch_identity_digest()
    (launch / "elsewhere").mkdir()
    monkeypatch.chdir(launch / "elsewhere")
    assert native_launch_identity_digest() != before, "a different directory"
    monkeypatch.setenv(name, str(launch / "elsewhere" / "account"))
    stable = native_launch_identity_digest()
    monkeypatch.chdir(launch)
    assert native_launch_identity_digest() == stable, "absolute is cwd-independent"


@pytest.mark.parametrize("name", ["CODEX_HOME", "OPENAI_BASE_URL"])
def test_codex_binds_its_account_and_namespace(launch, monkeypatch, name):
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    before = native_launch_identity_digest()
    monkeypatch.setenv(name, str(launch / "synthetic"))
    assert native_launch_identity_digest() != before


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("XDG_DATA_HOME", (".local", "share")),
        ("XDG_CONFIG_HOME", (".config",)),
    ],
)
def test_opencode_binds_its_effective_xdg_directories(
    launch, monkeypatch, name, default
):
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/opencode")
    unset = native_launch_identity_digest()
    # The documented default is the same location as leaving it unset.
    monkeypatch.setenv(name, str((launch / "home").joinpath(*default)))
    assert native_launch_identity_digest() == unset
    monkeypatch.setenv(name, str(launch / "other-xdg"))
    assert native_launch_identity_digest() != unset


def test_xdg_directories_do_not_bind_other_hosts(launch, monkeypatch):
    before = native_launch_identity_digest()
    monkeypatch.setenv("XDG_DATA_HOME", str(launch / "other-xdg"))
    assert native_launch_identity_digest() == before


def test_opencode_binds_its_namespace(launch, monkeypatch):
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/opencode")
    before = native_launch_identity_digest()
    monkeypatch.setenv("OPENCODE_CONFIG", str(launch / "opencode.json"))
    assert native_launch_identity_digest() != before


def test_a_path_lookup_binds_its_absolute_result_not_symlink_targets(
    launch, monkeypatch
):
    # The launch path is the identity: an in-place upgrade that re-points a
    # symlink at the same path does not move it.
    real_a = stub_executable(launch / "versions" / "a")
    real_b = stub_executable(launch / "versions" / "b")
    link_dir = launch / "links"
    link_dir.mkdir()
    link = link_dir / executable_name("claude")
    try:
        link.symlink_to(real_a)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create symlinks")
    monkeypatch.setenv("PATH", str(link_dir))
    before = native_launch_identity_digest()
    link.unlink()
    link.symlink_to(real_b)
    assert native_launch_identity_digest() == before


def test_cloud_account_variables_bind_only_on_a_cloud_provider(launch, monkeypatch):
    direct = native_launch_identity_digest()
    monkeypatch.setenv("AWS_PROFILE", "synthetic-profile-a")
    assert native_launch_identity_digest() == direct, "inert for the direct API"
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    bedrock = native_launch_identity_digest()
    monkeypatch.setenv("AWS_PROFILE", "synthetic-profile-b")
    assert native_launch_identity_digest() != bedrock
    # Rotating a cloud secret does not move it; only its presence does.
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-token-one")
    rotated = native_launch_identity_digest()
    monkeypatch.setenv("AWS_SESSION_TOKEN", "synthetic-token-two")
    assert native_launch_identity_digest() == rotated


def test_a_relative_path_entry_resolves_to_one_absolute_executable(launch, monkeypatch):
    # A relative PATH entry is resolved against this process's working
    # directory by the lookup; the launched and fingerprinted path is the
    # absolute result, so a later cwd change cannot silently re-point it.
    monkeypatch.setenv("PATH", "bin")
    resolved = _resolve_cli_path("claude-code")
    assert resolved == str((launch / "bin" / executable_name("claude")).absolute())
    assert resolved is not None and Path(resolved).is_absolute()


@pytest.mark.parametrize(
    "name",
    ["CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_TOOL_CALL_ID", "Codex_Thread_Id"],
)
def test_a_parent_codex_session_id_does_not_move_the_codex_identity(
    launch, monkeypatch, name
):
    # Codex injects these into every command it runs; restarting the daemon
    # from another Codex session must keep the same identity.
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    monkeypatch.setenv(name, "synthetic-session-one")
    before = native_launch_identity_digest()
    monkeypatch.setenv(name, "synthetic-session-two")
    assert native_launch_identity_digest() == before
    monkeypatch.delenv(name)
    assert native_launch_identity_digest() == before


def test_codex_sandbox_settings_stay_bound(launch, monkeypatch):
    monkeypatch.setenv(ENV_NATIVE_MODEL, "local-cli/codex")
    before = native_launch_identity_digest()
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    assert native_launch_identity_digest() != before


def test_claude_code_oauth_token_binds_presence_not_value(launch, monkeypatch):
    unset = native_launch_identity_digest()
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-oauth-one")
    present = native_launch_identity_digest()
    assert present != unset, "a subscription token selects the credential source"
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-oauth-two")
    assert native_launch_identity_digest() == present, "rotation does not move it"
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
    assert native_launch_identity_digest() == unset


@pytest.mark.parametrize(
    ("switch", "name"),
    [
        ("CLAUDE_CODE_USE_VERTEX", "GOOGLE_APPLICATION_CREDENTIALS"),
        ("CLAUDE_CODE_USE_BEDROCK", "AWS_SHARED_CREDENTIALS_FILE"),
        ("CLAUDE_CODE_USE_BEDROCK", "AWS_CONFIG_FILE"),
        ("CLAUDE_CODE_USE_BEDROCK", "AWS_WEB_IDENTITY_TOKEN_FILE"),
    ],
)
def test_a_credential_file_variable_binds_its_absolute_path(
    launch, monkeypatch, switch, name
):
    monkeypatch.setenv(switch, "1")
    # The file is never opened: a path that does not exist binds the same way.
    monkeypatch.setenv(name, str(launch / "account-a" / "credentials.json"))
    absolute = native_launch_identity_digest()
    monkeypatch.setenv(name, str(launch / "account-b" / "credentials.json"))
    assert native_launch_identity_digest() != absolute, "another file, another account"
    monkeypatch.setenv(name, str(Path("account-a") / "credentials.json"))
    assert native_launch_identity_digest() == absolute, "same file, relative spelling"


def test_cloud_secrets_stay_presence_only_beside_file_paths(launch, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret-one")
    before = native_launch_identity_digest()
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret-two")
    assert native_launch_identity_digest() == before
