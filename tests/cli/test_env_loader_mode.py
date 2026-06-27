import os

import pytest

from reflexio.cli import env_loader


def test_resolve_mode_prefers_flag_then_env(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_MODE", "self_host")
    assert env_loader.resolve_mode(cli_mode="platform") == "platform"
    assert env_loader.resolve_mode(cli_mode=None) == "self_host"
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    assert env_loader.resolve_mode(cli_mode=None) is None


def test_resolve_mode_blank_is_none(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    # A whitespace-only flag must not fall through to a ".env." filename.
    assert env_loader.resolve_mode(cli_mode="   ") is None
    monkeypatch.setenv("DEPLOYMENT_MODE", "  ")
    assert env_loader.resolve_mode(cli_mode=None) is None


def test_resolve_mode_rejects_path_traversal(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    for unsafe in ("../etc/passwd", "a/b", "self_host/..", "mode$"):
        with pytest.raises(ValueError, match="Invalid deployment mode"):
            env_loader.resolve_mode(cli_mode=unsafe)


def test_mode_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.platform").write_text(
        "DEPLOYMENT_MODE=platform\nBACKEND_PORT=8091\n"
    )
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    # The loader uses override=False, so an ambient BACKEND_PORT already present in
    # the process environment (e.g. exported by the shell running the test suite)
    # would win over the file value. Clear it first so this test exercises file
    # loading rather than whatever the surrounding shell happens to export.
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    loaded = env_loader.load_reflexio_env_for_mode()
    assert loaded is not None and loaded.name == ".env.platform"
    assert os.environ["BACKEND_PORT"] == "8091"


def test_override_false_process_env_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.platform").write_text("BACKEND_PORT=8091\n")
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setenv("BACKEND_PORT", "9999")
    env_loader.load_reflexio_env_for_mode()
    assert os.environ["BACKEND_PORT"] == "9999"


def test_load_drops_blank_assignments(tmp_path, monkeypatch):
    # A blank ``KEY=`` line must behave as unset, not as ``""`` — otherwise it
    # silently defeats every ``os.getenv(KEY, default)`` fallback (the HF_HOME
    # cache-in-repo-root bug class).
    env_file = tmp_path / ".env"
    env_file.write_text("BLANK=\nWS=   \nREAL=value\n")
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [env_file])
    monkeypatch.delenv("BLANK", raising=False)
    monkeypatch.delenv("WS", raising=False)
    monkeypatch.delenv("REAL", raising=False)
    env_loader.load_reflexio_env()
    assert os.getenv("BLANK") is None
    assert os.getenv("WS") is None
    assert os.environ["REAL"] == "value"


def test_blank_assignment_does_not_clobber_process_env(tmp_path, monkeypatch):
    # A real value already in the process environment must survive a later blank
    # file assignment of the same key.
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP=\n")
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [env_file])
    monkeypatch.setenv("KEEP", "real")
    env_loader.load_reflexio_env()
    assert os.environ["KEEP"] == "real"


def test_autogen_writes_home_not_committed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.platform").write_text("DEPLOYMENT_MODE=platform\n")
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", tmp_path / "home")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    env_loader.load_reflexio_env_for_mode(auto_generate_keys=["JWT_SECRET_KEY"])
    assert "JWT_SECRET_KEY" not in (tmp_path / ".env.platform").read_text()
    assert (tmp_path / "home" / ".env.platform").exists()
    assert "JWT_SECRET_KEY" in (tmp_path / "home" / ".env.platform").read_text()
