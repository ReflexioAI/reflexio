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
