import os

from reflexio.cli import env_loader


def test_resolve_mode_prefers_flag_then_env(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_MODE", "self_host")
    assert env_loader.resolve_mode(cli_mode="platform") == "platform"
    assert env_loader.resolve_mode(cli_mode=None) == "self_host"
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    assert env_loader.resolve_mode(cli_mode=None) is None


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
