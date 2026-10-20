"""Runtime configuration reads never provision an operator environment."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from reflexio.cli import env_loader


def test_plain_read_does_not_create_missing_user_environment(monkeypatch, tmp_path):
    target = tmp_path / "home" / ".env"
    monkeypatch.setattr(env_loader, "_find_env_example", lambda _package: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [target])
    assert env_loader.load_existing_reflexio_env() is None
    assert env_loader.get_loaded_env_path() is None
    assert not target.parent.exists()


def test_plain_read_preserves_file_bytes_and_process_precedence(monkeypatch, tmp_path):
    target = tmp_path / ".env"
    content = b"READ_ONLY_VALUE=file\nREAD_ONLY_BLANK=\n"
    target.write_bytes(content)
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [target])
    monkeypatch.setenv("READ_ONLY_VALUE", "process")
    monkeypatch.delenv("READ_ONLY_BLANK", raising=False)
    assert env_loader.load_existing_reflexio_env() == target
    assert os.environ["READ_ONLY_VALUE"] == "process"
    assert "READ_ONLY_BLANK" not in os.environ
    assert target.read_bytes() == content


@pytest.mark.parametrize("mode", ["platform", "self_host"])
def test_bundled_mode_is_read_in_memory_with_identical_interpolation(
    monkeypatch, tmp_path, mode
):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", home)
    monkeypatch.setattr(
        env_loader,
        "_find_mode_template",
        lambda _mode, _package: (
            "READ_ONLY_ROOT=bundled\nREAD_ONLY_CHILD=${READ_ONLY_ROOT}\nREAD_ONLY_BLANK=\n"
        ),
    )
    monkeypatch.setenv("DEPLOYMENT_MODE", mode)
    monkeypatch.setenv("READ_ONLY_ROOT", "process")
    monkeypatch.delenv("READ_ONLY_CHILD", raising=False)
    monkeypatch.delenv("READ_ONLY_BLANK", raising=False)
    assert env_loader.load_existing_reflexio_env_for_mode() is None
    assert os.environ["READ_ONLY_ROOT"] == "process"
    assert os.environ["READ_ONLY_CHILD"] == "process"
    assert "READ_ONLY_BLANK" not in os.environ
    assert not home.exists()
    assert env_loader.get_loaded_env_path() is None


def test_mode_read_preserves_existing_home_and_cwd_precedence(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    secrets = home / ".env.platform"
    secrets.write_text("READ_ONLY_SECRET=home\nREAD_ONLY_SHARED=home\n")
    preset = tmp_path / ".env.platform"
    preset.write_text("READ_ONLY_SHARED=cwd\nREAD_ONLY_PORT=8100\n")
    original = (secrets.read_bytes(), preset.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", home)
    monkeypatch.setenv("DEPLOYMENT_MODE", "platform")
    for name in ("READ_ONLY_SECRET", "READ_ONLY_SHARED", "READ_ONLY_PORT"):
        monkeypatch.delenv(name, raising=False)
    loaded = env_loader.load_existing_reflexio_env_for_mode()
    assert loaded is not None
    assert loaded.resolve() == preset
    assert os.environ["READ_ONLY_SHARED"] == "home"
    assert os.environ["READ_ONLY_SECRET"] == "home"
    assert os.environ["READ_ONLY_PORT"] == "8100"
    assert (secrets.read_bytes(), preset.read_bytes()) == original


def test_plain_mode_fallback_does_not_provision(monkeypatch, tmp_path):
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    monkeypatch.setattr(env_loader, "_find_env_example", lambda _package: None)
    target = tmp_path / "missing" / ".env"
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [target])
    assert env_loader.load_existing_reflexio_env_for_mode() is None
    assert not target.parent.exists()


@pytest.mark.parametrize("setup", [False, True])
def test_bundled_plain_defaults_are_evaluated_once(monkeypatch, tmp_path, setup):
    """A second parse could resolve forward references differently."""
    target = tmp_path / "home" / ".env"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(target))
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [target])
    monkeypatch.setattr(
        env_loader,
        "_find_env_example",
        lambda _package: (
            "READ_ONLY_FORWARD=${READ_ONLY_LATER}\nREAD_ONLY_LATER=value\n"
        ),
    )
    monkeypatch.delenv("READ_ONLY_FORWARD", raising=False)
    monkeypatch.delenv("READ_ONLY_LATER", raising=False)
    if setup:
        assert env_loader.load_reflexio_env() == target
    else:
        assert env_loader.load_existing_reflexio_env() is None
    assert "READ_ONLY_FORWARD" not in os.environ
    assert os.environ["READ_ONLY_LATER"] == "value"
    assert target.exists() is setup


def test_actual_server_import_does_not_create_a_user_env(tmp_path):
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / "home" / ".env"
    result = subprocess.run(  # noqa: S603 - fixed import against an isolated test home
        [sys.executable, "-c", "import reflexio.server; print('imported')"],
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "REFLEXIO_ENV_FILE": str(target),
            "PYTHONPATH": str(root),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "imported\n"
    assert not target.parent.exists()
    assert not list(tmp_path.rglob(".env*"))
