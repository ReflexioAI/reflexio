"""Tests for the ``REFLEXIO_ENV_FILE`` override of the user-level .env path.

Lets a second local backend keep its env separate from the OSS reflexio default
under ``~/.reflexio`` (claude-smart points it at ``~/.claude-smart/.env``) while
still sharing ``~/.reflexio/data``.
"""

import os

from reflexio.cli import env_loader

_ENV = "REFLEXIO_ENV_FILE"


def test_user_env_file_defaults_when_unset(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert env_loader.user_env_file() == env_loader._USER_ENV_FILE
    assert env_loader.get_env_path() == env_loader._USER_ENV_FILE


def test_user_env_file_honors_override(monkeypatch, tmp_path):
    target = tmp_path / "claude-smart" / ".env"
    monkeypatch.setenv(_ENV, str(target))
    assert env_loader.user_env_file() == target
    assert env_loader.get_env_path() == target
    # Search order: ./.env first, then the (overridden) user-level file.
    assert env_loader._env_search_paths()[-1] == target


def test_user_env_file_expands_user(monkeypatch):
    monkeypatch.setenv(_ENV, "~/.claude-smart/.env")
    assert env_loader.user_env_file() == (env_loader.Path.home() / ".claude-smart" / ".env")


def test_blank_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(_ENV, "   ")
    assert env_loader.user_env_file() == env_loader._USER_ENV_FILE


def test_load_reflexio_env_prefers_override_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no ./.env present
    override = tmp_path / ".env.cs"
    override.write_text("REFLEXIO_ENV_FILE_TEST_MARKER=present\n")
    monkeypatch.setenv(_ENV, str(override))
    monkeypatch.delenv("REFLEXIO_ENV_FILE_TEST_MARKER", raising=False)
    try:
        loaded = env_loader.load_reflexio_env()
        assert loaded == override
        assert os.environ.get("REFLEXIO_ENV_FILE_TEST_MARKER") == "present"
    finally:
        os.environ.pop("REFLEXIO_ENV_FILE_TEST_MARKER", None)


def test_load_reflexio_env_auto_creates_at_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no ./.env present
    override = tmp_path / "claude-smart" / ".env"  # parent dir does not exist yet
    monkeypatch.setenv(_ENV, str(override))
    # Don't pull the freshly written template into the process env.
    monkeypatch.setattr(env_loader, "_load_dotenv_pruned", lambda *_, **__: None)

    created = env_loader.load_reflexio_env()

    assert created == override
    assert override.exists()  # parent dir + file created at the override location
