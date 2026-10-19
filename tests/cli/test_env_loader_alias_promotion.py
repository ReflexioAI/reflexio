"""Removed database names cannot be redirected by dotenv or silently promoted."""

import os

import pytest

from reflexio.cli import env_loader
from reflexio.removed_env import REMOVED_ENV_ALIASES, reject_removed_env


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (*REMOVED_ENV_ALIASES, "DATA_DB_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["DATA_SUPABASE_DB_URL", "POSTGRES_DB_URL"])
@pytest.mark.parametrize("mode", [None, "platform", "self_host"])
def test_old_process_target_fails_before_a_dotenv_target_is_loaded(
    monkeypatch, tmp_path, name, mode
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / (f".env.{mode}" if mode else ".env")).write_text(
        "DATA_DB_URL=local-target\n"
    )
    monkeypatch.setenv(name, "private-prod-target")
    with pytest.raises(ValueError, match=name) as error:
        env_loader.load_reflexio_env_for_mode(cli_mode=mode)
    assert "DATA_DB_URL" not in os.environ
    assert "private-prod-target" not in str(error.value)


@pytest.mark.parametrize("name", list(REMOVED_ENV_ALIASES))
@pytest.mark.parametrize("value", ["", " ", "private-value"])
def test_removed_names_rejected_in_process_and_file(monkeypatch, tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        reject_removed_env({name: value})
    env_file = tmp_path / ".env"
    env_file.write_text(f"{name}={value}\nDATA_DB_URL=local-target\n")
    with pytest.raises(ValueError, match=name) as error:
        env_loader._load_dotenv_pruned(env_file)
    assert "DATA_DB_URL" not in os.environ
    assert "private-value" not in str(error.value)


def test_canonical_process_target_keeps_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DB_URL", "process-target")
    env_file = tmp_path / ".env"
    env_file.write_text("DATA_DB_URL=file-target\n")
    env_loader._load_dotenv_pruned(env_file)
    assert os.environ["DATA_DB_URL"] == "process-target"
