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


@pytest.mark.parametrize("source", ["process", "project", "home"])
@pytest.mark.parametrize("legacy", ["off", "jev"])
def test_launch_consumes_only_the_shipped_inactive_search_judge_default(
    monkeypatch, tmp_path, source, legacy
):
    mode = "self_host"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", home_dir)
    monkeypatch.setattr(env_loader, "_loaded_env_path", None)
    monkeypatch.setenv("DEPLOYMENT_MODE", mode)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("REFLEXIO_SEARCH_JUDGE_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DB_URL", "isolated-for-cleanup")
    monkeypatch.delenv("DATA_DB_URL")
    name = "REFLEXIO_SEARCH_JUDGE"
    selected = (home_dir if source == "home" else tmp_path) / f".env.{mode}"
    original = "DATA_DB_URL=fixture-target\n"
    if source == "process":
        monkeypatch.setenv(name, legacy)
    else:
        original = f"{name}={legacy}\n" + original
    selected.write_text(original)
    if legacy == "jev":
        with pytest.raises(ValueError, match=name):
            env_loader.load_existing_reflexio_env_for_mode()
        assert "DATA_DB_URL" not in os.environ
    else:
        env_loader.load_existing_reflexio_env_for_mode()
        assert name not in os.environ
        assert "REFLEXIO_SEARCH_JUDGE_API_KEY" not in os.environ
        assert os.environ["DATA_DB_URL"] == "fixture-target"
        with pytest.raises(ValueError, match=name):
            reject_removed_env({name: legacy})
    assert selected.read_text() == original


def test_canonical_process_target_keeps_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DB_URL", "process-target")
    env_file = tmp_path / ".env"
    env_file.write_text("DATA_DB_URL=file-target\n")
    env_loader._load_dotenv_pruned(env_file)
    assert os.environ["DATA_DB_URL"] == "process-target"


def test_scoped_rejection_observes_live_assignments_without_rejecting_other_scopes():
    canonical = "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS"
    removed = "REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS"
    env: dict[str, str] = {}
    reject_removed_env(env, canonical=canonical)
    env[removed] = ""
    with pytest.raises(ValueError, match=removed):
        reject_removed_env(env, canonical=canonical)
    del env[removed]
    env["DATA_SUPABASE_DB_URL"] = "private-target"
    reject_removed_env(env, canonical=canonical)
    with pytest.raises(ValueError, match="DATA_SUPABASE_DB_URL") as error:
        reject_removed_env(env)
    assert "private-target" not in str(error.value)


@pytest.mark.parametrize("disabled", ["1", "true", "T", "yes", "Y"])
def test_disabled_dotenv_does_not_validate_or_prune_file(
    monkeypatch, tmp_path, disabled
):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", disabled)
    monkeypatch.setenv("KEEP_PROCESS_VALUE", "")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_DB_URL=private-target\nDATA_DB_URL=wrong-target\nKEEP_PROCESS_VALUE=\n"
    )
    env_loader._load_dotenv_pruned(env_file)
    assert "DATA_DB_URL" not in os.environ
    assert os.environ["KEEP_PROCESS_VALUE"] == ""
    monkeypatch.setenv("POSTGRES_DB_URL", "private-process-target")
    with pytest.raises(ValueError, match="POSTGRES_DB_URL"):
        env_loader._load_dotenv_pruned(env_file)


def test_worker_upgrade_diagnostic_preserves_execution_count():
    with pytest.raises(ValueError, match="explicit server --workers N") as error:
        reject_removed_env({"WEB_CONCURRENCY": "private-value"})
    assert "never selected execution capacity" in str(error.value)
    assert "private-value" not in str(error.value)


def test_statement_timeout_retirement_explains_the_policy_change():
    with pytest.raises(
        ValueError, match="REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS"
    ) as error:
        reject_removed_env({"REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS": "private-value"})
    message = str(error.value)
    assert "remaining checked-out connection lease" in message
    assert "do not automatically copy the old value" in message
    assert "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS" in message
    assert "private-value" not in message


@pytest.mark.parametrize("value", ["", " ", "private-batch-value"])
def test_regeneration_batch_upgrade_names_existing_job_option(value):
    with pytest.raises(
        ValueError, match="REFLEXIO_EMBEDDING_REGENERATION_BATCH_SIZE"
    ) as error:
        reject_removed_env({"REFLEXIO_EMBEDDING_REGENERATION_BATCH_SIZE": value})
    assert "batch_size" in str(error.value)
    assert "existing jobs" in str(error.value)
    assert "private-batch-value" not in str(error.value)
