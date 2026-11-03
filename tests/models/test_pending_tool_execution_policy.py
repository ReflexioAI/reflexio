"""Historical timing conversion never restores a second execution policy."""

import json
import os
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from reflexio.cli.bootstrap_config import load_storage_from_config
from reflexio.config_file import LocalConfigFile
from reflexio.config_migration import (
    ConfigurationMigrationError,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.pending_tool_policy import warn_retired_resume_timing
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

FIELDS = ("resume_poll_interval_seconds", "resume_claim_ttl_seconds")
ENV = ("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "REFLEXIO_RESUME_CLAIM_TTL_SECONDS")


@pytest.mark.parametrize("poll,ttl", [(5.0, 600), (0.25, 1800), ("3", "1")])
def test_explicit_conversion_retires_valid_custom_policy_without_mutation(poll, ttl):
    pending = {
        "enabled": False,
        FIELDS[0]: poll,
        FIELDS[1]: ttl,
        "hmac_secrets": ["synthetic-signing-secret"],
        "opaque": {"retain": "unchanged"},
        "max_resume_attempts": 7,
    }
    original = deepcopy(pending)
    converted = strip_retired_config_fields({"pending_tool_call_config": pending})
    assert converted == {
        "pending_tool_call_config": {
            **{k: v for k, v in pending.items() if k not in (*FIELDS, "enabled")},
            "pending_ttl_seconds": 0,
        }
    }
    assert pending == original
    for name in FIELDS:
        assert name not in PendingToolCallConfig.model_json_schema()["properties"]


@pytest.mark.parametrize("key", FIELDS)
@pytest.mark.parametrize("value", [5, 600, None, "invalid"])
def test_current_model_rejects_even_historical_defaults(key, value):
    with pytest.raises(ValueError, match=key):
        PendingToolCallConfig.model_validate({key: value})


@pytest.mark.parametrize(
    "key,value",
    [
        (FIELDS[0], value)
        for value in (
            0,
            -1,
            None,
            "PRIVATE_INVALID",
            "nan",
            "inf",
            threading.TIMEOUT_MAX * 2,
        )
    ]
    + [
        (FIELDS[1], value)
        for value in (0, -1, None, "PRIVATE_INVALID", 1.5, ((1 << 31) - 1) // 6 + 1)
    ],
)
def test_conversion_rejects_malformed_history_even_when_disabled(key, value):
    source = {"pending_tool_call_config": {"enabled": False, key: value}}
    before = deepcopy(source)
    with pytest.raises(ConfigurationMigrationError, match=key) as failure:
        strip_retired_config_fields(source)
    assert "PRIVATE_INVALID" not in str(failure.value)
    assert source == before


@pytest.mark.parametrize("ttl", [600, 60])
def test_local_policy_requires_preparation_and_preserves_exact_original(tmp_path, ttl):
    storage = LocalFileConfigStorage("pending-history", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "storage_config": {
                "type": "sqlite",
                "db_path": str(tmp_path / "owned.sqlite"),
            },
            "pending_tool_call_config": {
                "enabled": False,
                "resume_claim_ttl_seconds": ttl,
            },
        }
    )
    path.write_text(original)
    for load in (
        storage.load_config,
        lambda: load_storage_from_config(
            org_id="pending-history", base_dir=str(tmp_path)
        ),
    ):
        with pytest.raises(ValueError, match="pending_tool_call_config"):
            load()
        assert path.read_text() == original
    assert LocalConfigFile(path).prepare()
    assert not (storage.load_config().pending_tool_call_config.pending_ttl_seconds > 0)
    assert (
        "resume_claim_ttl_seconds"
        not in json.loads(path.read_text())["pending_tool_call_config"]
    )
    assert not LocalConfigFile(path).prepare()


def test_file_and_cli_propagate_new_migration_error_families(tmp_path):
    storage = LocalFileConfigStorage("future-policy", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    before = json.dumps({"storage_config": {"type": "sqlite"}})
    path.write_text(before)
    with patch(
        "reflexio.server.services.configurator.local_file_config_storage.validate_stored_config",
        side_effect=ConfigurationMigrationError("new policy family"),
    ):
        for load in (
            storage.load_config,
            lambda: load_storage_from_config("future-policy", base_dir=str(tmp_path)),
        ):
            with pytest.raises(ConfigurationMigrationError, match="new policy family"):
                load()
            assert path.read_text() == before


@pytest.mark.parametrize("blank", ["", "  "])
def test_obsolete_diagnostic_uses_only_supplied_mapping_and_blank_is_unset(
    monkeypatch, caplog, blank
):
    for name in ENV:
        monkeypatch.setenv(name, "PRIVATE_AMBIENT")
    warn_retired_resume_timing({})
    warn_retired_resume_timing(dict.fromkeys(ENV, blank))
    assert not caplog.records


@pytest.mark.parametrize("value", ["0.25", "1800", "PRIVATE_INVALID", "nan", "1e100"])
def test_obsolete_assignment_is_value_free_diagnostic_not_policy(value: str, caplog):
    from reflexio import durable_learning_policy

    before = durable_learning_policy.resolve_discovery_timing({})
    mapping: dict[str, str] = dict.fromkeys(ENV, value)
    assert warn_retired_resume_timing(mapping) is None
    assert durable_learning_policy.resolve_discovery_timing(mapping) == before
    assert len(caplog.records) == 1
    assert all(name in caplog.text for name in ENV)
    assert "release_blocking=false" in caplog.text
    assert "PRIVATE_INVALID" not in caplog.text


@pytest.mark.parametrize("name", ENV)
def test_oss_startup_ignores_obsolete_value_with_diagnostic(tmp_path, name):
    empty = tmp_path / "empty-env"
    empty.touch()
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(  # noqa: S603 -- Fixed interpreter/script, isolated HOME/env.
        [sys.executable, "-c", "import reflexio.server; print('startup-ready')"],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "REFLEXIO_ENV_FILE": str(empty),
            "PYTHON_DOTENV_DISABLED": "1",
            "REFLEXIO_LOG_DIR": str(tmp_path / "logs"),
            "OPENAI_API_KEY": "sk-test-not-real",
            "REFLEXIO_NATIVE_COMPUTE_THREADS": "1",
            name: "PRIVATE_OBSOLETE",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "startup-ready" in output and name in output
    assert "obsolete_resume_timing" in output
    assert "PRIVATE_OBSOLETE" not in output
