"""Retired pending-tool execution tuning must never silently change a saved claim policy."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.cli.bootstrap_config import load_storage_from_config
from reflexio.config_migration import strip_retired_config_fields
from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

LEGACY_DEFAULTS = {
    "resume_poll_interval_seconds": 5.0,
    "resume_claim_ttl_seconds": 600,
}


@pytest.mark.parametrize("strings", [False, True])
def test_defaults_canonicalize_without_mutating_input(strings):
    legacy = {k: str(v) if strings else v for k, v in LEGACY_DEFAULTS.items()}
    payload = {"pending_tool_call_config": {**legacy, "enabled": False}}
    original = deepcopy(payload)
    normalized = strip_retired_config_fields(payload)
    assert normalized == {"pending_tool_call_config": {"enabled": False}}
    policy = PendingToolCallConfig.model_validate(payload["pending_tool_call_config"])
    for key in LEGACY_DEFAULTS:
        assert key not in policy.model_dump()
        assert key not in policy.model_json_schema()["properties"]
    assert payload == original


@pytest.mark.parametrize("key", LEGACY_DEFAULTS)
@pytest.mark.parametrize("value", [0, 3, None, "invalid", -1, 1.5])
def test_custom_or_invalid_policy_is_rejected_even_when_disabled(key, value):
    legacy = {"enabled": False, key: value}
    with pytest.raises(ValueError, match=key):
        PendingToolCallConfig.model_validate(legacy)
    with pytest.raises(ValueError, match=key):
        strip_retired_config_fields({"pending_tool_call_config": legacy})


def test_local_saved_default_policy_is_canonicalized(tmp_path):
    storage = LocalFileConfigStorage("pending-defaults", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "pending_tool_call_config": LEGACY_DEFAULTS,
            }
        )
    )
    original = path.read_bytes()
    for _ in range(2):
        assert not storage.load_config().pending_tool_call_config.enabled
    assert path.read_bytes() == original
    from reflexio.cli.config_upgrade import convert_config_file

    canonical = tmp_path / "canonical.json"
    convert_config_file(path, canonical)
    assert path.read_bytes() == original
    saved = json.loads(canonical.read_text())["pending_tool_call_config"]
    assert not set(LEGACY_DEFAULTS).intersection(saved)


def test_custom_saved_policy_is_not_rewritten(tmp_path):
    storage = LocalFileConfigStorage("pending-custom", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "storage_config": {"type": "sqlite"},
            "pending_tool_call_config": {"resume_claim_ttl_seconds": 60},
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="pending_tool_call_config"):
        storage.load_config()
    assert path.read_text() == original

    with pytest.raises(ValueError, match="pending_tool_call_config"):
        load_storage_from_config(org_id="pending-custom", base_dir=str(tmp_path))
    assert path.read_text() == original


def test_file_and_cli_propagate_new_migration_error_families(tmp_path):
    from unittest.mock import patch

    from reflexio.config_migration import ConfigurationMigrationError

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
def test_resume_timing_supplied_mapping_defaults_ignore_ambient(monkeypatch, blank):
    from reflexio.pending_tool_policy import resolve_resume_timing

    monkeypatch.setenv("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "invalid-private-poll")
    monkeypatch.setenv("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", "invalid-private-ttl")
    for values in (
        {},
        {
            "REFLEXIO_RESUME_POLL_INTERVAL_SECONDS": blank,
            "REFLEXIO_RESUME_CLAIM_TTL_SECONDS": blank,
        },
    ):
        policy = resolve_resume_timing(values)
        assert policy.poll_seconds == 5.0
        assert policy.claim_ttl_seconds == 600


@pytest.mark.parametrize("ttl", [1, 1800, ((1 << 31) - 1) // 6])
def test_resume_timing_preserves_independent_custom_policy(ttl):
    from reflexio.pending_tool_policy import resolve_resume_timing

    policy = resolve_resume_timing(
        {
            "REFLEXIO_RESUME_POLL_INTERVAL_SECONDS": "0.25",
            "REFLEXIO_RESUME_CLAIM_TTL_SECONDS": str(ttl),
        }
    )
    assert policy.poll_seconds == 0.25
    assert policy.claim_ttl_seconds == ttl


@pytest.mark.parametrize(
    "name,value",
    [
        ("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", value)
        for value in ["0", "-1", "nan", "inf", "private-invalid", "1e100"]
    ]
    + [
        ("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", value)
        for value in [
            "0",
            "-1",
            "1.5",
            "inf",
            "private-invalid",
            str(((1 << 31) - 1) // 6 + 1),
        ]
    ],
)
def test_resume_timing_rejects_invalid_policy_without_values(name, value):
    from reflexio.pending_tool_policy import resolve_resume_timing

    with pytest.raises(ValueError, match=name) as error:
        resolve_resume_timing({name: value})
    assert "private-invalid" not in str(error.value)


@pytest.mark.parametrize(
    "name",
    ["REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "REFLEXIO_RESUME_CLAIM_TTL_SECONDS"],
)
def test_oss_startup_refuses_invalid_resume_timing_before_serving(tmp_path, name):
    import os
    import subprocess
    import sys

    empty = tmp_path / "empty-env"
    empty.write_text("")
    result = subprocess.run(  # noqa: S603 - fixed interpreter/script, isolated env.
        [sys.executable, "-c", "import reflexio.server"],
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "REFLEXIO_ENV_FILE": str(empty),
            "REFLEXIO_LOG_DIR": str(tmp_path / "logs"),
            name: "invalid-private-timing",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert name in result.stderr
    assert "invalid-private-timing" not in result.stderr
