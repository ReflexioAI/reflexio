"""Saved scheduling choices remain effective and malformed policies fail closed."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.cli.bootstrap_config import load_storage_from_config
from reflexio.models.config_schema import (
    PlaybookOptimizerConfig,
    strip_retired_config_fields,
)
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

SCHEDULING_DEFAULTS = {
    "scheduler_jitter_seconds": 1.0,
    "abort_cooldown_threshold": 2,
    "cooldown_after_aborts_seconds": 3600,
}


@pytest.mark.parametrize("strings", [False, True])
def test_defaults_remain_supported_without_mutating_input(strings):
    legacy = {k: str(v) if strings else v for k, v in SCHEDULING_DEFAULTS.items()}
    payload = {"playbook_optimizer_config": {**legacy, "enabled": False}}
    original = deepcopy(payload)
    normalized = strip_retired_config_fields(payload)
    assert normalized == payload
    policy = PlaybookOptimizerConfig.model_validate(
        payload["playbook_optimizer_config"]
    )
    for key in SCHEDULING_DEFAULTS:
        assert policy.model_dump()[key] == SCHEDULING_DEFAULTS[key]
        assert key in policy.model_json_schema()["properties"]
    assert payload == original


@pytest.mark.parametrize(
    "key,value",
    [
        *[
            (key, value)
            for key in SCHEDULING_DEFAULTS
            for value in (
                None,
                "private-invalid-policy",
                -1,
                float("inf"),
                float("nan"),
            )
        ],
        ("abort_cooldown_threshold", 0),
        ("abort_cooldown_threshold", 1.5),
        ("cooldown_after_aborts_seconds", 1.5),
        ("scheduler_jitter_seconds", 10**100),
        ("cooldown_after_aborts_seconds", 10**100),
    ],
)
def test_invalid_policy_is_rejected_even_when_disabled(key, value):
    from reflexio.config_migration import ConfigurationMigrationError

    policy = {"enabled": False, key: value}
    for load, error_type in (
        (lambda: PlaybookOptimizerConfig.model_validate(policy), ValueError),
        (
            lambda: strip_retired_config_fields({"playbook_optimizer_config": policy}),
            ConfigurationMigrationError,
        ),
    ):
        with pytest.raises(error_type, match=key) as error:
            load()
        assert "private-invalid-policy" not in str(error.value)


def test_local_saved_default_policy_is_canonicalized(tmp_path):
    storage = LocalFileConfigStorage("scheduler-defaults", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "playbook_optimizer_config": SCHEDULING_DEFAULTS,
            }
        )
    )
    for _ in range(2):
        assert not storage.load_config().playbook_optimizer_config.enabled
    saved = json.loads(path.read_text())["playbook_optimizer_config"]
    assert {key: saved[key] for key in SCHEDULING_DEFAULTS} == SCHEDULING_DEFAULTS


def test_invalid_saved_policy_is_not_rewritten(tmp_path):
    storage = LocalFileConfigStorage("scheduler-custom", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "storage_config": {"type": "sqlite"},
            "playbook_optimizer_config": {
                "cooldown_after_aborts_seconds": "private-invalid-policy"
            },
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="cooldown_after_aborts_seconds"):
        storage.load_config()
    assert path.read_text() == original

    with pytest.raises(ValueError, match="cooldown_after_aborts_seconds"):
        load_storage_from_config(org_id="scheduler-custom", base_dir=str(tmp_path))
    assert path.read_text() == original


def test_custom_scheduling_policy_is_supported_and_saved(tmp_path):
    custom = {
        "scheduler_jitter_seconds": 0.25,
        "abort_cooldown_threshold": 4,
        "cooldown_after_aborts_seconds": 60,
    }
    storage = LocalFileConfigStorage("supported-scheduler", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "playbook_optimizer_config": custom,
            }
        )
    )
    for _ in range(2):
        loaded = storage.load_config()
        assert {
            key: getattr(loaded.playbook_optimizer_config, key) for key in custom
        } == custom
        storage.save_config(loaded)
    saved = json.loads(path.read_text())["playbook_optimizer_config"]
    assert {key: saved[key] for key in custom} == custom
    assert (
        load_storage_from_config(org_id="supported-scheduler", base_dir=str(tmp_path))
        is not None
    )


@pytest.mark.parametrize(
    "key", ["scheduler_jitter_seconds", "cooldown_after_aborts_seconds"]
)
def test_zero_delay_remains_supported(key):
    policy = PlaybookOptimizerConfig.model_validate({key: 0}).scheduling_policy
    assert getattr(policy, key) == 0
