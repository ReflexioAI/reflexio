"""Only the historical upgrade owner consumes retired optimizer timer inputs."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.cli.config_upgrade import prepare_local_config
from reflexio.config_migration import (
    ConfigurationMigrationError,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.mark.parametrize("jitter", [0, 1.0, 17.5, "300"])
def test_explicit_upgrade_retires_timer_without_resetting_other_policy(jitter):
    policy = {
        "metric_call_budget": 0,
        "scheduler_jitter_seconds": jitter,
        "abort_cooldown_threshold": 4,
        "cooldown_after_aborts_seconds": 17,
    }
    payload = {
        "playbook_optimizer_config": policy,
        "metadata": {"scheduler_jitter_seconds": 123},
    }
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="scheduler_jitter_seconds"):
        PlaybookOptimizerConfig.model_validate(policy)
    converted = strip_retired_config_fields(payload)
    assert converted["playbook_optimizer_config"] == {
        key: value for key, value in policy.items() if key != "scheduler_jitter_seconds"
    }
    assert converted["metadata"] == payload["metadata"]
    assert payload == original


@pytest.mark.parametrize(
    "jitter", [-1, None, float("nan"), float("inf"), 10**100, "private-policy"]
)
def test_malformed_historical_timer_is_not_silently_discarded(jitter):
    with pytest.raises(
        ConfigurationMigrationError, match="scheduler_jitter_seconds"
    ) as error:
        strip_retired_config_fields(
            {"playbook_optimizer_config": {"scheduler_jitter_seconds": jitter}}
        )
    assert "private-policy" not in str(error.value)


def test_local_startup_owner_upgrades_saved_timer_before_strict_read(tmp_path):
    storage = LocalFileConfigStorage("timer-upgrade", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": 5},
            "storage_config": {"type": "sqlite"},
            "playbook_optimizer_config": {
                "metric_call_budget": 0,
                "scheduler_jitter_seconds": 300,
                "abort_cooldown_threshold": 4,
                "cooldown_after_aborts_seconds": 17,
            },
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="playbook_optimizer_config"):
        storage.load_config()
    assert path.read_text() == original
    assert prepare_local_config("timer-upgrade", base_dir=str(tmp_path))
    loaded = storage.load_config().playbook_optimizer_config
    assert loaded.metric_call_budget == 0
    assert loaded.abort_cooldown_threshold == 4
    assert loaded.cooldown_after_aborts_seconds == 17
    assert (
        "scheduler_jitter_seconds"
        not in json.loads(path.read_text())["playbook_optimizer_config"]
    )
    current = path.read_bytes()
    assert not prepare_local_config("timer-upgrade", base_dir=str(tmp_path))
    assert path.read_bytes() == current
