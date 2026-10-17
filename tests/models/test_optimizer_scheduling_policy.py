"""Retired scheduler tuning must never silently change a saved retry policy."""

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

LEGACY_DEFAULTS = {
    "scheduler_jitter_seconds": 1.0,
    "abort_cooldown_threshold": 2,
    "cooldown_after_aborts_seconds": 3600,
}


@pytest.mark.parametrize("strings", [False, True])
def test_defaults_canonicalize_without_mutating_input(strings):
    legacy = {k: str(v) if strings else v for k, v in LEGACY_DEFAULTS.items()}
    payload = {"playbook_optimizer_config": {**legacy, "enabled": False}}
    original = deepcopy(payload)
    normalized = strip_retired_config_fields(payload)
    assert normalized == {"playbook_optimizer_config": {"enabled": False}}
    policy = PlaybookOptimizerConfig.model_validate(
        payload["playbook_optimizer_config"]
    )
    for key in LEGACY_DEFAULTS:
        assert key not in policy.model_dump()
        assert key not in policy.model_json_schema()["properties"]
    assert payload == original


@pytest.mark.parametrize("key", LEGACY_DEFAULTS)
@pytest.mark.parametrize("value", [0, 3, None, "invalid", -1, 1.5])
def test_custom_or_invalid_policy_is_rejected_even_when_disabled(key, value):
    legacy = {"enabled": False, key: value}
    with pytest.raises(ValueError, match=key):
        PlaybookOptimizerConfig.model_validate(legacy)
    with pytest.raises(ValueError, match=key):
        strip_retired_config_fields({"playbook_optimizer_config": legacy})


def test_local_saved_default_policy_is_canonicalized(tmp_path):
    storage = LocalFileConfigStorage("scheduler-defaults", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "playbook_optimizer_config": LEGACY_DEFAULTS,
            }
        )
    )
    for _ in range(2):
        assert not storage.load_config().playbook_optimizer_config.enabled
    saved = json.loads(path.read_text())["playbook_optimizer_config"]
    assert not set(LEGACY_DEFAULTS).intersection(saved)


def test_custom_saved_policy_is_not_rewritten(tmp_path):
    storage = LocalFileConfigStorage("scheduler-custom", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "storage_config": {"type": "sqlite"},
            "playbook_optimizer_config": {"cooldown_after_aborts_seconds": 60},
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError, match="cooldown_after_aborts_seconds"):
        storage.load_config()
    assert path.read_text() == original

    with pytest.raises(ValueError, match="cooldown_after_aborts_seconds"):
        load_storage_from_config(org_id="scheduler-custom", base_dir=str(tmp_path))
    assert path.read_text() == original
