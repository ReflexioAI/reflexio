"""Replay retirement is a versioned policy change, never a runtime alias."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.cli.config_upgrade import prepare_local_config
from reflexio.config_file import FILE_FORMAT_VERSION
from reflexio.config_migration import (
    ConfigurationMigrationError,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


@pytest.mark.parametrize("retries,backoff", [(0, 0), (3, 1), (7, 2.5), ("4", "0.25")])
def test_upgrade_retires_only_delivery_policy_and_preserves_history(retries, backoff):
    source = {
        "playbook_optimizer_config": {
            "metric_call_budget": 0,
            "webhook_timeout_seconds": 23,
            "webhook_max_retries": retries,
            "webhook_backoff_base_seconds": backoff,
            "abort_cooldown_threshold": 4,
        },
        "opaque": {
            "gepa_publication_authority": {
                "backend_identity": {"max_retries": 7, "backoff_base_seconds": "2.5"},
                "digest": "historical",
            }
        },
    }
    original = deepcopy(source)
    with pytest.raises(ValueError, match="webhook_"):
        PlaybookOptimizerConfig.model_validate(source["playbook_optimizer_config"])
    converted = strip_retired_config_fields(source)
    expected = deepcopy(source)
    for name in ("webhook_max_retries", "webhook_backoff_base_seconds"):
        expected["playbook_optimizer_config"].pop(name)
    assert converted == expected and source == original
    assert strip_retired_config_fields(converted) == converted


@pytest.mark.parametrize(
    "field", ["webhook_max_retries", "webhook_backoff_base_seconds"]
)
@pytest.mark.parametrize("value", [-1, None, "private-invalid"])
def test_malformed_historical_delivery_is_not_discarded(field, value):
    with pytest.raises(ConfigurationMigrationError, match=field) as error:
        strip_retired_config_fields({"playbook_optimizer_config": {field: value}})
    assert "private-invalid" not in str(error.value)


def test_local_generation6_upgrade_retains_original_bytes(tmp_path):
    storage = LocalFileConfigStorage("delivery", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": 6},
            "storage_config": {"type": "sqlite"},
            "playbook_optimizer_config": {
                "metric_call_budget": 0,
                "webhook_max_retries": 7,
                "webhook_backoff_base_seconds": 2.5,
                "webhook_timeout_seconds": 23,
            },
        }
    )
    path.write_text(original)
    with pytest.raises(ValueError):
        storage.load_config()
    assert prepare_local_config("delivery", base_dir=str(tmp_path))
    assert storage.load_config().playbook_optimizer_config.webhook_timeout_seconds == 23
    assert (
        json.loads(path.read_text())["_reflexio_config_file"]["version"]
        == FILE_FORMAT_VERSION
    )
    backups = list((path.parent / ".config-migrations" / path.name).rglob("*"))
    assert any(p.is_file() and p.read_bytes() == original.encode() for p in backups)
    current = path.read_bytes()
    assert not prepare_local_config("delivery", base_dir=str(tmp_path))
    assert path.read_bytes() == current
