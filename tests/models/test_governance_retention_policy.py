"""Deletion consent is owned by one age; historical defaults never grant it."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflexio.config_env import ConfigurationMigrationError, StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile
from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import (
    Config,
    GovernanceRetentionConfig,
    StorageConfigSQLite,
    validate_stored_config,
)


@pytest.mark.parametrize("days", [0, 1, 45, 4000])
def test_current_policy_preserves_explicit_age_and_independent_batch(days):
    cfg = GovernanceRetentionConfig(
        audit_events_max_age_days=days, audit_events_delete_batch_limit=2
    )
    assert cfg.audit_events_max_age_days == days
    assert cfg.audit_events_delete_batch_limit == 2
    assert GovernanceRetentionConfig().audit_events_max_age_days == 0


@pytest.mark.parametrize(
    "old",
    [
        {},
        {"audit_events_retention_days": 45},
        {"audit_events_retention_enabled": False, "audit_events_retention_days": 45},
        {"audit_events_retention_enabled": True},
        {"audit_events_retention_enabled": True, "audit_events_retention_days": 45},
    ],
)
def test_historical_conversion_preserves_consent_and_unknown_fields(old):
    raw = {
        "storage_config": None,
        "governance_retention": {
            **old,
            "audit_events_delete_batch_limit": 2,
            "opaque": {"keep": True},
        },
    }
    before = json.dumps(raw)
    result = canonicalize_config_document(raw)
    cfg = Config.model_validate(result).governance_retention
    assert cfg.audit_events_max_age_days == (
        old.get("audit_events_retention_days", 365)
        if old.get("audit_events_retention_enabled", False)
        else 0
    )
    assert result["governance_retention"]["opaque"] == {"keep": True}
    assert cfg.audit_events_delete_batch_limit == 2
    assert json.dumps(raw) == before


@pytest.mark.parametrize(
    "old",
    [
        {"audit_events_retention_enabled": "private-invalid"},
        {"audit_events_retention_enabled": None},
        {"audit_events_retention_enabled": False, "audit_events_retention_days": 0},
        {"audit_events_retention_days": -1},
        {"audit_events_retention_days": 1.5},
        {"audit_events_retention_days": None},
        {"audit_events_max_age_days": 7, "audit_events_retention_enabled": True},
    ],
)
def test_invalid_or_conflicting_history_never_grants_deletion(old):
    with pytest.raises(ConfigurationMigrationError) as error:
        canonicalize_config_document(
            {"storage_config": None, "governance_retention": old}
        )
    assert "private-invalid" not in str(error.value)


@pytest.mark.parametrize(
    "old",
    [
        {"audit_events_retention_enabled": False},
        {"audit_events_retention_days": 45},
        {"audit_events_max_age_days": 7, "audit_events_retention_enabled": True},
    ],
)
def test_runtime_rejects_retired_names_even_for_stored_extra_ignore(old):
    with pytest.raises(ValidationError, match="Obsolete configuration fields"):
        GovernanceRetentionConfig.model_validate(old, extra="ignore")


@pytest.mark.parametrize("version", [0, 1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_file_upgrade_preserves_consent_and_exact_originals(tmp_path, version, enabled):
    path = tmp_path / "config.json"
    raw = {
        "storage_config": None,
        "governance_retention": {
            "audit_events_retention_enabled": enabled,
            "audit_events_retention_days": 45,
            "audit_events_delete_batch_limit": 2,
            "opaque": {"keep": True},
        },
    }
    if version:
        raw["_reflexio_config_file"] = {"version": version}
    original = json.dumps(raw).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    assert owner.prepare()
    assert Config.model_validate(
        owner.read_document()
    ).governance_retention.audit_events_max_age_days == (45 if enabled else 0)
    digest = hashlib.sha256(original).hexdigest()
    assert (
        tmp_path / ".config-migrations/config.json" / f"{digest}.json"
    ).read_bytes() == original
    converted = path.read_bytes()
    assert not owner.prepare()
    assert path.read_bytes() == converted


@pytest.mark.parametrize("version", [0, 1, 2])
def test_ignored_new_name_in_older_file_cannot_authorize_deletion(tmp_path, version):
    path = tmp_path / "config.json"
    raw = {
        "storage_config": None,
        "governance_retention": {"audit_events_max_age_days": 7},
    }
    if version:
        raw["_reflexio_config_file"] = {"version": version}
    original = json.dumps(raw).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError, match="audit_events_max_age_days"):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


def test_explicit_current_publication_stamps_age_policy_generation(tmp_path):
    path = tmp_path / "config.json"
    owner = LocalConfigFile(path)
    owner.publish_document(
        json.dumps(
            {
                "storage_config": None,
                "governance_retention": {"audit_events_max_age_days": 7},
            }
        )
    )
    assert (
        json.loads(path.read_text())["_reflexio_config_file"]["version"]
        == FILE_FORMAT_VERSION
        == 4
    )
    assert not owner.prepare()


def test_frozen_generation_two_preserves_floor_and_disabled_retention(tmp_path):
    fixture = Path(__file__).parents[1] / "fixtures/local_config_generation2.json"
    path = tmp_path / "config.json"
    path.write_bytes(fixture.read_bytes())
    owner = LocalConfigFile(path)
    assert owner.prepare()
    config = validate_stored_config(owner.read_document())
    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == "generation2-owned.sqlite"
    assert config.retrieval_floor.candidate_pool_size == 73
    assert config.retrieval_floor.user_playbook_floor == 0
    assert config.governance_retention.audit_events_max_age_days == 0
    assert config.governance_retention.audit_events_delete_batch_limit == 2
    assert owner.read_document()["opaque_extension"] == {
        "preserve": ["generation2", False]
    }


def test_generation_two_does_not_replay_retired_retrieval_conversion(tmp_path):
    path = tmp_path / "config.json"
    original = json.dumps(
        {
            "storage_config": None,
            "_reflexio_config_file": {"version": 2},
            "retrieval_floor": {"enabled": True, "pool_size": 73},
        }
    ).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()
