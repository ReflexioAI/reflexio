"""Retired phase clocks cannot re-enter current storage configuration."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import StorageConfigPostgres


@pytest.mark.parametrize("field", ["pool_acquire_timeout", "read_pool_acquire_timeout"])
def test_current_storage_refuses_retired_queue_clock(field):
    with pytest.raises(ValidationError):
        StorageConfigPostgres.model_validate(
            {
                "type": "postgres",
                "db_url": "postgresql://opaque/target",
                field: 7.5,
            }
        )


def test_explicit_upgrade_retires_phase_clocks_without_changing_routes_or_capacity():
    original = {
        "storage_config": {
            "type": "postgres",
            "db_url": "postgresql://opaque/target",
            "schema": "tenant",
            "pool_size": 3,
            "pool_acquire_timeout": 7.5,
            "read_db_url": "postgresql://opaque/replica",
            "read_pool_size": 2,
            "read_pool_acquire_timeout": 4.5,
        }
    }
    snapshot = deepcopy(original)
    converted = canonicalize_config_document(original)
    assert converted["storage_config"] == {
        key: value
        for key, value in snapshot["storage_config"].items()
        if key not in {"pool_acquire_timeout", "read_pool_acquire_timeout"}
    }
    assert original == snapshot
    assert canonicalize_config_document(converted) == converted


@pytest.mark.parametrize("field", ["pool_acquire_timeout", "read_pool_acquire_timeout"])
@pytest.mark.parametrize(
    "value", [-1, 0, float("nan"), float("inf"), "private-invalid"]
)
def test_upgrade_refuses_invalid_historical_queue_policy(field, value):
    with pytest.raises(ValueError, match=field) as error:
        canonicalize_config_document(
            {
                "storage_config": {
                    "type": "postgres",
                    "db_url": "postgresql://opaque/target",
                    field: value,
                }
            }
        )
    assert "private-invalid" not in str(error.value)


def test_inherited_historical_reader_queue_clock_is_removed():
    converted = canonicalize_config_document(
        {
            "storage_config": {
                "type": "postgres",
                "db_url": "postgresql://opaque/target",
                "read_pool_acquire_timeout": None,
            }
        }
    )
    assert "read_pool_acquire_timeout" not in converted["storage_config"]


def test_file_upgrade_preserves_original_and_current_generation_refuses_old_fields(
    tmp_path,
):
    import json

    from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile

    path = tmp_path / "config.json"
    storage = {
        "type": "postgres",
        "db_url": "postgresql://opaque/target",
        "pool_size": 3,
        "read_pool_size": 2,
        "pool_acquire_timeout": 7.5,
        "read_pool_acquire_timeout": 4.5,
    }
    original = json.dumps(
        {"storage_config": storage, "_reflexio_config_file": {"version": 13}}
    ).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    with pytest.raises(ValueError):
        owner.read_document()
    assert owner.prepare()
    current = owner.read_document()["storage_config"]
    assert current == {
        key: value
        for key, value in storage.items()
        if key not in {"pool_acquire_timeout", "read_pool_acquire_timeout"}
    }
    assert any(
        p.read_bytes() == original
        for p in (tmp_path / ".config-migrations").rglob("*.json")
        if p.is_file()
    )
    assert not owner.prepare()
    forbidden = json.dumps(
        {
            "storage_config": storage,
            "_reflexio_config_file": {"version": FILE_FORMAT_VERSION},
        }
    ).encode()
    path.write_bytes(forbidden)
    with pytest.raises(ValueError):
        owner.prepare()
    assert path.read_bytes() == forbidden


@pytest.mark.parametrize("backend", ["postgres", "supabase"])
@pytest.mark.parametrize("field", ["pool_acquire_timeout", "read_pool_acquire_timeout"])
def test_managed_reference_never_gains_native_queue_fields(tmp_path, backend, field):
    import json

    from reflexio.config_file import LocalConfigFile

    document = {
        "storage_config": {"type": backend, "managed_by": "platform", field: 7.5}
    }
    snapshot = deepcopy(document)
    with pytest.raises(ValueError):
        canonicalize_config_document(document)
    assert document == snapshot
    path = tmp_path / "config.json"
    original = json.dumps(
        {**document, "_reflexio_config_file": {"version": 13}}
    ).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()
