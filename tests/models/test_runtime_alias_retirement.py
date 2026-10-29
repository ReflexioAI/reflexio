"""Frozen historical inputs remain upgradeable, never ordinary runtime choices."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from reflexio.config_migration import (
    StoredConfigurationError,
    canonicalize_config_document,
)
from reflexio.models.config_schema import Config, validate_stored_config

CASES = json.loads(
    (Path(__file__).parent / "fixtures/runtime_alias_retirement.json").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["alias"])
def test_explicit_upgrade_preserves_reviewed_policy(case):
    source = deepcopy(case["input"])
    converted = canonicalize_config_document(source)
    expected = deepcopy(case["expected"])
    # The frozen alias corpus predates retirement of the consolidation method
    # selector. Keep that historical witness; assert the new method upgrade
    # independently rather than changing its recorded expected document.
    extractor = expected.get("user_playbook_extractor_config")
    if isinstance(extractor, dict) and isinstance(
        policy := extractor.get("deduplication_config"), dict
    ):
        policy.pop("strategy", None)
    assert converted == expected
    assert source == case["input"]
    assert validate_stored_config(converted) == validate_stored_config(expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["alias"])
@pytest.mark.parametrize("collision", [False, True])
def test_normal_readers_reject_every_retired_alias(case, collision):
    source = deepcopy(case["input"])
    if collision:

        def merge(left, right):
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(left.get(key), dict):
                    merge(left[key], value)
                else:
                    left[key] = deepcopy(value)

        merge(source, case["expected"])
    for reader in (Config.model_validate, validate_stored_config):
        with pytest.raises(ValueError):
            reader(source)


def test_frozen_manifest_has_169_unique_paths():
    assert len(CASES) == 169
    assert len({case["alias"] for case in CASES}) == 169


@pytest.mark.parametrize("value", [None, False, 0, "", {}, []])
@pytest.mark.parametrize(
    "root,field",
    [
        (None, "batch_size"),
        ("profile_extractor_config", "batch_size_override"),
        ("user_playbook_extractor_config", "feedback_aggregator_config"),
        ("playbook_optimizer_config", "use_merge"),
        ("storage_config", "storage_type"),
    ],
)
def test_obsolete_presence_is_rejected_even_when_empty(root, field, value):
    payload: dict[str, Any] = {"storage_config": None}
    if root is None:
        payload[field] = value
    else:
        payload[root] = {field: value}
    with pytest.raises(StoredConfigurationError):
        validate_stored_config(payload)


def test_opaque_lookalikes_are_not_configuration_paths():
    payload = {
        "storage_config": None,
        "opaque": {"batch_size": 17, "storage_type": "private", "use_merge": False},
    }
    before = deepcopy(payload)
    assert validate_stored_config(payload).window_size == 10
    assert payload == before
    assert canonicalize_config_document(payload) == payload


def test_direct_nested_model_rejection_is_value_free():
    from reflexio.models.config_schema import (
        PlaybookAggregatorConfig,
        PlaybookOptimizerConfig,
        ProfileExtractorConfig,
        StorageConfigPostgres,
    )

    for model, payload in (
        (
            ProfileExtractorConfig,
            {
                "extraction_definition_prompt": "private-secret",
                "batch_size_override": 23,
            },
        ),
        (PlaybookAggregatorConfig, {"min_feedback_threshold": "private-secret"}),
        (PlaybookOptimizerConfig, {"use_merge": "private-secret"}),
        (
            StorageConfigPostgres,
            {"db_url": "private-secret", "schema_name": "private-secret"},
        ),
    ):
        with pytest.raises(ValueError) as error:
            model.model_validate(payload, extra="ignore")
        assert "private-secret" not in str(error.value)
        assert "input_value" not in str(error.value)


def test_native_storage_requires_a_canonical_raw_tag_but_keeps_typed_defaults():
    from pydantic import TypeAdapter

    from reflexio.models.config_schema import StorageConfig, StorageConfigPostgres

    raw = {
        "db_url": "postgresql://user:password@localhost/db",
        "pool_size": 7,
        "read_pool_size": 3,
    }
    with pytest.raises(ValueError):
        TypeAdapter(StorageConfig).validate_python(raw)
    current = canonicalize_config_document({"storage_config": raw})
    assert current["storage_config"] == {**raw, "type": "postgres"}
    stored = validate_stored_config(current)
    assert isinstance(stored.storage_config, StorageConfigPostgres)
    assert (
        stored.storage_config.pool_size == 7
        and stored.storage_config.read_pool_size == 3
    )
    typed = StorageConfigPostgres(
        db_url="postgresql://user:password@localhost/db", pool_size=7, read_pool_size=3
    )
    assert Config(storage_config=typed).storage_config is typed


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", 23),
        ("batch_interval", 7),
        ("extraction_window_size", 23),
        ("extraction_window_stride", 7),
        ("extraction_preset", "long_form"),
        ("profile_extractor_configs", []),
        ("user_playbook_extractor_configs", []),
        ("profile_extractor_config", {"profile_content_definition_prompt": "obsolete"}),
        (
            "user_playbook_extractor_config",
            {"extraction_definition_prompt": "Keep", "batch_size_override": 17},
        ),
        ("storage_config", {"type": "sqlite", "schema_name": "obsolete"}),
    ],
)
def test_named_writes_reject_aliases_without_mutating_live_or_saved_policy(
    tmp_path, name, value
):
    from pathlib import Path

    from reflexio.server.services.configurator.configurator import (
        DefaultConfigurator,
    )

    configurator = DefaultConfigurator("named", base_dir=str(tmp_path))
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    assert isinstance(configurator.config_storage, LocalFileConfigStorage)
    path = Path(configurator.config_storage.config_file)
    before = path.read_bytes()
    served = configurator.get_config().model_dump(mode="json")
    with pytest.raises(ValueError):
        configurator.set_config_by_name(name, value)
    assert path.read_bytes() == before
    assert configurator.get_config().model_dump(mode="json") == served
    configurator.set_config_by_name("window_size", 31)
    assert configurator.get_config().window_size == 31
    assert configurator.config_storage.load_config().window_size == 31


def test_documented_evaluation_file_is_accepted_by_the_current_runtime():
    from pathlib import Path

    from reflexio.config_document import decode_config_document

    path = Path(__file__).resolve().parents[1] / "test_data/scenarios/eval_config.json"
    config = Config.model_validate(decode_config_document(path.read_text()))
    assert config.profile_extractor_config is not None
    assert config.user_playbook_extractor_config is not None
    assert config.profile_extractor_config.window_size_override == 20
    assert config.user_playbook_extractor_config.stride_size_override == 1
