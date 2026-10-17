"""Session evaluation accepts retired window input without advertising it."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reflexio.models.config_schema import (
    AgentSuccessConfig,
    strip_retired_config_fields,
)
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

LEGACY_FIELDS = (
    "window_size_override",
    "stride_size_override",
    "batch_size_override",
    "batch_interval_override",
    "extraction_window_size_override",
    "extraction_window_stride_override",
)


@pytest.mark.parametrize("key", LEGACY_FIELDS)
@pytest.mark.parametrize("value", [23, None, "obsolete"])
def test_window_inputs_are_ignored_without_affecting_customer_policy(key, value):
    payload = {
        "success_definition_prompt": "Judge the final session outcome.",
        "tagging_definition_prompt": "Tag by outcome.",
        "sampling_rate": 0.25,
        "evaluation_only_sampling_rate": 0.75,
        "retrieved_learning_sampling_rate": 1.0,
        "request_sources_enabled": ["api"],
        key: value,
    }
    before = deepcopy(payload)
    config = AgentSuccessConfig.model_validate(payload)
    assert not set(LEGACY_FIELDS).intersection(config.model_dump())
    assert not set(LEGACY_FIELDS).intersection(config.model_json_schema()["properties"])
    assert config.model_dump() == {k: v for k, v in payload.items() if k != key}
    assert payload == before


def test_raw_cleanup_is_scoped_and_preserves_extraction_windows_and_metadata():
    payload = {
        "agent_success_configs": [
            {"success_definition_prompt": "Keep", **dict.fromkeys(LEGACY_FIELDS, 23)}
        ],
        "agent_success_config": {
            "success_definition_prompt": "Keep",
            **dict.fromkeys(LEGACY_FIELDS, 23),
        },
        "profile_extractor_config": {
            "window_size_override": 23,
            "stride_size_override": 7,
        },
        "extension": {"window_size_override": "opaque"},
    }
    before = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    assert cleaned["agent_success_config"] == {"success_definition_prompt": "Keep"}
    assert cleaned["agent_success_configs"] == [{"success_definition_prompt": "Keep"}]
    assert cleaned["profile_extractor_config"] == before["profile_extractor_config"]
    assert cleaned["extension"] == before["extension"]
    assert payload == before


def test_saved_evaluation_windows_canonicalize_without_losing_policy(tmp_path):
    storage = LocalFileConfigStorage("evaluation-session", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite"},
                "agent_success_config": {
                    "success_definition_prompt": "Keep",
                    "window_size_override": 23,
                    "stride_size_override": 7,
                },
            }
        )
    )
    for _ in range(2):
        config = storage.load_config().agent_success_config
        assert config is not None
        assert config.success_definition_prompt == "Keep"
    saved = json.loads(path.read_text())["agent_success_config"]
    assert not set(LEGACY_FIELDS).intersection(saved)
