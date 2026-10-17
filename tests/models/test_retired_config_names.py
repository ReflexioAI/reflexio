"""Retired singleton names stay readable without surviving canonical saves."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
    strip_retired_config_fields,
    validate_stored_config,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name


@pytest.mark.parametrize(
    ("model", "field", "prompt", "identity"),
    [
        (
            ProfileExtractorConfig,
            "extractor_name",
            "extraction_definition_prompt",
            "profile",
        ),
        (
            UserPlaybookExtractorConfig,
            "extractor_name",
            "extraction_definition_prompt",
            "playbook",
        ),
        (
            AgentSuccessConfig,
            "evaluation_name",
            "success_definition_prompt",
            "agent_success",
        ),
    ],
)
def test_retired_names_are_input_only(model, field, prompt, identity):
    payload = {
        field: "old customer label",
        prompt: "Keep customer instructions",
        "window_size_override": 23,
        "stride_size_override": 7,
    }
    before = deepcopy(payload)
    config = model.model_validate(payload)
    assert payload == before
    assert field not in config.model_dump()
    assert field not in model.model_json_schema()["properties"]
    assert getattr(config, prompt) == "Keep customer instructions"
    if model is AgentSuccessConfig:
        assert "window_size_override" not in config.model_dump()
        assert "stride_size_override" not in config.model_dump()
    else:
        assert config.window_size_override == 23
        assert config.stride_size_override == 7
    assert get_extractor_name(config) == identity


@pytest.mark.parametrize("field", ["playbook_name", "feedback_name"])
def test_legacy_playbook_names_do_not_survive_alias_migration(field):
    payload = {field: "old name", "playbook_definition_prompt": "Keep rules"}
    before = deepcopy(payload)
    config = UserPlaybookExtractorConfig.model_validate(payload)
    assert payload == before
    assert config.extraction_definition_prompt == "Keep rules"
    assert field not in config.model_dump()
    assert "extractor_name" not in config.model_dump()


@pytest.mark.parametrize("legacy", [False, True])
def test_cleanup_preserves_opaque_metadata_and_all_legacy_entries(legacy):
    fields = [
        "profile_extractor_config",
        "user_playbook_extractor_config",
        "agent_success_config",
    ]
    payload = {
        "storage_config": None,
        "custom_extension": {"extractor_name": "keep opaque"},
        "shadow_mode_enabled": True,
    }
    for field in fields:
        item = {
            "extractor_name"
            if field != "agent_success_config"
            else "evaluation_name": "old",
            "extraction_definition_prompt"
            if field != "agent_success_config"
            else "success_definition_prompt": "Keep prompt",
        }
        payload[field + "s" if legacy else field] = (
            [item, deepcopy(item)] if legacy else item
        )
    before = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    assert payload == before
    assert "shadow_mode_enabled" not in cleaned
    assert cleaned["custom_extension"] == payload["custom_extension"]
    for field in fields:
        value = cleaned[field + "s" if legacy else field]
        for item in value if legacy else [value]:
            assert "extractor_name" not in item and "evaluation_name" not in item
            assert "Keep prompt" in item.values()
    loaded = validate_stored_config(payload)
    assert loaded.profile_extractor_config is not None
    assert loaded.user_playbook_extractor_config is not None
    assert loaded.agent_success_config is not None
    assert loaded.profile_extractor_config.extraction_definition_prompt == "Keep prompt"
    assert (
        loaded.user_playbook_extractor_config.extraction_definition_prompt
        == "Keep prompt"
    )
    assert loaded.agent_success_config.success_definition_prompt == "Keep prompt"


def test_retirement_does_not_relax_unknown_top_level_validation():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Config.model_validate({"storage_config": None, "unexpected_new_knob": True})
