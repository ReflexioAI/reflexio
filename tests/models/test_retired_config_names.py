"""Retired singleton names stay readable without surviving canonical saves."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.config_migration import (
    canonicalize_config_document,
    strip_retired_config_fields,
)
from reflexio.models.config_schema import (
    AgentSuccessConfig,
    AzureOpenAIConfig,
    Config,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
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
    from reflexio.config_migration import canonicalize_config_document

    current = canonicalize_config_document(
        {"storage_config": None, "user_playbook_extractor_config": payload}
    )
    config = UserPlaybookExtractorConfig.model_validate(
        current["user_playbook_extractor_config"]
    )
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
    loaded = validate_stored_config(canonicalize_config_document(payload))
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


@pytest.mark.parametrize("loader", [Config.model_validate, validate_stored_config])
def test_azure_model_id_is_the_only_deployment_declaration(loader, monkeypatch):
    monkeypatch.setattr(
        "reflexio.models.api_schema.validators.socket.getaddrinfo",
        lambda _host, port, *_args, **_kwargs: [
            (2, 1, 0, "", ("93.184.216.34", port or 443))
        ],
    )
    payload = {
        "storage_config": None,
        "api_key_config": {
            "openai": {
                "azure_config": {
                    "api_key": "fixture-azure-key",
                    "endpoint": "https://fixture.openai.azure.com/",
                    "api_version": "2024-02-15-preview",
                    "deployment_name": "ignored-legacy-deployment",
                }
            }
        },
    }
    original = deepcopy(payload)
    config = loader(payload)
    azure = config.model_dump(mode="json")["api_key_config"]["openai"]["azure_config"]
    assert "deployment_name" not in azure
    assert "deployment_name" not in AzureOpenAIConfig.model_fields
    assert "deployment_name" not in AzureOpenAIConfig.model_json_schema()["properties"]
    assert azure == {
        key: value
        for key, value in original["api_key_config"]["openai"]["azure_config"].items()
        if key != "deployment_name"
    }
    assert payload == original


def test_raw_upgrade_discards_only_the_unused_azure_declaration():
    payload = {
        "api_key_config": {
            "provider_metadata": {"keep": True},
            "openai": {
                "openai_metadata": [1, 2],
                "azure_config": {
                    "deployment_name": "unused",
                    "api_key": "fixture-key",
                    "endpoint": "https://fixture.openai.azure.com/",
                    "custom_metadata": {"keep": [1, 2]},
                },
            },
        }
    }
    original = deepcopy(payload)
    cleaned = strip_retired_config_fields(payload)
    expected = deepcopy(original)
    expected["api_key_config"]["openai"]["azure_config"].pop("deployment_name")
    assert cleaned == expected
    assert payload == original
