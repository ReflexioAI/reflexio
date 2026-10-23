"""Only profile context is executable; playbook context was a stored no-op."""

from copy import deepcopy
from decimal import Decimal

import pytest

from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
    validate_stored_config,
)

pytestmark = pytest.mark.unit


def test_playbook_context_is_not_a_supported_input_or_serialized_policy() -> None:
    raw = {
        "extraction_definition_prompt": "Preserve customer-selected playbook goals",
        "context_prompt": "This value never entered the playbook prompt",
    }
    model = UserPlaybookExtractorConfig.model_validate(raw)
    assert "context_prompt" not in model.model_dump()
    assert (
        "context_prompt"
        not in UserPlaybookExtractorConfig.model_json_schema()["properties"]
    )
    assert model.extraction_definition_prompt == raw["extraction_definition_prompt"]


def test_old_saved_context_is_inert_and_raw_upgrade_preserves_opaque_data() -> None:
    raw = {
        "storage_config": {"type": "sqlite"},
        "agent_context_prompt": "Shared agent policy remains",
        "profile_extractor_config": {
            "extraction_definition_prompt": "Extract profile facts",
            "context_prompt": "Live profile context remains",
        },
        "user_playbook_extractor_config": {
            "extraction_definition_prompt": "Extract playbook rules",
            "context_prompt": "Previously ignored playbook context",
            "opaque": {"precise": Decimal("0.12345678901234567890123456789")},
        },
    }
    before = deepcopy(raw)
    loaded = validate_stored_config(raw)
    assert loaded.profile_extractor_config is not None
    assert (
        loaded.profile_extractor_config.context_prompt == "Live profile context remains"
    )
    assert loaded.user_playbook_extractor_config is not None
    assert "context_prompt" not in loaded.user_playbook_extractor_config.model_dump()
    converted = canonicalize_config_document(raw)
    assert raw == before
    # Raw upgrades preserve ignored data, including frozen historical alias cases.
    assert converted == raw
    assert canonicalize_config_document(converted) == converted
    assert (
        Config.model_validate(converted).agent_context_prompt
        == raw["agent_context_prompt"]
    )


def test_profile_context_remains_a_supported_serialized_input() -> None:
    config = ProfileExtractorConfig(
        extraction_definition_prompt="Extract profile facts",
        context_prompt="A customer's custom profile instructions",
    )
    assert config.model_dump()["context_prompt"] == config.context_prompt
    assert "context_prompt" in ProfileExtractorConfig.model_json_schema()["properties"]
