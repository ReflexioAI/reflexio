"""Retired no-op stays readable without widening the current write schema."""

import pytest
from pydantic import ValidationError

from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import Config, validate_stored_config


@pytest.mark.parametrize("value", [True, False])
def test_retired_shadow_input_is_accepted_but_not_serialized(value: bool) -> None:
    payload = {
        "storage_config": {"type": "sqlite"},
        "shadow_mode_enabled": value,
        "agent_context_prompt": "keep me",
    }
    for config in (
        Config.model_validate(canonicalize_config_document(payload)),
        validate_stored_config(payload),
    ):
        assert config.agent_context_prompt == "keep me"
        assert "shadow_mode_enabled" not in config.model_dump()
    assert payload["shadow_mode_enabled"] is value
    assert "shadow_mode_enabled" not in Config.model_json_schema()["properties"]


def test_retired_input_does_not_allow_arbitrary_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Config.model_validate(
            {"storage_config": None, "shadow_mode_enabled": True, "typo_config": True}
        )
