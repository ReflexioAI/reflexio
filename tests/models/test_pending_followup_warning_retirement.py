"""Retiring observational policy preserves saved execution choices and metadata."""

from copy import deepcopy

import pytest

from reflexio.config_migration import strip_retired_config_fields
from reflexio.models.config_schema import (
    PendingToolCallConfig,
    PendingToolCallToolOverrideConfig,
)


def test_pending_warning_is_not_supported_policy():
    for model in (PendingToolCallConfig, PendingToolCallToolOverrideConfig):
        assert (
            "max_pending_followups_per_scope"
            not in model.model_json_schema()["properties"]
        )


@pytest.mark.parametrize("historical_value", [1, 37, None, {"obsolete": True}])
def test_raw_pending_warning_cleanup_is_narrow_and_nonmutating(historical_value):
    raw = {
        "pending_tool_call_config": {
            "enabled": True,
            "max_pending_followups_per_scope": historical_value,
            "pending_ttl_seconds": 700,
            "extension": {"max_pending_followups_per_scope": "opaque"},
            "tool_overrides": {
                "ask_human": {
                    "max_pending_followups_per_scope": historical_value,
                    "pending_ttl_seconds": 90,
                    "extension": {"max_pending_followups_per_scope": "opaque"},
                },
                "attach_pending_info_request": {"similarity_threshold": 0.8},
            },
        }
    }
    original = deepcopy(raw)
    cleaned = strip_retired_config_fields(raw)
    expected = deepcopy(original)
    expected["pending_tool_call_config"].pop("max_pending_followups_per_scope")
    expected["pending_tool_call_config"]["tool_overrides"]["ask_human"].pop(
        "max_pending_followups_per_scope"
    )
    assert cleaned == expected
    assert raw == original
    config = PendingToolCallConfig.model_validate(cleaned["pending_tool_call_config"])
    assert "max_pending_followups_per_scope" not in config.model_dump()
    assert config.for_tool("ask_human").pending_ttl_seconds == 90
    assert config.for_tool("attach_pending_info_request").similarity_threshold == 0.8
    assert config.for_tool("other").pending_ttl_seconds == 700
