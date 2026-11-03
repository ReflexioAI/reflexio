"""One lifetime owns new-call consent without reopening tool overrides."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.config_migration import (
    ConfigurationMigrationError,
    canonicalize_pending_creation_policy,
)
from reflexio.models.config_schema import (
    PendingToolCallConfig,
    PendingToolCallToolOverrideConfig,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("seconds", [0, 37, 172_801])
def test_creation_policy_resolves_exact_base_and_tool_lifetimes(seconds):
    config = PendingToolCallConfig(
        pending_ttl_seconds=seconds,
        tool_overrides={
            "ask_human": PendingToolCallToolOverrideConfig(pending_ttl_seconds=53)
        },
    )
    assert config.creation_lifetime_seconds() == seconds
    assert config.creation_lifetime_seconds("unknown") == seconds
    assert config.for_tool("ask_human").pending_ttl_seconds == (53 if seconds else 0)
    assert "enabled" not in config.model_dump()


def test_default_is_off_without_disabling_callback_and_reuse_policy():
    config = PendingToolCallConfig(
        prior_answer_valid_seconds=719,
        tool_overrides={
            "ask_human": PendingToolCallToolOverrideConfig(
                prior_answer_valid_seconds=347
            )
        },
    )
    assert config.creation_lifetime_seconds() == 0
    assert config.for_tool("ask_human").pending_ttl_seconds == 0
    assert config.for_tool("ask_human").prior_answer_valid_seconds == 347
    assert config.for_tool("unknown").prior_answer_valid_seconds == 719


@pytest.mark.parametrize("value", [True, False, None, ""])
def test_current_reader_rejects_retired_creation_switch(value):
    with pytest.raises(ValidationError, match="enabled"):
        PendingToolCallConfig.model_validate({"enabled": value})


@pytest.mark.parametrize("enabled,expected", [(True, 37), (False, 0)])
def test_explicit_conversion_preserves_effective_creation_and_opaque_data(
    enabled, expected
):
    source = {
        "enabled": enabled,
        "pending_ttl_seconds": 37,
        "tool_overrides": {"ask_human": {"pending_ttl_seconds": 53}},
        "hmac_secrets": ["synthetic-secret"],
        "extension": {"keep": [17, False]},
    }
    original = deepcopy(source)
    converted = canonicalize_pending_creation_policy(source)
    assert source == original
    assert converted == {
        **{key: value for key, value in original.items() if key != "enabled"},
        "pending_ttl_seconds": expected,
    }


def test_implicit_old_opt_out_and_explicit_current_duration_are_distinct():
    current = {"pending_ttl_seconds": 37}
    assert canonicalize_pending_creation_policy(current) == current
    assert canonicalize_pending_creation_policy(current, historical=True) == {
        "pending_ttl_seconds": 0
    }
    assert canonicalize_pending_creation_policy({"enabled": True}) == {
        "pending_ttl_seconds": 86_400
    }


@pytest.mark.parametrize("value", [0, "0", False, -1, None, "PRIVATE_INVALID_TTL"])
@pytest.mark.parametrize("enabled", [False, True])
def test_invalid_dormant_lifetime_cannot_become_current_consent(value, enabled):
    source = {"enabled": enabled, "pending_ttl_seconds": value}
    before = deepcopy(source)
    with pytest.raises(ConfigurationMigrationError) as error:
        canonicalize_pending_creation_policy(source, historical=True)
    assert source == before
    assert "PRIVATE_INVALID_TTL" not in str(error.value)
