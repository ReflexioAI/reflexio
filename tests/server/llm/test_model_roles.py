"""Tests for the agentic tool-calling ModelRole additions."""

from reflexio.server.llm.model_defaults import _PROVIDER_DEFAULTS, ModelRole


def test_new_roles_exist():
    assert ModelRole.ANGLE_READER.value == "angle_reader"
    assert ModelRole.CRITIC.value == "critic"
    assert ModelRole.SYNTHESIZER.value == "synthesizer"
    assert ModelRole.RECONCILER.value == "reconciler"


def test_anthropic_defaults_cover_new_roles():
    anthropic = _PROVIDER_DEFAULTS["anthropic"]
    assert anthropic.angle_reader == "claude-haiku-4-5-20251001"
    assert anthropic.critic == "claude-sonnet-4-6"
    assert anthropic.synthesizer == "claude-sonnet-4-6"
    assert anthropic.reconciler == "claude-sonnet-4-6"


def test_claude_code_defaults_cover_new_roles():
    cc = _PROVIDER_DEFAULTS["claude-code"]
    assert cc.angle_reader == "claude-code/default"
    assert cc.critic == "claude-code/default"
    assert cc.synthesizer == "claude-code/default"
    assert cc.reconciler == "claude-code/default"


def test_unpopulated_providers_default_to_none():
    """Providers that haven't opted into tool-calling fall through to next priority provider."""
    local = _PROVIDER_DEFAULTS["local"]
    assert local.angle_reader is None
    assert local.critic is None
    assert local.synthesizer is None
    assert local.reconciler is None
