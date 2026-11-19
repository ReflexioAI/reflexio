"""The model availability check that guards offline tuner admission."""

from __future__ import annotations

import pytest

from reflexio.models.config_schema import (
    Config,
    LLMConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.offline_tuner.config import (
    OfflineTunerConfig,
    generation_model_available,
    sanitize_platform_offline_tuner_config,
)


def _config(generation_model_name: str | None = None) -> Config:
    """A minimal valid Config. `storage_config` is required and irrelevant here."""
    return Config(
        storage_config=StorageConfigSQLite(),
        llm_config=(
            LLMConfig(generation_model_name=generation_model_name)
            if generation_model_name is not None
            else None
        ),
    )


_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
    "MINIMAX_API_KEY",
    "DASHSCOPE_API_KEY",
    "XAI_API_KEY",
    "MOONSHOT_API_KEY",
    "ZAI_API_KEY",
)


@pytest.fixture
def no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear EVERY provider key.

    The credential floor pins placeholder keys for the suite, so a test that
    means "no providers" has to say so explicitly -- otherwise it silently
    tests the opposite of what it claims.
    """
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def test_the_production_case_resolves_through_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider key suffices even without an explicit organization model."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-resolution-only")

    config = _config()
    assert config.llm_config is None

    assert generation_model_available(config) is True


def test_no_model_and_no_provider_key_is_still_refused(no_provider_keys: None) -> None:
    """The gate must still SHUT. Widening it into "always true" would let
    tuning be switched on where nothing could ever run it, moving the failure
    from a visible refusal to a silent no-op."""
    assert generation_model_available(_config()) is False


def test_an_org_override_needs_no_provider_key(no_provider_keys: None) -> None:
    """Highest-priority branch: an explicit org model short-circuits detection."""
    assert generation_model_available(_config("openai/gpt-4o-mini")) is True


def test_enabled_survives_the_save_when_a_model_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end shape of the bug: a save that echoed `enabled: false`."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-resolution-only")

    sanitized = sanitize_platform_offline_tuner_config(
        OfflineTunerConfig(enabled=True), shared_config=_config()
    )
    assert sanitized.enabled is True


def test_enabled_is_stripped_when_nothing_resolves(no_provider_keys: None) -> None:
    sanitized = sanitize_platform_offline_tuner_config(
        OfflineTunerConfig(enabled=True), shared_config=_config()
    )
    assert sanitized.enabled is False


def test_a_disabled_config_is_returned_untouched(no_provider_keys: None) -> None:
    """Switching tuning OFF must never depend on model availability -- an org
    that cannot resolve a model must still be able to turn it off."""
    original = OfflineTunerConfig(enabled=False)
    assert (
        sanitize_platform_offline_tuner_config(original, shared_config=_config())
        is original
    )
