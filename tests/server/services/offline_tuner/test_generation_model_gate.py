"""The gate that decides whether tuning may be switched on at all.

This gate had NO tests, and that is how it came to disagree with the resolver
it was standing in for. `generation_model_available` reimplemented two of
`resolve_model_name`'s four branches and omitted auto-detect-from-provider-keys
-- a branch a deployment can resolve through exclusively -- so
`sanitize_platform_offline_tuner_config` silently flipped `enabled` back to
False and nobody could turn tuning on.

`test_the_production_case_*` is the regression: no org override, an empty
site var and a provider key, which fails against the old implementation.
"""

from __future__ import annotations

from collections.abc import Mapping

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


@pytest.fixture
def empty_site_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """`llm_model_setting.json` ships with every value empty. Reproduce that."""
    monkeypatch.setattr(
        "reflexio.server.services.offline_tuner.config.SiteVarManager",
        lambda: _FakeSiteVars({"default_generation_model_name": ""}),
    )


class _FakeSiteVars:
    """`get_site_var` returns parsed JSON, so a value here can be ANY type.

    The payload is typed `Mapping[str, object]` rather than `dict[str, str]`
    precisely so a malformed-site-var test can pass a non-string without
    fighting `dict`'s invariance.
    """

    def __init__(self, payload: Mapping[str, object]) -> None:
        self._payload = payload

    def get_site_var(self, name: str) -> Mapping[str, object]:
        assert name == "llm_model_setting"
        return self._payload


def test_the_production_case_resolves_through_provider_keys(
    monkeypatch: pytest.MonkeyPatch, empty_site_var: None
) -> None:
    """THE REGRESSION. No org llm_config, empty site var, a provider key set.

    An organization can carry no `llm_config` while the shipped site var has
    an empty `default_generation_model_name`. Generation works there
    regardless, because the resolver auto-detects from provider keys.

    The old implementation checked only the org override and the site var, so
    it answered False here and tuning could not be enabled by anyone.
    """
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-resolution-only")

    config = _config()
    assert config.llm_config is None

    assert generation_model_available(config) is True


def test_no_model_and_no_provider_key_is_still_refused(
    no_provider_keys: None, empty_site_var: None
) -> None:
    """The gate must still SHUT. Widening it into "always true" would let
    tuning be switched on where nothing could ever run it, moving the failure
    from a visible refusal to a silent no-op."""
    assert generation_model_available(_config()) is False


def test_an_org_override_needs_no_provider_key(
    no_provider_keys: None, empty_site_var: None
) -> None:
    """Highest-priority branch: an explicit org model short-circuits detection."""
    assert generation_model_available(_config("openai/gpt-4o-mini")) is True


def test_the_site_var_default_is_honoured(
    no_provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated platform default resolves with no org config and no keys."""
    monkeypatch.setattr(
        "reflexio.server.services.offline_tuner.config.SiteVarManager",
        lambda: _FakeSiteVars({"default_generation_model_name": "openai/gpt-4o-mini"}),
    )
    assert generation_model_available(_config()) is True


def test_a_malformed_non_string_site_var_does_not_admit(
    no_provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed `llm_model_setting` must not open the gate.

    `get_site_var` returns parsed JSON, so this value can be any type, and
    `resolve_model_name`'s site-var branch returns a truthy value VERBATIM --
    an unnormalized `123` would resolve to `123` and admit the save. The
    execution path, `open_world_generation_client`, drops every non-string
    before resolving, so with no provider key it raises `RuntimeError` and the
    run cannot start.

    Admitting something the run then refuses is the exact disagreement this
    gate exists to end, so it must normalize the way the run does.
    """
    monkeypatch.setattr(
        "reflexio.server.services.offline_tuner.config.SiteVarManager",
        lambda: _FakeSiteVars({"default_generation_model_name": 123}),
    )
    assert generation_model_available(_config()) is False


def test_enabled_survives_the_save_when_a_model_resolves(
    monkeypatch: pytest.MonkeyPatch, empty_site_var: None
) -> None:
    """The end-to-end shape of the bug: a save that echoed `enabled: false`."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-resolution-only")

    sanitized = sanitize_platform_offline_tuner_config(
        OfflineTunerConfig(enabled=True), shared_config=_config()
    )
    assert sanitized.enabled is True


def test_enabled_is_stripped_when_nothing_resolves(
    no_provider_keys: None, empty_site_var: None
) -> None:
    sanitized = sanitize_platform_offline_tuner_config(
        OfflineTunerConfig(enabled=True), shared_config=_config()
    )
    assert sanitized.enabled is False


def test_a_disabled_config_is_returned_untouched(
    no_provider_keys: None, empty_site_var: None
) -> None:
    """Switching tuning OFF must never depend on model availability -- an org
    that cannot resolve a model must still be able to turn it off."""
    original = OfflineTunerConfig(enabled=False)
    assert (
        sanitize_platform_offline_tuner_config(original, shared_config=_config())
        is original
    )
