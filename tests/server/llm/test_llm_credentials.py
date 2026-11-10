"""Tests for the test-suite provider-credential floor."""

from __future__ import annotations

import os

import pytest

from reflexio.server.llm.model_defaults import _ENV_TO_PROVIDER
from reflexio.test_support.llm_credentials import (
    _PLACEHOLDER_KEY,
    ensure_provider_credential,
    real_generation_provider,
    real_provider_key,
)


@pytest.fixture(autouse=True)
def _bare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every provider key; the directory conftest strips the CLI opt-ins."""
    for var in _ENV_TO_PROVIDER:
        monkeypatch.delenv(var, raising=False)


def test_fills_a_bare_environment() -> None:
    ensure_provider_credential()
    assert os.environ["OPENAI_API_KEY"] == _PLACEHOLDER_KEY


def test_leaves_a_real_key_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-real")
    ensure_provider_credential()
    assert os.environ.get("OPENAI_API_KEY") is None


def test_fills_when_only_an_embedding_provider_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``local`` resolves embeddings but no generation-role model."""
    monkeypatch.setattr(
        "reflexio.test_support.llm_credentials.detect_available_providers",
        lambda: ["local"],
    )
    ensure_provider_credential()
    assert os.environ["OPENAI_API_KEY"] == _PLACEHOLDER_KEY


def test_placeholder_does_not_read_as_a_real_credential() -> None:
    """The floor must not silently un-skip the ``requires_credentials`` tier."""
    ensure_provider_credential()
    # Detection still sees it — that is what keeps model resolution working.
    assert os.environ["OPENAI_API_KEY"] == _PLACEHOLDER_KEY
    assert real_provider_key("OPENAI_API_KEY") is None


def test_real_provider_key_returns_a_configured_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert real_provider_key("OPENAI_API_KEY") == "sk-real"
    assert real_provider_key("ANTHROPIC_API_KEY") is None


def test_no_generation_provider_in_a_bare_environment() -> None:
    assert real_generation_provider() is None


def test_generation_provider_ignores_the_placeholder_floor() -> None:
    """The regression this helper exists to prevent, in reverse.

    The floor pins a key that ``detect_available_providers`` counts, so a gate
    built on detection alone would report a provider and let a live test run
    against a credential that authenticates with nothing.
    """
    ensure_provider_credential()
    assert real_generation_provider() is None


def test_generation_provider_finds_a_non_openai_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: a MiniMax-only machine has a usable provider."""
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real")
    assert real_generation_provider() == "minimax"


def test_generation_provider_honors_the_allowed_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real")
    assert real_generation_provider({"openai", "anthropic"}) is None
    assert real_generation_provider({"openai", "minimax"}) == "minimax"


def test_generation_provider_rejects_an_embedding_only_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``local`` is detected but serves no generation-role model."""
    monkeypatch.setattr(
        "reflexio.test_support.llm_credentials.detect_available_providers",
        lambda: ["local"],
    )
    assert real_generation_provider() is None


@pytest.mark.parametrize("value", ["", "0", "1"])
def test_existing_key_still_rejects_retired_embedding_setting(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", _PLACEHOLDER_KEY)
    monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", value)
    with pytest.raises(ValueError, match="REFLEXIO_EMBEDDING_PROVIDER"):
        ensure_provider_credential()


def test_generation_provider_refuses_a_resolved_zai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Z.ai subscription: a live test that resolution would route to zai skips.

    ``zai`` outranks ``openai`` in resolution, so returning ``openai`` here
    would let the test run while the call still went to Z.ai.
    """
    monkeypatch.setenv("ZAI_API_KEY", "zai-real")
    assert real_generation_provider() is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert real_generation_provider() is None
    assert real_generation_provider({"openai"}) is None


def test_generation_provider_ignores_zai_below_a_resolved_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A higher-priority real key wins resolution, so the zai key is inert."""
    monkeypatch.setenv("ZAI_API_KEY", "zai-real")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real")
    assert real_generation_provider() == "minimax"
    assert real_generation_provider({"minimax"}) == "minimax"
