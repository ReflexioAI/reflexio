"""Tests for the test-suite provider-credential floor."""

from __future__ import annotations

import os

import pytest

from reflexio.server.llm.model_defaults import _ENV_TO_PROVIDER
from reflexio.test_support.llm_credentials import (
    _PLACEHOLDER_KEY,
    ensure_provider_credential,
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
