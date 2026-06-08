"""Tests for the Nomic local embedding provider's batch-size cap."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder


def _embedder_with_fake_model() -> tuple[NomicEmbedder, MagicMock]:
    """Build a NomicEmbedder whose model is a stub recording encode() calls."""
    embedder = NomicEmbedder()
    model = MagicMock()
    # One native 768-dim row; embed() slices to 512 and renormalises.
    model.encode.return_value = np.array([[0.1] * 768])
    embedder._model = model
    return embedder, model


def test_embed_uses_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env override, encode() is called with batch_size=4."""
    monkeypatch.delenv("REFLEXIO_EMBED_BATCH_SIZE", raising=False)
    embedder, model = _embedder_with_fake_model()

    out = embedder.embed(["hello"])

    assert len(out) == 1
    assert len(out[0]) == 512
    assert model.encode.call_args.kwargs["batch_size"] == 4


def test_embed_respects_batch_size_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """REFLEXIO_EMBED_BATCH_SIZE overrides the default mini-batch size."""
    monkeypatch.setenv("REFLEXIO_EMBED_BATCH_SIZE", "16")
    embedder, model = _embedder_with_fake_model()

    embedder.embed(["hello"])

    assert model.encode.call_args.kwargs["batch_size"] == 16


def test_embed_ignores_invalid_batch_size_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer override falls back to the default rather than crashing."""
    monkeypatch.setenv("REFLEXIO_EMBED_BATCH_SIZE", "not-a-number")
    embedder, model = _embedder_with_fake_model()

    embedder.embed(["hello"])

    assert model.encode.call_args.kwargs["batch_size"] == 4
