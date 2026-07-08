"""Tests for the Nomic local embedding provider's batch-size cap."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder


class _ReentrancyDetectingModel:
    """Fake sentence-transformers model that flags concurrent ``encode()``.

    The real ``nomic-bert`` model mutates instance-level rotary caches
    (``_cos_cached``/``_sin_cached``) on every forward pass, so two threads
    inside ``encode()`` at once corrupt each other. This stub makes that
    corruption observable: it raises (and records a flag) the moment a second
    thread enters while the first has not yet exited.
    """

    def __init__(self, hold: float = 0.02) -> None:
        self._hold = hold
        self._guard = threading.Lock()
        self._active = False
        self.reentered = False
        self.calls = 0

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        with self._guard:
            if self._active:
                self.reentered = True
                raise AssertionError("model.encode() entered by two threads at once")
            self._active = True
            self.calls += 1
        try:
            time.sleep(self._hold)  # widen the race window
        finally:
            with self._guard:
                self._active = False
        return np.array([[0.1] * 768 for _ in texts])


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


def test_encode_is_serialized_across_threads() -> None:
    """Concurrent ``embed()`` calls on ONE embedder never overlap in encode().

    Regression for the prod race: the shared singleton model is not
    thread-safe, so the embedder must serialize ``model.encode()`` itself.
    Reverting the ``self._model_lock`` guard in ``embed()`` makes this fail.
    """
    embedder = NomicEmbedder()
    model = _ReentrancyDetectingModel(hold=0.02)
    embedder._model = model  # pre-load so _load() is a no-op fast path

    # Mixed-length payloads: differing text counts and differing char lengths.
    payloads = [
        ["short"],
        ["x" * 500],
        ["a", "b", "c"],
        ["", "y" * 50],
        ["only-one-longer-input " * 10],
        ["p", "q"],
        ["z" * 2000],
        ["m", "n", "o", "p"],
    ]
    results: list[list[list[float]]] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker(texts: list[str]) -> None:
        try:
            out = embedder.embed(texts)
        except BaseException as exc:  # noqa: BLE001
            with results_lock:
                errors.append(exc)
            return
        with results_lock:
            results.append(out)

    threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert model.reentered is False
    assert model.calls == len(payloads)
    assert len(results) == len(payloads)
    assert all(len(vec) == 512 for out in results for vec in out)
