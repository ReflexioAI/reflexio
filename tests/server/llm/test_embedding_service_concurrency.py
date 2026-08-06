"""Tests for the embedding daemon's bounded, queueing concurrency control."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from reflexio.server.llm import embedding_service as es


class _ConcurrencyModel:
    """Fake sentence-transformers model recording peak simultaneous encodes.

    Injected into a real ``NomicEmbedder`` so the assertions exercise the
    embedder-level ``_model_lock`` — the invariant that survives after the
    daemon's own ``_MODEL_ENCODE_LOCK`` was removed (the embedder, not the
    daemon, now owns serialization of the non-thread-safe model).
    """

    def __init__(self, hold: float = 0.1) -> None:
        self._hold = hold
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.calls = 0
        self.encode_calls: list[list[str]] = []

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        with self._lock:
            self.current += 1
            self.calls += 1
            self.peak = max(self.peak, self.current)
            self.encode_calls.append(list(texts))
        # Hold the slot so concurrent callers would overlap if unserialized.
        time.sleep(self._hold)
        with self._lock:
            self.current -= 1
        return np.array([[0.1] * 768 for _ in texts])


def _reset_service_state(
    monkeypatch: pytest.MonkeyPatch, model: _ConcurrencyModel
) -> None:
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    monkeypatch.setattr(es, "_ACTIVE_BATCH_PROCESSORS", 0)
    embedder = es.NomicEmbedder()
    embedder._model = model  # pre-load so _load() is a no-op fast path
    monkeypatch.setattr(es.NomicEmbedder, "get", classmethod(lambda _cls: embedder))


def test_max_concurrency_defaults_to_4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REFLEXIO_EMBED_MAX_CONCURRENCY", raising=False)
    assert es._max_concurrency() == 4


def test_max_concurrency_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "2")
    assert es._max_concurrency() == 2


def test_max_concurrency_ignores_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "bogus")
    assert es._max_concurrency() == 4


def test_micro_batch_delay_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "25")
    assert es._micro_batch_delay_seconds() == 0.025


def test_micro_batch_max_texts_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    assert es._micro_batch_max_texts() == 8


def test_embed_texts_caps_and_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Excess concurrent requests queue (never reject) and encodes serialize."""
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "1")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "1")
    fake_model = _ConcurrencyModel(hold=0.1)
    _reset_service_state(monkeypatch, fake_model)

    model = "local/nomic-embed-text-v1.5"
    errors: list[Exception] = []

    def worker() -> None:
        try:
            es._embed_texts(model, ["x"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # All six requests completed — they queued, none were rejected.
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert fake_model.calls == 6
    # Model inference is serialized by the embedder's own ``_model_lock`` (see
    # PR #153: concurrent encodes on the shared, non-thread-safe model corrupt
    # each other's tensors). The daemon no longer holds a redundant lock — the
    # embedder owns serialization by construction. The fake model counts time
    # inside encode(), so the peak simultaneous-encode count must be exactly 1.
    assert fake_model.peak == 1


def test_micro_batches_concurrent_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent small requests can share one encode call."""
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "50")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    fake_model = _ConcurrencyModel(hold=0.0)
    _reset_service_state(monkeypatch, fake_model)

    model = "local/nomic-embed-text-v1.5"
    barrier = threading.Barrier(2)
    results: list[list[list[float]]] = []
    errors: list[Exception] = []

    def worker(text: str) -> None:
        try:
            barrier.wait(timeout=1)
            results.append(es._embed_texts(model, [text]))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("first",)),
        threading.Thread(target=worker, args=("second",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == 2
    assert len(fake_model.encode_calls) == 1
    assert set(fake_model.encode_calls[0]) == {"first", "second"}


def test_micro_batches_requests_for_registered_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider extension keeps the daemon's request coalescing behavior."""
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "50")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    monkeypatch.setattr(es, "_ACTIVE_BATCH_PROCESSORS", 0)
    model = "custom/enterprise-model"
    encode_calls: list[list[str]] = []
    barrier = threading.Barrier(2)

    def encode(texts: list[str]) -> list[list[float]]:
        encode_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def worker(text: str) -> None:
        barrier.wait(timeout=1)
        es._embed_texts(model, [text], encoder=encode, allowed_models={model})

    threads = [
        threading.Thread(target=worker, args=("first",)),
        threading.Thread(target=worker, args=("second",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(encode_calls) == 1
    assert set(encode_calls[0]) == {"first", "second"}
