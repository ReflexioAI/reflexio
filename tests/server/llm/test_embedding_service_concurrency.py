"""Tests for the embedding daemon's bounded, queueing concurrency control."""

from __future__ import annotations

import threading
import time

import pytest

from reflexio.server.llm import embedding_service as es


class _ConcurrencyRecorder:
    """Stand-in embedder that records the peak number of simultaneous calls."""

    def __init__(self, hold: float = 0.1) -> None:
        self._hold = hold
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.current += 1
            self.calls += 1
            self.peak = max(self.peak, self.current)
        # Hold the slot so concurrent callers actually overlap.
        time.sleep(self._hold)
        with self._lock:
            self.current -= 1
        return [[0.0] * 512 for _ in texts]


class _BatchRecorder:
    """Stand-in embedder that records each encode call's texts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.calls.append(list(texts))
        return [[float(index)] * 512 for index, _text in enumerate(texts)]


def _reset_service_state(
    monkeypatch: pytest.MonkeyPatch, recorder: _ConcurrencyRecorder | _BatchRecorder
) -> None:
    monkeypatch.setattr(es, "_ENCODE_SEMAPHORE", None)
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    monkeypatch.setattr(es, "_ACTIVE_BATCH_PROCESSORS", 0)
    monkeypatch.setattr(es.NomicEmbedder, "get", classmethod(lambda _cls: recorder))


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
    """Excess concurrent requests queue (never reject) and the cap is honored."""
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "1")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "1")
    recorder = _ConcurrencyRecorder(hold=0.1)
    _reset_service_state(monkeypatch, recorder)

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
        thread.join()

    # All six requests completed — they queued, none were rejected.
    assert errors == []
    assert recorder.calls == 6
    # The semaphore never let more than the configured limit run at once.
    assert recorder.peak <= 2
    # ...and concurrency was actually exercised (not accidentally serialized).
    assert recorder.peak >= 2


def test_micro_batches_concurrent_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent small requests can share one encode call."""
    monkeypatch.setenv("REFLEXIO_EMBED_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS", "50")
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    recorder = _BatchRecorder()
    _reset_service_state(monkeypatch, recorder)

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
        thread.join()

    assert errors == []
    assert len(results) == 2
    assert len(recorder.calls) == 1
    assert set(recorder.calls[0]) == {"first", "second"}
