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
    monkeypatch.setattr(es, "_BATCH_WORKER_RUNNING", False)
    embedder = es.NomicEmbedder()
    embedder._model = model  # pre-load so _load() is a no-op fast path
    monkeypatch.setattr(es.NomicEmbedder, "get", classmethod(lambda _cls: embedder))


def test_micro_batch_max_texts_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    assert es._micro_batch_max_texts() == 8


def test_embed_texts_caps_and_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Excess concurrent requests queue (never reject) and encodes serialize."""
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


@pytest.mark.parametrize("old_marker", [None, "1"])
def test_service_owns_nomic_lifecycle_without_registration_marker(
    monkeypatch, old_marker
):
    from fastapi.testclient import TestClient

    if old_marker is None:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_DAEMON", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_DAEMON", old_marker)
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "false")
    model = _ConcurrencyModel(hold=0)
    _reset_service_state(monkeypatch, model)
    app = es.create_embedding_app(default_model=es.NOMIC_TEXT_MODEL)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post(
            "/v1/embeddings", json={"model": es.NOMIC_TEXT_MODEL, "input": ["hello"]}
        )
        assert response.status_code == 200
        assert len(response.json()["data"][0]["embedding"]) == 512
        assert model.calls >= 2  # Startup warmup and the HTTP request both encode.


@pytest.mark.parametrize("batch_limit", [2, 8])
def test_queue_has_one_owner_and_batches_waiting_requests(monkeypatch, batch_limit):
    """Serialization belongs to the queue, including registered encoders."""
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", str(batch_limit))
    entered = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    calls = []
    active = 0
    peak = 0

    def encode(texts):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            calls.append(list(texts))
        try:
            if texts == ["blocker"]:
                entered.set()
                assert release.wait(5), "test did not release encoder"
            return [[float(len(text))] for text in texts]
        finally:
            with guard:
                active -= 1

    def embed(text):
        return es._embed_texts(
            "custom/test", [text], encoder=encode, allowed_models={"custom/test"}
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(embed, "blocker")
        others = []
        try:
            assert entered.wait(2)
            others = [pool.submit(embed, text) for text in ("one", "second", "third")]
            deadline = time.monotonic() + 2
            while (
                len(es._MICRO_BATCH_QUEUE) != 3
                and peak == 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            assert peak == 1, "competing processors entered the registered encoder"
            assert len(es._MICRO_BATCH_QUEUE) == 3
        finally:
            release.set()
        assert first.result(timeout=3) == [[7.0]]
        assert [future.result(timeout=3) for future in others] == [
            [[3.0]],
            [[6.0]],
            [[5.0]],
        ]
    assert calls[0] == ["blocker"]
    assert len(calls[1]) == min(3, batch_limit)
    assert {text for batch in calls[1:] for text in batch} == {"one", "second", "third"}
    assert len(calls) == (3 if batch_limit == 2 else 2)
    assert peak == 1


def test_worker_start_failure_discards_unowned_work_and_recovers(monkeypatch):
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return [[1.0] for _ in texts]

    def unavailable(_thread):
        raise RuntimeError("worker unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", unavailable)
        with pytest.raises(RuntimeError, match="worker unavailable"):
            es._embed_texts(
                "custom/test",
                ["failed"],
                encoder=encode,
                allowed_models={"custom/test"},
            )
    assert es._MICRO_BATCH_QUEUE == [], "failed admission left ghost work queued"
    assert es._embed_texts(
        "custom/test", ["next"], encoder=encode, allowed_models={"custom/test"}
    ) == [[1.0]]
    assert calls == [["next"]]


@pytest.mark.parametrize("failure", ["exception", "short-result"])
def test_failed_encode_reaches_its_caller_and_queue_can_restart(monkeypatch, failure):
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])

    def encode(texts):
        if texts == ["failed"]:
            if failure == "exception":
                raise ValueError("encode failed")
            return []
        return [[1.0] for _ in texts]

    with pytest.raises(
        (ValueError, RuntimeError), match="encode failed|count mismatch"
    ):
        es._embed_texts(
            "custom/test", ["failed"], encoder=encode, allowed_models={"custom/test"}
        )
    assert es._embed_texts(
        "custom/test", ["next"], encoder=encode, allowed_models={"custom/test"}
    ) == [[1.0]]


def test_registered_encoder_identity_is_not_coalesced(monkeypatch):
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    first = es._EmbeddingJob(
        "custom/test", ["a"], encoder=lambda texts: [[1.0] for _ in texts]
    )
    other = es._EmbeddingJob(
        "custom/test", ["b"], encoder=lambda texts: [[2.0] for _ in texts]
    )
    es._MICRO_BATCH_QUEUE.extend([first, other])
    batch = es._take_micro_batch()
    assert batch == [first]
    es._process_micro_batch(batch)
    assert first.done.is_set() and first.result == [[1.0]]
    assert not other.done.is_set()
    es._process_micro_batch(es._take_micro_batch())
    assert other.done.is_set() and other.result == [[2.0]]


def test_worker_crash_fails_active_and_queued_callers_and_restarts(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(es, "_JOB_WAIT_TIMEOUT_SECONDS", 4)
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    entered, release = threading.Event(), threading.Event()
    original = es._process_micro_batch

    def crash(_jobs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("worker processing failed")

    def encode(texts):
        return [[1.0] for _ in texts]

    def embed(text):
        return es._embed_texts(
            "custom/test", [text], encoder=encode, allowed_models={"custom/test"}
        )

    monkeypatch.setattr(es, "_process_micro_batch", crash)
    with ThreadPoolExecutor(max_workers=3) as pool:
        active = pool.submit(embed, "active")
        queued = []
        try:
            assert entered.wait(2)
            queued = [pool.submit(embed, text) for text in ("queued1", "queued2")]
            deadline = time.monotonic() + 2
            while len(es._MICRO_BATCH_QUEUE) != 2 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert len(es._MICRO_BATCH_QUEUE) == 2
        finally:
            release.set()
        for future in [active, *queued]:
            with pytest.raises(RuntimeError, match="worker processing failed"):
                future.result(timeout=3)
    assert es._MICRO_BATCH_QUEUE == []
    monkeypatch.setattr(es, "_process_micro_batch", original)
    assert embed("restart") == [[1.0]]


@pytest.mark.parametrize("failure", ["exception", "short-result"])
def test_failed_coalesced_batch_fails_all_borrowers_but_not_other_encoder(
    monkeypatch, failure
):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(es, "_JOB_WAIT_TIMEOUT_SECONDS", 4)
    monkeypatch.setattr(es, "_ACTIVE_MODEL", None)
    monkeypatch.setattr(es, "_MICRO_BATCH_QUEUE", [])
    monkeypatch.setenv("REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS", "8")
    entered, release = threading.Event(), threading.Event()
    batches = []

    def encode(texts):
        batches.append(list(texts))
        if texts == ["blocker"]:
            entered.set()
            assert release.wait(5)
        if any(text.startswith("bad") for text in texts):
            if failure == "exception":
                raise ValueError("combined encode failed")
            return []
        return [[1.0] for _ in texts]

    def isolated(texts):
        return [[2.0] for _ in texts]

    def embed(text, encoder=encode):
        return es._embed_texts(
            "custom/test", [text], encoder=encoder, allowed_models={"custom/test"}
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        active = pool.submit(embed, "blocker")
        failed = []
        try:
            assert entered.wait(2)
            failed = [pool.submit(embed, text) for text in ("bad1", "bad2")]
            unrelated = pool.submit(embed, "other", isolated)
            deadline = time.monotonic() + 2
            while len(es._MICRO_BATCH_QUEUE) != 3 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert len(es._MICRO_BATCH_QUEUE) == 3
        finally:
            release.set()
        assert active.result(timeout=3) == [[1.0]]
        for future in failed:
            with pytest.raises(
                (ValueError, RuntimeError),
                match="combined encode failed|count mismatch",
            ):
                future.result(timeout=3)
        assert unrelated.result(timeout=3) == [[2.0]]
    assert len(batches) == 2
    assert set(batches[1]) == {"bad1", "bad2"}
