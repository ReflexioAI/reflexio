"""Execution policy and concurrency shared by native sentence models."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:2"])
def test_actual_loader_honors_process_device(monkeypatch, device):
    monkeypatch.setenv("REFLEXIO_EMBEDDING_DEVICE", device)
    model = MagicMock()
    model.get_embedding_dimension.return_value = 768
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    NomicEmbedder()._load()
    assert constructor.call_args.kwargs["device"] == device


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_device_uses_same_cpu_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_DEVICE", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_DEVICE", value)
    constructor = MagicMock()
    constructor.return_value.get_embedding_dimension.return_value = 768
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    NomicEmbedder()._load()
    assert constructor.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize("old", ["", "cpu", "cuda"])
def test_old_device_never_overrides_or_disappears(monkeypatch, old):
    monkeypatch.setenv("NOMIC_EMBED_DEVICE", old)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_DEVICE", "cpu")
    constructor = MagicMock()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    with pytest.raises(
        ValueError, match="NOMIC_EMBED_DEVICE.*REFLEXIO_EMBEDDING_DEVICE"
    ):
        NomicEmbedder()._load()
    constructor.assert_not_called()


def test_concurrent_first_use_constructs_once_and_serializes_encoding(monkeypatch):
    monkeypatch.setenv("REFLEXIO_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("REFLEXIO_EMBED_BATCH_SIZE", "7")
    barrier = threading.Barrier(2)
    guard = threading.Lock()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    model = MagicMock()
    model.get_embedding_dimension.return_value = 768

    def encode(texts, **kwargs):
        assert guard.acquire(blocking=False), "concurrent native encode"
        try:
            entered.set()
            assert release.wait(3)
            calls.append((texts, kwargs))
            return np.ones((len(texts), 768))
        finally:
            guard.release()

    model.encode.side_effect = encode
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    embedder = NomicEmbedder()

    def embed(text):
        barrier.wait(3)
        return embedder.embed([text])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(embed, text) for text in ["one", "two"]]
        try:
            assert entered.wait(3)
        finally:
            release.set()
        results = [future.result(timeout=3) for future in futures]
    constructor.assert_called_once()
    assert sorted(texts[0] for texts, _ in calls) == ["one", "two"]
    assert all(kwargs["batch_size"] == 7 for _, kwargs in calls)
    assert all(len(rows[0]) == 512 for rows in results)
    assert all(sum(x * x for x in rows[0]) == pytest.approx(1) for rows in results)
