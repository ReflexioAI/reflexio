from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from reflexio.server.llm.providers.multilingual_e5_embedding_provider import (
    MULTILINGUAL_E5_HF_MODEL,
    MULTILINGUAL_E5_HF_REVISION,
    MultilingualE5Embedder,
    MultilingualE5EmbedderError,
    _pad_unit_embedding,
)


def test_pad_unit_embedding_preserves_unit_norm_and_storage_width() -> None:
    padded = _pad_unit_embedding([0.5] * 384)

    assert len(padded) == 512
    assert sum(value * value for value in padded) == pytest.approx(1.0)
    assert padded[384:] == [0.0] * 128


def test_pad_unit_embedding_rejects_wrong_native_width() -> None:
    with pytest.raises(MultilingualE5EmbedderError, match="384-dimensional"):
        _pad_unit_embedding([0.5] * 383)


def test_embed_uses_normalized_sentence_transformer_output(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_EMBED_BATCH_SIZE", "8")
    embedder = MultilingualE5Embedder()
    model = MagicMock()
    model.encode.return_value = np.array([[0.5] * 384])
    embedder._model = model

    result = embedder.embed(["query: 用户使用什么数据库？"])

    assert len(result[0]) == 512
    model.encode.assert_called_once()
    assert model.encode.call_args.kwargs == {
        "batch_size": 8,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_load_rejects_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(MultilingualE5EmbedderError, match="dedicated CUDA"):
        MultilingualE5Embedder()._load()


def test_load_pins_the_baked_model_revision(monkeypatch) -> None:
    loaded = MagicMock()
    sentence_transformer = MagicMock(return_value=loaded)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", sentence_transformer
    )

    assert MultilingualE5Embedder()._load() is loaded
    sentence_transformer.assert_called_once_with(
        MULTILINGUAL_E5_HF_MODEL,
        device="cuda",
        revision=MULTILINGUAL_E5_HF_REVISION,
    )
