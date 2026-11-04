"""Cached/direct paths must apply tokenizer policy before native execution."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from reflexio.server.llm import _litellm_embedding, _model_compat, input_budget
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers.local_embedding_provider import ONNXMiniLM
from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder
from reflexio.server.llm.rerank.common import MULTILINGUAL_RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_model import (
    CrossEncoderRunner,
    RerankWork,
)


def _assert_policy() -> None:
    assert os.environ["RAYON_NUM_THREADS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


@pytest.fixture(autouse=True)
def lower_budget(monkeypatch):
    monkeypatch.setenv("REFLEXIO_NATIVE_COMPUTE_THREADS", "1")
    monkeypatch.setenv("RAYON_NUM_THREADS", "3")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")


class _PolicyLock:
    def __enter__(self):
        _assert_policy()

    def __exit__(self, *args):
        return False


def test_direct_complete_prompt_count_configures_before_both_counters(monkeypatch):
    def count(**kwargs):
        _assert_policy()
        return 10

    counter = Mock(side_effect=count)
    monkeypatch.setattr(input_budget.litellm, "token_counter", counter)
    assert (
        input_budget.measure_request_input_tokens(
            {
                "model": "synthetic",
                "messages": [],
                "response_format": {"type": "json_object"},
            }
        )
        == 24
    )
    assert counter.call_count == 2


def test_cached_sentence_model_configures_before_model_lock(monkeypatch):
    encoder = NomicEmbedder()
    encoder._model = Mock()
    encoder._model.encode.return_value = np.ones((1, 768))
    monkeypatch.setattr(encoder, "_model_lock", _PolicyLock())
    assert len(encoder.embed(["synthetic"])[0]) == 512
    encoder._model.encode.assert_called_once()


def test_cached_reranker_planning_configures_before_preprocess_and_lock(monkeypatch):
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    model = Mock()
    mask = Mock()
    mask.sum.return_value.tolist.return_value = [2] * 8
    model.preprocess.return_value = {"attention_mask": mask}
    runner._model = model
    monkeypatch.setattr(runner, "_predict_lock", _PolicyLock())
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    assert RerankWork(runner, "query").select_chunk_size(["doc"] * 8) == 16
    model.preprocess.assert_called_once()


def test_cached_onnx_model_configures_before_individual_encode(monkeypatch):
    encoder = ONNXMiniLM.__new__(ONNXMiniLM)
    monkeypatch.setattr(encoder, "_np", np, raising=False)
    encoder._session = Mock()
    encoder._session.run.return_value = [np.ones((1, 2, 3))]

    def encode(text):
        _assert_policy()
        return SimpleNamespace(ids=[1, 2], attention_mask=[1, 1])

    encoder._tokenizer = Mock(encode=Mock(side_effect=encode))
    assert len(encoder(["synthetic"])[0]) == 3
    encoder._tokenizer.encode.assert_called_once_with("synthetic")


@pytest.mark.parametrize("batch", [False, True])
def test_both_cloud_embedding_methods_configure_before_vendor_dispatch(
    monkeypatch, batch
):
    monkeypatch.setattr(
        _litellm_embedding, "_truncate_for_embedding", lambda text, _: text
    )
    client = LiteLLMClient(LiteLLMConfig(model="synthetic"))

    def dispatch(**kwargs):
        _assert_policy()
        return SimpleNamespace(data=[{"index": 0, "embedding": [0.5, 0.5]}])

    vendor = Mock(side_effect=dispatch)
    monkeypatch.setattr(_litellm_embedding.litellm, "embedding", vendor)
    if batch:
        assert client.get_embeddings(["text"], "text-embedding-3-small") == [[0.5, 0.5]]
    else:
        assert client.get_embedding("text", "text-embedding-3-small") == [0.5, 0.5]
    vendor.assert_called_once()


def test_shared_completion_configures_before_vendor_dispatch(monkeypatch):
    def dispatch(**kwargs):
        _assert_policy()
        return "synthetic"

    vendor = Mock(side_effect=dispatch)
    monkeypatch.setattr(_model_compat.litellm, "completion", vendor)
    assert (
        _model_compat.model_completion({"model": "synthetic", "messages": []})
        == "synthetic"
    )
    vendor.assert_called_once()
