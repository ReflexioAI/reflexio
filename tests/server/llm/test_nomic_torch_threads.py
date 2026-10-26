"""Model constructors use the shared numerical-library owner before loading."""

from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def clean_thread_inputs(monkeypatch):
    for name in ("REFLEXIO_EMBED_TORCH_THREADS", "REFLEXIO_INFERENCE_TORCH_THREADS"):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "REFLEXIO_NATIVE_COMPUTE_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "2")


@pytest.fixture
def torch_budget(monkeypatch):
    import torch
    from threadpoolctl import threadpool_limits

    current = [8]
    setter = Mock(side_effect=lambda count: current.__setitem__(0, count))
    monkeypatch.setattr(torch, "get_num_threads", lambda: current[0])
    monkeypatch.setattr(torch, "set_num_threads", setter)
    # These constructor-order tests inspect the Torch API; the real native pools
    # are qualified in separate subprocesses by test_native_compute.py.
    monkeypatch.setattr("threadpoolctl.threadpool_limits", Mock(spec=threadpool_limits))
    return current, setter


def test_nomic_constructor_observes_the_shared_budget(monkeypatch, torch_budget):
    from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder

    monkeypatch.setenv("REFLEXIO_NATIVE_COMPUTE_THREADS", "3")
    model = Mock()
    model.get_embedding_dimension.return_value = 768

    def construct(*args, **kwargs):
        assert torch_budget[0] == [3]
        return model

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", construct)
    assert NomicEmbedder()._load() is model


@pytest.mark.parametrize("name", ["REFLEXIO_NATIVE_COMPUTE_THREADS"])
def test_reranker_first_load_applies_budget_before_model_import(
    monkeypatch, torch_budget, name
):
    from reflexio.server.llm.rerank import cross_encoder_model
    from reflexio.server.llm.rerank.common import ENGLISH_RERANK_MODEL

    monkeypatch.setenv(name, "3")
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    model = Mock()
    constructor = Mock(return_value=model)

    def import_model():
        # A MiniLM deployment has not constructed either Torch embedding model.
        assert torch_budget[0] == [3]
        return constructor

    monkeypatch.setattr(cross_encoder_model, "_import_cross_encoder", import_model)
    runner = cross_encoder_model.CrossEncoderRunner(ENGLISH_RERANK_MODEL)
    assert runner._get_model() is model
    assert runner._get_model() is model
    constructor.assert_called_once()
    torch_budget[1].assert_called_once_with(3)
