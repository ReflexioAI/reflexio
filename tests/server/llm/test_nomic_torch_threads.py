"""One Torch thread budget, including compatible Nomic configuration."""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from reflexio.server.llm.torch_threads import configure_torch_threads


@pytest.fixture(autouse=True)
def clean_thread_inputs(monkeypatch):
    monkeypatch.delenv("REFLEXIO_EMBED_TORCH_THREADS", raising=False)
    monkeypatch.delenv("REFLEXIO_INFERENCE_TORCH_THREADS", raising=False)


@pytest.fixture
def torch_budget(monkeypatch):
    import torch

    current = [8]

    def set_threads(count):
        # Force overlapping callers to observe the old count without the owner lock.
        time.sleep(0.01)
        current[0] = count

    setter = Mock(side_effect=set_threads)
    monkeypatch.setattr(torch, "get_num_threads", lambda: current[0])
    monkeypatch.setattr(torch, "set_num_threads", setter)
    return current, setter


@pytest.mark.parametrize(
    "name", ["REFLEXIO_EMBED_TORCH_THREADS", "REFLEXIO_INFERENCE_TORCH_THREADS"]
)
@pytest.mark.parametrize("count", [1, 3])
def test_both_names_select_the_same_budget_once(monkeypatch, torch_budget, name, count):
    monkeypatch.setenv(name, str(count))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: configure_torch_threads(default=2), range(8)))
    current, setter = torch_budget
    assert current == [count]
    setter.assert_called_once_with(count)


def test_unconfigured_nomic_keeps_torch_default(torch_budget):
    configure_torch_threads()
    torch_budget[1].assert_not_called()


def test_unconfigured_e5_keeps_existing_two_thread_default(torch_budget):
    configure_torch_threads(default=2)
    torch_budget[1].assert_called_once_with(2)


@pytest.mark.parametrize("default", [None, 2])
@pytest.mark.parametrize("invalid", ["", "not-a-number", "0", "-1"])
def test_invalid_budget_keeps_model_default(
    monkeypatch, torch_budget, default, invalid
):
    monkeypatch.setenv("REFLEXIO_INFERENCE_TORCH_THREADS", invalid)
    configure_torch_threads(default=default)
    if default is None:
        torch_budget[1].assert_not_called()
    else:
        torch_budget[1].assert_called_once_with(default)


def test_matching_legacy_and_canonical_values_are_accepted(monkeypatch, torch_budget):
    monkeypatch.setenv("REFLEXIO_INFERENCE_TORCH_THREADS", "3")
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "3")
    configure_torch_threads()
    torch_budget[1].assert_called_once_with(3)


def test_conflicting_budgets_fail_before_changing_torch(monkeypatch, torch_budget):
    monkeypatch.setenv("REFLEXIO_INFERENCE_TORCH_THREADS", "2")
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "3")
    with pytest.raises(ValueError, match="Conflicting Torch thread budgets"):
        configure_torch_threads()
    torch_budget[1].assert_not_called()


def test_legacy_budget_is_not_overwritten_by_e5_default(monkeypatch, torch_budget):
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "3")
    configure_torch_threads()
    configure_torch_threads(default=2)
    torch_budget[1].assert_called_once_with(3)


def test_actual_torch_budget_and_tensor_operation_in_fresh_process(monkeypatch):
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "2")
    result = subprocess.run(  # noqa: S603 — fixed interpreter and literal source
        [
            sys.executable,
            "-c",
            "from reflexio.server.llm.torch_threads import configure_torch_threads; configure_torch_threads(); import torch; assert torch.get_num_threads() == 2; assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8; configure_torch_threads(default=2); assert torch.get_num_threads() == 2",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_unconfigured_budget_does_not_import_torch():
    result = subprocess.run(  # noqa: S603 — fixed interpreter and literal source
        [
            sys.executable,
            "-c",
            "import sys; from reflexio.server.llm.torch_threads import configure_torch_threads; configure_torch_threads(); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_nomic_constructor_observes_the_shared_budget(monkeypatch, torch_budget):
    from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder

    monkeypatch.setenv("REFLEXIO_INFERENCE_TORCH_THREADS", "3")
    model = Mock()
    model.get_embedding_dimension.return_value = 768

    def construct(*args, **kwargs):
        assert torch_budget[0] == [3]
        return model

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", construct)
    assert NomicEmbedder()._load() is model


def test_valid_alias_survives_invalid_canonical_input(monkeypatch, torch_budget):
    monkeypatch.setenv("REFLEXIO_INFERENCE_TORCH_THREADS", "invalid")
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "3")
    configure_torch_threads(default=2)
    torch_budget[1].assert_called_once_with(3)


@pytest.mark.parametrize(
    "name", ["REFLEXIO_EMBED_TORCH_THREADS", "REFLEXIO_INFERENCE_TORCH_THREADS"]
)
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
