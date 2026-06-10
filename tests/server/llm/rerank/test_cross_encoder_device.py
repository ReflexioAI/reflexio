"""Device selection for the cross-encoder reranker singleton."""

from unittest.mock import MagicMock, patch

import pytest

import reflexio.server.llm.rerank.cross_encoder_reranker as reranker


@pytest.fixture(autouse=True)
def _reset_model_singleton():
    """Isolate each test from the process-wide model singleton."""
    reranker._MODEL = None
    yield
    reranker._MODEL = None


def _load_model_with_mock() -> MagicMock:
    cross_encoder_cls = MagicMock()
    with patch.object(
        reranker, "_import_cross_encoder", return_value=cross_encoder_cls
    ):
        reranker._get_model()
    return cross_encoder_cls


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("REFLEXIO_RERANK_DEVICE", raising=False)
    cross_encoder_cls = _load_model_with_mock()
    cross_encoder_cls.assert_called_once_with(reranker._MODEL_NAME, device="cpu")


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_DEVICE", "mps")
    cross_encoder_cls = _load_model_with_mock()
    cross_encoder_cls.assert_called_once_with(reranker._MODEL_NAME, device="mps")


def test_score_pairs_disables_progress_bar():
    model = MagicMock()
    model.predict.return_value = [0.5]
    reranker._MODEL = model
    reranker.score_pairs("query", ["doc"])
    model.predict.assert_called_once_with([("query", "doc")], show_progress_bar=False)
