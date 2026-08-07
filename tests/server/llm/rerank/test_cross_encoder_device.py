from unittest.mock import ANY, MagicMock, patch

import pytest

import reflexio.server.llm.rerank.cross_encoder_model as model_module
from reflexio.server.llm.rerank.common import (
    MULTILINGUAL_RERANK_MODEL,
    MULTILINGUAL_RERANK_REVISION,
    CrossEncoderUnavailableError,
)
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner


def test_model_load_uses_selected_revision_and_device(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_DEVICE", "mps")
    cross_encoder_cls = MagicMock()
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    with patch.object(
        model_module, "_import_cross_encoder", return_value=cross_encoder_cls
    ):
        runner._get_model()

    cross_encoder_cls.assert_called_once_with(
        MULTILINGUAL_RERANK_MODEL,
        device="mps",
        revision=MULTILINGUAL_RERANK_REVISION,
    )


def test_score_pairs_disables_progress_and_forces_raw_signed_logits():
    from torch import nn

    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    model = MagicMock()
    model.predict.return_value = [-3.5]
    runner._model = model

    assert runner.score_pairs("query", ["doc"]) == [-3.5]
    model.predict.assert_called_once_with(
        [("query", "doc")],
        show_progress_bar=False,
        activation_fn=ANY,
    )
    _, kwargs = model.predict.call_args
    assert isinstance(kwargs["activation_fn"], nn.Identity)


def test_score_pairs_predict_failure_raises_unavailable():
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    model = MagicMock()
    model.predict.side_effect = RuntimeError("cuda oom")
    runner._model = model

    with pytest.raises(CrossEncoderUnavailableError):
        runner.score_pairs("query", ["doc"])
