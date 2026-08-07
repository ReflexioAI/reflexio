from unittest.mock import patch

from reflexio.server.llm.rerank.common import RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner


def test_prewarm_returns_true_when_model_scores():
    runner = CrossEncoderRunner(RERANK_MODEL)
    with patch.object(runner, "score_pairs", return_value=[0.0]) as score:
        assert runner.prewarm() is True
    score.assert_called_once_with("warmup", ["warmup"])


def test_prewarm_returns_false_when_scoring_fails():
    runner = CrossEncoderRunner(RERANK_MODEL)
    with patch.object(runner, "score_pairs", side_effect=RuntimeError("failed")):
        assert runner.prewarm() is False


def test_disabled_prewarm_does_not_load_model(monkeypatch):
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "false")
    runner = CrossEncoderRunner(RERANK_MODEL)
    with patch.object(runner, "_get_model") as get_model:
        assert runner.prewarm() is False
    get_model.assert_not_called()
