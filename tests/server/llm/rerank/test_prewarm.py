from unittest.mock import patch

from reflexio.server.llm.rerank.cross_encoder_reranker import (
    CrossEncoderUnavailableError,
    prewarm,
)


def test_prewarm_returns_true_when_model_loads():
    with patch(
        "reflexio.server.llm.rerank.cross_encoder_reranker.score_pairs",
        return_value=[0.0],
    ) as mock_score:
        assert prewarm() is True
    mock_score.assert_called_once()


def test_prewarm_returns_false_when_unavailable():
    with patch(
        "reflexio.server.llm.rerank.cross_encoder_reranker.score_pairs",
        side_effect=CrossEncoderUnavailableError("no model"),
    ):
        assert prewarm() is False


def test_prewarm_returns_false_on_unexpected_error():
    with patch(
        "reflexio.server.llm.rerank.cross_encoder_reranker.score_pairs",
        side_effect=RuntimeError("predict blew up"),
    ):
        assert prewarm() is False
