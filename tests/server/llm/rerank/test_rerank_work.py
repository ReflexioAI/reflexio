"""Larger turns must preserve the padding of every original four-pair group."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.rerank.common import MULTILINGUAL_RERANK_MODEL
from reflexio.server.llm.rerank.cross_encoder_model import (
    CrossEncoderRunner,
    RerankWork,
)


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ([32] * 16, 16),
        ([128] * 16, 8),
        ([256] * 16, 4),
        ([512] * 16, 4),
        ([32] * 4 + [64] * 12, 4),
        ([32] * 8 + [512] * 8, 8),
        ([32] * 10, 16),
    ],
)
def test_planner_preserves_padding_and_token_budget(lengths, expected):
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    model = MagicMock()
    model.preprocess.return_value[
        "attention_mask"
    ].sum.return_value.tolist.return_value = lengths
    runner._model = model
    documents = [str(i) for i in range(len(lengths))]

    assert RerankWork(runner, "query").select_chunk_size(documents) == expected
    model.preprocess.assert_called_once_with([("query", doc) for doc in documents])
    model.preprocess.return_value["attention_mask"].sum.assert_called_once_with(dim=1)


def test_planning_looks_ahead_at_most_sixteen_pairs():
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    model = MagicMock()
    model.preprocess.return_value[
        "attention_mask"
    ].sum.return_value.tolist.return_value = [32] * 16
    runner._model = model
    documents = [str(i) for i in range(90)]
    assert RerankWork(runner, "query").select_chunk_size(documents) == 16
    model.preprocess.assert_called_once_with([("query", doc) for doc in documents[:16]])


def test_short_tail_does_not_load_or_preprocess_model():
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    assert RerankWork(runner, "query").select_chunk_size(["tail"] * 4) == 4
    assert runner._model is None


def test_older_runtime_keeps_original_four_pair_turns():
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    runner._model = SimpleNamespace()
    assert RerankWork(runner, "query").select_chunk_size(["doc"] * 16) == 4


def test_callable_preserves_query_documents_and_scores():
    runner = MagicMock(spec=CrossEncoderRunner)
    runner.score_pairs.return_value = [-2.5, 1.0, -2.5]
    documents = ["first", "second", "first"]
    assert RerankWork(runner, "query")(documents) == [-2.5, 1.0, -2.5]
    runner.score_pairs.assert_called_once_with("query", documents)
