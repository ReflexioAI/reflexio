from dataclasses import dataclass
from unittest.mock import patch

from reflexio.server.services.retrieval.relevance_floor import apply_relevance_floor


@dataclass
class _Item:
    content: str


def _items(*texts: str) -> list[_Item]:
    return [_Item(content=t) for t in texts]


def test_drops_below_floor_and_sorts_desc():
    items = _items("a", "b", "c")
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[-2.0, -8.0, 1.0],
    ):
        out = apply_relevance_floor("q", items, floor=-5.0, top_k=10, arm="test")
    assert [i.content for i in out] == ["c", "a"]


def test_returns_zero_when_all_below_floor():
    items = _items("a", "b")
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[-9.0, -7.0],
    ):
        out = apply_relevance_floor("q", items, floor=-5.0, top_k=10, arm="test")
    assert out == []


def test_caps_to_top_k():
    items = _items("a", "b", "c")
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[1.0, 2.0, 3.0],
    ):
        out = apply_relevance_floor("q", items, floor=-5.0, top_k=2, arm="test")
    assert [i.content for i in out] == ["c", "b"]


def test_empty_items_returns_empty_without_scoring():
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs"
    ) as mock_score:
        out = apply_relevance_floor("q", [], floor=-5.0, top_k=10, arm="test")
    assert out == []
    mock_score.assert_not_called()


def test_unavailable_reranker_degrades_to_unfiltered():
    from reflexio.server.llm.rerank.cross_encoder_reranker import (
        CrossEncoderUnavailableError,
    )

    items = _items("a", "b", "c")
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=CrossEncoderUnavailableError("no model"),
    ):
        out = apply_relevance_floor("q", items, floor=-5.0, top_k=2, arm="test")
    assert [i.content for i in out] == ["a", "b"]
