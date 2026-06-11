from dataclasses import dataclass
from unittest.mock import patch

from reflexio.server.services.retrieval.relevance_floor import (
    apply_relevance_floor,
    apply_relevance_floors,
)


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


def test_batched_floors_score_once_and_apply_per_arm_floors():
    arms = [
        ("a1", _items("p1", "p2"), -5.0),
        ("a2", _items("q1"), 0.5),
    ]
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[1.0, -7.0, 0.0],
    ) as mock_score:
        out = apply_relevance_floors("q", arms, top_k=10)
    mock_score.assert_called_once_with("q", ["p1", "p2", "q1"])
    assert [i.content for i in out[0]] == ["p1"]  # -7.0 below -5.0 floor
    assert out[1] == []  # 0.0 below the 0.5 floor


def test_batched_floors_sort_desc_and_cap_per_arm():
    arms = [("a", _items("x", "y", "z"), -5.0)]
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[1.0, 3.0, 2.0],
    ):
        out = apply_relevance_floors("q", arms, top_k=2)
    assert [i.content for i in out[0]] == ["y", "z"]


def test_batched_floors_unavailable_degrades_each_arm():
    from reflexio.server.llm.rerank.cross_encoder_reranker import (
        CrossEncoderUnavailableError,
    )

    arms = [
        ("a1", _items("a", "b", "c"), -5.0),
        ("a2", _items("d"), -5.0),
    ]
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        side_effect=CrossEncoderUnavailableError("no model"),
    ):
        out = apply_relevance_floors("q", arms, top_k=2)
    assert [i.content for i in out[0]] == ["a", "b"]
    assert [i.content for i in out[1]] == ["d"]


def test_batched_floors_all_empty_skips_scoring():
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs"
    ) as mock_score:
        out = apply_relevance_floors("q", [("a1", [], -5.0), ("a2", [], -5.0)], top_k=5)
    assert out == [[], []]
    mock_score.assert_not_called()


def test_batched_floors_empty_arm_keeps_offsets_aligned():
    arms = [
        ("a1", [], -5.0),
        ("a2", _items("x"), -5.0),
    ]
    with patch(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs",
        return_value=[2.0],
    ):
        out = apply_relevance_floors("q", arms, top_k=5)
    assert out[0] == []
    assert [i.content for i in out[1]] == ["x"]
