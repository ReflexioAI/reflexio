"""Tests for _filter_citations_by_horizon helper."""

from datetime import UTC, datetime

from reflexio.models.api_schema.domain.entities import Citation, Interaction
from reflexio.server.services.reflection.service import (
    _filter_citations_by_horizon,
)


def _interaction(idx: int, role: str = "Assistant", citations=()) -> Interaction:
    """Build a minimal Interaction for tests."""
    return Interaction(
        interaction_id=idx,
        user_id="u1",
        request_id="req",
        role=role,
        content=f"turn {idx}",
        tools_used=[],
        citations=list(citations),
        created_at=int(datetime.now(UTC).timestamp()) + idx,
    )


def test_filter_drops_citation_with_insufficient_horizon():
    """Citation on the last turn of a 10-turn window with stride 5 should be
    deferred (position 9 >= stride 5, only 0 turns after)."""
    cite = Citation(kind="playbook", real_id="42")
    window = [_interaction(i) for i in range(9)] + [_interaction(9, citations=[cite])]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=3,
        stride_size=5,
    )
    assert eligible == []


def test_filter_keeps_citation_with_full_horizon():
    """Citation on turn 0 has 9 turns after it. With post_horizon_size=3, eligible."""
    cite = Citation(kind="playbook", real_id="42")
    window = [_interaction(0, citations=[cite])] + [
        _interaction(i) for i in range(1, 10)
    ]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=3,
        stride_size=5,
    )
    assert len(eligible) == 1
    assert eligible[0].citation.real_id == "42"
    assert eligible[0].has_full_horizon is True


def test_filter_last_chance_when_about_to_fall_out():
    """Citation at position 9 of 10 with stride_size=10 means it's about to
    fall out next stride. Should be eligible as last_chance (has_full_horizon=False)."""
    cite = Citation(kind="playbook", real_id="42")
    window = [_interaction(i) for i in range(9)] + [_interaction(9, citations=[cite])]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=3,
        stride_size=10,
    )
    assert len(eligible) == 1
    assert eligible[0].has_full_horizon is False


def test_filter_zero_horizon_disables_filter():
    """post_horizon_size=0 → every citation is full-horizon eligible."""
    cite = Citation(kind="playbook", real_id="42")
    window = [_interaction(i) for i in range(9)] + [_interaction(9, citations=[cite])]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=0,
        stride_size=5,
    )
    assert len(eligible) == 1
    assert eligible[0].has_full_horizon is True


def test_filter_dedupes_by_kind_and_real_id_picking_earliest():
    """Same citation appearing on multiple turns → only emitted once,
    using the EARLIEST occurrence (maximizes after-context)."""
    cite = Citation(kind="playbook", real_id="42")
    window = [
        _interaction(0, citations=[cite]),
        _interaction(1),
        _interaction(2),
        _interaction(3, citations=[cite]),  # later occurrence — should be ignored
        _interaction(4),
        _interaction(5),
        _interaction(6),
        _interaction(7),
        _interaction(8),
        _interaction(9),
    ]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=3,
        stride_size=5,
    )
    assert len(eligible) == 1
    assert eligible[0].position == 0  # earliest, not 3


def test_filter_ignores_citations_on_non_assistant_turns():
    """Only Assistant turns contribute citations to the filter."""
    cite = Citation(kind="playbook", real_id="42")
    # User turn has the citation reference — should be ignored
    window = [_interaction(0, role="User", citations=[cite])] + [
        _interaction(i) for i in range(1, 10)
    ]
    eligible = _filter_citations_by_horizon(
        citations=[cite],
        window=window,
        post_horizon_size=3,
        stride_size=5,
    )
    assert eligible == []
