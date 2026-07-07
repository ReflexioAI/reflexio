"""Unit tests for temporal post-processing (retrieval/temporal.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile
from reflexio.server.services.retrieval.temporal import (
    filter_current,
    freshness_collapse,
    sort_by_recency,
    window_bounds,
)

_NOW = int(datetime.now(UTC).timestamp())
_DAY = 86_400


def _profile(
    pid: str,
    age_days: int,
    *,
    content: str | None = None,
    superseded_by: str | None = None,
    expiration_timestamp: int | None = None,
) -> UserProfile:
    profile = UserProfile(
        profile_id=pid,
        user_id="u1",
        content=content or f"content-{pid}",
        last_modified_timestamp=_NOW - age_days * _DAY,
        generated_from_request_id="r1",
        superseded_by=superseded_by,
    )
    if expiration_timestamp is not None:
        profile.expiration_timestamp = expiration_timestamp
    return profile


def _playbook(upid: int, age_days: int, *, trigger: str, content: str) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=upid,
        user_id="u1",
        agent_version="v1",
        request_id="r1",
        trigger=trigger,
        content=content,
        created_at=_NOW - age_days * _DAY,
    )


# ---------------------------------------------------------------------------
# window_bounds
# ---------------------------------------------------------------------------


def test_window_bounds_full_window():
    now = datetime.now(UTC)
    start, end = window_bounds(7, 0, now)
    assert start == now - timedelta(days=7)
    assert end == now


def test_window_bounds_as_of_only():
    now = datetime.now(UTC)
    start, end = window_bounds(None, 30, now)
    assert start is None
    assert end == now - timedelta(days=30)


def test_window_bounds_unset():
    assert window_bounds(None, None) == (None, None)


def test_window_bounds_swaps_inverted_offsets():
    now = datetime.now(UTC)
    start, end = window_bounds(0, 7, now)
    assert start is not None and end is not None
    assert start < end


# ---------------------------------------------------------------------------
# filter_current
# ---------------------------------------------------------------------------


def test_filter_current_drops_superseded_and_expired():
    live = _profile("live", 5)
    superseded = _profile("old", 50, superseded_by="live")
    expired = _profile("gone", 50, expiration_timestamp=_NOW - _DAY)
    kept = filter_current([live, superseded, expired], _NOW)
    assert [p.profile_id for p in kept] == ["live"]


def test_filter_current_keeps_playbooks_without_expiry():
    playbook = _playbook(1, 5, trigger="t", content="c")
    assert filter_current([playbook], _NOW) == [playbook]


# ---------------------------------------------------------------------------
# freshness_collapse
# ---------------------------------------------------------------------------


def test_freshness_collapse_promotes_fresh_near_duplicate():
    stale = _playbook(
        1, 200, trigger="user says ship", content="Skip tests and deploy immediately."
    )
    fresh = _playbook(2, 5, trigger="user says ship", content="Run tests then deploy.")
    unrelated = _playbook(
        3, 100, trigger="code review", content="Require two approvals."
    )
    collapsed = freshness_collapse([stale, fresh, unrelated])
    assert [p.user_playbook_id for p in collapsed] == [2, 1, 3]


def test_freshness_collapse_leaves_distinct_facts_alone():
    a = _profile("a", 90, content="User prefers postgres for OLTP work.")
    b = _profile("b", 1, content="User plays tennis on weekends.")
    collapsed = freshness_collapse([a, b])
    assert [p.profile_id for p in collapsed] == ["a", "b"]


def test_freshness_collapse_group_anchored_at_best_rank():
    unrelated_top = _profile("top", 10, content="User is a backend engineer.")
    stale = _profile(
        "stale", 90, content="User uses pip for Python package management."
    )
    fresh = _profile(
        "fresh", 2, content="User switched to uv for Python package management."
    )
    collapsed = freshness_collapse([unrelated_top, stale, fresh])
    assert [p.profile_id for p in collapsed] == ["top", "fresh", "stale"]


# ---------------------------------------------------------------------------
# sort_by_recency
# ---------------------------------------------------------------------------


def test_sort_by_recency_orders_newest_first():
    old = _profile("old", 120)
    new = _profile("new", 1)
    assert [p.profile_id for p in sort_by_recency([old, new])] == ["new", "old"]
