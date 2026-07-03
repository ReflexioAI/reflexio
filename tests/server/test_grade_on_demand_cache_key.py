"""Unit tests for the grade-on-demand cache key builder.

The key namespaces the on-demand grading cache in ``operation_state``. It MUST
be injective across its four components (org_id, session_id, agent_version,
evaluation_name) — a collision serves a cached verdict for one input against a
DIFFERENT input.
"""

from __future__ import annotations

from reflexio.server.routes.evaluation import (
    _GRADE_ON_DEMAND_CACHE_KEY_PREFIX,
    _grade_on_demand_cache_key,
)


def test_cache_key_is_injective_across_delimiter_boundaries() -> None:
    """Distinct component tuples must never collapse to the same key.

    ``session_id="a::b", agent_version="c"`` and
    ``session_id="a", agent_version="b::c"`` are DIFFERENT inputs; with a naive
    ``::`` join they produce an identical key and cross-serve verdicts.
    """
    key_a = _grade_on_demand_cache_key("org", "a::b", "c", "eval")
    key_b = _grade_on_demand_cache_key("org", "a", "b::c", "eval")
    assert key_a != key_b


def test_cache_key_injective_org_vs_session_boundary() -> None:
    """The org/session boundary must not be forgeable either."""
    key_a = _grade_on_demand_cache_key("o::x", "s", "v", "e")
    key_b = _grade_on_demand_cache_key("o", "x::s", "v", "e")
    assert key_a != key_b


def test_cache_key_injective_version_vs_eval_boundary() -> None:
    key_a = _grade_on_demand_cache_key("o", "s", "v::x", "e")
    key_b = _grade_on_demand_cache_key("o", "s", "v", "x::e")
    assert key_a != key_b


def test_cache_key_keeps_prefix_for_filtering() -> None:
    key = _grade_on_demand_cache_key("org", "sess", "v1", "eval")
    assert key.startswith(f"{_GRADE_ON_DEMAND_CACHE_KEY_PREFIX}::")


def test_cache_key_stable_for_identical_inputs() -> None:
    assert _grade_on_demand_cache_key("o", "s", "v", "e") == (
        _grade_on_demand_cache_key("o", "s", "v", "e")
    )
