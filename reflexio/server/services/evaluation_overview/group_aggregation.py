"""Pure aggregation helpers for F2's session-level A/B groups.

These functions are deliberately DB-free and LLM-free so they can be unit
tested in milliseconds and exercised by mutmut. The storage layer hands
in raw tuples; we hand back SuccessRateTrendByGroup.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from reflexio.models.api_schema.eval_overview_schema import (
    SuccessRateTrendByGroup,
    TrendPoint,
)

# Z-value for a 95% Wald CI on a binomial proportion (two-sided).
_Z_95 = 1.96

# Cap on the lift CI half-width. With very small sample sizes the Wald CI
# computes to absurd widths (e.g., ±70pp), and rendering that on a
# dashboard implies false precision. 0.5 (= ±50pp) is the product-meaningful
# threshold: any wider than that, the user should think "we don't know"
# and the frontend should consider rendering a low-confidence indicator.
_MAX_CI_HALF_WIDTH = 0.5


class GroupAssignment(StrEnum):
    """Group label derived from `Request.metadata.reflexio_retrieval_enabled`."""

    TREATMENT = "treatment"
    CONTROL = "control"
    UNTAGGED = "untagged"


def assign_group_from_metadata(metadata: dict[str, Any]) -> GroupAssignment:
    """
    Bucket a session by its first request's metadata.

    Reads ``metadata.reflexio_retrieval_enabled`` as a true bool. Anything
    else (missing, string, int, None) maps to UNTAGGED — we never silently
    coerce non-bool values, so customers see how many of their sessions
    are tagged inconsistently.

    Args:
        metadata (dict[str, Any]): The first request's metadata dict. Non-dict
            input is tolerated and routed to UNTAGGED (defensive — the caller
            may pass a value read from a JSON column with no type guarantee).

    Returns:
        GroupAssignment: TREATMENT, CONTROL, or UNTAGGED.
    """
    if not isinstance(metadata, dict):
        return GroupAssignment.UNTAGGED
    value = metadata.get("reflexio_retrieval_enabled")
    if value is True:
        return GroupAssignment.TREATMENT
    if value is False:
        return GroupAssignment.CONTROL
    return GroupAssignment.UNTAGGED


def group_session_outcomes_by_metadata(
    outcomes: Iterable[tuple[str, int, bool, dict[str, Any]]],
) -> dict[GroupAssignment, list[tuple[str, int, bool]]]:
    """
    Split (session_id, ts, is_success, metadata) tuples into three groups.

    Args:
        outcomes (Iterable[tuple[str, int, bool, dict[str, Any]]]): Tuples of
            (session_id, ts, is_success, first_request_metadata).

    Returns:
        dict[GroupAssignment, list[tuple[str, int, bool]]]: Dict keyed by every
        GroupAssignment member (TREATMENT, CONTROL, UNTAGGED) — always present,
        possibly empty list — with values being (session_id, ts, is_success)
        tuples (metadata dropped after binning).
    """
    out: dict[GroupAssignment, list[tuple[str, int, bool]]] = {
        GroupAssignment.TREATMENT: [],
        GroupAssignment.CONTROL: [],
        GroupAssignment.UNTAGGED: [],
    }
    for session_id, ts, is_success, metadata in outcomes:
        group = assign_group_from_metadata(metadata)
        out[group].append((session_id, ts, is_success))
    return out


def bucket_outcomes(
    outcomes: Iterable[tuple[str, int, bool]],
    bucket_seconds: int,
) -> list[dict[str, int]]:
    """
    Bucket outcomes by time, returning one dict per bucket.

    Args:
        outcomes (Iterable[tuple[str, int, bool]]): Tuples of (session_id, ts,
            is_success). ``session_id`` is unused at this stage; kept for future
            per-session breakdowns.
        bucket_seconds (int): Bucket width in seconds (e.g., 86400 for daily,
            604800 for weekly). Must be positive.

    Returns:
        list[dict[str, int]]: List of ``{"ts": bucket_start, "n": int,
        "successes": int}``, sorted by ``ts`` ascending. Empty buckets are
        omitted.

    Raises:
        ValueError: When ``bucket_seconds`` is not positive.
    """
    if bucket_seconds <= 0:
        raise ValueError(f"bucket_seconds must be positive, got {bucket_seconds}")
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "successes": 0})
    for _, ts, is_success in outcomes:
        bucket_start = (ts // bucket_seconds) * bucket_seconds
        counts[bucket_start]["n"] += 1
        if is_success:
            counts[bucket_start]["successes"] += 1
    return [
        {"ts": bucket_start, "n": d["n"], "successes": d["successes"]}
        for bucket_start, d in sorted(counts.items())
    ]


def compute_trend_by_group(
    outcomes: Iterable[tuple[str, int, bool, dict[str, Any]]],
    bucket_seconds: int,
) -> SuccessRateTrendByGroup:
    """
    Build the dual+untagged-curve trend payload from raw session outcomes.

    Args:
        outcomes (Iterable[tuple[str, int, bool, dict[str, Any]]]): Tuples of
            (session_id, ts, is_success, first_request_metadata).
        bucket_seconds (int): Bucket width passed to :func:`bucket_outcomes`.

    Returns:
        SuccessRateTrendByGroup: Three curves (any of which may be empty).
    """
    grouped = group_session_outcomes_by_metadata(outcomes)
    return SuccessRateTrendByGroup(
        treatment=_to_trend_points(grouped[GroupAssignment.TREATMENT], bucket_seconds),
        control=_to_trend_points(grouped[GroupAssignment.CONTROL], bucket_seconds),
        untagged=_to_trend_points(grouped[GroupAssignment.UNTAGGED], bucket_seconds),
    )


def _to_trend_points(
    outcomes: list[tuple[str, int, bool]],
    bucket_seconds: int,
) -> list[TrendPoint]:
    """Convert raw outcomes into bucketed TrendPoints; omit empty buckets."""
    return [
        TrendPoint(ts=b["ts"], rate=b["successes"] / b["n"], n=b["n"])
        for b in bucket_outcomes(outcomes, bucket_seconds)
        if b["n"] > 0
    ]


def compute_lift_with_ci(
    n_t: int, p_t: float, n_c: int, p_c: float
) -> tuple[float | None, float | None]:
    """
    Compute success-rate lift (treatment − control) and a 95% CI half-width.

    Uses the Wald approximation for the difference of two binomial
    proportions. Returns ``(None, None)`` if either group has zero samples —
    surfaces honest "no estimate" rather than fake zeros.

    Args:
        n_t (int): Treatment group size.
        p_t (float): Treatment success rate, in ``[0.0, 1.0]``.
        n_c (int): Control group size.
        p_c (float): Control success rate, in ``[0.0, 1.0]``.

    Returns:
        tuple[float | None, float | None]: ``(lift, ci_half_width)`` — both in
        proportion units (e.g., 0.14 for 14pp), or ``(None, None)`` when there
        isn't enough data.
    """
    if n_t <= 0 or n_c <= 0:
        return None, None
    lift = p_t - p_c
    var = (p_t * (1 - p_t)) / n_t + (p_c * (1 - p_c)) / n_c
    ci_half = _Z_95 * math.sqrt(var)
    ci_half = min(ci_half, _MAX_CI_HALF_WIDTH)
    return lift, ci_half
