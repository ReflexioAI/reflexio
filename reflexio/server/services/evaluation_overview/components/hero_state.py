"""Pure functions deriving the /evaluations hero block state and delta.

These have no IO; they're invoked from EvaluationOverviewService with already-
loaded aggregate numbers. Separated so the spec's 4-state machine can be
unit-tested cheaply.
"""

from __future__ import annotations

from enum import StrEnum

_FULL_MIN_DAYS = 14
_FULL_MIN_SHADOW_N = 500


class HeroState(StrEnum):
    """The four mutually exclusive hero block configurations from spec §3.2.

    Members serialize to the snake_case strings the frontend switches on.
    """

    FULL = "full"
    EARLY = "early"
    SHADOW_OFF = "shadow_off"
    EMPTY = "empty"


def compute_hero_state(
    *,
    shadow_enabled: bool,
    days_since_first_eval: int | None,
    n_shadow_in_window: int,
    total_results: int,
) -> HeroState:
    """Return the hero state for the current org and time window.

    Args:
        shadow_enabled (bool): Value of `Config.shadow_mode_enabled`.
        days_since_first_eval (int | None): Wall-clock days since the first
            ever evaluated session for this org. None when no results exist.
        n_shadow_in_window (int): Count of evaluation results that have a
            shadow grade inside the trend window. Caller is responsible for
            filtering; this function only consumes the count.
            Why graded-only: prevents the FULL gate from tripping mid-window-
            flag-flip when sessions exist with shadow_content but no shadow
            grade. (Note: the direct-grade `shadow_is_success` /
            `shadow_is_escalated` columns added briefly in May 2026 were
            retracted before any production deploy — see the deleted
            migration pair in supabase/data/supabase/migrations/. The
            per-turn shadow grade in F1 lives on a different surface.)
        total_results (int): Total AgentSuccessEvaluationResult rows in the
            trend window, used to distinguish EMPTY from states with data.

    Returns:
        HeroState: The single applicable state. Empty wins over everything;
            after that the order is shadow_off → early → full per spec.
    """
    if total_results == 0:
        return HeroState.EMPTY
    if not shadow_enabled:
        return HeroState.SHADOW_OFF
    if days_since_first_eval is None or days_since_first_eval < _FULL_MIN_DAYS:
        return HeroState.EARLY
    if n_shadow_in_window < _FULL_MIN_SHADOW_N:
        return HeroState.EARLY
    return HeroState.FULL
