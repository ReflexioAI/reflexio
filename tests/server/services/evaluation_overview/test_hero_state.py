"""Unit tests for hero state derivation (the 4 states from spec §3.2)."""

from reflexio.server.services.evaluation_overview.components.hero_state import (
    HeroState,
    compute_hero_state,
)


def test_empty_when_no_evaluated_sessions_ever() -> None:
    """State 4: no results ever → empty."""
    state = compute_hero_state(
        shadow_enabled=False,
        days_since_first_eval=None,
        n_shadow_in_window=0,
        total_results=0,
    )
    assert state == HeroState.EMPTY


def test_shadow_off_when_disabled_and_some_data() -> None:
    """State 3: shadow off, >=7 days of trend → shadow_off."""
    state = compute_hero_state(
        shadow_enabled=False,
        days_since_first_eval=10,
        n_shadow_in_window=0,
        total_results=42,
    )
    assert state == HeroState.SHADOW_OFF


def test_early_when_shadow_just_enabled() -> None:
    """State 2: shadow on but <14 days since enabled → early."""
    state = compute_hero_state(
        shadow_enabled=True,
        days_since_first_eval=3,
        n_shadow_in_window=120,
        total_results=120,
    )
    assert state == HeroState.EARLY


def test_early_when_shadow_volume_low() -> None:
    """State 2: shadow on, 30 days, but n_shadow < 500 → early."""
    state = compute_hero_state(
        shadow_enabled=True,
        days_since_first_eval=30,
        n_shadow_in_window=300,
        total_results=300,
    )
    assert state == HeroState.EARLY


def test_full_when_all_thresholds_met() -> None:
    """State 1: shadow on, >=14 days, >=500 shadow sessions → full."""
    state = compute_hero_state(
        shadow_enabled=True,
        days_since_first_eval=21,
        n_shadow_in_window=800,
        total_results=800,
    )
    assert state == HeroState.FULL
