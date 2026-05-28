"""Unit tests for the pure F1 outcome helpers."""

import random
from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.services.shadow_comparison.outcome import (
    Outcome,
    assign_positions,
    derive_reflexio_outcome,
)

# --- derive_reflexio_outcome truth table ---------------------------------


@pytest.mark.parametrize(
    "better, reflexio_is_request_1, expected",
    [
        ("1", True, Outcome.WIN),
        ("2", True, Outcome.LOSS),
        ("1", False, Outcome.LOSS),
        ("2", False, Outcome.WIN),
        ("tie", True, Outcome.TIE),
        ("tie", False, Outcome.TIE),
    ],
)
def test_derive_reflexio_outcome(better, reflexio_is_request_1, expected):
    v = ShadowComparisonVerdict(
        verdict_id=1,
        interaction_id="i",
        session_id="s",
        agent_version="v1",
        reflexio_is_request_1=reflexio_is_request_1,
        output=ShadowComparisonOutput(
            better_request=better,
            is_significantly_better=False,
        ),
        judge_prompt_version="v1.0.0",
        created_at=datetime.now(UTC),
    )
    assert derive_reflexio_outcome(v) == expected


# --- assign_positions ---------------------------------------------------


def test_assign_positions_returns_both_responses():
    """Both responses must be returned exactly as passed; only order varies."""
    request_1, request_2, _ = assign_positions(
        reflexio_response="REFLEX",
        shadow_response="SHADOW",
        rng=random.Random(0),  # noqa: S311 — position randomization, not crypto
    )
    pair = {request_1, request_2}
    assert pair == {"REFLEX", "SHADOW"}


def test_assign_positions_records_assignment_correctly():
    """The bool returned must accurately describe the assignment."""
    request_1, request_2, is_request_1 = assign_positions(
        reflexio_response="REFLEX",
        shadow_response="SHADOW",
        rng=random.Random(0),  # noqa: S311 — position randomization, not crypto
    )
    if is_request_1:
        assert request_1 == "REFLEX"
        assert request_2 == "SHADOW"
    else:
        assert request_1 == "SHADOW"
        assert request_2 == "REFLEX"


def test_assign_positions_is_roughly_balanced_over_n_calls():
    """Over many seeded calls, mean(reflexio_is_request_1) is roughly 0.5."""
    rng = random.Random(0)  # noqa: S311 — position randomization, not crypto
    count_request_1 = 0
    n = 2_000
    for _ in range(n):
        _, _, is_r1 = assign_positions("REFLEX", "SHADOW", rng=rng)
        if is_r1:
            count_request_1 += 1
    p = count_request_1 / n
    assert 0.45 < p < 0.55, f"position bias: {p}"


def test_assign_positions_is_reproducible_with_same_seed():
    rng1 = random.Random(42)  # noqa: S311 — position randomization, not crypto
    rng2 = random.Random(42)  # noqa: S311 — position randomization, not crypto
    a = [assign_positions("R", "S", rng=rng1)[2] for _ in range(50)]
    b = [assign_positions("R", "S", rng=rng2)[2] for _ in range(50)]
    assert a == b
