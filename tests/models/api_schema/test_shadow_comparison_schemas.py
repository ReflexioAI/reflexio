"""Verify F1 schemas for per-turn shadow comparison."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)


def test_shadow_comparison_output_minimal():
    o = ShadowComparisonOutput(
        better_request="1",
        is_significantly_better=True,
    )
    assert o.better_request == "1"
    assert o.is_significantly_better is True
    assert o.comparison_reason is None


def test_shadow_comparison_output_accepts_all_three_better_request_values():
    for value in ("1", "2", "tie"):
        o = ShadowComparisonOutput(
            better_request=value,
            is_significantly_better=False,
        )
        assert o.better_request == value


def test_shadow_comparison_output_rejects_invalid_better_request():
    with pytest.raises(ValidationError):
        ShadowComparisonOutput(
            better_request="3",  # type: ignore[arg-type]
            is_significantly_better=False,
        )


def test_shadow_comparison_verdict_round_trip():
    v = ShadowComparisonVerdict(
        verdict_id=42,
        interaction_id="int-1",
        session_id="sess-1",
        agent_version="v1",
        reflexio_is_request_1=True,
        output=ShadowComparisonOutput(
            better_request="1",
            is_significantly_better=True,
            comparison_reason="Request 1 directly addressed the question",
        ),
        judge_prompt_version="v1.0.0",
        created_at=datetime.now(UTC),
    )
    re_parsed = ShadowComparisonVerdict(**v.model_dump())
    assert re_parsed.verdict_id == 42
    assert re_parsed.reflexio_is_request_1 is True
    assert re_parsed.output.better_request == "1"


def test_shadow_comparison_verdict_requires_judge_prompt_version():
    """The pinned prompt version must be set — it's how the dashboard
    filters out verdicts produced under a prior rubric epoch."""
    with pytest.raises(ValidationError):
        ShadowComparisonVerdict(
            verdict_id=1,
            interaction_id="i",
            session_id="s",
            agent_version="v1",
            reflexio_is_request_1=True,
            output=ShadowComparisonOutput(
                better_request="tie",
                is_significantly_better=False,
            ),
            created_at=datetime.now(UTC),
            judge_prompt_version="",  # empty string rejected by NonEmptyStr
        )


def test_shadow_comparison_verdict_rejects_whitespace_judge_prompt_version():
    """NonEmptyStr rejects whitespace-only strings; Field(min_length=1) wouldn't."""
    with pytest.raises(ValidationError):
        ShadowComparisonVerdict(
            verdict_id=1,
            interaction_id="i",
            session_id="s",
            agent_version="v1",
            reflexio_is_request_1=True,
            output=ShadowComparisonOutput(
                better_request="tie",
                is_significantly_better=False,
            ),
            judge_prompt_version="   ",  # whitespace-only
            created_at=datetime.now(UTC),
        )


def test_shadow_comparison_output_extras_runtime_policy():
    """Runtime is lenient — extras pass through without raising.

    The JSON schema sent to the LLM advertises strictness; this asymmetry
    is intentional (server-side resilience + model-side constraint).
    """
    o = ShadowComparisonOutput(
        better_request="1",
        is_significantly_better=True,
        unexpected_field="surprise",  # type: ignore[call-arg]
    )
    # No raise; the field is accepted via extra='allow'.
    assert o.better_request == "1"


def test_shadow_comparison_output_json_schema_forbids_extras():
    """The JSON schema (what we send to the LLM) advertises strictness."""
    schema = ShadowComparisonOutput.model_json_schema()
    assert schema.get("additionalProperties") is False
