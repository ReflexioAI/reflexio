"""Tests for evaluation overview source-set schema behavior."""

from reflexio.models.api_schema.eval_overview_schema import (
    ContextTile,
    EvaluationSourceSetRequest,
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    HeroBlock,
    NumberWithDelta,
    PercentWithDelta,
    ScoreDistribution,
    SourceSetComparison,
    SourceSetEvaluationMetrics,
)


def _minimal_response_kwargs() -> dict:
    """Build minimal valid response payload kwargs."""
    return {
        "hero": HeroBlock(
            state="empty",
            regular_success_rate_pp=0.0,
            shadow_success_rate_pp=None,
            delta_pp=None,
            buckets=[],
        ),
        "context_tiles": ContextTile(
            success=PercentWithDelta(current=0.0, delta_pp=0.0),
            corrections=NumberWithDelta(current=0.0, delta=0.0),
            turns=NumberWithDelta(current=0.0, delta=0.0),
            escalation=PercentWithDelta(current=0.0, delta_pp=0.0),
        ),
        "rule_attribution": [],
        "score_distribution": ScoreDistribution(
            current_bins=[], baseline_bins=[], labels=[]
        ),
    }


def test_source_set_request_accepts_empty_source_value():
    request = GetEvaluationOverviewRequest(
        from_ts=0,
        to_ts=1,
        source_sets=[EvaluationSourceSetRequest(label="empty", sources=[""])],
    )
    assert request.source_sets[0].sources == [""]


def test_source_set_request_rejects_duplicate_labels():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GetEvaluationOverviewRequest(
            from_ts=0,
            to_ts=1,
            source_sets=[
                EvaluationSourceSetRequest(label="same", sources=["a"]),
                EvaluationSourceSetRequest(label="same", sources=["b"]),
            ],
        )


def test_source_set_request_rejects_empty_sources():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvaluationSourceSetRequest(label="empty", sources=[])


def test_source_set_request_rejects_overlapping_sources():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GetEvaluationOverviewRequest(
            from_ts=0,
            to_ts=1,
            source_sets=[
                EvaluationSourceSetRequest(label="a", sources=["shared"]),
                EvaluationSourceSetRequest(label="b", sources=["shared"]),
            ],
        )


def test_overview_response_defaults_to_empty_source_set_comparison():
    r = GetEvaluationOverviewResponse(**_minimal_response_kwargs())
    assert r.source_set_comparison.available_sources == []
    assert r.source_set_comparison.sets == []
    assert r.source_set_comparison.unmatched_session_count == 0


def test_overview_response_carries_source_set_comparison():
    payload = _minimal_response_kwargs()
    payload["source_set_comparison"] = SourceSetComparison(
        available_sources=["a", "b"],
        sets=[
            SourceSetEvaluationMetrics(
                label="A",
                sources=["a"],
                session_count=1,
                session_ids=["s1"],
                success_rate_pp=100.0,
                buckets=[],
                context_tiles=payload["context_tiles"],
                score_distribution=payload["score_distribution"],
                rule_attribution=[],
            )
        ],
        unmatched_session_count=2,
    )
    r = GetEvaluationOverviewResponse(**payload)
    assert r.source_set_comparison.available_sources == ["a", "b"]
    assert r.source_set_comparison.sets[0].session_ids == ["s1"]
    assert r.source_set_comparison.unmatched_session_count == 2
