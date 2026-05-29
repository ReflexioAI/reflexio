"""Schema sanity for /api/get_evaluation_overview request/response models."""

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
)


def test_request_defaults_bucket_to_week_and_include_shadow_to_true() -> None:
    req = GetEvaluationOverviewRequest(from_ts=0, to_ts=10)
    assert req.bucket == "week"
    assert req.include_shadow is True


def test_request_rejects_inverted_time_window() -> None:
    with pytest.raises(ValidationError, match="from_ts must be <= to_ts"):
        GetEvaluationOverviewRequest(from_ts=10, to_ts=1)


def test_response_round_trips_full_payload() -> None:
    payload = {
        "hero": {
            "state": "full",
            "regular_success_rate_pp": 87.4,
            "shadow_success_rate_pp": 73.2,
            "delta_pp": 14.2,
            "buckets": [
                {
                    "ts": 1700000000,
                    "regular_rate": 0.85,
                    "shadow_rate": 0.70,
                    "regular_n": 100,
                    "shadow_n": 80,
                },
            ],
        },
        "context_tiles": {
            "success": {"current": 87.4, "delta_pp": 6.2},
            "corrections": {"current": 0.4, "delta": -0.7},
            "turns": {"current": 3.2, "delta": -1.4},
            "escalation": {"current": 4.1, "delta_pp": -6.9},
        },
        "rule_attribution": [
            {
                "rule_id": "rule_42",
                "kind": "playbook",
                "title": "Confirm address",
                "successes_with": 42,
                "failures_with": 4,
                "net_sessions": 38,
            },
        ],
        "score_distribution": {
            "current_bins": [50, 20, 15, 10, 3, 2],
            "baseline_bins": [30, 25, 20, 15, 5, 5],
            "labels": ["0", "1", "2", "3", "4", "5+"],
        },
    }
    resp = GetEvaluationOverviewResponse(**payload)
    assert resp.hero.state == "full"
    assert resp.context_tiles.success.current == 87.4
    assert resp.rule_attribution[0].net_sessions == 38
    assert resp.score_distribution.labels == ["0", "1", "2", "3", "4", "5+"]
