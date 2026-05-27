"""Tests for the Braintrust-tile aggregation in EvaluationOverviewService.

Plan C-overview: when imported_score rows exist, the response now carries a
`braintrust_tiles` list — one row per scorer with current mean + count +
delta vs prior window.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from reflexio.models.api_schema.braintrust_schema import ImportedScore
from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
    _aggregate_imported_scores,
)


def _score(
    *,
    scorer_name: str,
    value: float,
    ts: int,
    session_id: str | None = None,
) -> ImportedScore:
    return ImportedScore(
        org_id="org_t",
        source_run_id=f"span_{scorer_name}_{ts}",
        session_id=session_id,
        scorer_name=scorer_name,
        value=value,
        ts=ts,
    )


def test_aggregate_groups_by_scorer_name() -> None:
    """`_aggregate_imported_scores` returns {scorer_name: (mean, count)}."""
    scores = [
        _score(scorer_name="hallucination", value=0.1, ts=100),
        _score(scorer_name="hallucination", value=0.3, ts=200),
        _score(scorer_name="factuality", value=0.9, ts=150),
    ]
    out = _aggregate_imported_scores(scores)
    assert out["hallucination"] == (0.2, 2)
    assert out["factuality"] == (0.9, 1)


def test_service_returns_empty_braintrust_tiles_with_no_imported_scores() -> None:
    """Default no-op storage returns []; tiles are empty."""
    storage = MagicMock()
    storage.org_id = "org_t"
    storage.get_agent_success_evaluation_results.return_value = []
    storage.get_imported_scores.return_value = []
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=False)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.braintrust_tiles == []


def test_service_emits_braintrust_tiles_with_delta() -> None:
    """When imported_score rows exist, tiles report current mean, n, and delta."""
    storage = MagicMock()
    storage.org_id = "org_t"
    storage.get_agent_success_evaluation_results.return_value = []

    # Storage returns different score sets per (from_ts, to_ts) window
    def get_imported_scores_router(
        _org_id: str, from_ts: int, _to_ts: int
    ) -> list[ImportedScore]:
        if from_ts >= 1_000_000:  # current window
            return [
                _score(scorer_name="hallucination", value=0.1, ts=1_000_010),
                _score(scorer_name="hallucination", value=0.3, ts=1_000_020),
                _score(scorer_name="factuality", value=0.9, ts=1_000_030),
            ]
        # prior window
        return [
            _score(scorer_name="hallucination", value=0.5, ts=900_010),
        ]

    storage.get_imported_scores.side_effect = get_imported_scores_router
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=False)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=1_000_000, to_ts=2_000_000))

    tiles_by_name = {t.scorer_name: t for t in response.braintrust_tiles}
    assert tiles_by_name["hallucination"].current == 0.2
    assert tiles_by_name["hallucination"].n == 2
    # prior mean was 0.5 → delta = 0.2 - 0.5 = -0.3
    assert abs(tiles_by_name["hallucination"].delta - (-0.3)) < 1e-9
    # factuality has no prior data → delta equals current (signals "no baseline")
    assert tiles_by_name["factuality"].current == 0.9
    assert tiles_by_name["factuality"].delta == 0.9
