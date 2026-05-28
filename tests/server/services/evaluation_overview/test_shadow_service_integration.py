"""Service-level wire-up: ``service.run()`` populates shadow_win_rate_trend.

These tests exercise the full ``EvaluationOverviewService.run`` path with a
real :class:`SQLiteStorage` so the integration between storage filtering,
the aggregator, and the response payload is covered end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(tmp_path, worker_id):
    """Per-worker isolated SQLite store."""
    db_path = tmp_path / f"f1_shadow_{worker_id}.db"
    return SQLiteStorage(org_id=worker_id, db_path=str(db_path))


def _verdict(
    *,
    interaction_id: str,
    session_id: str,
    better: str,
    reflexio_is_r1: bool,
    ts: int,
    judge_prompt_version: str = "v1.0.0",
) -> ShadowComparisonVerdict:
    return ShadowComparisonVerdict(
        verdict_id=0,
        interaction_id=interaction_id,
        session_id=session_id,
        agent_version="v1",
        reflexio_is_request_1=reflexio_is_r1,
        output=ShadowComparisonOutput(
            better_request=better,  # type: ignore[arg-type]
            is_significantly_better=True,
        ),
        judge_prompt_version=judge_prompt_version,
        created_at=datetime.fromtimestamp(ts, tz=UTC),
    )


def test_service_returns_populated_shadow_win_rate_trend(storage) -> None:
    base_ts = 1_700_000_000
    # i1: judge said "1", reflexio was request 1 → WIN
    storage.save_shadow_comparison_verdict(
        _verdict(
            interaction_id="i1",
            session_id="s1",
            better="1",
            reflexio_is_r1=True,
            ts=base_ts,
        )
    )
    # i2: judge said "1", reflexio was request 2 → LOSS
    storage.save_shadow_comparison_verdict(
        _verdict(
            interaction_id="i2",
            session_id="s2",
            better="1",
            reflexio_is_r1=False,
            ts=base_ts,
        )
    )

    service = EvaluationOverviewService(
        storage=storage,
        config=Config(storage_config=StorageConfigSQLite()),
    )
    response = service.run(
        GetEvaluationOverviewRequest(from_ts=base_ts - 1, to_ts=base_ts + 1)
    )

    trend = response.shadow_win_rate_trend
    assert trend.window_total.n == 2
    assert trend.window_total.wins == 1
    assert trend.window_total.losses == 1
    assert trend.window_total.ties == 0
    assert trend.window_total.win_rate == 0.5
    assert trend.window_total.net_win == 0.0
    assert trend.judge_prompt_version == "v1.0.0"
    assert len(trend.daily) == 1
    assert trend.daily[0].n == 2


def test_service_empty_window_returns_empty_trend(storage) -> None:
    service = EvaluationOverviewService(
        storage=storage,
        config=Config(storage_config=StorageConfigSQLite()),
    )
    response = service.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=1))
    trend = response.shadow_win_rate_trend
    assert trend.daily == []
    assert trend.window_total.n == 0
    assert trend.judge_prompt_version == "v1.0.0"


def test_service_filters_out_stale_prompt_versions(storage) -> None:
    """A verdict graded under an older rubric must not contaminate the headline."""
    base_ts = 1_700_000_000
    storage.save_shadow_comparison_verdict(
        _verdict(
            interaction_id="i_current",
            session_id="s_current",
            better="1",
            reflexio_is_r1=True,
            ts=base_ts,
            judge_prompt_version="v1.0.0",
        )
    )
    storage.save_shadow_comparison_verdict(
        _verdict(
            interaction_id="i_stale",
            session_id="s_stale",
            better="2",
            reflexio_is_r1=True,
            ts=base_ts,
            judge_prompt_version="v0.9.0",
        )
    )
    service = EvaluationOverviewService(
        storage=storage,
        config=Config(storage_config=StorageConfigSQLite()),
    )
    response = service.run(
        GetEvaluationOverviewRequest(from_ts=base_ts - 1, to_ts=base_ts + 1)
    )
    trend = response.shadow_win_rate_trend
    # Only the v1.0.0 verdict should show up.
    assert trend.window_total.n == 1
    assert trend.window_total.wins == 1
    assert trend.window_total.losses == 0
    assert trend.judge_prompt_version == "v1.0.0"
