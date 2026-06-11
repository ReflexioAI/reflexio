"""Integration test for EvaluationOverviewService with a mocked storage."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from reflexio.models.api_schema.domain.entities import AgentSuccessEvaluationResult
from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
)


def _eval_result(
    *,
    result_id: int,
    session_id: str,
    is_success: bool,
    corrections: int = 0,
    created_at: int = 1700000000,
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        result_id=result_id,
        agent_version="v_e2e",
        session_id=session_id,
        is_success=is_success,
        evaluation_name="overall",
        created_at=created_at,
        number_of_correction_per_session=corrections,
    )


def test_service_returns_full_response_with_shadow_enabled_and_data() -> None:
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        _eval_result(result_id=1, session_id="s1", is_success=True, corrections=0),
        _eval_result(result_id=2, session_id="s2", is_success=True, corrections=1),
        _eval_result(result_id=3, session_id="s3", is_success=False, corrections=3),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=True)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.hero.state in ("full", "early", "shadow_off", "empty")
    assert response.context_tiles.success.current >= 0.0
    assert len(response.score_distribution.current_bins) == 6
    assert response.score_distribution.labels == ["0", "1", "2", "3", "4", "5+"]


def test_service_hero_shadow_fields_always_null_after_direct_grade_removal() -> None:
    """Direct shadow grading was removed; hero's shadow_success_rate_pp / delta_pp
    must always be None until a methodologically sound replacement ships.

    See docs/superpowers/specs/2026-05-27-shadow-mode-validity-and-alternatives.md
    for the rationale (multi-turn trajectory contamination + applicability gap).
    """
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        _eval_result(result_id=1, session_id="s1", is_success=True),
        _eval_result(result_id=2, session_id="s2", is_success=False),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.hero.shadow_success_rate_pp is None
    assert response.hero.delta_pp is None
    for bucket in response.hero.buckets:
        assert bucket.shadow_rate is None
        assert bucket.shadow_n == 0


def test_service_returns_empty_state_when_no_results() -> None:
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = []
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=False)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.hero.state == "empty"
    assert response.context_tiles.success.current == 0.0
    assert response.rule_attribution == []


def test_service_reports_single_recent_success_as_100_percent() -> None:
    now = int(time.time())
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        _eval_result(
            result_id=1,
            session_id="recent-success",
            is_success=True,
            created_at=now - 60,
        ),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    storage.get_imported_scores.return_value = []
    storage.get_sessions.return_value = {}
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(from_ts=now - 3600, to_ts=now, bucket="day")
    )

    assert response.hero.regular_success_rate_pp == 100.0
    assert response.context_tiles.success.current == 100.0


def test_service_uses_first_ever_eval_for_hero_age() -> None:
    """A narrow window should not make a mature org look newly onboarded."""
    now = int(time.time())
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        _eval_result(
            result_id=1,
            session_id="old",
            is_success=True,
            created_at=now - 10 * 24 * 60 * 60,
        ),
        _eval_result(
            result_id=2, session_id="current", is_success=True, created_at=now
        ),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    storage.get_imported_scores.return_value = []
    storage.get_sessions.return_value = {}
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=now - 60, to_ts=now))

    assert response.hero.state == "shadow_off"


def test_service_honors_day_bucket_for_hero_trend() -> None:
    day = 24 * 60 * 60
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        _eval_result(result_id=1, session_id="s1", is_success=True, created_at=day),
        _eval_result(
            result_id=2,
            session_id="s2",
            is_success=False,
            created_at=day + 3600,
        ),
        _eval_result(
            result_id=3,
            session_id="s3",
            is_success=True,
            created_at=2 * day,
        ),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    storage.get_imported_scores.return_value = []
    storage.get_sessions.return_value = {}
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(from_ts=0, to_ts=3 * day, bucket="day")
    )

    assert [bucket.ts for bucket in response.hero.buckets] == [day, 2 * day]
    assert [bucket.regular_n for bucket in response.hero.buckets] == [2, 1]
