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
    shadow_is_success: bool | None = None,
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        result_id=result_id,
        agent_version="v_e2e",
        session_id=session_id,
        is_success=is_success,
        evaluation_name="overall",
        created_at=created_at,
        number_of_correction_per_session=corrections,
        shadow_is_success=shadow_is_success,
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


def test_service_hero_shadow_rate_non_null_when_shadow_graded() -> None:
    """When results include shadow_is_success grades, hero exposes non-null shadow data."""
    storage = MagicMock()
    storage.get_agent_success_evaluation_results.return_value = [
        # 2 shadow-graded successes, 1 shadow-graded failure, 1 ungraded
        _eval_result(
            result_id=1, session_id="s1", is_success=True, shadow_is_success=True
        ),
        _eval_result(
            result_id=2, session_id="s2", is_success=True, shadow_is_success=True
        ),
        _eval_result(
            result_id=3, session_id="s3", is_success=False, shadow_is_success=False
        ),
        _eval_result(
            result_id=4, session_id="s4", is_success=True, shadow_is_success=None
        ),
    ]
    storage.get_playbook_application_stats.return_value = []
    storage.get_interactions_by_session.return_value = []
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=True)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    # 2/3 shadow-graded are successes → 66.67%
    assert response.hero.shadow_success_rate_pp is not None
    assert abs(response.hero.shadow_success_rate_pp - 200 / 3) < 0.01
    assert response.hero.delta_pp is not None
    # regular is 3/4 → 75%; shadow is 66.67%; delta ≈ −8.33
    assert abs(response.hero.delta_pp - (200 / 3 - 75.0)) < 0.01


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
