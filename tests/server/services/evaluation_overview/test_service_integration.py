"""Integration test for EvaluationOverviewService with a mocked storage."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import AgentSuccessEvaluationResult
from reflexio.models.api_schema.eval_overview_schema import (
    EvaluationSourceSetRequest,
    GetEvaluationOverviewRequest,
)
from reflexio.models.api_schema.internal_schema import (
    SessionCitation,
    SessionFirstRequest,
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
    user_id: str = "u1",
    corrections: int = 0,
    failure_type: str | None = None,
    created_at: int = 1700000000,
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        result_id=result_id,
        user_id=user_id,
        agent_version="v_e2e",
        session_id=session_id,
        is_success=is_success,
        failure_type=failure_type,
        evaluation_name="overall",
        created_at=created_at,
        number_of_correction_per_session=corrections,
    )


def _storage_with_results(
    results: list[AgentSuccessEvaluationResult],
) -> MagicMock:
    storage = MagicMock()
    storage.org_id = ""
    storage.get_agent_success_evaluation_results_in_window.return_value = results
    storage.get_first_requests_by_user_session_pairs.return_value = {
        (r.user_id, r.session_id): SessionFirstRequest(
            session_id=r.session_id,
            user_id=r.user_id,
            source="api",
            created_at=r.created_at,
        )
        for r in results
    }
    storage.get_citations_by_session_ids.return_value = []
    storage.get_imported_scores.return_value = []
    storage.get_shadow_comparison_verdicts.return_value = []
    return storage


def test_service_returns_full_response_with_shadow_enabled_and_data() -> None:
    storage = _storage_with_results(
        [
            _eval_result(result_id=1, session_id="s1", is_success=True, corrections=0),
            _eval_result(result_id=2, session_id="s2", is_success=True, corrections=1),
            _eval_result(result_id=3, session_id="s3", is_success=False, corrections=3),
        ]
    )
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
    storage = _storage_with_results(
        [
            _eval_result(result_id=1, session_id="s1", is_success=True),
            _eval_result(result_id=2, session_id="s2", is_success=False),
        ]
    )
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.hero.shadow_success_rate_pp is None
    assert response.hero.delta_pp is None
    for bucket in response.hero.buckets:
        assert bucket.shadow_rate is None
        assert bucket.shadow_n == 0


def test_service_returns_empty_state_when_no_results() -> None:
    storage = _storage_with_results([])
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=False)

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.hero.state == "empty"
    assert response.context_tiles.success.current == 0.0
    assert response.rule_attribution == []
    assert response.recent_results == []


def test_service_returns_newest_100_embedding_free_results() -> None:
    results = [
        _eval_result(
            result_id=index,
            session_id=f"s{index}",
            is_success=index % 2 == 0,
            created_at=1_700_000_000 + index,
        )
        for index in range(1, 106)
    ]
    for result in results:
        result.embedding = [0.25] * 512
    storage = _storage_with_results(list(reversed(results[::2])) + results[1::2])
    config = Config(storage_config=StorageConfigSQLite())

    response = EvaluationOverviewService(storage=storage, config=config).run(
        GetEvaluationOverviewRequest(
            from_ts=1_700_000_000,
            to_ts=1_700_000_200,
            include_shadow=False,
        )
    )

    assert [row.result_id for row in response.recent_results] == list(range(105, 5, -1))
    assert all("embedding" not in row.model_dump() for row in response.recent_results)
    storage.get_agent_success_evaluation_results_in_window.assert_called_once_with(
        from_ts=1_699_395_200,
        to_ts=1_700_000_200,
        agent_version=None,
        include_embedding=False,
    )


def test_service_reports_single_recent_success_as_100_percent() -> None:
    now = int(time.time())
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="recent-success",
                is_success=True,
                created_at=now - 60,
            ),
        ]
    )
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(from_ts=now - 3600, to_ts=now, bucket="day")
    )

    assert response.hero.state == "shadow_off"
    assert response.hero.regular_success_rate_pp == 100.0
    assert response.context_tiles.success.current == 100.0


def test_service_reports_task_and_behavior_success_separately() -> None:
    now = int(time.time())
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="success",
                is_success=True,
                created_at=now - 60,
            ),
            _eval_result(
                result_id=2,
                session_id="agent-failure",
                is_success=False,
                failure_type="wrong_answer",
                created_at=now - 60,
            ),
            _eval_result(
                result_id=3,
                session_id="system-failure",
                is_success=False,
                failure_type="system_error",
                created_at=now - 60,
            ),
        ]
    )
    config = Config(storage_config=StorageConfigSQLite())

    response = EvaluationOverviewService(storage=storage, config=config).run(
        GetEvaluationOverviewRequest(from_ts=now - 3600, to_ts=now, bucket="day")
    )

    assert response.hero.regular_success_rate_pp == pytest.approx(100 / 3)
    assert response.context_tiles.success.current == pytest.approx(100 / 3)
    assert response.context_tiles.behavior_success.current == 50.0
    assert response.context_tiles.behavior_success.eligible_sessions == 2
    assert response.context_tiles.behavior_success.excluded_system_errors == 1


def test_behavior_success_is_null_when_all_rows_are_system_errors() -> None:
    now = int(time.time())
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="system-failure",
                is_success=False,
                failure_type="SYSTEM_ERROR",
                created_at=now - 60,
            )
        ]
    )
    config = Config(storage_config=StorageConfigSQLite())

    response = EvaluationOverviewService(storage=storage, config=config).run(
        GetEvaluationOverviewRequest(from_ts=now - 3600, to_ts=now)
    )

    assert response.context_tiles.success.current == 0.0
    assert response.context_tiles.behavior_success.current is None
    assert response.context_tiles.behavior_success.delta_pp is None
    assert response.context_tiles.behavior_success.eligible_sessions == 0
    assert response.context_tiles.behavior_success.excluded_system_errors == 1


def test_service_shows_recent_evaluations_immediately() -> None:
    """A new org with results should render its available trend immediately."""
    now = int(time.time())
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="recent",
                is_success=True,
                created_at=now,
            )
        ]
    )
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(from_ts=now - 3600, to_ts=now, bucket="day")
    )

    assert response.hero.state == "shadow_off"
    assert response.hero.regular_success_rate_pp == 100.0
    assert sum(bucket.regular_n for bucket in response.hero.buckets) == 1


def test_service_honors_day_bucket_for_hero_trend() -> None:
    day = 24 * 60 * 60
    storage = _storage_with_results(
        [
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
    )
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(from_ts=0, to_ts=3 * day, bucket="day")
    )

    assert [bucket.ts for bucket in response.hero.buckets] == [day, 2 * day]
    assert [bucket.regular_n for bucket in response.hero.buckets] == [2, 1]
    assert [bucket.regular_rate for bucket in response.hero.buckets] == [0.5, 1.0]


def test_service_uses_bulk_storage_methods_without_per_session_reads() -> None:
    storage = _storage_with_results(
        [
            _eval_result(result_id=1, session_id="s1", is_success=True),
            _eval_result(result_id=2, session_id="s2", is_success=False),
        ]
    )
    storage.get_citations_by_session_ids.return_value = [
        SessionCitation(
            user_id="u1",
            session_id="s1",
            kind="playbook",
            real_id="42",
            title="Keep answers short",
        )
    ]
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=int(time.time())))

    assert response.rule_attribution[0].rule_id == "42"
    storage.get_agent_success_evaluation_results.assert_not_called()
    storage.get_sessions.assert_not_called()
    storage.get_interactions_by_session.assert_not_called()
    storage.get_playbook_application_stats.assert_not_called()
    storage.get_first_requests_by_user_session_pairs.assert_called_once()
    storage.get_citations_by_session_ids.assert_called_once()


def test_service_limits_source_lookup_to_window_when_no_source_sets() -> None:
    now = int(time.time())
    day = 24 * 60 * 60
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="current",
                is_success=True,
                created_at=now,
            ),
            _eval_result(
                result_id=2,
                session_id="previous",
                is_success=False,
                created_at=now - 10 * day,
            ),
        ]
    )
    storage.get_first_requests_by_user_session_pairs.return_value = {
        ("u1", "current"): SessionFirstRequest(
            session_id="current",
            user_id="u1",
            source="current-source",
            created_at=now,
        ),
        ("u1", "previous"): SessionFirstRequest(
            session_id="previous",
            user_id="u1",
            source="previous-source",
            created_at=now - 10 * day,
        ),
    }
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(
            from_ts=now - 60,
            to_ts=now,
            include_shadow=False,
        )
    )

    assert response.source_set_comparison.available_sources == ["current-source"]
    assert [
        group.model_dump() for group in response.source_set_comparison.source_sessions
    ] == [
        {
            "source": "current-source",
            "session_ids": ["current"],
            "sessions": [{"user_id": "u1", "session_id": "current"}],
        }
    ]
    storage.get_first_requests_by_user_session_pairs.assert_called_once_with(
        [("u1", "current")]
    )


def test_service_groups_source_sessions_without_duplicate_session_ids() -> None:
    now = int(time.time())
    results = [
        _eval_result(result_id=1, session_id="shared", is_success=True, created_at=now),
        _eval_result(
            result_id=2, session_id="shared", is_success=False, created_at=now
        ),
        _eval_result(result_id=3, session_id="other", is_success=True, created_at=now),
    ]
    storage = _storage_with_results(results)
    storage.get_first_requests_by_user_session_pairs.return_value = {
        ("u1", "shared"): SessionFirstRequest(
            session_id="shared",
            user_id="u1",
            source="api",
            created_at=now,
        ),
        ("u1", "other"): SessionFirstRequest(
            session_id="other",
            user_id="u1",
            source="web",
            created_at=now,
        ),
    }

    response = EvaluationOverviewService(
        storage=storage,
        config=Config(storage_config=StorageConfigSQLite()),
    ).run(
        GetEvaluationOverviewRequest(
            from_ts=now - 60,
            to_ts=now,
            include_shadow=False,
        )
    )

    assert [
        group.model_dump() for group in response.source_set_comparison.source_sessions
    ] == [
        {
            "source": "api",
            "session_ids": ["shared"],
            "sessions": [{"user_id": "u1", "session_id": "shared"}],
        },
        {
            "source": "web",
            "session_ids": ["other"],
            "sessions": [{"user_id": "u1", "session_id": "other"}],
        },
    ]


def test_service_preserves_user_identity_when_session_ids_collide() -> None:
    now = int(time.time())
    results = [
        _eval_result(
            result_id=1,
            user_id="user-api",
            session_id="shared",
            is_success=True,
            created_at=now,
        ),
        _eval_result(
            result_id=2,
            user_id="user-web",
            session_id="shared",
            is_success=False,
            created_at=now,
        ),
    ]
    storage = _storage_with_results(results)
    storage.get_first_requests_by_user_session_pairs.return_value = {
        ("user-api", "shared"): SessionFirstRequest(
            session_id="shared",
            user_id="user-api",
            source="api",
            created_at=now,
        ),
        ("user-web", "shared"): SessionFirstRequest(
            session_id="shared",
            user_id="user-web",
            source="web",
            created_at=now,
        ),
    }

    response = EvaluationOverviewService(
        storage=storage,
        config=Config(storage_config=StorageConfigSQLite()),
    ).run(
        GetEvaluationOverviewRequest(
            from_ts=now - 60,
            to_ts=now,
            include_shadow=False,
        )
    )

    groups = {
        group.source: [session.model_dump() for session in group.sessions]
        for group in response.source_set_comparison.source_sessions
    }
    assert groups == {
        "api": [{"user_id": "user-api", "session_id": "shared"}],
        "web": [{"user_id": "user-web", "session_id": "shared"}],
    }


def test_service_loads_baseline_sources_when_source_sets_requested() -> None:
    now = int(time.time())
    day = 24 * 60 * 60
    storage = _storage_with_results(
        [
            _eval_result(
                result_id=1,
                session_id="current",
                is_success=True,
                created_at=now,
            ),
            _eval_result(
                result_id=2,
                session_id="previous",
                is_success=False,
                created_at=now - 10 * day,
            ),
        ]
    )
    storage.get_first_requests_by_user_session_pairs.return_value = {
        ("u1", "current"): SessionFirstRequest(
            session_id="current",
            user_id="u1",
            source="candidate",
            created_at=now,
        ),
        ("u1", "previous"): SessionFirstRequest(
            session_id="previous",
            user_id="u1",
            source="candidate",
            created_at=now - 10 * day,
        ),
    }
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(
            from_ts=now - 7 * day,
            to_ts=now,
            include_shadow=False,
            source_sets=[
                EvaluationSourceSetRequest(label="Candidate", sources=["candidate"])
            ],
        )
    )

    assert (
        response.source_set_comparison.sets[0].context_tiles.success.delta_pp == 100.0
    )
    storage.get_first_requests_by_user_session_pairs.assert_called_once_with(
        [("u1", "current"), ("u1", "previous")]
    )


def test_service_skips_shadow_storage_when_shadow_excluded() -> None:
    storage = _storage_with_results(
        [_eval_result(result_id=1, session_id="s1", is_success=True)]
    )
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(
            from_ts=0,
            to_ts=int(time.time()),
            include_shadow=False,
        )
    )

    assert response.shadow_win_rate_trend.window_total.n == 0
    storage.get_shadow_comparison_verdicts.assert_not_called()


def test_service_handles_5k_sessions_with_bulk_call_shape() -> None:
    now = int(time.time())
    results = [
        _eval_result(
            result_id=i + 1,
            session_id=f"s{i}",
            is_success=(i % 2 == 0),
            corrections=i % 6,
            created_at=now - (i % 3600),
        )
        for i in range(5000)
    ]
    storage = _storage_with_results(results)
    config = Config(storage_config=StorageConfigSQLite())

    svc = EvaluationOverviewService(storage=storage, config=config)
    response = svc.run(
        GetEvaluationOverviewRequest(
            from_ts=now - 7200,
            to_ts=now,
            include_shadow=False,
        )
    )

    assert response.hero.regular_success_rate_pp == 50.0
    storage.get_agent_success_evaluation_results_in_window.assert_called_once()
    storage.get_first_requests_by_user_session_pairs.assert_called_once()
    storage.get_citations_by_session_ids.assert_called_once()
    storage.get_sessions.assert_not_called()
    storage.get_interactions_by_session.assert_not_called()
