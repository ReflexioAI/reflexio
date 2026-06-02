"""Integration tests for POST /api/evaluations/grade_on_demand.

The endpoint grades a single session synchronously and caches the verdict
for 24h via the existing ``operation_state`` storage. F3 Task 8 ships it
so Plan 3 (F1)'s bounded-list "click-through" UX can pull a fresh grade
for any session that wasn't in the regen sampler's chosen subset.

These tests focus on the handler contract:
  - Cache miss → fresh grade is computed and persisted.
  - Cache hit (within 24h) → returns cached without re-grading.
  - Unknown session → returns ``skipped_reason`` instead of 5xx.

The actual judge call inside ``run_group_evaluation`` is mocked at the
runner boundary — the runner is exercised exhaustively in its own
``test_group_evaluation_runner_*`` suite, and end-to-end LLM behaviour is
covered by the e2e tier. Here we only verify the endpoint correctly
wires storage → runner → cache → response.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
    Interaction,
    Request,
)
from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.cache.reflexio_cache import get_reflexio

pytestmark = pytest.mark.integration


def _seed_session(
    storage,
    *,
    session_id: str,
    user_id: str,
    agent_version: str,
    created_at: int = 1_700_000_000,
) -> None:
    """Insert one Request + one Interaction so the runner sees data."""
    storage.add_request(
        Request(
            request_id=f"req-{session_id}",
            user_id=user_id,
            created_at=created_at,
            source="api",
            agent_version=agent_version,
            session_id=session_id,
        )
    )
    storage.add_user_interaction(
        user_id,
        Interaction(
            interaction_id=0,
            user_id=user_id,
            request_id=f"req-{session_id}",
            content="hello",
            role="user",
            created_at=created_at + 1,
        ),
    )


def _configure_evaluator(org_id: str, *, evaluation_name: str) -> None:
    """Register a single AgentSuccessConfig so the handler passes the
    'known evaluation_name' gate."""
    reflexio = get_reflexio(org_id=org_id)
    reflexio.request_context.configurator.set_config_by_name(
        "agent_success_config",
        AgentSuccessConfig(
            evaluation_name=evaluation_name,
            success_definition_prompt=(
                "Evaluate whether the agent successfully completed the task."
            ),
        ),
    )


def _fake_runner_factory(storage, *, agent_version: str, evaluation_name: str):
    """Build a side-effect callable that imitates ``run_group_evaluation``.

    The real runner writes an ``AgentSuccessEvaluationResult`` row. We
    short-circuit the LLM call but keep the persistence contract intact
    so the handler's ``get_agent_success_evaluation_results`` lookup
    finds a fresh row to report as ``result_id``.
    """

    def _fake(*, session_id: str, **_kwargs: object) -> None:
        storage.save_agent_success_evaluation_results(
            [
                AgentSuccessEvaluationResult(
                    result_id=0,
                    session_id=session_id,
                    agent_version=agent_version,
                    evaluation_name=evaluation_name,
                    is_success=True,
                    failure_type="",
                    failure_reason="",
                    regular_vs_shadow=None,
                    number_of_correction_per_session=0,
                    user_turns_to_resolution=None,
                    is_escalated=False,
                    embedding=[],
                    created_at=1_700_000_500,
                )
            ]
        )

    return _fake


def test_grade_on_demand_first_call_returns_fresh_result(client_with_org):
    """First call grades the session and writes a fresh result row.

    Verifies the cache-miss path: no prior ``operation_state`` entry, so
    the handler must invoke the runner, locate the freshly-written row,
    and surface ``cached=False`` with a non-None ``result_id``.
    """
    client, org_id = client_with_org
    _configure_evaluator(org_id, evaluation_name="overall_success")
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None
    _seed_session(
        storage,
        session_id="grade-on-demand-fresh",
        user_id="user-fresh",
        agent_version="v1",
    )

    with patch(
        "reflexio.server.api.run_group_evaluation",
        side_effect=_fake_runner_factory(
            storage, agent_version="v1", evaluation_name="overall_success"
        ),
    ) as runner:
        resp = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "grade-on-demand-fresh",
                "agent_version": "v1",
                "evaluation_name": "overall_success",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == "grade-on-demand-fresh"
    assert body["cached"] is False
    assert body["skipped_reason"] is None
    assert body["result_id"] is not None
    runner.assert_called_once()


def test_grade_on_demand_second_call_returns_cached(client_with_org):
    """Second call within the 24h window short-circuits the runner.

    The first call populates the cache; the second must serve from it
    without re-invoking ``run_group_evaluation``. Cached responses echo
    the prior ``result_id`` so the frontend can re-link to the verdict.
    """
    client, org_id = client_with_org
    _configure_evaluator(org_id, evaluation_name="overall_success")
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None
    _seed_session(
        storage,
        session_id="grade-on-demand-cached",
        user_id="user-cached",
        agent_version="v1",
    )

    fake = _fake_runner_factory(
        storage, agent_version="v1", evaluation_name="overall_success"
    )
    with patch("reflexio.server.api.run_group_evaluation", side_effect=fake) as runner:
        first = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "grade-on-demand-cached",
                "agent_version": "v1",
                "evaluation_name": "overall_success",
            },
        )
        assert first.status_code == 200, first.text
        first_body = first.json()
        assert first_body["cached"] is False
        assert runner.call_count == 1

        second = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "grade-on-demand-cached",
                "agent_version": "v1",
                "evaluation_name": "overall_success",
            },
        )

    assert second.status_code == 200, second.text
    second_body = second.json()
    assert second_body["cached"] is True
    assert second_body["session_id"] == "grade-on-demand-cached"
    assert second_body["result_id"] == first_body["result_id"]
    # Runner only ran once across both calls — the cache short-circuit
    # is the whole point of this endpoint vs. /regenerate.
    assert runner.call_count == 1


def test_grade_on_demand_scopes_cache_and_readback_by_evaluation_name(client_with_org):
    """Rows for another evaluator in the same session/version must not be reused."""
    client, org_id = client_with_org
    _configure_evaluator(org_id, evaluation_name="overall_success")
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None
    _seed_session(
        storage,
        session_id="grade-on-demand-scoped",
        user_id="user-scoped",
        agent_version="v1",
    )

    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                result_id=0,
                session_id="grade-on-demand-scoped",
                agent_version="v1",
                evaluation_name="other_evaluator",
                is_success=False,
                failure_type="old",
                failure_reason="Historical row for another evaluator",
                regular_vs_shadow=None,
                number_of_correction_per_session=0,
                user_turns_to_resolution=None,
                is_escalated=False,
                embedding=[],
                created_at=1_700_000_900,
            )
        ]
    )
    historical_other_id = next(
        row.result_id
        for row in storage.get_agent_success_evaluation_results(
            limit=10, agent_version="v1"
        )
        if row.session_id == "grade-on-demand-scoped"
        and row.evaluation_name == "other_evaluator"
    )

    with patch(
        "reflexio.server.api.run_group_evaluation",
        side_effect=_fake_runner_factory(
            storage, agent_version="v1", evaluation_name="overall_success"
        ),
    ):
        first = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "grade-on-demand-scoped",
                "agent_version": "v1",
                "evaluation_name": "overall_success",
            },
        )
        second = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "grade-on-demand-scoped",
                "agent_version": "v1",
                "evaluation_name": "other_evaluator",
            },
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body["result_id"] is not None
    assert first_body["result_id"] != historical_other_id
    assert second_body["result_id"] is None


def test_grade_on_demand_unknown_session_returns_skipped(client_with_org):
    """Unknown ``session_id`` returns ``skipped_reason`` with a 200.

    Plan 3's bounded-list UX may surface stale session ids from the
    customer's URL bar; surfacing this as 200 + ``skipped_reason`` keeps
    the frontend's error-handling local and avoids polluting 5xx telemetry.
    """
    client, org_id = client_with_org
    _configure_evaluator(org_id, evaluation_name="overall_success")

    with patch("reflexio.server.api.run_group_evaluation") as runner:
        resp = client.post(
            "/api/evaluations/grade_on_demand",
            json={
                "session_id": "missing-session-grade-on-demand",
                "agent_version": "v1",
                "evaluation_name": "overall_success",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped_reason"] is not None
    assert body["result_id"] is None
    assert body["cached"] is False
    # Handler must not invoke the runner when no requests exist for the
    # session — saves the LLM call on a guaranteed-empty workload.
    runner.assert_not_called()
