"""Tests for force_regenerate + evaluation_name on run_group_evaluation.

These cover the regenerate flow's behavior at the runner layer:
- force_regenerate=True bypasses both the operation-state "already evaluated"
  short-circuit and the completeness-delay gate so an operator can re-evaluate
  any session.
- evaluation_name propagates into AgentSuccessEvaluationRequest.evaluation_name_filter
  so the downstream service narrows to a single evaluator instead of running
  every configured rubric.
- When both kwargs are set, prior result rows for that
  (session_id, evaluation_name, agent_version) tuple are deleted before the new
  run so the regenerated verdict cleanly replaces the old one.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)


def _make_request(request_id: str, user_id: str, session_id: str) -> Request:
    """Create a request old enough to pass the completion delay gate.

    Args:
        request_id (str): Request identifier.
        user_id (str): Owner user identifier.
        session_id (str): Session identifier.

    Returns:
        Request: A Request with created_at well in the past.
    """
    now = int(datetime.now(UTC).timestamp())
    return Request(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        created_at=now - 10000,
    )


def _make_interaction(request_id: str, user_id: str) -> Interaction:
    """Create a minimal interaction tied to a request.

    Args:
        request_id (str): Owning request identifier.
        user_id (str): Owner user identifier.

    Returns:
        Interaction: A populated Interaction instance.
    """
    now = int(datetime.now(UTC).timestamp())
    return Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id=request_id,
        content="test content",
        role="user",
        created_at=now - 9999,
    )


def _make_storage(*, with_evaluated_marker: bool) -> MagicMock:
    """Build a storage mock seeded with one request + one interaction.

    Args:
        with_evaluated_marker (bool): When True, get_operation_state returns a
            payload whose operation_state.evaluated is True — simulating a
            session that's already been evaluated.

    Returns:
        MagicMock: A storage stub wired with the standard return values.
    """
    storage = MagicMock()
    if with_evaluated_marker:
        storage.get_operation_state.return_value = {
            "operation_state": {"evaluated": True, "evaluated_at": 1}
        }
    else:
        storage.get_operation_state.return_value = None
    storage.get_requests_by_session.return_value = [
        _make_request("req_1", "user_a", "session_a")
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_1", "user_a")
    ]
    storage.delete_agent_success_evaluation_results_for_session.return_value = 1
    return storage


def test_force_regenerate_bypasses_already_evaluated_short_circuit() -> None:
    """force_regenerate=True must still invoke the service on a marked session."""
    storage = _make_storage(with_evaluated_marker=True)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    # Service was constructed and invoked despite the existing evaluated marker.
    service.run.assert_called_once()


def test_evaluation_name_propagates_into_evaluation_request() -> None:
    """evaluation_name kwarg must flow into AgentSuccessEvaluationRequest.evaluation_name_filter."""
    storage = _make_storage(with_evaluated_marker=False)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
            evaluation_name="overall_success",
        )

    service.run.assert_called_once()
    eval_request = service.run.call_args.args[0]
    assert eval_request.evaluation_name_filter == "overall_success"
    assert eval_request.session_id == "session_a"
    assert eval_request.agent_version == "1.0.0"


def test_force_regenerate_with_evaluation_name_deletes_prior_results() -> None:
    """Both kwargs set => prior result rows deleted before re-running."""
    storage = _make_storage(with_evaluated_marker=True)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
            evaluation_name="overall_success",
        )

    # Scoped delete called exactly once with the targeted tuple.
    storage.delete_agent_success_evaluation_results_for_session.assert_called_once_with(
        session_id="session_a",
        evaluation_name="overall_success",
        agent_version="1.0.0",
    )


def test_force_regenerate_without_evaluation_name_does_not_delete() -> None:
    """force_regenerate alone (no evaluator name) must NOT call the scoped delete.

    Deletion is scoped to a single evaluation_name. A regenerate run that
    re-runs every configured evaluator should leave the storage layer to
    overwrite (or accumulate) results as it normally does, rather than
    silently wipe rows belonging to evaluators the caller didn't target.
    """
    storage = _make_storage(with_evaluated_marker=True)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    storage.delete_agent_success_evaluation_results_for_session.assert_not_called()
    service.run.assert_called_once()
    eval_request = service.run.call_args.args[0]
    assert eval_request.evaluation_name_filter is None


def test_force_regenerate_bypasses_completeness_delay_gate() -> None:
    """force_regenerate=True must skip the delay check even for a fresh session."""
    # Build a request just a moment old — would normally trip the delay gate.
    storage = MagicMock()
    storage.get_operation_state.return_value = None
    now = int(datetime.now(UTC).timestamp())
    fresh_request = Request(
        request_id="req_fresh",
        user_id="user_a",
        session_id="session_a",
        created_at=now,
    )
    storage.get_requests_by_session.return_value = [fresh_request]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_fresh", "user_a")
    ]

    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    service.run.assert_called_once()
