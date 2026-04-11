"""Tests for group_evaluation_runner behavior."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    _build_state_key,
    run_group_evaluation,
)


def _make_request(request_id: str, user_id: str, session_id: str) -> Request:
    """Create a request object old enough to pass completion delay checks."""
    now = int(datetime.now(UTC).timestamp())
    return Request(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        created_at=now - 10000,
    )


def _make_interaction(request_id: str, user_id: str) -> Interaction:
    """Create an interaction object tied to a request."""
    now = int(datetime.now(UTC).timestamp())
    return Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id=request_id,
        content="test content",
        role="user",
        created_at=now - 9999,
    )


def test_build_state_key_includes_user_id() -> None:
    """Ensure state keys are scoped to org + user + session_id."""
    key = _build_state_key("org_a", "user_a", "group_a")
    assert key == "agent_success_group_eval::org_a::user_a::group_a"


def test_run_group_evaluation_skips_mark_when_service_has_failures() -> None:
    """Do not mark evaluated=true when evaluation run reports failures."""
    org_id = "org_a"
    user_id = "user_a"
    session_id = "group_a"
    state_key = _build_state_key(org_id, user_id, session_id)

    storage = MagicMock()
    storage.get_operation_state.return_value = None
    storage.get_requests_by_session.return_value = [
        _make_request("req_1", user_id, session_id)
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_1", user_id)
    ]

    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation.group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = True
        service._last_extractor_run_stats = {"failed": 1}
        service.last_run_save_failed = False
        service.last_run_saved_result_count = 0
        service_cls.return_value = service

        run_group_evaluation(
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
        )

    storage.get_operation_state.assert_called_once_with(state_key)
    storage.upsert_operation_state.assert_not_called()


def test_run_group_evaluation_marks_state_when_service_succeeds() -> None:
    """Mark evaluated=true only when evaluation run succeeds."""
    org_id = "org_a"
    user_id = "user_a"
    session_id = "group_a"
    state_key = _build_state_key(org_id, user_id, session_id)

    storage = MagicMock()
    storage.get_operation_state.return_value = None
    storage.get_requests_by_session.return_value = [
        _make_request("req_1", user_id, session_id)
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_1", user_id)
    ]

    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation.group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
        )

    storage.get_operation_state.assert_called_once_with(state_key)
    storage.upsert_operation_state.assert_called_once()
    upsert_key, upsert_payload = storage.upsert_operation_state.call_args[0]
    assert upsert_key == state_key
    assert upsert_payload["evaluated"] is True


def test_effective_delay_seconds_symbol_exists() -> None:
    """Pin the path that ``tests/e2e_tests/conftest.py`` patches.

    The autouse `_zero_group_evaluation_delay` fixture in the e2e conftest
    patches `_EFFECTIVE_DELAY_SECONDS` on this module to disable the
    "session in progress" gate during e2e tests. If a future rename
    moves or deletes that constant, the patch silently becomes a no-op,
    every e2e test that publishes interactions then has to wait through
    the real delay before evaluation can run, and the suite quietly
    regresses. This test makes the rename visible by failing immediately
    on import-path drift.
    """
    from reflexio.server.services.agent_success_evaluation import (
        group_evaluation_runner,
    )

    assert hasattr(group_evaluation_runner, "_EFFECTIVE_DELAY_SECONDS"), (
        "tests/e2e_tests/conftest.py patches "
        "reflexio.server.services.agent_success_evaluation."
        "group_evaluation_runner._EFFECTIVE_DELAY_SECONDS — keep that "
        "symbol name in sync if you rename it."
    )
    assert isinstance(group_evaluation_runner._EFFECTIVE_DELAY_SECONDS, int)


def test_run_group_evaluation_skips_mark_when_nothing_saved() -> None:
    """Do not mark evaluated=true when evaluation produced no persisted results."""
    org_id = "org_a"
    user_id = "user_a"
    session_id = "group_a"
    state_key = _build_state_key(org_id, user_id, session_id)

    storage = MagicMock()
    storage.get_operation_state.return_value = None
    storage.get_requests_by_session.return_value = [
        _make_request("req_1", user_id, session_id)
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_1", user_id)
    ]

    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation.group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 0
        service_cls.return_value = service

        run_group_evaluation(
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
        )

    storage.get_operation_state.assert_called_once_with(state_key)
    storage.upsert_operation_state.assert_not_called()
