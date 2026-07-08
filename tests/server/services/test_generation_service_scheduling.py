"""Tests for `_schedule_group_evaluation_if_needed` in GenerationService.

Guards against regression of the bug where the agentic extraction backend
silently bypassed the scheduler call, leaving /evaluations permanently empty.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import (
    Interaction,
    InteractionData,
    Request,
)
from reflexio.models.api_schema.service_schemas import PublishUserInteractionRequest
from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    StorageConfigSQLite,
)
from reflexio.server.services.generation_service import GenerationService


@pytest.fixture
def service() -> GenerationService:
    """Build a bare GenerationService with the attributes the helper reads.

    Bypasses __init__ entirely because the helper only reads three instance
    attributes: org_id, request_context, and client. The full constructor would
    require a configurator + storage + LLM client that aren't needed here.
    """
    svc = GenerationService.__new__(GenerationService)
    svc.org_id = "org_test"
    svc.request_context = MagicMock(name="request_context")
    svc.client = MagicMock(name="llm_client")
    svc.storage = MagicMock(name="storage")
    svc.configurator = MagicMock(name="configurator")
    svc.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="Evaluate whether the agent succeeded.",
            sampling_rate=1.0,
        ),
    )
    return svc


def test_schedules_when_session_id_is_required(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper schedules because publish requests require a session_id."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    request_with_session = MagicMock(session_id="sess_required")

    service._schedule_group_evaluation_if_needed(
        new_request=request_with_session,
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )

    scheduler.schedule.assert_called_once()


def test_schedules_with_correct_key_when_session_id_present(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper schedules with key=(org_id, user_id, session_id)."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    request_with_session = MagicMock(session_id="sess_42")

    service._schedule_group_evaluation_if_needed(
        new_request=request_with_session,
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    scheduler.schedule.assert_called_once()
    call_args = scheduler.schedule.call_args
    key = call_args[0][0] if call_args[0] else call_args.kwargs.get("key")
    callback = (
        call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("callback")
    )
    assert key == ("org_test", "user_test", "sess_42")
    assert callable(callback)


def test_evaluation_only_does_not_bypass_sampling_rate(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 0% sample rate skips scheduling even for evaluation-only requests."""
    cast(Any, service.configurator.get_config).return_value = Config(
        storage_config=StorageConfigSQLite(),
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="Evaluate whether the agent succeeded.",
            sampling_rate=0.0,
        ),
    )
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    request_with_session = MagicMock(session_id="sess_42", evaluation_only=True)

    service._schedule_group_evaluation_if_needed(
        new_request=request_with_session,
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    scheduler.schedule.assert_not_called()


def test_shadow_content_bypasses_sampling_for_shadow_only(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shadow comparison enqueues even when session-level eval sampling is 0%."""
    cast(Any, service.configurator.get_config).return_value = Config(
        storage_config=StorageConfigSQLite(),
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="Evaluate whether the agent succeeded.",
            sampling_rate=0.0,
        ),
    )
    scheduler = MagicMock()
    shadow_enqueue = MagicMock(return_value=True)
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.enqueue_shadow_comparison",
        shadow_enqueue,
    )

    service._schedule_post_publish_evaluations(
        new_request=Request(
            request_id="req", user_id="user_test", session_id="sess_42"
        ),
        interactions=[
            Interaction(
                user_id="user_test",
                request_id="req",
                role="assistant",
                content="regular",
                shadow_content="shadow",
            )
        ],
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    shadow_enqueue.assert_called_once()
    scheduler.schedule.assert_not_called()


def test_no_shadow_and_zero_sampling_schedules_nothing(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    cast(Any, service.configurator.get_config).return_value = Config(
        storage_config=StorageConfigSQLite(),
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="Evaluate whether the agent succeeded.",
            sampling_rate=0.0,
        ),
    )
    scheduler = MagicMock()
    shadow_enqueue = MagicMock(return_value=True)
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.enqueue_shadow_comparison",
        shadow_enqueue,
    )

    service._schedule_post_publish_evaluations(
        new_request=Request(
            request_id="req", user_id="user_test", session_id="sess_42"
        ),
        interactions=[
            Interaction(user_id="user_test", request_id="req", role="assistant")
        ],
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    shadow_enqueue.assert_not_called()
    scheduler.schedule.assert_not_called()


def test_shadow_content_and_full_sampling_schedule_both(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = MagicMock()
    shadow_enqueue = MagicMock(return_value=True)
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.enqueue_shadow_comparison",
        shadow_enqueue,
    )

    service._schedule_post_publish_evaluations(
        new_request=Request(
            request_id="req", user_id="user_test", session_id="sess_42"
        ),
        interactions=[
            Interaction(
                user_id="user_test",
                request_id="req",
                role="assistant",
                content="regular",
                shadow_content="shadow",
            )
        ],
        user_id="user_test",
        agent_version="v_test",
        source="ide",
    )

    shadow_enqueue.assert_called_once()
    scheduler.schedule.assert_called_once()


def test_learning_stall_path_calls_post_publish_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MagicMock(name="storage")
    storage.get_request.return_value = None
    storage.get_stall_state.return_value = SimpleNamespace(
        stalled=True,
        reason="auth_error",
    )
    request_context = SimpleNamespace(
        storage=storage,
        org_id="org_test",
        configurator=MagicMock(name="configurator"),
    )
    service = GenerationService(
        llm_client=MagicMock(name="llm_client"),
        request_context=cast(Any, request_context),
    )
    service._cleanup_storage_tables_if_needed = MagicMock()  # type: ignore[method-assign]
    post_publish = MagicMock()
    monkeypatch.setattr(
        GenerationService,
        "_schedule_post_publish_evaluations",
        post_publish,
    )

    result = service.run(
        PublishUserInteractionRequest(
            user_id="user_test",
            session_id="sess_42",
            interaction_data_list=[
                InteractionData(
                    role="assistant",
                    content="regular",
                    shadow_content="shadow",
                )
            ],
        ),
        use_publish_limiter=False,
    )

    assert result.request_id is not None
    storage.add_request.assert_called_once()
    storage.add_user_interactions_bulk.assert_called_once()
    post_publish.assert_called_once()
