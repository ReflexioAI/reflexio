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


def test_evaluation_only_without_override_inherits_sampling_rate(
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


def _set_rates(
    service: GenerationService,
    *,
    success: float,
    retrieved: float | None,
    evaluation_only: float | None = None,
) -> None:
    cast(Any, service.configurator.get_config).return_value = Config(
        storage_config=StorageConfigSQLite(),
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="Evaluate whether the agent succeeded.",
            sampling_rate=success,
            evaluation_only_sampling_rate=evaluation_only,
            retrieved_learning_sampling_rate=retrieved,
        ),
    )


def _schedule_and_capture(
    service: GenerationService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluation_only: bool = False,
) -> MagicMock:
    """Schedule, then run the queued callback with run_group_evaluation stubbed."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    runner = MagicMock(name="run_group_evaluation")
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.run_group_evaluation", runner
    )

    service._schedule_group_evaluation_if_needed(
        new_request=MagicMock(session_id="sess_split", evaluation_only=evaluation_only),
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )
    if scheduler.schedule.call_count:
        scheduler.schedule.call_args[0][1]()  # invoke the queued callback
    return runner


def test_retrieved_only_sampling_schedules_but_skips_the_success_judge(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core coverage guarantee: dense tuner signal without the success bill.

    A session sampled ONLY for retrieved-learning must still be scheduled, and
    the runner must be told to skip the session-success judge.
    """
    _set_rates(service, success=0.0, retrieved=1.0)

    runner = _schedule_and_capture(service, monkeypatch)

    runner.assert_called_once()
    kwargs = runner.call_args.kwargs
    assert kwargs["run_agent_success"] is False
    assert kwargs["run_retrieved_learning"] is True


def test_success_only_sampling_skips_the_retrieved_learning_judge(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rates(service, success=1.0, retrieved=0.0)

    runner = _schedule_and_capture(service, monkeypatch)

    runner.assert_called_once()
    kwargs = runner.call_args.kwargs
    assert kwargs["run_agent_success"] is True
    assert kwargs["run_retrieved_learning"] is False


def test_evaluation_only_override_runs_only_success_judge(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rates(service, success=0.0, retrieved=0.0, evaluation_only=1.0)

    runner = _schedule_and_capture(service, monkeypatch, evaluation_only=True)

    runner.assert_called_once()
    kwargs = runner.call_args.kwargs
    assert kwargs["run_agent_success"] is True
    assert kwargs["run_retrieved_learning"] is False


def test_evaluation_only_override_does_not_change_regular_sampling(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rates(service, success=0.0, retrieved=0.0, evaluation_only=1.0)

    runner = _schedule_and_capture(service, monkeypatch, evaluation_only=False)

    runner.assert_not_called()


def test_neither_family_sampled_does_not_schedule_at_all(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rates(service, success=0.0, retrieved=0.0)

    runner = _schedule_and_capture(service, monkeypatch)

    runner.assert_not_called()


@pytest.mark.parametrize(
    ("retrieved_rate", "expected_sampled", "expected_schedule_count"),
    [(1.0, True, 1), (0.0, False, 0)],
)
def test_publish_persists_the_retrieved_learning_sampling_decision(
    service: GenerationService,
    monkeypatch: pytest.MonkeyPatch,
    retrieved_rate: float,
    expected_sampled: bool,
    expected_schedule_count: int,
) -> None:
    _set_rates(service, success=0.0, retrieved=retrieved_rate)
    storage = cast(MagicMock, service.storage)
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )

    service._schedule_post_publish_evaluations(
        new_request=Request(
            request_id="request-sampling-decision",
            user_id="user_test",
            session_id="session-sampling-decision",
        ),
        interactions=[],
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )

    storage.record_retrieved_learning_sampling_decision.assert_called_once_with(
        user_id="user_test",
        session_id="session-sampling-decision",
        request_id="request-sampling-decision",
        sampled=expected_sampled,
    )
    assert scheduler.schedule.call_count == expected_schedule_count


def test_sampling_decision_persistence_failure_does_not_break_publish_scheduling(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_rates(service, success=0.0, retrieved=1.0)
    storage = cast(MagicMock, service.storage)
    storage.record_retrieved_learning_sampling_decision.side_effect = RuntimeError(
        "sampling persistence unavailable"
    )
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )

    service._schedule_post_publish_evaluations(
        new_request=Request(
            request_id="request-sampling-failure",
            user_id="user_test",
            session_id="session-sampling-failure",
        ),
        interactions=[],
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )

    storage.record_retrieved_learning_sampling_decision.assert_called_once()
    scheduler.schedule.assert_called_once()


def test_unset_retrieved_rate_inherits_success_rate(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Orgs that have not opted in keep exactly their previous behavior."""
    _set_rates(service, success=1.0, retrieved=None)

    runner = _schedule_and_capture(service, monkeypatch)

    kwargs = runner.call_args.kwargs
    assert kwargs["run_agent_success"] is True
    assert kwargs["run_retrieved_learning"] is True


def _drain_scheduler(scheduler: MagicMock, *, max_rounds: int = 10) -> int:
    """Run queued callbacks until the scheduler stops re-arming. Returns rounds."""
    rounds = 0
    seen = 0
    while scheduler.schedule.call_count > seen and rounds < max_rounds:
        seen = scheduler.schedule.call_count
        scheduler.schedule.call_args_list[seen - 1][0][1]()
        rounds += 1
    return rounds


def _run_with_outcome(
    service: GenerationService,
    monkeypatch: pytest.MonkeyPatch,
    retrieved_status: str,
) -> tuple[MagicMock, MagicMock]:
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    runner = MagicMock(
        name="run_group_evaluation",
        return_value=SimpleNamespace(
            agent_success_status="complete",
            retrieved_learning_status=retrieved_status,
            retrieved_learning_fingerprint=None,
        ),
    )
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.run_group_evaluation", runner
    )
    _set_rates(service, success=1.0, retrieved=1.0)

    service._schedule_group_evaluation_if_needed(
        new_request=MagicMock(session_id="sess_retry"),
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )
    _drain_scheduler(scheduler)
    return scheduler, runner


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_a_run_that_committed_nothing_is_retried(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """`failed` and `pending` persisted NO rows, so nothing else re-triggers them
    unless the session happens to get more traffic. These are the retriable ones."""
    _, runner = _run_with_outcome(service, monkeypatch, status)

    # 1 initial + 3 bounded retries.
    assert runner.call_count == 1 + 3

    retries = runner.call_args_list[1:]
    for call in retries:
        # A retry must not re-pay the session-success judge.
        assert call.kwargs["run_agent_success"] is False
        assert call.kwargs["run_retrieved_learning"] is True


def test_degraded_is_never_retried(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`degraded` is an APPLIED, fingerprint-fenced commit — its rows are already
    persisted, and only the chunks whose judge failed carry NULL impact.

    Retrying it would re-execute EVERY relevance + impact chunk for the session
    and delete/re-insert rows that are already committed. A deterministically
    degrading chunk (an over-length learning, a content-filter refusal) degrades
    again on every attempt, so a bounded 3-retry sweep would burn 4x the judge
    bill on that slice and buy nothing. It is re-judged fresh on the next
    scheduled run and self-heals to "complete" once the transient failure clears.
    """
    _, runner = _run_with_outcome(service, monkeypatch, "degraded")

    assert runner.call_count == 1


@pytest.mark.parametrize("status", ["complete", "not_applicable"])
def test_terminal_retrieved_learning_is_not_retried(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    _, runner = _run_with_outcome(service, monkeypatch, status)

    assert runner.call_count == 1


def test_retrieved_learning_retry_does_not_fire_when_family_not_sampled(
    service: GenerationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session not admitted for retrieved-learning must never retry it."""
    scheduler = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.GroupEvaluationScheduler.get_instance",
        lambda: scheduler,
    )
    runner = MagicMock(
        return_value=SimpleNamespace(
            agent_success_status="complete",
            retrieved_learning_status="skipped",
            retrieved_learning_fingerprint=None,
        ),
    )
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.run_group_evaluation", runner
    )
    _set_rates(service, success=1.0, retrieved=0.0)

    service._schedule_group_evaluation_if_needed(
        new_request=MagicMock(session_id="sess_norl"),
        user_id="user_test",
        agent_version="v_test",
        source=None,
    )
    _drain_scheduler(scheduler)

    assert runner.call_count == 1
