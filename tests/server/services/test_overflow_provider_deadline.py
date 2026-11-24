"""Window replay and postprocessing share the original provider deadline."""

import time as real_time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm import _litellm_text_generation as generation
from reflexio.server.llm import call_deadline
from reflexio.server.llm._litellm_types import LiteLLMConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMClientError
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    owned_agent_run_write,
    owned_user_work_context,
)
from reflexio.server.services.extraction import (
    overflow_coordinator as coordinator_module,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
)
from reflexio.server.services.extraction.resumable_agent import ResumableExtractionAgent
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.test_durable_window_pipeline import _overflow_coordinator
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline

pytestmark = pytest.mark.integration


def test_saved_agent_output_then_real_client_postprocessing_fits_remaining_turn(
    pipeline, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-network")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    storage, window, token = _overflow_coordinator(pipeline)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", timeout=300, max_tokens=128, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"profiles":[]}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    wire = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    with owned_user_work_context(storage, window.user_id, token, 300):
        parent = adopt_agent_run_for_execution(
            storage, storage.get_agent_run("window:" + window.window_id)
        )
        with owned_agent_run_write(storage, parent):
            parent = storage.update_agent_run_status(
                parent.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output={"profiles": []},
            )
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            checkpoint={"stage": "recovered", "completed_steps": {}},
        )
        parent = adopt_agent_run_for_execution(
            storage, storage.get_agent_run(parent.id)
        )
        coordinator = OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={"gpt-4o": 128000},
            turn_seconds=90,
        )
        coordinator.saved_run = parent
        with coordinator.provider_scope():
            cached = ResumableExtractionAgent(client=client, storage=storage).start(
                run=parent,
                messages=[{"role": "user", "content": "synthetic saved source"}],
                history_inputs=None,
                output_schema=StructuredProfilesOutput,
            )
            assert cached.output == StructuredProfilesOutput(profiles=[])
            wire.assert_not_called()
            result = client.generate_chat_response_with_provenance(
                messages=[{"role": "user", "content": "synthetic consolidation"}],
                response_format=StructuredProfilesOutput,
            )
        assert result.value == StructuredProfilesOutput(profiles=[])
        wire.assert_called_once()
        assert 0 < wire.call_args.args[1] < 90
        assert len(coordinator._manifest["attempts"]) == 1
        saved = storage.get_agent_run(parent.id)
        assert len(saved.generation_request_snapshot["overflow"]["attempts"]) == 1
        assert saved.committed_output == {"profiles": []}
        assert (
            saved.generation_request_snapshot["overflow"]["checkpoint"]["stage"]
            == "recovered"
        )
        assert storage.count_all_profiles() == 0


@pytest.fixture
def bounded_turn(pipeline, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-network")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    clock = SimpleNamespace(now=1000.0)
    local_time = SimpleNamespace(
        monotonic=lambda: clock.now,
        perf_counter=real_time.perf_counter,
        time=real_time.time,
        sleep=real_time.sleep,
    )
    monkeypatch.setattr(generation, "time", local_time)
    monkeypatch.setattr(coordinator_module, "time", local_time)
    monkeypatch.setattr(call_deadline, "monotonic", local_time.monotonic)
    storage, window, token = _overflow_coordinator(pipeline)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", timeout=300, max_tokens=128, fallback_models=[])
    )
    with owned_user_work_context(storage, window.user_id, token, 300):
        parent = adopt_agent_run_for_execution(
            storage, storage.get_agent_run("window:" + window.window_id)
        )
        coordinator = OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={"gpt-4o": 128000, "gpt-4o-mini": 128000},
            turn_seconds=90,
        )
        coordinator.saved_run = parent
        yield SimpleNamespace(
            clock=clock,
            coordinator=coordinator,
            client=client,
            storage=storage,
            parent=parent,
        )


def _response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )


def test_pre_admission_elapsed_time_does_not_double_reserve_finish_margin(
    bounded_turn, monkeypatch
):
    turn = bounded_turn
    original = coordinator_module.check_request_input_budget

    def validation(*args, **kwargs):
        turn.clock.now += 0.05
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator_module, "check_request_input_budget", validation)
    wire = Mock(return_value=_response())
    monkeypatch.setattr(turn.client, "_completion_with_hard_timeout", wire)
    with turn.coordinator.provider_scope():
        assert (
            turn.client.generate_chat_response(
                [{"role": "user", "content": "synthetic"}]
            )
            == "ok"
        )
    assert wire.call_args.args[1] == pytest.approx(84.95)
    assert turn.coordinator._deadline == 1090
    assert len(turn.coordinator._manifest["attempts"]) == 1


def test_slot_wait_and_fallback_share_original_absolute_window_budget(
    bounded_turn, monkeypatch
):
    turn = bounded_turn
    turn.client.config.fallback_models = ["gpt-4o-mini"]
    waits, sent = [], []

    @contextmanager
    def slot(_model, *, max_wait):
        waits.append(max_wait)
        turn.clock.now += 20
        yield

    monkeypatch.setattr(generation, "provider_slot", slot)

    def wire(params, hard_timeout):
        sent.append((turn.clock.now, hard_timeout))
        if len(sent) == 1:
            turn.clock.now += 10
            raise LiteLLMClientError("synthetic primary failure")
        return _response()

    monkeypatch.setattr(turn.client, "_completion_with_hard_timeout", wire)
    with turn.coordinator.provider_scope():
        assert (
            turn.client.generate_chat_response(
                [{"role": "user", "content": "synthetic"}]
            )
            == "ok"
        )
    assert waits == [pytest.approx(85), pytest.approx(55)]
    assert sent == [(1020, pytest.approx(65)), (1050, pytest.approx(35))]
    assert len(turn.coordinator._manifest["attempts"]) == 2
    assert (
        len(
            turn.storage.get_agent_run(turn.parent.id).generation_request_snapshot[
                "overflow"
            ]["attempts"]
        )
        == 2
    )
    assert turn.client.config.timeout == 300 and turn.coordinator._deadline == 1090


@pytest.mark.parametrize("where", ["before-scope", "slot-wait"])
def test_expired_window_never_dispatches_or_reserves_another_attempt(
    bounded_turn, monkeypatch, where
):
    turn = bounded_turn
    wire = Mock(side_effect=AssertionError("expired work must not dispatch"))
    monkeypatch.setattr(turn.client, "_completion_with_hard_timeout", wire)
    if where == "before-scope":
        turn.clock.now += 90
        expected = OverflowSuspendedError
    else:

        @contextmanager
        def slot(*_a, **_kw):
            turn.clock.now += 90
            yield

        monkeypatch.setattr(generation, "provider_slot", slot)
        expected = LiteLLMClientError
    with pytest.raises(expected), turn.coordinator.provider_scope():
        turn.client.generate_chat_response([{"role": "user", "content": "synthetic"}])
    wire.assert_not_called()
    assert turn.coordinator._manifest["attempts"] == []
    assert call_deadline.model_call_deadline.get() is None


def test_tighter_caller_and_ordinary_nonwindow_timeout_are_unchanged(
    bounded_turn, monkeypatch
):
    turn = bounded_turn
    wire = Mock(return_value=_response())
    monkeypatch.setattr(turn.client, "_completion_with_hard_timeout", wire)
    with call_deadline.model_deadline(1030):
        with turn.coordinator.provider_scope():
            turn.client.generate_chat_response(
                [{"role": "user", "content": "synthetic"}]
            )
        assert call_deadline.model_call_deadline.get() == 1030
    assert wire.call_args.args[1] == 30
    assert call_deadline.model_call_deadline.get() is None
    turn.client.generate_chat_response([{"role": "user", "content": "ordinary"}])
    assert wire.call_args.args[1] == 300
    assert len(turn.coordinator._manifest["attempts"]) == 1
