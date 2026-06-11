"""Integration coverage for evaluation-only publish scheduling."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import InteractionData
from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    StorageConfigSQLite,
)
from reflexio.server.services.agent_success_evaluation import (
    delayed_group_evaluator,
    group_evaluation_runner,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
    AgentSuccessEvaluationService,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    GroupEvaluationScheduler,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.test_support.llm_mock import patched_litellm


def _wait_until(assertion: Callable[[], None], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: AssertionError | None = None
    while time.monotonic() < deadline:
        try:
            assertion()
            return
        except AssertionError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    assertion()


def _interaction_pair(label: str) -> list[InteractionData]:
    return [
        InteractionData(content=f"Please help with {label}", role="User"),
        InteractionData(content=f"Resolved {label} with steps A and B.", role="Assistant"),
    ]


@pytest.fixture(autouse=True)
def _reset_group_scheduler() -> Iterator[None]:
    GroupEvaluationScheduler._instance = None
    yield
    GroupEvaluationScheduler._instance = None


def test_evaluation_only_publish_waits_and_batches_followup_session_requests(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation-only publishes wait for inactivity and evaluate the session."""
    monkeypatch.setattr(delayed_group_evaluator, "_EFFECTIVE_DELAY_SECONDS", 1)
    monkeypatch.setattr(group_evaluation_runner, "_EFFECTIVE_DELAY_SECONDS", 1)

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        agent_context_prompt="this is a coding assistant",
        agent_success_config=AgentSuccessConfig(
            evaluation_name="overall_success",
            success_definition_prompt="agent completes the requested task",
            sampling_rate=1.0,
        ),
    )
    instance = Reflexio(
        org_id="org_eval_only_delay",
        configurator=DefaultConfigurator("org_eval_only_delay", config=config),
    )
    storage = instance.request_context.storage
    assert storage is not None

    user_id = "user_eval_only_delay"
    session_id = "session_eval_only_delay"
    agent_version = "agent_eval_only_delay"
    observed_request_model_counts: list[int] = []
    original_run = AgentSuccessEvaluationService.run

    def recording_run(
        self: AgentSuccessEvaluationService,
        request: AgentSuccessEvaluationRequest,
    ) -> None:
        observed_request_model_counts.append(len(request.request_interaction_data_models))
        return original_run(self, request)

    monkeypatch.setattr(AgentSuccessEvaluationService, "run", recording_run)

    with patched_litellm():
        first = instance.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": _interaction_pair("first issue"),
                "source": "integration",
                "agent_version": agent_version,
                "session_id": session_id,
                "evaluation_only": True,
            }
        )
        assert first.success is True
        assert (
            storage.get_agent_success_evaluation_results(
                agent_version=agent_version, limit=10
            )
            == []
        )

        time.sleep(0.55)
        second = instance.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": _interaction_pair("follow-up issue"),
                "source": "integration",
                "agent_version": agent_version,
                "session_id": session_id,
                "evaluation_only": True,
            }
        )
        assert second.success is True

        # This passes the first publish's original fire time but is still before
        # the second publish's rescheduled fire time.
        time.sleep(0.65)
        assert (
            storage.get_agent_success_evaluation_results(
                agent_version=agent_version, limit=10
            )
            == []
        )

        def assert_evaluated_once() -> None:
            results = storage.get_agent_success_evaluation_results(
                agent_version=agent_version, limit=10
            )
            assert len(results) == 1
            assert results[0].session_id == session_id

        _wait_until(assert_evaluated_once)

    stored_requests = storage.get_requests_by_session(user_id, session_id)
    assert len(stored_requests) == 2
    assert all(request.evaluation_only is True for request in stored_requests)
    assert observed_request_model_counts == [2]
