"""Integration coverage for evaluation-only publish scheduling."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import InteractionData
from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    StorageConfigSQLite,
)
from reflexio.server.services.agent_success_evaluation import runner, scheduler
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.scheduler import (
    GroupEvaluationScheduler,
)
from reflexio.server.services.agent_success_evaluation.service import (
    AgentSuccessEvaluationService,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.test_support.llm_mock import patched_litellm


class _SchedulerWaiting(BaseException):
    """Stop a synchronous scheduler pass when it would wait for time or work."""


def _interaction_pair(label: str) -> list[InteractionData]:
    return [
        InteractionData(content=f"Please help with {label}", role="User"),
        InteractionData(
            content=f"Resolved {label} with steps A and B.", role="Assistant"
        ),
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
    monkeypatch.setattr(scheduler, "_EFFECTIVE_DELAY_SECONDS", 10)
    monkeypatch.setattr(runner, "_EFFECTIVE_DELAY_SECONDS", 10)

    # Run the real heap/dispatch loop synchronously at explicit deadlines. Patch
    # module references, never global time/threading functions used by SQLite or
    # other background services. No scheduler daemon escapes this test.
    clock = SimpleNamespace(now=0.0)
    wake_event = MagicMock()
    wake_event.wait.side_effect = _SchedulerWaiting
    monkeypatch.setattr(
        scheduler,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=time.sleep),
    )
    monkeypatch.setattr(
        scheduler,
        "threading",
        SimpleNamespace(
            Lock=threading.Lock, Event=lambda: wake_event, Thread=MagicMock()
        ),
    )
    callbacks: list[Callable[[], None]] = []
    monkeypatch.setattr(
        scheduler, "submit_callback", lambda _name, callback: callbacks.append(callback)
    )
    group_scheduler = GroupEvaluationScheduler.get_instance()

    def drain_scheduler() -> None:
        with pytest.raises(_SchedulerWaiting):
            group_scheduler._scheduler_loop()

    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "reflexio.db")),
        agent_context_prompt="this is a coding assistant",
        agent_success_config=AgentSuccessConfig(
            success_definition_prompt="agent completes the requested task",
            sampling_rate=0.0,
            evaluation_only_sampling_rate=1.0,
            retrieved_learning_sampling_rate=0.0,
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
    retrieved_learning_run = MagicMock()

    def recording_run(
        self: AgentSuccessEvaluationService,
        request: AgentSuccessEvaluationRequest,
    ) -> None:
        observed_request_model_counts.append(
            len(request.request_interaction_data_models)
        )
        return original_run(self, request)

    monkeypatch.setattr(AgentSuccessEvaluationService, "run", recording_run)
    monkeypatch.setattr(
        runner, "_run_retrieved_learning_evaluation", retrieved_learning_run
    )

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

        key = (instance.org_id, None, user_id, session_id)
        assert group_scheduler._scheduled[key][0] == 10.0
        drain_scheduler()
        assert callbacks == []

        clock.now = 5.0
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
        assert group_scheduler._scheduled[key][0] == 15.0
        latest_created_at = max(
            request.created_at
            for request in storage.get_requests_by_session(user_id, session_id)
        )
        # Keep the runner's separate wall-clock inactivity check enabled and
        # aligned to the latest real stored request and its virtual publish time.
        monkeypatch.setattr(
            runner,
            "datetime",
            SimpleNamespace(
                now=lambda _tz: datetime.fromtimestamp(
                    latest_created_at + clock.now - 5.0, tz=UTC
                )
            ),
        )

        ordinary_session_id = "session_ordinary"
        ordinary = instance.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": _interaction_pair("ordinary issue"),
                "source": "integration",
                "agent_version": agent_version,
                "session_id": ordinary_session_id,
            },
            defer_learning=True,
        )
        assert ordinary.success is True

        # Consume the superseded first deadline; the follow-up must still wait.
        clock.now = 10.0
        drain_scheduler()
        assert callbacks == []
        assert group_scheduler._heap == [(15.0, key)]
        clock.now = 14.0
        drain_scheduler()
        assert callbacks == []
        # Even an explicitly attempted early callback must respect the runner's
        # independent inactivity gate; the real judge cannot run yet.
        group_scheduler._run_callback(key, group_scheduler._scheduled[key][1])
        assert observed_request_model_counts == []
        assert (
            storage.get_agent_success_evaluation_results(
                agent_version=agent_version, limit=10
            )
            == []
        )

        clock.now = 15.0
        drain_scheduler()
        assert len(callbacks) == 1
        callbacks[0]()
        clock.now = 25.0
        drain_scheduler()
        assert len(callbacks) == 1
        assert group_scheduler._scheduled == {}
        assert group_scheduler._heap == []

        results = storage.get_agent_success_evaluation_results(
            agent_version=agent_version, limit=10
        )
        assert [result.session_id for result in results] == [session_id]
        assert results[0].trajectory_interaction_count == 4

    stored_requests = storage.get_requests_by_session(user_id, session_id)
    assert len(stored_requests) == 2
    assert all(request.evaluation_only is True for request in stored_requests)
    assert observed_request_model_counts == [2]
    retrieved_learning_run.assert_not_called()
