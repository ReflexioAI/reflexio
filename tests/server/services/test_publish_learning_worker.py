from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.operation_limiter import (
    operation_limit,
    reset_operation_limiters_for_tests,
)
from reflexio.server.services.publish_learning_worker import (
    PublishLearningJob,
    PublishLearningWorker,
    reset_publish_learning_worker_for_tests,
)
from reflexio.server.tracing import configure_tracer
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder


@pytest.fixture(autouse=True)
def _reset_worker_limiters_and_metrics():
    reset_publish_learning_worker_for_tests()
    reset_operation_limiters_for_tests()
    configure_tracer(None)
    configure_usage_event_recorder(None)
    yield
    reset_publish_learning_worker_for_tests()
    reset_operation_limiters_for_tests()
    configure_tracer(None)
    configure_usage_event_recorder(None)


def _job(
    *,
    org_id: str = "org_1",
    request_id: str = "req_1",
    enqueued_at: float | None = None,
) -> PublishLearningJob:
    return PublishLearningJob(
        org_id=org_id,
        user_id="user_1",
        request_id=request_id,
        session_id="sess_1",
        source="test",
        agent_version="v1",
        force_extraction=False,
        skip_aggregation=False,
        **({} if enqueued_at is None else {"enqueued_at": enqueued_at}),
    )


def test_enqueue_over_warning_threshold_keeps_job_and_records_pressure():
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = PublishLearningWorker(queue_warn_size=1, worker_count=0)

    assert worker.enqueue(_job(request_id="req_1")) is True
    assert worker.enqueue(_job(request_id="req_2")) is True

    assert worker._queue.qsize() == 2
    assert any(event.event_name == "learning_queue_pressure" for event in events)


def test_limiter_timeout_requeues_without_publish_timeout_warning(monkeypatch, caplog):
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_TIMEOUT_SECONDS", "0.01")
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = PublishLearningWorker(queue_warn_size=10, worker_count=0)
    job = _job()

    with (
        operation_limit("org_1", "publish", timeout_seconds=0.1),
        caplog.at_level(logging.WARNING, logger="reflexio.server.operation_limiter"),
    ):
        worker._process_job(job)

    assert "publish_limiter_timeout" not in caplog.text
    assert worker._queue.get_nowait() == job
    assert any(event.event_name == "limiter_acquire_timeout" for event in events)
    assert any(event.event_name == "learning_requeued_limiter_busy" for event in events)


def test_requeued_learning_runs_after_limiter_capacity_opens(monkeypatch):
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_TIMEOUT_SECONDS", "0.01")
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = PublishLearningWorker(queue_warn_size=10, worker_count=0)
    job = _job()

    with operation_limit("org_1", "publish", timeout_seconds=0.1):
        worker._process_job(job)

    assert worker._queue.get_nowait() == job

    mock_reflexio = MagicMock()
    mock_reflexio.llm_client = MagicMock()
    mock_reflexio.request_context = MagicMock()
    with (
        patch(
            "reflexio.server.services.publish_learning_worker.get_reflexio",
            return_value=mock_reflexio,
        ),
        patch(
            "reflexio.server.services.publish_learning_worker.GenerationService"
        ) as generation_service_cls,
    ):
        worker._process_job(job)

    generation_service_cls.return_value.run_deferred_learning.assert_called_once()
    assert any(event.event_name == "learning_requeued_limiter_busy" for event in events)
    assert any(event.event_name == "learning_succeeded" for event in events)
