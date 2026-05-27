"""Unit tests for RegenJobRegistry and run_regen."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    RegenJob,
    RegenJobRegistry,
    run_regen,
)


def test_registry_create_returns_job_and_records_active():
    reg = RegenJobRegistry()
    job = reg.create(
        org_id="org1", evaluation_name="overall", from_ts=0, to_ts=100, total=5
    )
    assert isinstance(job, RegenJob)
    assert job.total == 5 and job.status == "running"
    assert reg.has_active("org1", "overall")
    assert reg.get(job.job_id) is job


def test_registry_rejects_second_active_for_same_org_evaluator():
    reg = RegenJobRegistry()
    reg.create(org_id="o", evaluation_name="e", from_ts=0, to_ts=1, total=0)
    with pytest.raises(RuntimeError, match="already running"):
        reg.create(org_id="o", evaluation_name="e", from_ts=0, to_ts=1, total=0)


def test_registry_allows_different_evaluator():
    reg = RegenJobRegistry()
    reg.create(org_id="o", evaluation_name="e1", from_ts=0, to_ts=1, total=0)
    reg.create(org_id="o", evaluation_name="e2", from_ts=0, to_ts=1, total=0)


def test_cancel_sets_event_and_status():
    reg = RegenJobRegistry()
    job = reg.create(org_id="o", evaluation_name="e", from_ts=0, to_ts=1, total=0)
    reg.cancel(job.job_id)
    assert job.cancel_event.is_set()


def test_evict_completed_older_than_drops_old_jobs():
    reg = RegenJobRegistry()
    job = reg.create(org_id="o", evaluation_name="e", from_ts=0, to_ts=1, total=0)
    job.status = "completed"
    job.finished_at = time.monotonic() - 7200
    reg.evict_completed_older_than(3600)
    assert reg.get(job.job_id) is None
    assert not reg.has_active("o", "e")


def test_run_regen_processes_sessions_and_marks_completed():
    descriptors = [
        SessionDescriptor("u1", "s1", "v1", "src"),
        SessionDescriptor("u1", "s2", "v1", "src"),
    ]
    storage = MagicMock()
    storage.get_session_ids_in_window.return_value = descriptors
    rc = MagicMock(storage=storage)
    llm = MagicMock()
    job = RegenJob(
        job_id="j1",
        org_id="o",
        evaluation_name="e",
        from_ts=0,
        to_ts=1,
        status="running",
        total=2,
    )

    with patch(
        "reflexio.server.services.agent_success_evaluation.regen_jobs.run_group_evaluation"
    ) as runner:
        run_regen(job=job, request_context=rc, llm_client=llm)
        assert runner.call_count == 2
        first_call = runner.call_args_list[0]
        assert first_call.kwargs["force_regenerate"] is True
        assert first_call.kwargs["evaluation_name"] == "e"

    assert job.status == "completed"
    assert job.completed == 2
    assert job.failed == 0


def test_run_regen_records_failures_and_continues():
    storage = MagicMock()
    storage.get_session_ids_in_window.return_value = [
        SessionDescriptor("u", "good", "v1", ""),
        SessionDescriptor("u", "bad", "v1", ""),
        SessionDescriptor("u", "good2", "v1", ""),
    ]
    rc = MagicMock(storage=storage)
    job = RegenJob(
        job_id="j",
        org_id="o",
        evaluation_name="e",
        from_ts=0,
        to_ts=1,
        status="running",
        total=3,
    )

    def runner(**kwargs):
        if kwargs["session_id"] == "bad":
            raise RuntimeError("LLM timeout")

    with patch(
        "reflexio.server.services.agent_success_evaluation.regen_jobs.run_group_evaluation",
        side_effect=runner,
    ):
        run_regen(job=job, request_context=rc, llm_client=MagicMock())

    assert job.status == "completed"
    assert job.completed == 2
    assert job.failed == 1
    assert job.failures[0].session_id == "bad"
    assert "LLM timeout" in job.failures[0].reason


def test_run_regen_observes_cancel_between_sessions():
    storage = MagicMock()
    storage.get_session_ids_in_window.return_value = [
        SessionDescriptor("u", f"s{i}", "v1", "") for i in range(5)
    ]
    rc = MagicMock(storage=storage)
    job = RegenJob(
        job_id="j",
        org_id="o",
        evaluation_name="e",
        from_ts=0,
        to_ts=1,
        status="running",
        total=5,
    )
    call_count = {"n": 0}

    def runner(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            job.cancel_event.set()

    with patch(
        "reflexio.server.services.agent_success_evaluation.regen_jobs.run_group_evaluation",
        side_effect=runner,
    ):
        run_regen(job=job, request_context=rc, llm_client=MagicMock())

    assert job.status == "cancelled"
    assert job.completed == 2
