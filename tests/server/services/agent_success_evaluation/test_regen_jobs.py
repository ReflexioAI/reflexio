"""Unit tests for RegenJobRegistry and run_regen."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    RegenJob,
    RegenJobRegistry,
    run_regen,
)


def _stub_storage(descriptors: list[SessionDescriptor]) -> MagicMock:
    """Build a MagicMock storage that satisfies the F3 sampler.

    Returns the supplied descriptors from ``get_session_ids_in_window`` and
    one synthetic Request per (user, session) from ``get_requests_by_session``
    so the sampler can read a first-request created_at and metadata dict.
    """
    storage = MagicMock()
    storage.get_session_ids_in_window.return_value = descriptors

    def _by_session(user_id: str, session_id: str) -> list[Request]:
        return [
            Request(
                request_id=f"req-{session_id}",
                user_id=user_id,
                created_at=1_700_000_000,
                source="src",
                agent_version="v1",
                session_id=session_id,
                metadata={},
            )
        ]

    storage.get_requests_by_session.side_effect = _by_session
    return storage


def _request_context(storage: MagicMock) -> MagicMock:
    """Wire a MagicMock request_context that exposes the storage and a
    real Config (sampler reads ``eval_sample_n_per_stratum`` and
    ``eval_concurrency_limit`` off it).
    """
    rc = MagicMock(storage=storage)
    rc.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    return rc


def test_registry_create_returns_job_and_records_active():
    reg = RegenJobRegistry()
    job = reg.create(org_id="org1", from_ts=0, to_ts=100, total=5)
    assert isinstance(job, RegenJob)
    assert job.total == 5 and job.status == "running"
    assert reg.has_active("org1")
    assert reg.get(job.job_id) is job


def test_registry_rejects_second_active_for_same_org():
    reg = RegenJobRegistry()
    reg.create(org_id="o", from_ts=0, to_ts=1, total=0)
    with pytest.raises(RuntimeError, match="already running"):
        reg.create(org_id="o", from_ts=0, to_ts=1, total=0)


def test_cancel_sets_event_and_status():
    reg = RegenJobRegistry()
    job = reg.create(org_id="o", from_ts=0, to_ts=1, total=0)
    reg.cancel(job.job_id)
    assert job.cancel_event.is_set()


def test_evict_completed_older_than_drops_old_jobs():
    reg = RegenJobRegistry()
    job = reg.create(org_id="o", from_ts=0, to_ts=1, total=0)
    job.status = "completed"
    job.finished_at = time.monotonic() - 7200
    reg.evict_completed_older_than(3600)
    assert reg.get(job.job_id) is None
    assert not reg.has_active("o")


def test_run_regen_processes_sessions_and_marks_completed():
    descriptors = [
        SessionDescriptor("u1", "s1", "v1", "src"),
        SessionDescriptor("u1", "s2", "v1", "src"),
    ]
    storage = _stub_storage(descriptors)
    rc = _request_context(storage)
    llm = MagicMock()
    job = RegenJob(
        job_id="j1",
        org_id="o",
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

    assert job.status == "completed"
    assert job.completed == 2
    assert job.failed == 0


def test_run_regen_records_failures_and_continues():
    storage = _stub_storage(
        [
            SessionDescriptor("u", "good", "v1", ""),
            SessionDescriptor("u", "bad", "v1", ""),
            SessionDescriptor("u", "good2", "v1", ""),
        ]
    )
    rc = _request_context(storage)
    job = RegenJob(
        job_id="j",
        org_id="o",
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
    storage = _stub_storage(
        [SessionDescriptor("u", f"s{i}", "v1", "") for i in range(5)]
    )
    rc = _request_context(storage)
    job = RegenJob(
        job_id="j",
        org_id="o",
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


def test_create_succeeds_after_previous_job_completed_within_ttl():
    """Bug fix: a completed job in the registry must NOT block a new regen.

    Pre-fix: ``has_active`` and ``create`` both checked
    ``key in _by_org``, which stayed populated until TTL eviction
    (default 1h). That meant users couldn't regenerate again for an hour
    after a successful run. After the fix, only a job whose ``status`` is
    still ``"running"`` blocks new submissions.
    """
    from reflexio.server.services.agent_success_evaluation.regen_jobs import (
        RegenJobRegistry,
    )

    reg = RegenJobRegistry()
    first = reg.create(org_id="o", from_ts=0, to_ts=1, total=2)
    first.status = "completed"
    # Registry still holds the finished job (TTL hasn't run); has_active must
    # report False because nothing is running anymore.
    assert reg.has_active("o") is False
    # And a fresh create must succeed instead of raising.
    second = reg.create(org_id="o", from_ts=0, to_ts=1, total=3)
    assert second.job_id != first.job_id
    assert second.status == "running"
    assert reg.has_active("o") is True
    # The previous completed job is still fetchable until TTL eviction.
    assert reg.get(first.job_id) is not None


def test_create_still_blocks_when_previous_job_is_running():
    """Sanity-check: the gate still fires when the previous job is actually running."""
    from reflexio.server.services.agent_success_evaluation.regen_jobs import (
        RegenJobRegistry,
    )

    reg = RegenJobRegistry()
    reg.create(org_id="o", from_ts=0, to_ts=1, total=2)
    import pytest

    with pytest.raises(RuntimeError, match="already running"):
        reg.create(org_id="o", from_ts=0, to_ts=1, total=2)
