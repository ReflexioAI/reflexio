"""The regen pool must carry the caller's ContextVars into its workers.

Why this file exists: ``run_regen`` dispatches through a ``ThreadPoolExecutor``,
and a ``ContextVar`` is isolated per thread. Without an explicit context copy
every var the caller bound is absent in the worker -- which in a deployment
that binds a per-project ContextVar means every project-scoped write in the job
fails closed. That shipped, and it failed 100% of sessions with
``agent_success_failed`` while logging nothing but "saved no results".

These tests use a plain local ContextVar rather than the enterprise project
binding: the propagation mechanism is what is under test, and this package must
not import enterprise modules.
"""

from __future__ import annotations

import contextvars
import threading
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.domain.entities import Request
from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    RegenJob,
    run_regen,
)
from reflexio.server.services.agent_success_evaluation.runner import (
    GroupEvaluationOutcome,
)

# Stands in for any request-scoped ContextVar a deployment binds -- the project
# id in enterprise, the correlation id in OSS.
_probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="")


def _stub_storage(descriptors: list[SessionDescriptor]) -> MagicMock:
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
            )
        ]

    storage.get_requests_by_session.side_effect = _by_session
    return storage


def _request_context(storage: MagicMock, concurrency: int = 4) -> MagicMock:
    rc = MagicMock(storage=storage)
    rc.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        eval_concurrency_limit=concurrency,
    )
    return rc


def _job(total: int) -> RegenJob:
    return RegenJob(
        job_id="j-ctx",
        org_id="o",
        from_ts=0,
        to_ts=1,
        status="running",
        total=total,
    )


def test_worker_sees_the_context_var_the_caller_bound() -> None:
    """The propagation guarantee. Fails before the context copy was added."""
    descriptors = [SessionDescriptor("u1", "s1", "v1", "src")]
    rc = _request_context(_stub_storage(descriptors))
    seen: list[str] = []

    def _capture(**_kwargs: object) -> GroupEvaluationOutcome:
        seen.append(_probe.get())
        return GroupEvaluationOutcome("complete", "complete", "fp")

    token = _probe.set("project-77")
    try:
        with patch(
            "reflexio.server.services.agent_success_evaluation"
            ".regen_jobs.run_group_evaluation",
            side_effect=_capture,
        ):
            run_regen(job=_job(1), request_context=rc, llm_client=MagicMock())
    finally:
        _probe.reset(token)

    assert seen == ["project-77"], (
        "worker did not observe the caller's ContextVar -- the pool submitted "
        "without copying context"
    )


def test_workers_run_genuinely_concurrently_under_separate_contexts() -> None:
    """The axis a single-candidate test cannot cover -- and it needs a barrier.

    ``Context.run`` raises "cannot enter context: ... is already entered" only
    while the same Context object is CURRENTLY entered elsewhere. A stubbed
    evaluator returns in microseconds, so workers never actually overlap and a
    single hoisted ``copy_context()`` survives -- verified: that mutant passed
    a version of this test that merely submitted six fast candidates.

    A ``Barrier`` forces real simultaneity: every worker blocks until all of
    them have entered, so a shared Context is necessarily entered twice at
    once. Under that mutant the second worker raises immediately, never reaches
    the barrier, and the others time out -- which surfaces as job failures
    rather than a clean assertion, so both are asserted.
    """
    worker_count = 3
    descriptors = [
        SessionDescriptor("u1", f"s{n}", "v1", "src") for n in range(worker_count)
    ]
    rc = _request_context(_stub_storage(descriptors), concurrency=worker_count)
    barrier = threading.Barrier(worker_count)
    seen: list[str] = []
    lock = threading.Lock()

    def _capture(**_kwargs: object) -> GroupEvaluationOutcome:
        value = _probe.get()
        # Hold every worker here until all of them have arrived. Timeout so a
        # failure is a bounded test failure, not a hung suite.
        barrier.wait(timeout=10)
        with lock:
            seen.append(value)
        return GroupEvaluationOutcome("complete", "complete", "fp")

    token = _probe.set("project-77")
    try:
        with patch(
            "reflexio.server.services.agent_success_evaluation"
            ".regen_jobs.run_group_evaluation",
            side_effect=_capture,
        ):
            job = _job(worker_count)
            run_regen(job=job, request_context=rc, llm_client=MagicMock())
    finally:
        _probe.reset(token)

    assert job.failed == 0, (
        "a worker failed while all were held at the barrier -- the likely "
        f"cause is one Context shared across threads. failures: {job.failures}"
    )
    assert len(seen) == worker_count
    assert set(seen) == {"project-77"}, (
        "at least one concurrent worker lost the ContextVar"
    )


def test_unbound_caller_still_runs() -> None:
    """No caller binding is not an error -- the worker just sees the default.

    Guards against the copy introducing a new failure shape (e.g. raising out
    of ``Context.run``) on the CLI and background paths that bind nothing.
    """
    descriptors = [SessionDescriptor("u1", "s1", "v1", "src")]
    rc = _request_context(_stub_storage(descriptors))
    seen: list[str] = []

    def _capture(**_kwargs: object) -> GroupEvaluationOutcome:
        seen.append(_probe.get())
        return GroupEvaluationOutcome("complete", "complete", "fp")

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".regen_jobs.run_group_evaluation",
        side_effect=_capture,
    ):
        job = _job(1)
        run_regen(job=job, request_context=rc, llm_client=MagicMock())

    assert seen == [""]
    assert job.status == "completed"
    assert job.failed == 0
