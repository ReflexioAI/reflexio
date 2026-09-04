"""Deferred work must carry its project, and must not swallow a scope failure.

Two properties are under test, and both are about attribution surviving the gap
between the request that queues work and the thread that runs it:

1. **Coalescing must not merge projects.** The debounce schedulers deliberately
   collapse repeated enqueues for one key into a single fire. If the key omits
   the project, two projects in one org publishing for the same user inside one
   window collapse into ONE callback, attributed to whichever request won the
   race. The tests below fail against a project-less key.

2. **A scope failure must surface.** Each deferred path used to funnel every
   exception into a blanket ``except`` + log. While that stands, no test
   asserting "the job raises" can go red, because the handler eats it one frame
   above the storage call. The tests below assert the escalation instead, and
   go red if the narrow ``WorkScopeError`` branch is folded back into the
   blanket one.

OSS registers no provider, so all of this is inert for a bare install: these
tests install a fake provider to stand in for the enterprise implementation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server import callback_executor
from reflexio.server.callback_executor import BoundedCallbackExecutor
from reflexio.server.extensions import register_service
from reflexio.server.services import generation_service as generation_service_module
from reflexio.server.services import publish_learning_worker as plw
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.playbook_optimizer import scheduler as pb_sched
from reflexio.server.services.shadow_comparison import worker as shadow_worker
from reflexio.server.services.tagging import tagging_scheduler
from reflexio.server.work_scope import (
    WORK_SCOPE_PROVIDER,
    WorkScope,
    WorkScopeError,
    bind_work_scope,
    current_project_id,
)


class _FakeProvider:
    """Stands in for the enterprise provider: a thread-local current scope."""

    def __init__(self, *, bind_raises: bool = False) -> None:
        self._local = threading.local()
        self._bind_raises = bind_raises

    def current(self) -> WorkScope | None:
        return getattr(self._local, "scope", None)

    @contextmanager
    def _bind(self, scope: WorkScope) -> Iterator[None]:
        if self._bind_raises:
            raise WorkScopeError(f"no project bound for {scope.org_id}")
        previous = getattr(self._local, "scope", None)
        self._local.scope = scope
        try:
            yield
        finally:
            self._local.scope = previous

    def bind(self, scope: WorkScope) -> Any:
        return self._bind(scope)


@pytest.fixture
def provider() -> _FakeProvider:
    """A provider that binds normally (the enterprise happy path)."""
    p = _FakeProvider()
    register_service(WORK_SCOPE_PROVIDER, p, override=True)
    return p


@pytest.fixture
def failing_provider() -> _FakeProvider:
    """A provider whose bind() rejects the scope, as enterprise does when a
    tenant write would otherwise be attributed to no project."""
    p = _FakeProvider(bind_raises=True)
    register_service(WORK_SCOPE_PROVIDER, p, override=True)
    return p


# ---------------------------------------------------------------------------
# 1. Coalescing must not merge two projects into one callback
# ---------------------------------------------------------------------------


def test_tagging_debounce_does_not_coalesce_two_projects(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE central assertion: same org/user/agent, two projects, one window.

    Against the old ``(org_id, user_id, agent_version)`` key the second schedule
    overwrites the first and exactly ONE callback fires.
    """
    monkeypatch.setattr(tagging_scheduler, "_EFFECTIVE_DELAY_SECONDS", 0.01)
    scheduler = tagging_scheduler.TaggingScheduler()

    class _FakeSchedulerClass:
        @staticmethod
        def get_instance() -> tagging_scheduler.TaggingScheduler:
            return scheduler

    monkeypatch.setattr(tagging_scheduler, "TaggingScheduler", _FakeSchedulerClass)
    # Neutralise the callback's real work; the key construction is what is
    # under test, and it must come from schedule_tagging itself.
    monkeypatch.setattr(tagging_scheduler, "RequestContext", lambda **_kwargs: object())

    fired: list[str | None] = []
    fired_lock = threading.Lock()

    class _FakeTaggingService:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> None:
            # Record the project the scheduler re-binds at FIRE time — that is
            # what attribution actually depends on.
            with fired_lock:
                fired.append(current_project_id())

    monkeypatch.setattr(tagging_scheduler, "TaggingService", _FakeTaggingService)

    for project in ("proj-a", "proj-b"):
        with bind_work_scope(WorkScope(org_id="org-1", project_id=project)):
            tagging_scheduler.schedule_tagging(
                org_id="org-1",
                user_id="user-1",
                agent_version="v1",
                request_context=None,  # type: ignore[arg-type]
                llm_client=None,  # type: ignore[arg-type]
            )

    assert scheduler.drain(timeout_seconds=5.0), "scheduler did not settle"

    with fired_lock:
        assert sorted(p for p in fired if p is not None) == ["proj-a", "proj-b"], (
            f"expected one callback per project, got {fired} — the debounce key "
            "coalesced two projects into a single fire"
        )


def test_group_evaluation_key_keeps_projects_distinct(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives the REAL key construction in GenerationService, not a hand-built key.

    A test that assembles the key itself would still pass after the project
    component was dropped from the production call site.
    """
    keys: list[tuple[Any, ...]] = []

    class _Recorder:
        def schedule(self, key: tuple[Any, ...], callback: Any) -> None:
            keys.append(key)

    recorder = _Recorder()

    class _FakeSchedulerClass:
        @staticmethod
        def get_instance() -> _Recorder:
            return recorder

    # Patch the name in the module that RESOLVES it. Patching
    # GroupEvaluationScheduler.get_instance itself did not survive full-suite
    # ordering, and the real singleton ran while the recorder stayed empty --
    # a green-looking test that measured nothing.
    monkeypatch.setattr(
        generation_service_module, "GroupEvaluationScheduler", _FakeSchedulerClass
    )

    fake_self = SimpleNamespace(
        org_id="org-1",
        storage=None,
        request_context=None,
        llm_client=None,
        client=None,
        _sampled_evaluation_families=lambda **_kwargs: (True, False),
    )
    new_request = SimpleNamespace(
        session_id="sess-1", request_id="req-1", evaluation_only=False
    )

    for project in ("proj-a", "proj-b"):
        with bind_work_scope(WorkScope(org_id="org-1", project_id=project)):
            GenerationService._schedule_group_evaluation_if_needed(
                fake_self,  # type: ignore[arg-type]
                new_request=new_request,  # type: ignore[arg-type]
                user_id="user-1",
                agent_version="v1",
                source=None,
            )

    assert len(set(keys)) == 2, (
        "two projects sharing org/user/session produced the same group-evaluation "
        f"key, so they would collapse into one fire: {keys}"
    )


def test_playbook_optimization_enqueue_keeps_projects_distinct(
    provider: _FakeProvider,
) -> None:
    scheduler = pb_sched.PlaybookOptimizationScheduler()
    target = pb_sched.PlaybookOptimizationTarget(kind="user_playbook", target_id=7)

    for project in ("proj-a", "proj-b"):
        with bind_work_scope(WorkScope(org_id="org-1", project_id=project)):
            scheduler.enqueue(org_id="org-1", target=target, callback=lambda: None)

    assert len(scheduler._scheduled) == 2, (
        "two projects optimizing the same target collapsed into one scheduled "
        f"run: {list(scheduler._scheduled)}"
    )


def test_enqueue_time_project_is_captured_not_fire_time(
    provider: _FakeProvider,
) -> None:
    """The key must record the project of the request that enqueued it.

    Resolving at fire time would return whichever request won the race — which
    is precisely the misattribution the payload/key change exists to prevent.
    """
    with bind_work_scope(WorkScope(org_id="org-1", project_id="proj-a")):
        captured = current_project_id()
    assert captured == "proj-a"
    # Outside the scope the ambient answer is gone; only the captured one survives.
    assert current_project_id() is None


# ---------------------------------------------------------------------------
# 2. A scope failure must surface on each deferred path
# ---------------------------------------------------------------------------


def test_callback_executor_escalates_a_scope_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anomalies: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        callback_executor,
        "capture_anomaly",
        lambda message, **tags: anomalies.append((message, tags)),
    )
    executor = BoundedCallbackExecutor(workers=1, queue_size=4)

    def raising() -> None:
        raise WorkScopeError("no project bound")

    executor.submit("deferred-work", raising)
    assert executor.drain(timeout_seconds=5.0), "executor did not settle"

    assert [m for m, _ in anomalies] == ["callback_executor.work_scope_failed"], (
        f"scope failure did not surface; anomalies={anomalies}"
    )

    # The pool must still be alive: escalation, not propagation. A worker that
    # died here would take all deferred work down with it after 16 failures.
    ran = threading.Event()
    executor.submit("after-failure", ran.set)
    assert ran.wait(timeout=5.0), "worker thread died on a scope failure"


def test_shadow_comparison_worker_escalates_a_scope_failure(
    failing_provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    anomalies: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        shadow_worker,
        "capture_anomaly",
        lambda message, **tags: anomalies.append((message, tags)),
    )
    worker = shadow_worker.ShadowComparisonWorker(worker_count=1, queue_size=4)
    worker.enqueue(
        shadow_worker.ShadowComparisonJob(
            org_id="org-1",
            interactions=[],
            session_id="sess-1",
            agent_version="v1",
            project_id="proj-a",
        )
    )

    deadline = time.monotonic() + 5.0
    while not anomalies and time.monotonic() < deadline:
        time.sleep(0.01)

    assert [m for m, _ in anomalies] == ["shadow_comparison.work_scope_failed"], (
        f"scope failure did not surface; anomalies={anomalies}"
    )
    assert anomalies[0][1]["project_id"] == "proj-a"


def test_publish_learning_worker_escalates_a_scope_failure(
    failing_provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    anomalies: list[tuple[str, dict[str, Any]]] = []
    events: list[str] = []
    monkeypatch.setattr(
        plw,
        "capture_anomaly",
        lambda message, **tags: anomalies.append((message, tags)),
    )
    monkeypatch.setattr(
        plw,
        "record_usage_event",
        lambda **kwargs: events.append(kwargs["event_name"]),
    )

    worker = plw.PublishLearningWorker(worker_count=1)
    worker._process_job(
        plw.PublishLearningJob(
            org_id="org-1",
            user_id="user-1",
            request_id="req-1",
            session_id="sess-1",
            source=None,
            agent_version="v1",
            force_extraction=False,
            skip_aggregation=False,
            project_id="proj-a",
        )
    )

    assert [m for m, _ in anomalies] == ["publish_learning.work_scope_failed"], (
        f"scope failure did not surface; anomalies={anomalies}"
    )
    # It must NOT be filed as a routine learning failure — that is the bucket
    # ordinary LLM/storage hiccups land in, where a dropped job is invisible.
    assert events == ["learning_scope_failed"], (
        f"scope failure was misfiled as a routine outcome: {events}"
    )


# ---------------------------------------------------------------------------
# 3. All of the above stays inert for a bare OSS install
# ---------------------------------------------------------------------------


def test_absent_project_is_normal_without_a_provider() -> None:
    """No provider registered (the OSS case): no scope, no error, no raise."""
    assert current_project_id() is None
    with bind_work_scope(WorkScope(org_id="org-1", project_id=None)):
        assert current_project_id() is None
    with bind_work_scope(None):
        assert current_project_id() is None


def test_empty_project_is_the_same_state_as_absent() -> None:
    """Unset and empty must not be two distinct projects.

    A provider reading a transaction-local Postgres GUC gets back the empty
    string, not NULL, when the GUC is unset on a pooled connection. If that
    reached the key/payload unnormalised, an empty project would debounce
    separately from an absent one, and would be stored as an empty string
    rather than NULL on the job row.
    """
    assert WorkScope(org_id="org-1", project_id="").project_id is None
    # Same value, therefore same debounce identity — not two separate fires.
    assert WorkScope(org_id="org-1", project_id="") == WorkScope(org_id="org-1")


def test_provider_returning_empty_string_is_normalised() -> None:
    """Even a provider that bypasses WorkScope's own coercion is normalised."""

    class _RawProvider:
        def current(self) -> Any:
            return SimpleNamespace(org_id="org-1", project_id="")

        def bind(self, scope: WorkScope) -> Any:
            raise AssertionError("not used")

    register_service(WORK_SCOPE_PROVIDER, _RawProvider(), override=True)
    assert current_project_id() is None
