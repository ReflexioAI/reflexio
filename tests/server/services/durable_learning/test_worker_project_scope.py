"""The durable-learning worker must BIND the project its job carries.

D10 put ``project_id`` on the LearningJob payload, but ``_process_job`` claimed
the job and ran the entire extract/persist cycle without ever binding it. The
payload existed and its consumer ignored it, so every row the cycle wrote took
whatever ambient scope happened to be in effect — which, on a worker thread
decoupled in time from the enqueueing request, is none.

Two properties are under test:

1. **The project is live where the work happens.** Asserting at the call site
   proves nothing: a bind that is entered and immediately discarded would still
   satisfy it. The scope is therefore sampled from *inside* the real
   ``compute_deferred_learning`` / ``persist_deferred_learning`` calls and from
   inside the fenced ``complete_learning_job`` write — one frame deeper than
   the binding, in the code that actually touches storage.

2. **A scope failure is visible.** ``_process_job`` funnelled every exception
   into a blanket ``except`` plus a routine ``learning_job_failed`` log, so an
   attribution failure was indistinguishable from an LLM or storage hiccup —
   and no test asserting "the job raises" could go red, because the handler ate
   it one frame above. The test below asserts the escalation instead, and goes
   red if the narrow ``WorkScopeError`` branch is folded back into the blanket
   one.

OSS registers no provider, so all of this is inert for a bare install: an
absent project is normal there, never an error. These tests install a fake
provider to stand in for the enterprise implementation, whose ``bind()``
rejects a scope with no project exactly as enterprise must.
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.extensions import register_service
from reflexio.server.services.durable_learning import worker as worker_module
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.work_scope import (
    WORK_SCOPE_PROVIDER,
    WorkScope,
    WorkScopeError,
    current_project_id,
)


@pytest.fixture(autouse=True)
def _disable_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the local ONNX embedder out of the run (see test_worker.py)."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")


class _FakeProvider:
    """Stands in for the enterprise provider.

    ``bind()`` fails closed on a scope with no project, which is the behaviour
    enterprise owns: a tenant write that cannot be attributed to a project must
    not proceed. OSS registers no provider at all, so this is never OSS's
    behaviour — see ``test_absent_project_is_inert_without_a_provider``.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def current(self) -> WorkScope | None:
        return getattr(self._local, "scope", None)

    @contextmanager
    def _bind(self, scope: WorkScope) -> Iterator[None]:
        if scope.project_id is None:
            raise WorkScopeError(f"no project on deferred work for {scope.org_id}")
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
    p = _FakeProvider()
    register_service(WORK_SCOPE_PROVIDER, p, override=True)
    return p


def _factory(tmp_dir: str):
    def _make(org_id: str) -> RequestContext:
        return RequestContext(org_id=org_id, storage_base_dir=tmp_dir)

    return _make


def _setup_job(
    storage: Any,
    *,
    org_id: str,
    user_id: str,
    request_id: str,
    project_id: str | None,
) -> None:
    """Seed request + interaction + a learning job carrying ``project_id``."""
    req = Request(
        request_id=request_id,
        user_id=user_id,
        session_id="sess1",
        agent_version="v1",
        source="test_src",
    )
    interaction = Interaction(
        user_id=user_id,
        request_id=request_id,
        content="test interaction content",
        embedding=[],
    )
    with storage.commit_scope():
        storage.add_request(req)
        storage.add_user_interactions_bulk(
            user_id, [interaction], embeddings_prepared=True
        )
        storage.enqueue_learning_job(
            org_id=org_id,
            user_id=user_id,
            request_id=request_id,
            covers_through=float(int(time.time())),
            project_id=project_id,
        )


def _sample_scope_inside_the_work(
    monkeypatch: pytest.MonkeyPatch, storage: Any
) -> dict[str, list[str | None]]:
    """Wrap the real work with spies that record the AMBIENT project.

    Each spy delegates to the real implementation, so the full cycle still
    runs; it only samples ``current_project_id()`` from inside. That is the
    point of the assertion — the project must be live in the frames that write,
    not merely passed to a context manager at the call site.
    """
    seen: dict[str, list[str | None]] = {
        "compute": [],
        "persist": [],
        "complete": [],
    }

    real_compute = GenerationService.compute_deferred_learning
    real_persist = GenerationService.persist_deferred_learning
    real_complete = type(storage).complete_learning_job

    def spy_compute(self: GenerationService, *args: Any, **kwargs: Any) -> Any:
        seen["compute"].append(current_project_id())
        return real_compute(self, *args, **kwargs)

    def spy_persist(self: GenerationService, *args: Any, **kwargs: Any) -> Any:
        seen["persist"].append(current_project_id())
        return real_persist(self, *args, **kwargs)

    def spy_complete(self: Any, *args: Any, **kwargs: Any) -> Any:
        seen["complete"].append(current_project_id())
        return real_complete(self, *args, **kwargs)

    monkeypatch.setattr(
        GenerationService, "compute_deferred_learning", spy_compute, raising=True
    )
    monkeypatch.setattr(
        GenerationService, "persist_deferred_learning", spy_persist, raising=True
    )
    monkeypatch.setattr(
        type(storage), "complete_learning_job", spy_complete, raising=True
    )
    return seen


# ---------------------------------------------------------------------------
# 1. A job carrying a project runs with that project bound
# ---------------------------------------------------------------------------


def test_job_with_a_project_runs_with_that_project_bound(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE central assertion: the payload's project is live where rows are written.

    Sampled inside compute, inside persist, and inside the fenced
    complete_learning_job — all one frame below the bind. Drop the
    ``bind_work_scope`` from ``_process_job`` and every sample reads ``None``.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_scoped")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_scoped",
            user_id="u_scoped",
            request_id="req_scoped",
            project_id="proj-a",
        )

        seen = _sample_scope_inside_the_work(monkeypatch, ctx.storage)

        worker = DurableLearningWorker(factory, instance_id="scoped")
        processed = worker.drain_org("org_scoped", batch_size=1, lease_seconds=300)

        assert processed == 1, (
            "the job must complete; a scope failure here would mean the payload's "
            "project never reached the provider"
        )
        assert seen["compute"] == ["proj-a"], (
            f"compute ran unscoped or misattributed: {seen['compute']}"
        )
        assert seen["persist"] == ["proj-a"], (
            f"persist ran unscoped or misattributed: {seen['persist']}"
        )
        assert seen["complete"] == ["proj-a"], (
            f"the fenced completion write ran unscoped: {seen['complete']}"
        )


def test_the_bound_scope_does_not_leak_past_the_job(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding is scoped to the job, not to the worker thread.

    A worker drains many orgs' jobs on one thread; a scope that outlived
    ``_process_job`` would attribute the NEXT job to the previous project.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_leak")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_leak",
            user_id="u_leak",
            request_id="req_leak",
            project_id="proj-b",
        )

        worker = DurableLearningWorker(factory, instance_id="leak")
        assert worker.drain_org("org_leak", batch_size=1, lease_seconds=300) == 1
        assert current_project_id() is None, (
            "the job's project outlived _process_job and would misattribute the next job"
        )


# ---------------------------------------------------------------------------
# 2. A job with NO project must surface a visible failure, not return cleanly
# ---------------------------------------------------------------------------


def test_job_without_a_project_escalates_instead_of_failing_quietly(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observability property.

    Under the blanket ``except`` this path was a routine ``learning_job_failed``
    — the same bucket an LLM timeout lands in — so a whole cycle dropped for
    want of attribution looked like a retryable blip. Restore the blanket
    handler and this test goes red on the empty anomaly list.
    """
    anomalies: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        worker_module,
        "capture_anomaly",
        lambda message, **tags: anomalies.append((message, tags)),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_unscoped")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_unscoped",
            user_id="u_unscoped",
            request_id="req_unscoped",
            project_id=None,
        )

        worker = DurableLearningWorker(factory, instance_id="unscoped")
        processed = worker.drain_org("org_unscoped", batch_size=1, lease_seconds=300)

        assert processed == 0, "an unattributable job must not be reported as done"
        assert [m for m, _ in anomalies] == ["durable_learning.work_scope_failed"], (
            f"scope failure did not surface; anomalies={anomalies}"
        )
        assert anomalies[0][1]["org_id"] == "org_unscoped"
        assert anomalies[0][1]["project_id"] is None

        # Escalated, not propagated: drain_org returned normally, so the daemon
        # loop is intact and the next job still runs. A raise here would kill
        # the worker thread and silently shrink the pool.
        _setup_job(
            ctx.storage,
            org_id="org_unscoped",
            user_id="u_after",
            request_id="req_after",
            project_id="proj-c",
        )
        assert worker.drain_org("org_unscoped", batch_size=5, lease_seconds=300) == 1, (
            "the worker stopped processing after a scope failure"
        )


def test_an_empty_project_is_treated_as_absent_not_as_a_project(
    provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``""`` and unset are the SAME state — do not reintroduce a path where
    they differ.

    A provider reading a transaction-local Postgres GUC gets back ``""``, not
    NULL, on a pooled connection. If an empty project slipped through as a
    distinct value, this job would run "scoped" to a project that does not
    exist instead of escalating.
    """
    anomalies: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        worker_module,
        "capture_anomaly",
        lambda message, **tags: anomalies.append((message, tags)),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_empty")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_empty",
            user_id="u_empty",
            request_id="req_empty",
            project_id="",
        )

        worker = DurableLearningWorker(factory, instance_id="empty")
        assert worker.drain_org("org_empty", batch_size=1, lease_seconds=300) == 0
        assert [m for m, _ in anomalies] == ["durable_learning.work_scope_failed"], (
            f"an empty project was treated as a real one; anomalies={anomalies}"
        )


# ---------------------------------------------------------------------------
# 3. Inert for a bare OSS install
# ---------------------------------------------------------------------------


def test_absent_project_is_inert_without_a_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No provider registered (the OSS case): the same job runs to completion.

    OSS has one org and no projects, so an absent project is normal there. The
    raising above belongs to the enterprise provider behind the seam, not to
    this module.
    """
    anomalies: list[str] = []
    monkeypatch.setattr(
        worker_module,
        "capture_anomaly",
        lambda message, **_tags: anomalies.append(message),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_oss")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_oss",
            user_id="u_oss",
            request_id="req_oss",
            project_id=None,
        )

        worker = DurableLearningWorker(factory, instance_id="oss")
        assert worker.drain_org("org_oss", batch_size=1, lease_seconds=300) == 1, (
            "a projectless job must be ordinary in OSS, not an error"
        )
        assert anomalies == [], f"OSS escalated a normal absence: {anomalies}"
