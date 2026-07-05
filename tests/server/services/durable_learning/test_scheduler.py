"""Tests for DurableLearningScheduler — Task 6.

Headline: one tick discovers orgs-with-work and drains each via the worker,
proving multi-ref capability (two org_ids backed by two storages, both drained
in a single _run_once).

SQLite storage; LLM mocked globally by conftest; embeddings disabled
(REFLEXIO_EMBEDDING_PROVIDER=off) so extraction writes land with empty vectors
without loading the local ONNX model.
"""

from __future__ import annotations

import tempfile
import time

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.scheduler import (
    DurableLearningScheduler,
)


@pytest.fixture(autouse=True)
def _disable_embeddings(monkeypatch):
    """Disable the local ONNX embedder so extraction runs without model load."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")


def _factory(base_dir: str):
    """Return a request-context factory pointing every org at ``base_dir``."""

    def _make(org_id: str) -> RequestContext:
        return RequestContext(org_id=org_id, storage_base_dir=base_dir)

    return _make


def _routing_factory(dir_by_org: dict[str, str]):
    """Return a factory that routes each org to its own storage base dir.

    Models the enterprise cross-ref case in miniature: each org resolves to a
    distinct storage (a distinct SQLite DB file == a distinct data ref).
    """

    def _make(org_id: str) -> RequestContext:
        return RequestContext(org_id=org_id, storage_base_dir=dir_by_org[org_id])

    return _make


def _seed_pending_job(
    storage,
    *,
    org_id: str,
    user_id: str,
    request_id: str,
) -> None:
    """Persist a request + interaction + a pending learning job (no embeddings)."""
    assert storage is not None
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
        )


def _still_pending(storage, user_id: str) -> bool:
    """Whether ``user_id`` still has a claimable (pending/failed) job."""
    claimed = [
        j
        for j in storage.claim_learning_jobs(
            claimed_by="probe", limit=10, lease_seconds=300
        )
        if j.user_id == user_id
    ]
    return bool(claimed)


# ---------------------------------------------------------------------------
# Test 1: gated off by default (flag unset → start() spawns no thread)
# ---------------------------------------------------------------------------


def test_scheduler_gated_off_by_default(monkeypatch):
    """With REFLEXIO_DURABLE_LEARNING_QUEUE unset/false, start() must not spawn
    the daemon thread (``_should_start`` vetoes)."""
    monkeypatch.delenv("REFLEXIO_DURABLE_LEARNING_QUEUE", raising=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        scheduler = DurableLearningScheduler(
            request_context_factory=_factory(tmp_dir),
            org_ids_provider=lambda: [],
        )
        scheduler.start()
        try:
            assert not scheduler.is_running(), (
                "scheduler must not start when the queue flag is off"
            )
        finally:
            scheduler.stop()


# ---------------------------------------------------------------------------
# Test 2: one tick drains the discovered orgs
# ---------------------------------------------------------------------------


def test_run_once_drains_discovered_orgs():
    """_run_once drives the worker over each org the provider yields; a seeded
    pending job is drained to status='done' and the poll interval returned."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_a")
        assert ctx.storage is not None
        _seed_pending_job(
            ctx.storage, org_id="org_a", user_id="u_a", request_id="req_a"
        )

        scheduler = DurableLearningScheduler(
            request_context_factory=factory,
            org_ids_provider=lambda: ["org_a"],
        )
        interval = scheduler._run_once()

        assert interval == 2.0, "default poll interval is 2.0s"
        assert (
            ctx.storage.get_learning_status_for_request(
                user_id="u_a", request_created_at=1.0
            )
            == "done"
        ), "the discovered org's pending job must reach status='done' after the tick"


# ---------------------------------------------------------------------------
# Test 3: one tick drains MULTIPLE sources (multi-ref capability)
# ---------------------------------------------------------------------------


def test_scheduler_drains_multiple_sources_in_one_tick():
    """A provider yielding two org_ids backed by two distinct storages (two DB
    files == two data refs) must have BOTH drained in a single _run_once.

    Proves the mechanism supports per-ref fan-out without wiring the enterprise
    cross-ref enumerator (that provider is a deferred follow-up).
    """
    with (
        tempfile.TemporaryDirectory() as dir_a,
        tempfile.TemporaryDirectory() as dir_b,
    ):
        factory = _routing_factory({"org_a": dir_a, "org_b": dir_b})

        ctx_a = factory("org_a")
        ctx_b = factory("org_b")
        assert ctx_a.storage is not None and ctx_b.storage is not None
        _seed_pending_job(
            ctx_a.storage, org_id="org_a", user_id="u_a", request_id="req_a"
        )
        _seed_pending_job(
            ctx_b.storage, org_id="org_b", user_id="u_b", request_id="req_b"
        )

        scheduler = DurableLearningScheduler(
            request_context_factory=factory,
            org_ids_provider=lambda: ["org_a", "org_b"],
        )
        scheduler._run_once()

        assert (
            ctx_a.storage.get_learning_status_for_request(
                user_id="u_a", request_created_at=1.0
            )
            == "done"
        ), "org_a's job must reach status='done'"
        assert (
            ctx_b.storage.get_learning_status_for_request(
                user_id="u_b", request_created_at=1.0
            )
            == "done"
        ), "org_b's job (a second ref) must also reach status='done' in the same tick"


# ---------------------------------------------------------------------------
# Test 4: one org failing does not abort the tick (per-org isolation)
# ---------------------------------------------------------------------------


def test_run_once_isolates_per_org_failures():
    """A provider whose first org raises inside the worker must not prevent the
    second, healthy org from being drained."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_ok")
        assert ctx.storage is not None
        _seed_pending_job(
            ctx.storage, org_id="org_ok", user_id="u_ok", request_id="req_ok"
        )

        def _provider():
            yield "org_boom"  # no storage seeded for this org, but drain is safe
            yield "org_ok"

        # Make the boom org explode in the factory to exercise isolation.
        real_factory = factory

        def _explode_factory(org_id: str):
            if org_id == "org_boom":
                raise RuntimeError("simulated per-org failure")
            return real_factory(org_id)

        scheduler = DurableLearningScheduler(
            request_context_factory=_explode_factory,
            org_ids_provider=_provider,
        )
        scheduler._run_once()

        assert not _still_pending(factory("org_ok").storage, "u_ok"), (
            "healthy org must still be drained after a prior org raised"
        )
