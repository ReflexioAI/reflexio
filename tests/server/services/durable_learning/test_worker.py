"""Tests for DurableLearningWorker — Task 5b.

Headline: exactly-once proof under a claim race (real threads).

Test taxonomy:
1. Exactly-once under claim race — profile+playbook counts match a single run.
2. Per-job flags honored — force_extraction/skip_aggregation round-trip.
3. Missing Request → job failed (not dead), not processed.
4. Dead transition — job goes dead after max_attempts threshold.
5. Superseded job — complete_learning_job=0 rolls back profile writes.
6. Failed job is re-claimable (no manual status reset required).

SQLite storage; LLM mocked globally by conftest; embeddings disabled
(REFLEXIO_EMBEDDING_PROVIDER=off) so _get_embedding returns [] immediately
without loading the local ONNX model.

Dead-transition semantics:
  Only claim_learning_jobs increments `attempts` (fail_learning_job does not).
  With max_attempts=3 (DB default):
    Claim 1 → attempts=1 (1<3, not dead) → fail(dead=False) → status='failed'
    Claim 2 → attempts=2 (2<3, not dead) → fail(dead=False) → status='failed'
    Claim 3 → attempts=3 (3>=3, dead)    → fail(dead=True)  → status='dead'
  So the job goes dead after 3 delivery attempts.
"""

from __future__ import annotations

import tempfile
import threading
import time
from unittest import mock

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.generation_service import (
    GenerationService,
    GenerationServiceResult,
)

# ---------------------------------------------------------------------------
# Module-level fixture: disable the local ONNX embedder for all tests.
# EmbeddingUnavailableError is caught at every call site so writes still land,
# just with empty embedding vectors (acceptable for count assertions).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_embeddings(monkeypatch):
    """Set REFLEXIO_EMBEDDING_PROVIDER=off so get_embedding/get_embeddings raise
    EmbeddingUnavailableError immediately (caught by callers -> empty vectors).
    This prevents loading the local ONNXMiniLM model during test runs."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factory(tmp_dir: str):
    """Return a request-context factory pointing at tmp_dir."""

    def _make(org_id: str) -> RequestContext:
        return RequestContext(org_id=org_id, storage_base_dir=tmp_dir)

    return _make


def _setup_job(
    storage,
    *,
    org_id: str,
    user_id: str,
    request_id: str,
    session_id: str = "sess1",
    agent_version: str = "v1",
    source: str = "test_src",
    force_extraction: bool = False,
    skip_aggregation: bool = False,
) -> None:
    """Directly add request + interaction + job to storage without embedding calls."""
    assert storage is not None, "_setup_job requires non-None storage"
    req = Request(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        source=source,
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
            force_extraction=force_extraction,
            skip_aggregation=skip_aggregation,
        )


# ---------------------------------------------------------------------------
# Test 1: Exactly-once under a claim race (the headline)
# ---------------------------------------------------------------------------


def test_exactly_once_under_claim_race():
    """Two workers racing the same job must produce profile+playbook counts
    identical to a single-worker run.

    Strategy:
    - Run a single worker on org_single to record baseline profile/playbook counts.
    - Set up an identical job on org_race; pre-claim with stale token, re-claim
      with live token so both workers hold distinct claim tokens for the same job.
    - Run both workers' _process_job concurrently in real threads.
    - Assert org_race counts == org_single counts (exactly one committed).

    The fenced complete_learning_job (raises _Superseded on rowcount=0 -> rollback)
    guarantees the stale worker's writes do not survive.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)

        # Baseline: single worker run
        ctx_single = factory("org_single")
        assert ctx_single.storage is not None
        _setup_job(
            ctx_single.storage,
            org_id="org_single",
            user_id="u_single",
            request_id="req_single",
        )

        worker_single = DurableLearningWorker(factory, instance_id="single")
        n_processed = worker_single.drain_org(
            "org_single", batch_size=1, lease_seconds=300
        )
        assert n_processed == 1, "single worker must complete the job"

        baseline_profiles = ctx_single.storage.count_all_profiles()
        baseline_playbooks = ctx_single.storage.count_user_playbooks()

        # Race: two workers, same job
        ctx_race = factory("org_race")
        assert ctx_race.storage is not None
        _setup_job(
            ctx_race.storage,
            org_id="org_race",
            user_id="u_race",
            request_id="req_race",
        )

        # Claim with an immediately-expired lease so a second claim can supersede it.
        [stale_job] = ctx_race.storage.claim_learning_jobs(
            claimed_by="pre_stale", limit=1, lease_seconds=-1
        )
        # Re-claim: the stale lease is already expired -> this gets the live token.
        [live_job] = ctx_race.storage.claim_learning_jobs(
            claimed_by="pre_live", limit=1, lease_seconds=300
        )

        assert stale_job.job_id == live_job.job_id, "same job re-claimed"
        assert stale_job.claim_token != live_job.claim_token, "tokens must differ"

        worker_stale = DurableLearningWorker(factory, instance_id="w_stale")
        worker_live = DurableLearningWorker(factory, instance_id="w_live")

        errors: list[BaseException] = []

        def run_stale() -> None:
            try:
                worker_stale._process_job(factory("org_race"), stale_job)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def run_live() -> None:
            try:
                worker_live._process_job(factory("org_race"), live_job)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t_stale = threading.Thread(target=run_stale, name="worker-stale")
        t_live = threading.Thread(target=run_live, name="worker-live")
        t_stale.start()
        t_live.start()
        t_stale.join(timeout=60)
        t_live.join(timeout=60)

        assert not t_stale.is_alive(), "stale worker thread timed out"
        assert not t_live.is_alive(), "live worker thread timed out"
        assert not errors, f"Worker threads raised: {errors}"

        # Assert exactly-once: counts must match baseline
        race_profiles = ctx_race.storage.count_all_profiles()
        race_playbooks = ctx_race.storage.count_user_playbooks()

        assert race_profiles == baseline_profiles, (
            f"Exactly-once violated: race produced {race_profiles} profiles "
            f"but baseline is {baseline_profiles}. "
            "The stale worker's writes must have been rolled back."
        )
        assert race_playbooks == baseline_playbooks, (
            f"Exactly-once violated: race produced {race_playbooks} playbooks "
            f"but baseline is {baseline_playbooks}. "
            "The stale worker's writes must have been rolled back."
        )


# ---------------------------------------------------------------------------
# Test 2: Per-job flags honored (force_extraction, skip_aggregation)
# ---------------------------------------------------------------------------


def test_per_job_flags_honored():
    """force_extraction and skip_aggregation from the job are passed to
    run_deferred_learning -- not hardcoded False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_flags")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_flags",
            user_id="u_flags",
            request_id="req_flags",
            force_extraction=True,
            skip_aggregation=True,
        )

        jobs = ctx.storage.claim_learning_jobs(
            claimed_by="flag_worker", limit=1, lease_seconds=300
        )
        assert jobs, "expected a job"
        job = jobs[0]
        assert job.force_extraction is True, "force_extraction must be True in job"
        assert job.skip_aggregation is True, "skip_aggregation must be True in job"

        call_kwargs: dict = {}

        def spy_run_deferred(
            _self,
            *,
            user_id,
            request_id,
            session_id,
            source,
            agent_version,
            force_extraction,
            skip_aggregation,
            sequential=False,
        ):
            call_kwargs.update(
                force_extraction=force_extraction,
                skip_aggregation=skip_aggregation,
            )
            return GenerationServiceResult(request_id=request_id)

        worker = DurableLearningWorker(factory, instance_id="flag_w")

        with mock.patch.object(
            GenerationService, "run_deferred_learning", spy_run_deferred
        ):
            worker._process_job(factory("org_flags"), job)

        assert call_kwargs.get("force_extraction") is True, (
            "force_extraction=True must reach run_deferred_learning"
        )
        assert call_kwargs.get("skip_aggregation") is True, (
            "skip_aggregation=True must reach run_deferred_learning"
        )


# ---------------------------------------------------------------------------
# Test 3: Missing Request fails the job (not dead on first attempt)
# ---------------------------------------------------------------------------


def test_missing_request_fails_job():
    """If the stored Request for latest_request_id is absent, the job is failed
    without calling run_deferred_learning (prevents a run with wrong agent_version).
    On the first attempt (attempts=1 < max_attempts=3) it is failed, not dead."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_missing")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_missing",
            user_id="u_missing",
            request_id="req_missing",
        )

        ctx.storage.delete_request("req_missing")
        assert ctx.storage.get_request("req_missing") is None

        [job] = ctx.storage.claim_learning_jobs(
            claimed_by="missing_w", limit=1, lease_seconds=300
        )

        ran_learning = False

        def _should_not_run(*args, **kwargs):
            nonlocal ran_learning
            ran_learning = True
            return  # type: ignore[return-value]

        worker = DurableLearningWorker(factory, instance_id="missing_w")

        with mock.patch.object(
            GenerationService, "run_deferred_learning", _should_not_run
        ):
            result = worker._process_job(factory("org_missing"), job)

        assert result is False, "missing request must not be counted as processed"
        assert not ran_learning, (
            "run_deferred_learning must NOT be called when Request is absent"
        )

        # First failure: attempts=1 < max_attempts=3 -> status='failed' (not dead).
        # get_learning_status_for_request returns "pending" for a failed non-dead job.
        status = ctx.storage.get_learning_status_for_request(
            user_id="u_missing",
            request_created_at=0.0,
        )
        assert status == "pending", (
            f"after first failure (not dead): expected 'pending', got {status!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Dead transition after max_attempts threshold
# ---------------------------------------------------------------------------


def test_dead_transition_after_max_attempts():
    """Job goes dead after exactly max_attempts (3) delivery attempts.

    Only claim_learning_jobs increments attempts; fail_learning_job does not.
    With max_attempts=3 (DB default):
      Cycle 1: claim -> attempts=1 (1<3, not dead) -> fail(dead=False) -> status='failed'
      Cycle 2: claim -> attempts=2 (2<3, not dead) -> fail(dead=False) -> status='failed'
      Cycle 3: claim -> attempts=3 (3>=3, dead)    -> fail(dead=True)  -> status='dead'
    Failed jobs are re-claimable naturally (no manual status reset needed).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_dead")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_dead",
            user_id="u_dead",
            request_id="req_dead",
        )

        worker = DurableLearningWorker(factory, instance_id="dead_w")

        def _always_raise(
            _self,
            *,
            user_id,
            request_id,
            session_id,
            source,
            agent_version,
            force_extraction,
            skip_aggregation,
            sequential=False,
        ):
            raise RuntimeError("simulated extraction failure")

        # Cycle 1: claim -> attempts=1 (1<3, not dead) -> fail(dead=False) -> status='failed'
        [job1] = ctx.storage.claim_learning_jobs(
            claimed_by="dead_w", limit=1, lease_seconds=300
        )
        assert job1.attempts == 1, f"cycle 1: expected attempts=1, got {job1.attempts}"
        with mock.patch.object(
            GenerationService, "run_deferred_learning", _always_raise
        ):
            r1 = worker._process_job(factory("org_dead"), job1)
        assert r1 is False

        # status='failed' (not dead); get_learning_status returns "pending"
        assert (
            ctx.storage.get_learning_status_for_request(
                user_id="u_dead", request_created_at=0.0
            )
            == "pending"
        ), "cycle 1 fail must leave job as 'pending' (reclaimable)"

        # Cycle 2: failed job is re-claimable without manual reset
        # claim -> attempts=2 (2<3, not dead) -> fail(dead=False) -> status='failed'
        [job2] = ctx.storage.claim_learning_jobs(
            claimed_by="dead_w", limit=1, lease_seconds=300
        )
        assert job2.attempts == 2, f"cycle 2: expected attempts=2, got {job2.attempts}"
        with mock.patch.object(
            GenerationService, "run_deferred_learning", _always_raise
        ):
            r2 = worker._process_job(factory("org_dead"), job2)
        assert r2 is False

        assert (
            ctx.storage.get_learning_status_for_request(
                user_id="u_dead", request_created_at=0.0
            )
            == "pending"
        ), "cycle 2 fail must leave job as 'pending' (reclaimable)"

        # Cycle 3: claim -> attempts=3 (3>=3, dead=True) -> fail(dead=True) -> status='dead'
        [job3] = ctx.storage.claim_learning_jobs(
            claimed_by="dead_w", limit=1, lease_seconds=300
        )
        assert job3.attempts == 3, f"cycle 3: expected attempts=3, got {job3.attempts}"
        with mock.patch.object(
            GenerationService, "run_deferred_learning", _always_raise
        ):
            r3 = worker._process_job(factory("org_dead"), job3)
        assert r3 is False

        # Job must now be dead
        assert (
            ctx.storage.get_learning_status_for_request(
                user_id="u_dead", request_created_at=0.0
            )
            == "failed"
        ), "cycle 3 dead-fail must surface as 'failed'"

        # Dead job must NOT be reclaimable
        still_claimable = ctx.storage.claim_learning_jobs(
            claimed_by="post_dead", limit=5, lease_seconds=300
        )
        assert not any(j.user_id == "u_dead" for j in still_claimable), (
            "dead job must not be reclaimable"
        )


# ---------------------------------------------------------------------------
# Test 5: Superseded job rolls back profile/playbook writes
# ---------------------------------------------------------------------------


def test_superseded_job_does_not_commit_outputs():
    """A worker whose complete_learning_job returns 0 (stale token) must NOT
    commit profile/playbook writes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_supersede")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_supersede",
            user_id="u_super",
            request_id="req_super",
        )

        # Claim with stale token (lease_seconds=-1 -> already expired).
        [stale_job] = ctx.storage.claim_learning_jobs(
            claimed_by="w_stale", limit=1, lease_seconds=-1
        )
        # Re-claim to supersede; the stale token is now invalid.
        [live_job] = ctx.storage.claim_learning_jobs(  # noqa: F841
            claimed_by="w_live", limit=1, lease_seconds=300
        )
        assert stale_job.claim_token != live_job.claim_token

        profiles_before = ctx.storage.count_all_profiles()

        worker = DurableLearningWorker(factory, instance_id="w_super")
        result = worker._process_job(factory("org_supersede"), stale_job)

        assert result is False, "_process_job must return False when superseded"
        profiles_after = ctx.storage.count_all_profiles()
        assert profiles_after == profiles_before, (
            "superseded worker must NOT commit profile writes: "
            f"before={profiles_before}, after={profiles_after}"
        )


# ---------------------------------------------------------------------------
# Test 6: Failed job is re-claimable without manual status reset
# ---------------------------------------------------------------------------


def test_failed_job_is_reclaimable():
    """A failed (not dead) job must be re-claimable directly.

    claim_learning_jobs includes status='failed' in its predicate, so a job
    that fails once is naturally picked up on the next poll without any manual
    status reset.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_reclaim")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_reclaim",
            user_id="u_reclaim",
            request_id="req_reclaim",
        )

        # Claim and fail the job (not dead — attempts=1 < max_attempts=3)
        [job1] = ctx.storage.claim_learning_jobs(
            claimed_by="w_reclaim", limit=1, lease_seconds=300
        )
        assert job1.attempts == 1
        assert job1.claim_token is not None
        ctx.storage.fail_learning_job(
            job_id=job1.job_id, claim_token=job1.claim_token, dead=False
        )

        # Without any manual reset, the job must be immediately re-claimable
        reclaimed = [
            j
            for j in ctx.storage.claim_learning_jobs(
                claimed_by="w_reclaim2", limit=5, lease_seconds=300
            )
            if j.user_id == "u_reclaim"
        ]
        assert len(reclaimed) == 1, "failed job must be re-claimable"
        assert reclaimed[0].attempts == 2, (
            f"re-claimed job must have attempts=2, got {reclaimed[0].attempts}"
        )
        assert reclaimed[0].status == "claimed"


# ---------------------------------------------------------------------------
# Test 7: schedule_tagging is called on the sequential (durable) path
# ---------------------------------------------------------------------------


def test_run_deferred_learning_schedules_tagging():
    """schedule_tagging must be called even when sequential=True (durable-worker path).

    Previously the sequential branch in _run_learning_steps returned early before
    reaching schedule_tagging — this test guards that regression.
    """
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = RequestContext(org_id="org_tag_seq", storage_base_dir=tmp_dir)
        gen = GenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=ctx,
        )

        tagging_calls: list[dict] = []

        def _mock_tagging(**kwargs):
            tagging_calls.append(kwargs)

        with (
            mock.patch(
                "reflexio.server.services.generation_service.schedule_tagging",
                side_effect=_mock_tagging,
            ),
            mock.patch(
                "reflexio.server.services.generation_service.ProfileGenerationService"
            ),
            mock.patch(
                "reflexio.server.services.generation_service.PlaybookGenerationService"
            ),
            mock.patch.object(gen, "_maybe_run_reflection"),
        ):
            gen.run_deferred_learning(
                user_id="u_tag",
                request_id="r_tag",
                session_id="s_tag",
                source="test",
                agent_version="v1",
                force_extraction=False,
                skip_aggregation=False,
                sequential=True,
            )

        assert len(tagging_calls) == 1, (
            f"schedule_tagging must be called exactly once on the sequential path, "
            f"got {len(tagging_calls)} calls"
        )
