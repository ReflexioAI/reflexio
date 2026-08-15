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
from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.base_generation_service import PreparedGenerationRun
from reflexio.server.services.deferred_learning_plan import (
    DeferredLearningPlan,
    GenerationComputePlan,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
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


def _org_operation_state_count(storage, org_id: str) -> int:
    """Count operation-state rows belonging to ``org_id``.

    get_all_operation_states() returns GLOBAL rows (SQLite shares one db file
    across orgs; org_id is embedded in each ``service_name`` key, e.g.
    ``profile_generation::<org>::<user>::lock``), so scope by the org marker to
    compare a single run against a race like-for-like.
    """
    return sum(
        1
        for s in storage.get_all_operation_states()
        if f"::{org_id}::" in s["service_name"]
    )


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


def _seed_completed_agent_run(
    storage,
    *,
    org_id: str,
    request_id: str,
    run_id: str,
    extractor_kind: str,
) -> None:
    storage.create_agent_run(
        AgentRunRecord(
            id=run_id,
            binding=AgentBinding(
                org_id=org_id,
                extractor_kind=extractor_kind,
                user_id="test_user",
                request_id=request_id,
                agent_version="v1",
                source="test_src",
            ),
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={"request_id": request_id},
            committed_output={f"{extractor_kind}s": []},
        )
    )


def _deferred_plan_with_runs(
    *,
    request_id: str,
    profile_run_id: str,
    playbook_run_id: str,
) -> DeferredLearningPlan:
    def generation_plan(run_id: str, extractor_name: str) -> GenerationComputePlan:
        return GenerationComputePlan(
            prepared=PreparedGenerationRun(
                extractor_config=mock.Mock(),
                extractor_name=extractor_name,
                identifier=f"{extractor_name}_generation",
            ),
            generated_count=0,
            billable_count=0,
            write_plan=None,
            bookmark_advance=None,
            generation_start=0.0,
            extraction_run_ids=[run_id],
            token_totals=None,
        )

    return DeferredLearningPlan(
        request_id=request_id,
        user_id="test_user",
        agent_version="v1",
        lock_acquired=True,
        profile=(mock.Mock(), generation_plan(profile_run_id, "profile")),
        playbook=(mock.Mock(), generation_plan(playbook_run_id, "playbook")),
    )


def _assert_runs_abandoned(storage, *run_ids: str) -> None:
    for run_id in run_ids:
        run = storage.get_agent_run(run_id)
        assert run is not None and run.status == AgentRunStatus.FAILED
    assert (
        storage.claim_finalization_failed_agent_run(
            org_id=storage.org_id,
            worker_id="resume-worker",
            now=datetime.now(UTC) + timedelta(days=1),
        )
        is None
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
        # Escaping side effects (lineage events + operation-state / bookmark rows):
        # a single clean run is the oracle. These are the writes that, on the
        # networked backends, escape the fence via _rpc/_table auto-commit; on
        # SQLite (single-connection atomic) the loser's copies always roll back,
        # so org_race must match the org_single baseline exactly.
        #
        # Scoping note: SQLite stores every org in one shared db file (org_id is a
        # column). count_all_profiles/count_user_playbooks already filter by org,
        # but get_lineage_events()/get_all_operation_states() return GLOBAL rows,
        # so we scope both to the org under test (org_id filter for lineage; a
        # service_name substring filter for operation-state locks) — otherwise the
        # baseline, captured before org_race exists, would spuriously differ.
        # Count-based comparison: org_single/org_race rows differ in
        # event_id/updated_at/request_id, so only the counts are the invariant.
        baseline_events = len(
            ctx_single.storage.get_lineage_events(org_id="org_single")
        )
        baseline_bookmarks = _org_operation_state_count(
            ctx_single.storage, "org_single"
        )

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

        # Build each worker's context BEFORE starting the threads: parallel
        # cold construction of SQLite storage on one db file is not safe
        # (concurrent schema DDL) — the server serializes it behind
        # get_reflexio's construction lock, which this direct factory
        # bypasses. The race under test is the claim-token fence, not
        # construction.
        ctx_stale = factory("org_race")
        ctx_live = factory("org_race")

        def run_stale() -> None:
            try:
                worker_stale._process_job(ctx_stale, stale_job)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def run_live() -> None:
            try:
                worker_live._process_job(ctx_live, live_job)
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

        # Exactly-once must ALSO hold for the ESCAPING side effects: the loser's
        # lineage events and operation-state / bookmark advance must have rolled
        # back with the fence, leaving org_race's counts identical to the
        # single-run oracle. (On the networked backends these auto-commit and leak
        # pre-fix — that is what gate (a) fixes; SQLite is the always-green parity
        # oracle: two SQLite connections race, but the loser's fenced scope rolls
        # back its whole transaction.)
        race_events = len(ctx_race.storage.get_lineage_events(org_id="org_race"))
        race_bookmarks = _org_operation_state_count(ctx_race.storage, "org_race")

        assert race_events == baseline_events, (
            f"Exactly-once violated: race produced {race_events} lineage events "
            f"but baseline is {baseline_events}. The stale worker's lineage "
            "events must have rolled back (no doubled events)."
        )
        assert race_bookmarks == baseline_bookmarks, (
            f"Exactly-once violated: race produced {race_bookmarks} operation-state "
            f"rows but baseline is {baseline_bookmarks}. The stale worker's "
            "bookmark / operation-state advance must have rolled back."
        )


# ---------------------------------------------------------------------------
# Test 2: Per-job flags honored (force_extraction, skip_aggregation)
# ---------------------------------------------------------------------------


def test_per_job_flags_honored():
    """force_extraction and skip_aggregation from the job reach the worker's
    compute entry point (compute_deferred_learning) -- not hardcoded False.

    Post gate-b the worker drives compute_deferred_learning (not
    run_deferred_learning), so the flags are asserted at that seam."""
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

        def spy_compute_deferred(
            _self,
            *,
            user_id,
            request_id,
            session_id,
            source,
            agent_version,
            force_extraction,
            skip_aggregation,
        ):
            call_kwargs.update(
                force_extraction=force_extraction,
                skip_aggregation=skip_aggregation,
            )
            # An all-None (lock_acquired=True) plan makes persist + emit no-ops,
            # so the worker fences + completes cleanly without touching storage.
            return DeferredLearningPlan(
                request_id=request_id,
                user_id=user_id,
                agent_version=agent_version,
                lock_acquired=True,
                profile=None,
                playbook=None,
            )

        worker = DurableLearningWorker(factory, instance_id="flag_w")

        with mock.patch.object(
            GenerationService, "compute_deferred_learning", spy_compute_deferred
        ):
            worker._process_job(factory("org_flags"), job)

        assert call_kwargs.get("force_extraction") is True, (
            "force_extraction=True must reach compute_deferred_learning"
        )
        assert call_kwargs.get("skip_aggregation") is True, (
            "skip_aggregation=True must reach compute_deferred_learning"
        )


# ---------------------------------------------------------------------------
# Test 3: Missing Request fails the job (not dead on first attempt)
# ---------------------------------------------------------------------------


def test_missing_request_fails_job():
    """If the stored Request for latest_request_id is absent, the job is failed
    without driving compute (prevents a run with wrong agent_version).
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
            GenerationService, "compute_deferred_learning", _should_not_run
        ):
            result = worker._process_job(factory("org_missing"), job)

        assert result is False, "missing request must not be counted as processed"
        assert not ran_learning, (
            "compute_deferred_learning must NOT be called when Request is absent"
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

        def _always_raise(*_args, **_kwargs):
            raise RuntimeError("simulated extraction failure")

        # Cycle 1: claim -> attempts=1 (1<3, not dead) -> fail(dead=False) -> status='failed'
        [job1] = ctx.storage.claim_learning_jobs(
            claimed_by="dead_w", limit=1, lease_seconds=300
        )
        assert job1.attempts == 1, f"cycle 1: expected attempts=1, got {job1.attempts}"
        with mock.patch.object(
            GenerationService, "compute_deferred_learning", _always_raise
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
            GenerationService, "compute_deferred_learning", _always_raise
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
            GenerationService, "compute_deferred_learning", _always_raise
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


def test_persist_failure_abandons_computed_agent_runs():
    """A rolled-back durable persist cannot leave resumable agent runs behind."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_persist_failure")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_persist_failure",
            user_id="test_user",
            request_id="req_persist_failure",
        )
        run_ids = ("profile-persist-failure", "playbook-persist-failure")
        _seed_completed_agent_run(
            ctx.storage,
            org_id="org_persist_failure",
            request_id="req_persist_failure",
            run_id=run_ids[0],
            extractor_kind="profile",
        )
        _seed_completed_agent_run(
            ctx.storage,
            org_id="org_persist_failure",
            request_id="req_persist_failure",
            run_id=run_ids[1],
            extractor_kind="playbook",
        )
        [job] = ctx.storage.claim_learning_jobs(
            claimed_by="persist-failure-worker", limit=1, lease_seconds=300
        )
        plan = _deferred_plan_with_runs(
            request_id="req_persist_failure",
            profile_run_id=run_ids[0],
            playbook_run_id=run_ids[1],
        )

        with (
            mock.patch.object(
                GenerationService, "compute_deferred_learning", return_value=plan
            ),
            mock.patch.object(
                GenerationService,
                "persist_deferred_learning",
                side_effect=RuntimeError("persist failed"),
            ),
        ):
            processed = DurableLearningWorker(factory)._process_job(
                factory("org_persist_failure"), job
            )

        assert processed is False
        _assert_runs_abandoned(ctx.storage, *run_ids)


def test_superseded_job_abandons_computed_agent_runs():
    """Fence loss abandons every run computed by the superseded attempt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_run_supersede")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_run_supersede",
            user_id="test_user",
            request_id="req_run_supersede",
        )
        [stale_job] = ctx.storage.claim_learning_jobs(
            claimed_by="stale-worker", limit=1, lease_seconds=-1
        )
        [live_job] = ctx.storage.claim_learning_jobs(
            claimed_by="live-worker", limit=1, lease_seconds=300
        )
        assert stale_job.claim_token != live_job.claim_token

        run_ids = ("profile-run-supersede", "playbook-run-supersede")
        _seed_completed_agent_run(
            ctx.storage,
            org_id="org_run_supersede",
            request_id="req_run_supersede",
            run_id=run_ids[0],
            extractor_kind="profile",
        )
        _seed_completed_agent_run(
            ctx.storage,
            org_id="org_run_supersede",
            request_id="req_run_supersede",
            run_id=run_ids[1],
            extractor_kind="playbook",
        )
        plan = _deferred_plan_with_runs(
            request_id="req_run_supersede",
            profile_run_id=run_ids[0],
            playbook_run_id=run_ids[1],
        )

        with mock.patch.object(
            GenerationService, "compute_deferred_learning", return_value=plan
        ):
            processed = DurableLearningWorker(factory)._process_job(
                factory("org_run_supersede"), stale_job
            )

        assert processed is False
        _assert_runs_abandoned(ctx.storage, *run_ids)


def test_post_commit_emit_failure_preserves_completed_job_and_agent_runs():
    """A post-commit side-effect failure cannot undo the durable winner."""
    from reflexio.models.api_schema.domain.entities import LineageContext
    from reflexio.models.api_schema.service_schemas import UserProfile
    from reflexio.server.services.deferred_learning_plan import (
        ProfileWritePlan,
    )
    from reflexio.server.services.profile.service import ProfileGenerationService

    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_post_commit_emit_failure")
        assert ctx.storage is not None
        _setup_job(
            ctx.storage,
            org_id="org_post_commit_emit_failure",
            user_id="test_user",
            request_id="req_post_commit_emit_failure",
            force_extraction=True,
            skip_aggregation=True,
        )
        [job] = ctx.storage.claim_learning_jobs(
            claimed_by="post-commit-worker", limit=1, lease_seconds=300
        )

        computed_plans: list[DeferredLearningPlan] = []
        run_id = "profile-post-commit-emit-failure"
        _seed_completed_agent_run(
            ctx.storage,
            org_id="org_post_commit_emit_failure",
            request_id="req_post_commit_emit_failure",
            run_id=run_id,
            extractor_kind="profile",
        )

        def build_plan(self, **_kwargs):
            profile = UserProfile(
                profile_id="profile-post-commit",
                user_id="test_user",
                content="This durable profile survived a post-commit emit failure.",
                last_modified_timestamp=1_000,
                generated_from_request_id="req_post_commit_emit_failure",
            )
            generation_plan = GenerationComputePlan(
                prepared=PreparedGenerationRun(
                    extractor_config=mock.Mock(),
                    identifier="profile_generation",
                    extractor_name="profile",
                ),
                generated_count=1,
                billable_count=1,
                write_plan=ProfileWritePlan(
                    user_id="test_user",
                    request_id="req_post_commit_emit_failure",
                    new_profiles=[profile],
                    superseded_ids=[],
                    lineage_contexts=[LineageContext(op_kind="create")],
                ),
                bookmark_advance=None,
                generation_start=0.0,
                extraction_run_ids=[run_id],
                token_totals=None,
            )
            plan = DeferredLearningPlan(
                request_id="req_post_commit_emit_failure",
                user_id="test_user",
                agent_version="v1",
                lock_acquired=True,
                profile=(
                    ProfileGenerationService(self.client, self.request_context),
                    generation_plan,
                ),
                playbook=None,
            )
            computed_plans.append(plan)
            return plan

        def emit_then_raise(self, plan):
            assert plan.profile is not None
            for extraction_run_id in plan.profile[1].extraction_run_ids:
                self.request_context.storage.update_agent_run_status(
                    extraction_run_id,
                    AgentRunStatus.FINALIZED,
                    pending_tool_call_ids=[],
                )
            raise RuntimeError("post-commit emit failed")

        with (
            mock.patch.object(
                GenerationService,
                "compute_deferred_learning",
                build_plan,
            ),
            mock.patch.object(
                GenerationService,
                "emit_deferred_learning_side_effects",
                emit_then_raise,
            ),
        ):
            processed = DurableLearningWorker(factory)._process_job(
                factory("org_post_commit_emit_failure"), job
            )

        assert len(computed_plans) == 1
        plan = computed_plans[0]
        assert plan.profile is not None
        generation_plan = plan.profile[1]
        assert generation_plan.finalization_result is not None
        assert generation_plan.finalization_result.won_receipt is True
        run = ctx.storage.get_agent_run(run_id)
        assert run is not None and run.status == AgentRunStatus.FINALIZED
        assert ctx.storage.get_agent_run_finalization_receipt(
            run_id=run_id,
            entity_type="profile",
        ) == ["profile-post-commit"]

        assert processed is True
        assert ctx.storage.count_learning_jobs_by_status("done") == 1
        assert ctx.storage.count_all_profiles() == 1


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
# Test 6b: same-user contention refunds the attempt → attempts stays bounded
# ---------------------------------------------------------------------------


def test_contention_refund_keeps_attempts_bounded():
    """Repeated same-user contention must NOT inflate ``attempts`` (F4 blocker).

    On the contention path (``compute_deferred_learning`` returns
    ``lock_acquired=False``) the worker fails the job with
    ``refund_attempt=True``, so ``claim_learning_jobs``'s ``attempts += 1`` is
    refunded and each contention cycle nets to zero. Without the refund a job
    that loses the per-user race every ~2s poll while the holder runs its ~60s
    compute would blow past ``max_attempts=3`` in seconds and dead-letter on its
    first real transient error with zero retries.

    Proof: run far more contention cycles than ``max_attempts``, assert every
    claim still sees ``attempts == 1``, then let the job finally win the lock but
    hit a transient persist error once — it must remain reclaimable (``failed``,
    not ``dead``) because its retry budget was never burned by contention.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_refund")
        assert ctx.storage is not None

        _setup_job(
            ctx.storage,
            org_id="org_refund",
            user_id="u_refund",
            request_id="req_refund",
        )

        worker = DurableLearningWorker(factory, instance_id="refund_w")

        def _contention(_self, *, user_id, request_id, agent_version, **_kwargs):
            # lock_acquired=False -> worker takes the contention requeue path.
            return DeferredLearningPlan(
                request_id=request_id,
                user_id=user_id,
                agent_version=agent_version,
                lock_acquired=False,
                profile=None,
                playbook=None,
            )

        # Many more contention cycles than max_attempts (3): each must net to 0.
        for cycle in range(6):
            [job] = ctx.storage.claim_learning_jobs(
                claimed_by="refund_w", limit=1, lease_seconds=300
            )
            assert job.attempts == 1, (
                f"cycle {cycle}: attempts must stay bounded at 1 "
                f"(prior cycle refunded to 0), got {job.attempts}"
            )
            with mock.patch.object(
                GenerationService, "compute_deferred_learning", _contention
            ):
                assert worker._process_job(factory("org_refund"), job) is False
            # Still reclaimable (failed, not dead) with full retry budget.
            assert (
                ctx.storage.get_learning_status_for_request(
                    user_id="u_refund", request_created_at=0.0
                )
                == "pending"
            ), f"cycle {cycle}: contention requeue must stay reclaimable"

        # The job finally WINS the lock but hits a transient persist error once.
        # attempts stayed bounded, so dead = attempts >= max_attempts is False:
        # the job is merely 'failed' (reclaimable), NOT dead-lettered.
        [win_job] = ctx.storage.claim_learning_jobs(
            claimed_by="refund_w", limit=1, lease_seconds=300
        )
        assert win_job.attempts == 1, (
            f"winning claim must still have attempts=1 after 6 contention cycles, "
            f"got {win_job.attempts}"
        )

        def _transient_raise(*_a, **_k):
            raise RuntimeError("transient persist error")

        with mock.patch.object(
            GenerationService, "compute_deferred_learning", _transient_raise
        ):
            assert worker._process_job(factory("org_refund"), win_job) is False

        # Not dead — retry budget intact despite many contention cycles.
        assert (
            ctx.storage.get_learning_status_for_request(
                user_id="u_refund", request_created_at=0.0
            )
            == "pending"
        ), "job must remain reclaimable (not dead) — contention did not burn attempts"


# ---------------------------------------------------------------------------
# Test 7: schedule_tagging is called on the sequential (durable) path
# ---------------------------------------------------------------------------


def test_run_deferred_learning_schedules_tagging():
    """run_deferred_learning must schedule tagging exactly once.

    Guards against a regression where a code path returned early before reaching
    schedule_tagging at the end of _run_learning_steps.
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
        ):
            gen.run_deferred_learning(
                user_id="u_tag",
                request_id="r_tag",
                session_id="s_tag",
                source="test",
                agent_version="v1",
                force_extraction=False,
                skip_aggregation=False,
            )

        assert len(tagging_calls) == 1, (
            f"schedule_tagging must be called exactly once, "
            f"got {len(tagging_calls)} calls"
        )
