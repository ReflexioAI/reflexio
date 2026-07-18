"""Gate (b) — compute/persist split invariant + atomicity contract tests (Task 1).

These are written FIRST (TDD RED). They exercise the compute→persist→side-effects
seam that Tasks 2-9 build:

    GenerationService.compute_deferred_learning(...)          -> DeferredLearningPlan
    GenerationService.persist_deferred_learning(plan)         -> None
    GenerationService.emit_deferred_learning_side_effects(plan)-> None
    DurableLearningWorker._process_job(ctx, job)              -> bool  (rewired Task 9)

None of the three new GenerationService methods exist yet, so every test that
references them fails with AttributeError until the implementation lands. That is
the correct Task-1 outcome (RED). ``test_side_effect_identity_vs_baseline`` is a
deliberate ``pytest.skip`` placeholder that Task 9 un-skips once a golden can be
captured against the real split path (see its docstring).

Invariants under test (spec §3, Round-3 folds F1/F3/F4):
- Compute issues NO learning DB-write (F3 tripwire — the 10 storage terminals).
- Persist issues NO LLM / embedding / document-expansion.
- The worker's commit_scope wraps ONLY persist (compute before, side-effects after).
- The extractor stride-bookmark advances IFF the persist rows commit (F1).
- Same-user contention leaves the job reclaimable and produces no duplicate (F4).
- A superseded (fence-lost) job commits nothing and emits no billing/tagging.

SQLite storage in ``tmp_path`` (single-connection-atomic, gate-(a)-independent);
``litellm.completion`` is mocked globally by the root conftest.
"""

from __future__ import annotations

import json
import tempfile
import time

import litellm
import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.base_generation_service import BaseGenerationService
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.generation_service import GenerationService

# ---------------------------------------------------------------------------
# F3 — the COMPLETE compute write-tripwire.
#
# Every learning-write storage terminal that compute must NOT trip. All ten
# names are confirmed to resolve on SQLiteStorage (grep of
# reflexio/server/services/storage/sqlite_storage/**). The bookmark's storage
# terminal is upsert_operation_state (via OperationStateManager.update_extractor
# _bookmark) — included below. The enterprise contract (Task 10) additionally
# adds "lineage_status_change_and_log" (the supabase/postgres supersede terminal).
# ---------------------------------------------------------------------------
_LEARNING_WRITES = (
    "add_user_profile",
    "supersede_profiles_by_ids",
    "delete_profiles_by_ids",
    "save_user_playbooks",
    "supersede_user_playbooks_by_ids",
    "delete_user_playbooks_by_ids",
    "merge_records",
    "supersede_record",
    "update_operation_state",
    "upsert_operation_state",  # bookmark terminal
)


@pytest.fixture(autouse=True)
def _disable_embeddings(monkeypatch):
    """Disable the local ONNX embedder (REFLEXIO_EMBEDDING_PROVIDER=off) so the
    embedding path short-circuits without loading the model. Mirrors
    test_worker.py — count/ordering assertions are unaffected."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factory(tmp_dir: str):
    def _make(org_id: str) -> RequestContext:
        return RequestContext(org_id=org_id, storage_base_dir=tmp_dir)

    return _make


def _make_gen(ctx: RequestContext) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=ctx,
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
    force_extraction: bool = True,
    skip_aggregation: bool = False,
) -> None:
    """Seed request + interaction + enqueue a learning job (no embedding calls).

    ``force_extraction`` defaults to True so a single seeded interaction still
    drives the extractor (bypassing the stride-size pre-filter, which would
    otherwise gate a 1-interaction window off and make the LLM/profile paths
    these tests exercise a no-op)."""
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


def _forbid_writes(storage, monkeypatch, phase: str) -> None:
    """Trip an AssertionError if any learning-write terminal fires during ``phase``."""
    for name in _LEARNING_WRITES:

        def spy(*_a, _n=name, **_k):
            raise AssertionError(f"{_n} called during {phase} phase")

        monkeypatch.setattr(storage, name, spy)


def _bookmark_snapshot(storage, org_id: str, user_id: str) -> dict[str, str]:
    """Snapshot the profile-extractor stride-bookmark operation-state rows for
    this org+user (key prefix ``profile_extractor::{org}::{user}::``).

    Returns {service_name: serialized operation_state} so an unchanged snapshot
    means the bookmark did not advance. SQLite shares one db file across orgs, so
    the org/user prefix scoping keeps the comparison like-for-like.
    """
    prefix = f"profile_extractor::{org_id}::{user_id}::"
    out: dict[str, str] = {}
    for row in storage.get_all_operation_states():
        name = row.get("service_name", "")
        if name.startswith(prefix):
            out[name] = json.dumps(row.get("operation_state"), sort_keys=True)
    return out


# ---------------------------------------------------------------------------
# Step 1 — the complete compute write-tripwire (F3)
# ---------------------------------------------------------------------------


def test_compute_issues_no_learning_write(monkeypatch):
    """compute_deferred_learning returns a plan and trips NONE of the ten
    learning-write storage terminals (F3). RED now: AttributeError — the method
    does not exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = _factory(tmp_dir)("org_compute")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_compute", user_id="u_c", request_id="req_c")
        gen = _make_gen(ctx)

        _forbid_writes(storage, monkeypatch, phase="compute")

        plan = gen.compute_deferred_learning(
            user_id="u_c",
            request_id="req_c",
            session_id="sess1",
            source="test_src",
            agent_version="v1",
            force_extraction=True,
            skip_aggregation=False,
        )
        assert type(plan).__name__ == "DeferredLearningPlan"


# ---------------------------------------------------------------------------
# Step 2 — persist issues no LLM / embedding / document-expansion
# ---------------------------------------------------------------------------


def test_persist_issues_no_llm_or_embedding(monkeypatch):
    """After a clean compute, persist must not call litellm.completion,
    storage._get_embedding, or storage._expand_document. RED now: AttributeError
    on compute_deferred_learning."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = _factory(tmp_dir)("org_persist")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_persist", user_id="u_p", request_id="req_p")
        gen = _make_gen(ctx)

        plan = gen.compute_deferred_learning(
            user_id="u_p",
            request_id="req_p",
            session_id="sess1",
            source="test_src",
            agent_version="v1",
            force_extraction=True,
            skip_aggregation=False,
        )

        def _boom(*_a, _n="", **_k):
            raise AssertionError(f"{_n} called during persist phase")

        monkeypatch.setattr(
            litellm, "completion", lambda *_a, **_k: _boom(_n="litellm.completion")
        )
        monkeypatch.setattr(
            storage, "_get_embedding", lambda *_a, **_k: _boom(_n="_get_embedding")
        )
        monkeypatch.setattr(
            storage, "_expand_document", lambda *_a, **_k: _boom(_n="_expand_document")
        )

        with storage.commit_scope():
            gen.persist_deferred_learning(plan)


# ---------------------------------------------------------------------------
# Step 3 — the worker's commit_scope wraps ONLY persist
# ---------------------------------------------------------------------------


def test_worker_scope_wraps_only_persist(monkeypatch):
    """Structural ordering: compute runs entirely before scope-enter, persist
    entirely inside, side-effects entirely after scope-exit. RED now:
    AttributeError capturing the (missing) phase methods."""
    # Capturing the originals fails with AttributeError today (methods absent) → RED.
    orig_compute = GenerationService.compute_deferred_learning
    orig_persist = GenerationService.persist_deferred_learning
    orig_emit = GenerationService.emit_deferred_learning_side_effects

    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_order")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_order", user_id="u_o", request_id="req_o")

        ticks: list[str] = []
        tick = {"n": 0}

        def _t() -> int:
            tick["n"] += 1
            return tick["n"]

        scope_enter = {"v": 0}
        scope_exit = {"v": 0}

        orig_scope = storage.commit_scope

        def wrapped_scope():
            cm = orig_scope()

            class _Tracker:
                def __enter__(self):
                    scope_enter["v"] = _t()
                    return cm.__enter__()

                def __exit__(self, *exc):
                    try:
                        return cm.__exit__(*exc)
                    finally:
                        scope_exit["v"] = _t()

            return _Tracker()

        monkeypatch.setattr(storage, "commit_scope", wrapped_scope)

        def wrap(name, fn):
            def _inner(self, *a, **k):
                ticks.append(f"{name}:{_t()}")
                return fn(self, *a, **k)

            return _inner

        monkeypatch.setattr(
            GenerationService,
            "compute_deferred_learning",
            wrap("compute", orig_compute),
        )
        monkeypatch.setattr(
            GenerationService,
            "persist_deferred_learning",
            wrap("persist", orig_persist),
        )
        monkeypatch.setattr(
            GenerationService,
            "emit_deferred_learning_side_effects",
            wrap("emit", orig_emit),
        )

        [job] = storage.claim_learning_jobs(
            claimed_by="w_order", limit=1, lease_seconds=300
        )
        DurableLearningWorker(factory, instance_id="w_order")._process_job(ctx, job)

        compute_tick = int(dict(t.split(":") for t in ticks)["compute"])
        persist_tick = int(dict(t.split(":") for t in ticks)["persist"])
        emit_tick = int(dict(t.split(":") for t in ticks)["emit"])

        assert compute_tick < scope_enter["v"], "compute must run before scope-enter"
        assert scope_enter["v"] < persist_tick < scope_exit["v"], (
            "persist must run strictly inside the scope"
        )
        assert emit_tick > scope_exit["v"], "side-effects must run after scope-exit"


# ---------------------------------------------------------------------------
# Step 4 — faithful-path tripwire: no LLM/embedding inside the REAL scope window
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_real_worker_no_llm_or_embedding_inside_scope(monkeypatch):
    """Drive the REAL _process_job (no phase patching). Assert litellm.completion,
    storage._get_embedding, and storage._expand_document are never called while
    the commit_scope is open. Proves the goal through the real scope window.

    RED now (via ``timeout`` marker): the current _process_job wraps
    run_deferred_learning inside the scope, and the extractor runs its LLM tool
    loop in a worker thread (base_generation `_execute_extractor`); that thread's
    agent_run writes block on the SQLite scope RLock held by the main thread →
    deadlock. That deadlock IS the in-scope-LLM bug this test forbids; the
    ``timeout`` marker turns it into a fast, clean failure. Post-Task-9 the LLM
    runs in compute (outside the scope) so this completes in seconds.

    Guard: the pre-split in-scope path deadlocks the extractor thread against the
    SQLite scope RLock, and a signal-interrupted scope leaves the lock poisoned
    for later tests. So this test fails FAST (before entering any scope) until the
    split API exists; once implemented, it drives the real _process_job and the
    scope-window tripwire runs for real."""
    assert hasattr(GenerationService, "compute_deferred_learning"), (
        "gate-(b) compute/persist split not implemented — faithful-path tripwire "
        "cannot run against the pre-split in-scope worker (it deadlocks)"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_faithful")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_faithful", user_id="u_f", request_id="req_f")

        scope_open = {"v": False}
        violations: list[str] = []

        orig_scope = storage.commit_scope

        def wrapped_scope():
            cm = orig_scope()

            class _Tracker:
                def __enter__(self):
                    scope_open["v"] = True
                    return cm.__enter__()

                def __exit__(self, *exc):
                    try:
                        return cm.__exit__(*exc)
                    finally:
                        scope_open["v"] = False

            return _Tracker()

        monkeypatch.setattr(storage, "commit_scope", wrapped_scope)

        orig_completion = litellm.completion
        # SQLite-only embedding helpers, not on the BaseStorage ABC surface.
        orig_embed = storage._get_embedding  # type: ignore[attr-defined]
        orig_expand = storage._expand_document  # type: ignore[attr-defined]

        def guard_completion(*a, **k):
            if scope_open["v"]:
                violations.append("litellm.completion inside scope")
            return orig_completion(*a, **k)

        def guard_embed(*a, **k):
            if scope_open["v"]:
                violations.append("_get_embedding inside scope")
            return orig_embed(*a, **k)

        def guard_expand(*a, **k):
            if scope_open["v"]:
                violations.append("_expand_document inside scope")
            return orig_expand(*a, **k)

        monkeypatch.setattr(litellm, "completion", guard_completion)
        monkeypatch.setattr(storage, "_get_embedding", guard_embed)
        monkeypatch.setattr(storage, "_expand_document", guard_expand)

        [job] = storage.claim_learning_jobs(
            claimed_by="w_faithful", limit=1, lease_seconds=300
        )
        DurableLearningWorker(factory, instance_id="w_faithful")._process_job(ctx, job)

        assert not violations, (
            "LLM / embedding / document-expansion must not run inside the "
            f"commit_scope window: {violations}"
        )


# ---------------------------------------------------------------------------
# Step 5 — the extractor stride-bookmark advances IFF the rows commit (F1)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_bookmark_advances_iff_rows_commit():
    """Two-phase (real fence-lost + retry), per the §3 F1 invariant:

    Phase 1 — complete_learning_job returns 0 (lease stolen) → _process_job
      returns False, no profile rows exist, and the profile_extractor bookmark
      is unchanged (its advance rolled back with the fenced scope).
    Phase 2 — with the real fence restored, re-claim + re-process the same job →
      profiles present AND the bookmark advanced.

    RED now (via ``timeout`` marker): Phase 1 drives the current in-scope
    _process_job with real extraction, which deadlocks the extractor worker thread
    against the SQLite scope RLock (see the faithful-path test). Post-Task-9 the
    extraction runs in compute (outside the scope), Phase 1 rolls the persist back
    on the fake fence loss, and Phase 2 commits — the green regression oracle for
    F1's move of the bookmark advance into persist.
    """
    from unittest import mock

    assert hasattr(GenerationService, "compute_deferred_learning"), (
        "gate-(b) compute/persist split not implemented — this test drives the "
        "real worker with extraction, which deadlocks under the pre-split "
        "in-scope path; it runs for real once the split exists"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_bm")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_bm", user_id="u_bm", request_id="req_bm")

        bm_before = _bookmark_snapshot(storage, "org_bm", "u_bm")

        # Phase 1: fence lost — complete_learning_job returns 0 for this attempt.
        [job1] = storage.claim_learning_jobs(
            claimed_by="w_bm1", limit=1, lease_seconds=300
        )
        with mock.patch.object(storage, "complete_learning_job", return_value=0):
            result1 = DurableLearningWorker(factory, instance_id="w_bm1")._process_job(
                ctx, job1
            )
        assert result1 is False, "fence-lost job must return False"
        assert storage.count_all_profiles() == 0, (
            "fence-lost persist must roll back — no profile rows"
        )
        assert _bookmark_snapshot(storage, "org_bm", "u_bm") == bm_before, (
            "bookmark must NOT advance when rows did not commit"
        )

        # The superseded path does not fail the job; return it to reclaimable so
        # Phase 2 can re-process the SAME job (mirrors a lost-lease retry).
        assert job1.claim_token is not None
        storage.fail_learning_job(
            job_id=job1.job_id, claim_token=job1.claim_token, dead=False
        )

        # Phase 2: real fence restored, re-claim + re-process the same job.
        [job2] = storage.claim_learning_jobs(
            claimed_by="w_bm2", limit=1, lease_seconds=300
        )
        result2 = DurableLearningWorker(factory, instance_id="w_bm2")._process_job(
            ctx, job2
        )
        assert result2 is True, "restored fence-winning job must complete"
        assert storage.count_all_profiles() > 0, "winning persist must land profiles"
        assert _bookmark_snapshot(storage, "org_bm", "u_bm") != bm_before, (
            "bookmark must advance once the rows commit"
        )


# ---------------------------------------------------------------------------
# Step 6 — same-user serialization → no duplicate (F4)
# ---------------------------------------------------------------------------


def test_same_user_two_workers_no_duplicate():
    """F4 same-user guard, exercised at the compute seam where it lives (Task 8).

    The guard acquires a DB-backed per-user in-progress lock at the START of
    compute_deferred_learning; a second concurrent compute for the same user is
    denied (``lock_acquired=False``) so its worker leaves the job pending rather
    than racing a duplicate write. We drive this deterministically (no threads,
    no dependence on the Task-8 lock key):

      1. compute job A (does NOT emit) → holds the per-user lock → lock_acquired True.
      2. compute job B for the SAME user while A's lock is held → lock_acquired False.
      3. emit A's side-effects → releases the lock.
      4. compute B again → now acquires (lock_acquired True).

    Then a sequential worker drain of two same-user jobs must leave a single
    clean run's profile count (no duplicate).

    RED now: AttributeError — compute_deferred_learning does not exist yet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)

        # --- Part 1: the per-user lock guard (the F4 mechanism) ---
        ctx = factory("org_f4")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_f4", user_id="u_f4", request_id="req_f4_a")
        _setup_job(storage, org_id="org_f4", user_id="u_f4", request_id="req_f4_b")
        gen = _make_gen(ctx)

        common = {
            "session_id": "sess1",
            "source": "test_src",
            "agent_version": "v1",
            "force_extraction": True,
            "skip_aggregation": False,
        }

        plan_a = gen.compute_deferred_learning(
            user_id="u_f4", request_id="req_f4_a", **common
        )
        assert plan_a.lock_acquired is True, (
            "first compute must acquire the per-user lock"
        )

        plan_b = gen.compute_deferred_learning(
            user_id="u_f4", request_id="req_f4_b", **common
        )
        assert plan_b.lock_acquired is False, (
            "a second same-user compute must be denied while the lock is held (F4)"
        )

        gen.emit_deferred_learning_side_effects(plan_a)  # releases the per-user lock

        plan_b2 = gen.compute_deferred_learning(
            user_id="u_f4", request_id="req_f4_b", **common
        )
        assert plan_b2.lock_acquired is True, (
            "after release, the same user's compute must acquire the lock again"
        )

        # --- Part 2: end-to-end no-duplicate oracle via QUEUE coalescing ---
        # F4's no-duplicate guarantee at the queue layer: two same-user jobs
        # enqueued while BOTH still pending coalesce into a SINGLE job (the
        # enqueue_learning_job ON CONFLICT upsert on (org,user,job_type) WHERE
        # status='pending'), so one drain runs a single extraction — no double
        # write. (Two NON-overlapping sequential jobs are legitimately NOT
        # coalesced and each learn; that is neither what F4 nor this oracle
        # covers, and under the mocked, dedup-skipping extractor it would just
        # re-extract — so this oracle deliberately exercises the coalescing path.)
        ctx2 = factory("org_f4_seq")
        storage2 = ctx2.storage
        assert storage2 is not None

        # Baseline: one clean single-job run for one user.
        _setup_job(
            storage2, org_id="org_f4_seq", user_id="u_base", request_id="req_base"
        )
        DurableLearningWorker(factory, instance_id="w_seq1").drain_org(
            "org_f4_seq", batch_size=5, lease_seconds=300
        )
        baseline = len(storage2.get_user_profile("u_base"))
        assert baseline > 0, "baseline single run must produce profiles"

        # Two same-user jobs enqueued while both pending coalesce to one job; a
        # single drain must therefore land the same per-user profile count.
        _setup_job(
            storage2, org_id="org_f4_seq", user_id="u_coal", request_id="req_coal_a"
        )
        _setup_job(
            storage2, org_id="org_f4_seq", user_id="u_coal", request_id="req_coal_b"
        )
        DurableLearningWorker(factory, instance_id="w_seq2").drain_org(
            "org_f4_seq", batch_size=5, lease_seconds=300
        )
        assert len(storage2.get_user_profile("u_coal")) == baseline, (
            "same-user coalescing must produce no duplicate profiles"
        )


# ---------------------------------------------------------------------------
# Step 7 — a CAS drop between compute and persist leaves no torn state
# ---------------------------------------------------------------------------


def test_cas_drop_leaves_no_torn_state(monkeypatch):
    """Compute a profile plan whose superseded_ids target an incumbent; supersede
    that incumbent out-of-band between compute and persist; persist must swallow
    the 0-rows/False from the guarded mutator (no raise) and leave no orphan.

    Regression guard for an incumbent changed between compute and persist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ctx = _factory(tmp_dir)("org_cas")
        storage = ctx.storage
        assert storage is not None
        _setup_job(storage, org_id="org_cas", user_id="u_cas", request_id="req_cas1")
        gen = _make_gen(ctx)

        # First clean run to create an incumbent profile to target. Call the
        # non-durable run_deferred_learning WITHOUT an outer commit_scope — the
        # extractor writes via its worker thread auto-commit normally; wrapping it
        # in a scope would deadlock that thread on the SQLite scope RLock.
        gen.run_deferred_learning(
            user_id="u_cas",
            request_id="req_cas1",
            session_id="sess1",
            source="test_src",
            agent_version="v1",
            force_extraction=True,
            skip_aggregation=False,
        )
        incumbents = storage.get_user_profile("u_cas")
        assert incumbents, "expected an incumbent profile from the first run"
        incumbent_ids = [p.profile_id for p in incumbents]

        _setup_job(storage, org_id="org_cas", user_id="u_cas", request_id="req_cas2")
        plan = gen.compute_deferred_learning(
            user_id="u_cas",
            request_id="req_cas2",
            session_id="sess1",
            source="test_src",
            agent_version="v1",
            force_extraction=True,
            skip_aggregation=False,
        )

        # Out-of-band: supersede the incumbents the plan may target.
        with storage.commit_scope():
            storage.supersede_profiles_by_ids(
                user_id="u_cas",
                profile_ids=incumbent_ids,
                request_id="req_oob",
            )

        # Persist must not raise on the CAS miss (0-rows/False → drop).
        with storage.commit_scope():
            gen.persist_deferred_learning(plan)

        # No torn state: each incumbent superseded exactly once (idempotent).
        remaining = {p.profile_id for p in storage.get_user_profile("u_cas")}
        assert not (set(incumbent_ids) & remaining), (
            "incumbents targeted by the CAS drop must not survive as live rows"
        )


# ---------------------------------------------------------------------------
# Step 8 — side-effect identity vs a captured golden (placeholder — Task 9)
# ---------------------------------------------------------------------------


def _norm_profiles(profiles) -> list[str]:
    """Content-stable profile projection (volatile ids/timestamps/embeddings
    dropped) as sorted JSON strings, for cross-run identity comparison."""
    return sorted(
        json.dumps(
            {
                "content": p.content,
                "source": p.source,
                "status": p.status.value if p.status is not None else None,
                "ttl": p.profile_time_to_live.value,
                "custom_features": p.custom_features,
                "extractor_names": p.extractor_names,
                "reader_angle": p.reader_angle,
                "tags": p.tags,
            },
            sort_keys=True,
        )
        for p in profiles
    )


def _norm_playbooks(playbooks) -> list[str]:
    """Content-stable playbook projection as sorted JSON strings."""
    return sorted(
        json.dumps(
            {
                "content": pb.content,
                "playbook_name": pb.playbook_name,
                "trigger": pb.trigger,
                "rationale": pb.rationale,
                "source": pb.source,
                "status": pb.status.value if pb.status is not None else None,
                "agent_version": pb.agent_version,
                "reader_angle": pb.reader_angle,
                "tags": pb.tags,
            },
            sort_keys=True,
        )
        for pb in playbooks
    )


def _norm_lineage(events) -> list[str]:
    """Content-free lineage projection (event_id / entity_id / request_id /
    org_id / created_at dropped — all volatile per-run) as sorted JSON strings."""
    return sorted(
        json.dumps(
            {
                "entity_type": e.entity_type,
                "op": e.op,
                "prov_relation": e.prov_relation,
                "actor": e.actor,
                "reason": e.reason,
                "from_status": e.from_status,
                "to_status": e.to_status,
            },
            sort_keys=True,
        )
        for e in events
    )


def test_side_effect_identity_vs_baseline():
    """The split durable ``_process_job`` writes the same profile/playbook rows +
    lineage events (by content, not just count) as the pre-split synchronous
    ``run_deferred_learning`` path on identical seeded input.

    Golden = a real drive of ``run_deferred_learning`` (unchanged by gate b —
    ``.run()`` / manual / rerun callers stay behavior-identical, spec §Global
    Constraints), in its OWN org+db. Split = a real drive of the rewired
    ``DurableLearningWorker._process_job`` in a SEPARATE org+db. The two paths are
    genuinely different orchestrations (single synchronous compute-persist-emit vs
    compute → fenced persist → post-commit emit), so this is not a self-compare.

    Volatile ids/timestamps/embeddings/request_ids are normalized out;
    ``REFLEXIO_EMBEDDING_PROVIDER=off`` keeps embeddings empty on both sides.

    Note (mock determinism): the root conftest mocks ``litellm.completion``, so
    extractor content is canned and identical across paths. This test proves the
    split path produces the SAME persisted side-effects as the synchronous path;
    it does not (and no unit test can) prove extraction semantics against a real
    LLM — that is the heavy-skill regression's job (Task 11)."""
    common = {
        "session_id": "sess1",
        "source": "test_src",
        "agent_version": "v1",
        "force_extraction": True,
        "skip_aggregation": False,
    }

    # --- Golden: synchronous run_deferred_learning in its own db ---
    with tempfile.TemporaryDirectory() as gold_dir:
        gctx = _factory(gold_dir)("org_gold")
        gstore = gctx.storage
        assert gstore is not None
        _setup_job(gstore, org_id="org_gold", user_id="u_g", request_id="req_g")
        _make_gen(gctx).run_deferred_learning(
            user_id="u_g", request_id="req_g", **common
        )
        gold_profiles = _norm_profiles(gstore.get_user_profile("u_g"))
        gold_playbooks = _norm_playbooks(gstore.get_user_playbooks(user_id="u_g"))
        gold_lineage = _norm_lineage(gstore.get_lineage_events(org_id="org_gold"))

    # --- Split: the rewired durable _process_job in a separate db ---
    with tempfile.TemporaryDirectory() as split_dir:
        factory = _factory(split_dir)
        sctx = factory("org_split")
        sstore = sctx.storage
        assert sstore is not None
        _setup_job(sstore, org_id="org_split", user_id="u_s", request_id="req_s")
        [job] = sstore.claim_learning_jobs(
            claimed_by="w_ident", limit=1, lease_seconds=300
        )
        assert (
            DurableLearningWorker(factory, instance_id="w_ident")._process_job(
                sctx, job
            )
            is True
        )
        split_profiles = _norm_profiles(sstore.get_user_profile("u_s"))
        split_playbooks = _norm_playbooks(sstore.get_user_playbooks(user_id="u_s"))
        split_lineage = _norm_lineage(sstore.get_lineage_events(org_id="org_split"))

    # Non-vacuous: the golden path must actually have produced rows to compare.
    assert gold_profiles, "golden run produced no profiles — nothing to compare"

    assert split_profiles == gold_profiles, (
        "split path profile rows differ from the synchronous baseline"
    )
    assert split_playbooks == gold_playbooks, (
        "split path playbook rows differ from the synchronous baseline"
    )
    assert split_lineage == gold_lineage, (
        "split path lineage events differ from the synchronous baseline"
    )


# ---------------------------------------------------------------------------
# Step 9 — a superseded job emits no billing / tagging (phantom-billing gate)
# ---------------------------------------------------------------------------


def test_superseded_job_emits_no_billing(monkeypatch):
    """Force complete_learning_job → 0 (fence lost); assert the post-commit
    side-effects never fire: _record_billing_learning_events and
    schedule_tagging.

    RED now: the current _process_job runs run_deferred_learning (which fires
    billing + schedule_tagging) INSIDE the scope, before complete_learning_job
    returns 0 — so a superseded job phantom-bills today. Task 9 moves these to
    the post-commit emit that only the fence winner reaches.

    Note (Task-1 fidelity): the plan text names ``record_usage_event`` as a spy
    target, but that terminal legitimately fires during compute (extraction
    billing), so spying it raw would false-positive under the eventual green
    design. The precise post-commit gate is ``_record_billing_learning_events``;
    that is what is asserted here. Flagged for Tasks 2-11 review.
    """
    import reflexio.server.services.generation_service as gen_mod

    fired: list[str] = []

    monkeypatch.setattr(
        BaseGenerationService,
        "_record_billing_learning_events",
        lambda _self, *_a, **_k: fired.append("billing"),
    )
    monkeypatch.setattr(
        gen_mod, "schedule_tagging", lambda *_a, **_k: fired.append("tagging")
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        factory = _factory(tmp_dir)
        ctx = factory("org_nobill")
        storage = ctx.storage
        assert storage is not None
        # force_extraction=False keeps extraction stride-gated (no real extractor
        # thread, no in-scope deadlock); schedule_tagging still fires unconditionally
        # inside run_deferred_learning, which is exactly what the phantom-billing
        # gate must suppress for a superseded job.
        _setup_job(
            storage,
            org_id="org_nobill",
            user_id="u_nb",
            request_id="req_nb",
            force_extraction=False,
        )

        [job] = storage.claim_learning_jobs(
            claimed_by="w_nb", limit=1, lease_seconds=300
        )
        monkeypatch.setattr(storage, "complete_learning_job", lambda **_k: 0)

        result = DurableLearningWorker(factory, instance_id="w_nb")._process_job(
            ctx, job
        )
        assert result is False, "fence-lost job must return False"
        assert fired == [], (
            f"a superseded job must emit no billing/tagging side-effects: {fired}"
        )
