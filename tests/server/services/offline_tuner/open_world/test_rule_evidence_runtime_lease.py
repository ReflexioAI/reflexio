"""Lease, stage and terminal-write semantics of the rule-evidence attempt runner.

The runner may only write under its own live fence. These tests drive the
in-memory store from `test_runner.py` through the lease edge cases: an expired
lease is refused, a stage write that lost the race is accepted only when the
job already sits at or beyond the requested stage, an expired lease is
reclaimed for the final fenced write unless a peer got there first, and a
diagnostic or terminal write that fails is reported without masking the
attempt's own failure.
"""

from __future__ import annotations

import logging

import pytest

from reflexio.models.api_schema.domain import OptimizationJobClaim
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    RuleEvidenceAttemptRunner,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationLeaseConflictError,
)
from reflexio.server.services.storage.error import StorageError
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)
from tests.server.services.offline_tuner.open_world.test_runner import (
    runtime as runtime,
)

_LOGGER = "reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime"


def _expire(runner: RuleEvidenceAttemptRunner, storage, *, now: int) -> None:
    runner.store.clock = lambda: now


# --- _claim ------------------------------------------------------------------------


def test_a_lease_that_cannot_be_renewed_past_now_is_refused(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    _expire(runner, storage, now=5000)

    def renew(job_id, owner, fence, seconds, *, now):
        # The ledger answers with the old (already expired) expiry.
        return OptimizationJobClaim(
            job_id=job_id, owner=owner, fence=fence, expires_at=4000
        )

    storage.renew_playbook_optimization_job_lease = renew
    with pytest.raises(OpenWorldInvocationLeaseConflictError, match="lease expired"):
        runner._claim()


def test_a_lease_far_from_expiry_is_not_renewed(runtime) -> None:
    runner, _, storage, _ = runtime
    storage.renew_playbook_optimization_job_lease = lambda *_a, **_k: pytest.fail(
        "a lease with more than ten minutes left must not be renewed"
    )
    assert runner._claim() == runner.lease.claim


# --- _advance ----------------------------------------------------------------------


def _refuse_advance(storage) -> None:
    storage.advance_playbook_optimization_stage = lambda *_a, **_k: False


@pytest.mark.parametrize("stage", ["discovery_analyzed", "candidate_generated"])
def test_a_lost_stage_write_is_accepted_when_the_job_is_already_there(
    runtime, stage
) -> None:
    runner, _, storage, _ = runtime
    _refuse_advance(storage)
    storage.job.stage = "candidate_generated"
    runner._advance(stage)  # no raise: a peer-free retry already advanced it


@pytest.mark.parametrize(
    ("job_changes", "requested"),
    [
        ({"stage": "evidence_frozen"}, "discovery_analyzed"),
        ({"stage": "discovery_analyzed"}, "held_out_analyzed"),
        ({"stage": "abstained"}, "discovery_analyzed"),
        ({"status": "failed"}, "discovery_analyzed"),
        ({"lease_owner": "peer"}, "discovery_analyzed"),
        ({"lease_fence": 2}, "discovery_analyzed"),
        ({"lease_expires_at": 9000}, "discovery_analyzed"),
        ({}, "failed"),
    ],
)
def test_a_lost_stage_write_behind_or_under_another_lease_is_refused(
    runtime, job_changes, requested
) -> None:
    runner, _, storage, _ = runtime
    _refuse_advance(storage)
    storage.job.stage = "held_out_analyzed"
    for name, value in job_changes.items():
        setattr(storage.job, name, value)
    with pytest.raises(OpenWorldInvocationLeaseConflictError, match="stage/lease"):
        runner._advance(requested)


def test_a_lost_stage_write_for_a_missing_job_is_refused(runtime) -> None:
    runner, _, storage, _ = runtime
    _refuse_advance(storage)
    storage.get_playbook_optimization_job = lambda _job_id: None
    with pytest.raises(OpenWorldInvocationLeaseConflictError):
        runner._advance("discovery_analyzed")


# --- _terminal_claim and terminal writes ---------------------------------------------


def _failing_preparation(before_call):
    before_call()
    raise ValueError("private preparation failure")


def test_an_expired_own_lease_is_reclaimed_for_the_terminal_write(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    reclaimed = []

    def reclaim(job_id, owner, seconds, *, now):
        reclaimed.append((job_id, owner, seconds, now))
        storage.job.lease_fence = 2
        return OptimizationJobClaim(
            job_id=job_id, owner=owner, fence=2, expires_at=now + seconds
        )

    storage.reclaim_playbook_optimization_job = reclaim
    fences = []
    advance = storage.advance_playbook_optimization_stage

    def recording_advance(job_id, fence, stage, **rest):
        fences.append((stage, fence))
        return advance(job_id, fence, stage, **rest)

    storage.advance_playbook_optimization_stage = recording_advance

    def expire_then_fail(before_call):
        before_call()
        storage.job.lease_expires_at = 150  # lapsed while preparing
        raise ValueError("private preparation failure")

    result = runner.run(**{**kwargs, "evidence": expire_then_fail})
    assert result.reason == "infrastructure_failure"
    assert reclaimed == [(1, "worker", 3600, 200)]
    assert runner.lease.claim.fence == 2
    assert fences == [("failed", 2)]
    assert storage.job.status == "failed"


def test_a_peer_reclaim_wins_and_no_terminal_write_is_made(runtime) -> None:
    runner, transport, storage, kwargs = runtime

    def reclaim(*_args, **_kwargs):
        raise StorageError("a peer holds the live lease")

    storage.reclaim_playbook_optimization_job = reclaim
    storage.advance_playbook_optimization_stage = lambda *_a, **_k: pytest.fail(
        "a peer's live lease must never be terminalized"
    )

    def expire_then_fail(before_call):
        before_call()
        storage.job.lease_expires_at = None
        raise ValueError("private preparation failure")

    result = runner.run(**{**kwargs, "evidence": expire_then_fail})
    assert result.reason == "infrastructure_failure"
    assert result.rate_limited is False
    assert not hasattr(storage, "failure")


@pytest.mark.parametrize("status", ["failed", "skipped", "succeeded"])
def test_a_job_already_terminal_is_never_rewritten(runtime, status) -> None:
    runner, transport, storage, kwargs = runtime

    def finished_then_fail(before_call):
        before_call()
        storage.job.status = status
        raise ValueError("private preparation failure")

    storage.advance_playbook_optimization_stage = lambda *_a, **_k: pytest.fail(
        "a terminal job must not be advanced again"
    )
    result = runner.run(**{**kwargs, "evidence": finished_then_fail})
    assert result.reason == "infrastructure_failure"


def test_an_unwritable_failure_diagnostic_still_terminalizes(runtime, caplog) -> None:
    runner, transport, storage, kwargs = runtime

    def broken_merge(job_id, patch):
        raise StorageError("metadata column unavailable")

    storage.merge_playbook_optimization_job_metadata = broken_merge
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = runner.run(**{**kwargs, "evidence": _failing_preparation})
    assert result.reason == "infrastructure_failure"
    assert storage.job.status == "failed"
    assert any(
        "failure diagnostic unavailable error_type=StorageError" in r.getMessage()
        for r in caplog.records
    )


def test_a_refused_terminal_write_is_logged_and_still_reports_failure(
    runtime, caplog
) -> None:
    runner, transport, storage, kwargs = runtime
    storage.advance_playbook_optimization_stage = lambda *_a, **_k: False
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = runner.run(**{**kwargs, "evidence": _failing_preparation})
    assert result.reason == "infrastructure_failure"
    assert storage.job.status == "running"
    assert any(
        "terminal write refused after lease changed" in r.getMessage()
        for r in caplog.records
    )


# --- persisted abstention replay -----------------------------------------------------


def _abstain_transport(transport) -> None:
    from dataclasses import replace

    from reflexio.server.services.playbook.rule_evidence.wire import (
        GlobalRuleAudit as Proposal,
    )

    original = transport.complete

    def abstain(**request):
        completion = original(**request)
        if issubclass(request["response_format"], Proposal):
            return replace(
                completion,
                value=Proposal(edits=[], hypothesis="", abstention_reason="None"),
            )
        return completion

    transport.complete = abstain


def test_an_abstained_job_without_its_decision_is_a_failure(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    _abstain_transport(transport)
    assert runner.run(**kwargs).decision.decision == "abstain"
    del storage.artifacts["open_world_attempt_decision"]
    calls = len(transport.calls)
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert len(transport.calls) == calls


@pytest.mark.parametrize(
    "job_changes",
    [
        {"status": "running"},
        {"lease_fence": 2},
        {"terminal_outcome": "infrastructure_failure"},
    ],
)
def test_an_abstained_job_whose_terminal_state_changed_is_a_failure(
    runtime, job_changes
) -> None:
    runner, transport, storage, kwargs = runtime
    _abstain_transport(transport)
    assert runner.run(**kwargs).decision.decision == "abstain"
    for name, value in job_changes.items():
        setattr(storage.job, name, value)
    calls = len(transport.calls)
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert len(transport.calls) == calls


def test_a_lost_abstention_write_is_accepted_when_already_recorded(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    _abstain_transport(transport)
    advance = storage.advance_playbook_optimization_stage

    def abstain_raced(job_id, fence, stage, **rest):
        if stage == "abstained":
            # A concurrent retry of this same attempt recorded it first.
            storage.job.terminal_outcome = "no_grounded_hypothesis"
            return False
        return advance(job_id, fence, stage, **rest)

    storage.advance_playbook_optimization_stage = abstain_raced
    assert runner.run(**kwargs).decision.decision == "abstain"


def test_a_lost_abstention_write_that_was_not_recorded_is_a_failure(
    runtime,
) -> None:
    runner, transport, storage, kwargs = runtime
    _abstain_transport(transport)
    advance = storage.advance_playbook_optimization_stage
    stages = []

    def abstain_refused(job_id, fence, stage, **rest):
        stages.append(stage)
        if stage == "abstained":
            return False
        return advance(job_id, fence, stage, **rest)

    storage.advance_playbook_optimization_stage = abstain_refused
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert stages[-1] == "failed"


def test_lazily_loaded_evidence_for_another_bundle_is_refused(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    evidence = kwargs["evidence"]
    evidence.bundle_digest = "f" * 64

    def load(before_call):
        before_call()
        return evidence

    result = runner.run(**{**kwargs, "evidence": load})
    assert result.reason == "infrastructure_failure"
    assert not transport.calls
    assert storage.job.status == "failed"
