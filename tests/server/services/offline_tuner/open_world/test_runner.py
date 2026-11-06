"""Single-method durable runner tests.

Deleted-policy execution assertions are retired; source/privacy/publication
contracts remain tested against the promoted method and SQL authorizers.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
)
from reflexio.server.llm import LiteLLMClientError
from reflexio.server.services.offline_tuner.evidence.compaction import CompactionError
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)
from reflexio.test_support.open_world_rule_evidence import runtime as runtime
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)


def test_native_method_records_each_diagnostic_and_proposal_once(runtime):
    runner, transport, storage, kwargs = runtime
    result = runner.run(**kwargs)
    assert result.decision.decision == "accept"
    assert len(transport.calls) == 2
    assert {row.component_role for row in storage.rows.values()} == {
        "discovery",
        "candidate",
    }
    assert all(row.state == "completed" for row in storage.rows.values())
    assert storage.job.stage == "held_out_analyzed"
    assert len(storage.artifacts) == 3


def test_long_preparation_renews_and_rebinds_the_current_lease(runtime):
    runner, transport, storage, kwargs = runtime
    clock = [200]
    runner.store.clock = lambda: clock[0]
    renewals = []

    def renew(job_id, owner, fence, seconds, *, now):
        assert (job_id, owner, fence) == (1, "worker", 1)
        assert now < storage.job.lease_expires_at
        storage.job.lease_expires_at = now + seconds
        claim = OptimizationJobClaim(
            job_id=job_id, owner=owner, fence=fence, expires_at=now + seconds
        )
        renewals.append(claim)
        return claim

    storage.renew_playbook_optimization_job_lease = renew
    bind = storage.bind_open_world_attempt_context
    bindings = []

    def live_bind(**request):
        assert request["now"] < storage.job.lease_expires_at
        assert request["fence"] == storage.job.lease_fence
        bindings.append(request["now"])
        return bind(**request)

    storage.bind_open_world_attempt_context = live_bind
    evidence = kwargs["evidence"]

    def prepare(before_call):
        # Model batches and receipt writes each run this same authority guard.
        for _ in range(25):
            clock[0] += 300
            before_call()
        return evidence

    result = runner.run(**{**kwargs, "evidence": prepare})
    assert result.decision.decision == "accept"
    assert len(renewals) >= 2
    assert len(bindings) >= 25
    assert runner.lease.claim == renewals[-1]
    assert storage.job.stage == "held_out_analyzed"
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        CompactionError("private malformed compaction"),
        LiteLLMClientError("private compaction provider failure"),
        ValueError("private invalid frozen trajectory"),
        RuntimeError("private preparation setup failure"),
    ],
)
def test_preparation_failures_durably_terminalize_without_judge_calls(runtime, error):
    runner, transport, storage, kwargs = runtime

    def prepare(before_call):
        before_call()
        raise error

    result = runner.run(**{**kwargs, "evidence": prepare})
    assert result.reason == "infrastructure_failure"
    assert storage.job.stage == storage.job.status == "failed"
    assert storage.job.terminal_outcome == "infrastructure_failure"
    assert "open_world_terminal_failure" in storage.failure
    assert "private" not in json.dumps(storage.failure)
    assert not transport.calls
    assert not storage.rows


@pytest.mark.parametrize("changed", [{"lease_owner": "peer"}, {"lease_fence": 2}])
def test_preparation_failure_never_terminalizes_a_peers_claim(runtime, changed):
    runner, transport, storage, kwargs = runtime

    def prepare(before_call):
        before_call()
        for name, value in changed.items():
            setattr(storage.job, name, value)
        raise CompactionError("preparation lost its authority")

    result = runner.run(**{**kwargs, "evidence": prepare})
    assert result.reason == "infrastructure_failure"
    assert storage.job.stage == "evidence_frozen"
    assert storage.job.status == "running"
    assert not hasattr(storage, "failure")
    assert not transport.calls


def test_resume_reuses_completed_calls_after_process_crash(runtime):
    runner, transport, storage, kwargs = runtime
    storage.crash_on_candidate = True
    with pytest.raises(KeyboardInterrupt):
        runner.run(**kwargs)
    assert len(transport.calls) == 2
    storage.crash_on_candidate = False
    result = runner.run(**kwargs)
    assert result.decision.decision == "accept"
    assert len(transport.calls) == 2


def test_missing_authority_refuses_without_provider_calls(runtime):
    runner, transport, storage, kwargs = runtime
    storage.authority = replace(storage.authority, authority_digest="e" * 64)
    result = runner.run(**kwargs)
    assert result.reason == "analyst_unqualified"
    assert not transport.calls
    assert storage.job.stage == "failed"


def test_tampered_completed_output_is_failure_not_abstention(runtime):
    runner, transport, storage, kwargs = runtime
    storage.crash_on_candidate = True
    with pytest.raises(KeyboardInterrupt):
        runner.run(**kwargs)
    key = next(iter(storage.rows))
    storage.rows[key] = replace(storage.rows[key], output_digest="f" * 64)
    storage.crash_on_candidate = False
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert len(transport.calls) == 2
    assert "open_world_terminal_failure" in storage.failure


def test_no_rule_gap_retains_the_required_durable_tuner_abstention(runtime):
    runner, transport, storage, kwargs = runtime
    original = transport.complete

    def adequate(**request):
        completion = original(**request)
        if issubclass(request["response_format"], Proposal):
            return replace(
                completion,
                value=Proposal(
                    edits=[],
                    hypothesis="",
                    abstention_reason="No grounded rule defect",
                ),
            )
        assert issubclass(request["response_format"], Diagnostic)
        diagnostic = completion.value
        return replace(
            completion,
            value=diagnostic.model_copy(
                update={
                    "rule_assessment": diagnostic.rule_assessment.model_copy(
                        update={"attribution": "no_supported_target_defect"}
                    )
                }
            ),
        )

    transport.complete = adequate
    result = runner.run(**kwargs)
    assert result.decision.decision == "abstain"
    assert len(transport.calls) == 2
    assert {row.component_role for row in storage.rows.values()} == {
        "discovery",
        "candidate",
    }
    assert all(row.state == "completed" for row in storage.rows.values())
    assert storage.job.stage == "abstained"
    assert "open_world_candidate" not in storage.artifacts
    artifact_bytes = {
        key: artifact.content_json for key, artifact in storage.artifacts.items()
    }
    calls = len(transport.calls)
    assert runner.run(**kwargs) == result
    assert (
        runner.run(
            **{
                **kwargs,
                "evidence": lambda _before_call: pytest.fail(
                    "terminal retry must not repeat preparation"
                ),
            }
        )
        == result
    )
    assert len(transport.calls) == calls
    assert {
        key: artifact.content_json for key, artifact in storage.artifacts.items()
    } == artifact_bytes


def test_proposal_abstention_retains_its_actual_completed_receipt(runtime):
    runner, transport, storage, kwargs = runtime
    original = transport.complete
    expected_output = Proposal(
        edits=[],
        hypothesis="",
        abstention_reason="Missing deciding information",
    )

    def abstain(**request):
        completion = original(**request)
        if issubclass(request["response_format"], Proposal):
            completion = replace(
                completion,
                value=expected_output,
            )
        return completion

    transport.complete = abstain
    result = runner.run(**kwargs)
    assert result.decision.decision == "abstain"
    assert len(transport.calls) == 2
    payload = json.loads(storage.artifacts["open_world_attempt_decision"].content_json)[
        "payload"
    ]
    assert payload["proposal_receipt"]["observed_output"] == expected_output.model_dump(
        mode="json"
    )
    assert payload["candidate_digest"] is None
    assert "open_world_candidate" not in storage.artifacts
    artifact_bytes = {
        key: artifact.content_json for key, artifact in storage.artifacts.items()
    }
    calls = len(transport.calls)
    assert runner.run(**kwargs) == result
    assert len(transport.calls) == calls
    assert {
        key: artifact.content_json for key, artifact in storage.artifacts.items()
    } == artifact_bytes


def test_changed_incumbent_is_a_stale_failure_without_model_calls(runtime):
    runner, transport, storage, kwargs = runtime
    kwargs["incumbent"] = kwargs["incumbent"].model_copy(update={"content": "changed"})
    result = runner.run(**kwargs)
    assert result.reason == "stale_incumbent"
    assert not transport.calls
    assert storage.job.stage == "failed"


def test_source_bundle_swap_is_refused_without_model_calls(runtime):
    runner, transport, _, kwargs = runtime
    kwargs["evidence"].bundle_digest = "f" * 64
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert not transport.calls
