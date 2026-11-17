"""The durable final-version stages of the maintained rule-evidence runner.

Uses the same store, lease, transport and publication owner. Every derived value
is reconstructed from completed receipts on load; hashes alone grant no authority.
"""

from __future__ import annotations

import difflib
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter

from reflexio.models.api_schema.domain import OptimizationJobClaim, UserPlaybook
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    PLAYBOOK_REVISION_POLICY_ID,
    OpenWorldEvidenceBundle,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    OpenWorldEvidenceBundleStore,
    _parse_canonical_bundle,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationConflictError,
    OpenWorldInvocationLeaseConflictError,
)
from reflexio.server.services.playbook.revision import FinalPlaybookRevision
from reflexio.server.services.playbook.rule_evidence import run_rule_evidence
from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    PROFILE,
    TriggerBatch,
    TriggerGate,
    TriggerMaintenanceResult,
    TriggerSourceFrame,
    run_trigger_maintenance,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)

from .models import AbstainDecision, AcceptDecision, EvidenceDocument
from .rule_evidence import (
    diagnostic_population_digest,
    native_rule_evidence_input,
    validate_evidence_edit,
)

if TYPE_CHECKING:
    from .rule_evidence_runtime import (
        RuleEvidenceAnalysisStore,
        RuleEvidenceAttemptRunner,
        RuleEvidenceDecision,
        RuleEvidencePublicationInputs,
    )


def _same(left: object, right: object) -> bool:
    # Python equality equates True/1 and False/0; persisted JSON identity does not.
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _diff(original: str, final: str) -> dict[str, str]:
    return {
        "incumbent_content_digest": sha256(original.encode()).hexdigest(),
        "candidate_content_digest": sha256(final.encode()).hexdigest(),
        "unified_diff": "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                final.splitlines(keepends=True),
                fromfile="incumbent",
                tofile="candidate",
            )
        ),
    }


def _common(
    *,
    job_id: int,
    bundle: OpenWorldEvidenceBundle,
    authority: Any,
    diagnostics_artifact_digest: str,
    population_digest: str,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "bundle_digest": bundle.digest,
        "task_context_digest": bundle.task_context.digest,
        "incumbent_full_version_fingerprint": bundle.target.incumbent_full_version_fingerprint,
        "authority_epoch": authority.epoch,
        "authority_digest": authority.authority_digest,
        "diagnostics_artifact_digest": diagnostics_artifact_digest,
        "diagnostic_population_digest": population_digest,
    }


def _frame(
    bundle: OpenWorldEvidenceBundle,
    incumbent: UserPlaybook,
    final_content: str,
    content_digest: str,
) -> TriggerSourceFrame:
    if not incumbent.user_id:
        raise ValueError("Trigger source requires the exact incumbent owner")
    return TriggerSourceFrame(
        incumbent_id=incumbent.user_playbook_id,
        owner_id=incumbent.user_id,
        incumbent_version=bundle.target.incumbent_full_version_fingerprint,
        evidence_bundle_digest=bundle.digest,
        original_content=incumbent.content,
        original_trigger=incumbent.trigger,
        final_content=final_content,
        final_content_artifact_digest=content_digest,
    )


def _content(
    common: dict[str, Any],
    incumbent: UserPlaybook,
    result: Any,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    final = (
        result.revised_content
        if result.revised_content is not None
        else incumbent.content
    )
    return {
        **common,
        **receipt,
        "original_content": incumbent.content,
        "final_content": final,
        "rule_audit": result.rule_audit.model_dump(mode="json"),
        "content_diff": _diff(incumbent.content, final),
    }


def _trigger(
    common: dict[str, Any],
    frame: TriggerSourceFrame,
    resolved: TriggerMaintenanceResult,
    receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        **common,
        "content_result_artifact_digest": frame.final_content_artifact_digest,
        "source_frame": frame.model_dump(mode="json"),
        "source_frame_digest": frame.digest,
        "profile": PROFILE,
        "gate_receipt": receipts.get("gate"),
        "builder_receipt": receipts.get("builder"),
        "resolved_result": resolved.model_dump(mode="json"),
    }


def _candidate(
    common: dict[str, Any],
    frame: TriggerSourceFrame,
    content: dict[str, Any],
    trigger_digest: str,
    resolved: TriggerMaintenanceResult,
) -> dict[str, Any]:
    changes = [
        name
        for name, old, new in (
            ("content", frame.original_content, resolved.revision.revised_content),
            ("trigger", frame.original_trigger, resolved.revision.revised_trigger),
        )
        if old != new
    ]
    if not changes:
        raise ValueError("unchanged final pair cannot be a candidate")
    return {
        **common,
        "content_result_artifact_digest": frame.final_content_artifact_digest,
        "trigger_result_artifact_digest": trigger_digest,
        **resolved.revision.model_dump(mode="json"),
        "content_diff": content["content_diff"],
        "change_fields": changes,
    }


def _decision(
    common: dict[str, Any],
    content_digest: str,
    trigger_digest: str,
    candidate_digest: str | None,
) -> dict[str, Any]:
    return {
        **common,
        "policy_id": PLAYBOOK_REVISION_POLICY_ID,
        "content_result_artifact_digest": content_digest,
        "trigger_result_artifact_digest": trigger_digest,
        "candidate_digest": candidate_digest,
        "decision": {"decision": "accept"}
        if candidate_digest
        else {"decision": "abstain", "reason": "no_grounded_hypothesis"},
    }


def finish_attempt(
    runner: RuleEvidenceAttemptRunner,
    *,
    bundle: OpenWorldEvidenceBundle,
    incumbent: UserPlaybook,
    diagnostics_artifact: Any,
    diagnostics: list[Diagnostic],
    result: Any,
    validation: Any,
    proposal_receipt: dict[str, Any],
    shared_protocol: bool,
) -> RuleEvidenceDecision:
    from .rule_evidence_runtime import RuleEvidenceDecision, artifact_payload

    store = runner.store
    _, authority = store.bind_context(bundle, runner._claim())
    if not proposal_receipt or not _same(
        result.model_dump(mode="json"), validation.result.model_dump(mode="json")
    ):
        raise OpenWorldInvocationConflictError(
            "Content result differs from verified source replay"
        )
    common = _common(
        job_id=runner.lease.claim.job_id,
        bundle=bundle,
        authority=authority,
        diagnostics_artifact_digest=diagnostics_artifact.content_digest,
        population_digest=diagnostic_population_digest(diagnostics),
    )
    content = _content(common, incumbent, result, proposal_receipt)
    content_artifact = store.write(
        "open_world_content_result",
        artifact_payload("rule_evidence_content_result", content, schema_version="v2"),
        runner._claim(),
    )
    frame = _frame(
        bundle, incumbent, content["final_content"], content_artifact.content_digest
    )
    receipts: dict[str, dict[str, Any]] = {}

    def invoke(
        system: str, payload: dict[str, object], schema: type[StrictStructuredOutput]
    ) -> StrictStructuredOutput:
        if schema not in (TriggerGate, TriggerBatch):
            raise ValueError("unexpected Trigger schema")
        from .shared_compaction import shared_consumer_admission

        admission = shared_consumer_admission.set(shared_protocol)
        try:
            value, receipt = store.invoke(
                bundle=bundle,
                claim=runner._claim(),
                system=system,
                payload=payload,
                schema=schema,
                source_frame_digest=frame.digest,
                content_result_artifact_digest=content_artifact.content_digest,
            )
        finally:
            shared_consumer_admission.reset(admission)
        key = "gate" if schema is TriggerGate else "builder"
        if key in receipts:
            raise OpenWorldInvocationConflictError("duplicate Trigger stage")
        receipts[key] = receipt
        return value

    resolved = run_trigger_maintenance(frame, invoke)
    trigger = _trigger(common, frame, resolved, receipts)
    trigger_artifact = store.write(
        "open_world_trigger_result",
        artifact_payload("rule_evidence_trigger_result", trigger, schema_version="v2"),
        runner._claim(),
    )
    candidate_digest = None
    if resolved.revision.changes(frame):
        candidate = _candidate(
            common, frame, content, trigger_artifact.content_digest, resolved
        )
        row = store.write(
            "open_world_candidate",
            artifact_payload("rule_evidence_candidate", candidate, schema_version="v2"),
            runner._claim(),
        )
        candidate_digest = row.content_digest
    decision = _decision(
        common,
        content_artifact.content_digest,
        trigger_artifact.content_digest,
        candidate_digest,
    )
    store.write(
        "open_world_attempt_decision",
        artifact_payload("rule_evidence_decision", decision, schema_version="v2"),
        runner._claim(),
    )
    if candidate_digest is not None:
        runner._advance("candidate_generated")
        runner._advance("held_out_analyzed")
    else:
        claim = runner._claim()
        if not store.storage.advance_playbook_optimization_stage(
            claim.job_id,
            claim.fence,
            "abstained",
            terminal_outcome="no_grounded_hypothesis",
            now=store.clock(),
        ):
            raise OpenWorldInvocationLeaseConflictError(
                "v2 abstention was not recorded"
            )
    return RuleEvidenceDecision(
        bundle.digest,
        candidate_digest,
        validation.diagnostic_population_digest,
        AcceptDecision(decision="accept")
        if candidate_digest
        else AbstainDecision(decision="abstain", reason="no_grounded_hypothesis"),
    )


def load_publication_inputs(
    store: RuleEvidenceAnalysisStore,
    *,
    job_id: int,
    claim: OptimizationJobClaim,
    now: int,
) -> RuleEvidencePublicationInputs:
    from .rule_evidence_runtime import (
        FinalContentDiff,
        RuleEvidenceCandidate,
        RuleEvidenceDecision,
        RuleEvidencePublicationInputs,
    )

    job = store.storage.get_playbook_optimization_job(job_id)
    if (
        claim.job_id != job_id
        or job is None
        or job.status != "running"
        or job.stage not in {"held_out_analyzed", "publishing"}
        or job.lease_owner != claim.owner
        or job.lease_fence != claim.fence
        or job.lease_expires_at != claim.expires_at
        or claim.expires_at <= now
    ):
        raise OpenWorldInvocationLeaseConflictError("publication lease/stage is stale")
    frozen = store.storage.load_open_world_evidence_bundle(job_id)
    bundle = _parse_canonical_bundle(frozen.artifact.content_json)
    if (
        frozen.artifact.content_digest != bundle.digest
        or bundle.target.tenant_id != store.storage.org_id
        or job.target_id != bundle.target.user_playbook_id
        or job.expected_population_manifest_digest != bundle.expected.digest
    ):
        raise OpenWorldInvocationConflictError("publication source bundle changed")
    incumbent = store.storage.get_user_playbook_by_id(bundle.target.user_playbook_id)
    if incumbent is None:
        raise OpenWorldInvocationConflictError("publication incumbent is missing")
    context, authority = store.bind_context(bundle, claim)
    diagnostic_row = store.read(
        "open_world_discovery_memo", "rule_evidence_diagnostics", claim
    )
    content_row = store.read(
        "open_world_content_result",
        "rule_evidence_content_result",
        claim,
        schema_version="v2",
    )
    trigger_row = store.read(
        "open_world_trigger_result",
        "rule_evidence_trigger_result",
        claim,
        schema_version="v2",
    )
    candidate_row = store.read(
        "open_world_candidate", "rule_evidence_candidate", claim, schema_version="v2"
    )
    decision_row = store.read(
        "open_world_attempt_decision",
        "rule_evidence_decision",
        claim,
        schema_version="v2",
    )
    if (
        diagnostic_row is None
        or content_row is None
        or trigger_row is None
        or candidate_row is None
        or decision_row is None
    ):
        raise OpenWorldInvocationConflictError(
            "v2 publication requires all admitted result artifacts"
        )
    diagnostic_artifact, collection = diagnostic_row
    content_artifact, content = content_row
    trigger_artifact, trigger = trigger_row
    candidate_artifact, candidate = candidate_row
    decision_artifact, decision = decision_row
    if any(
        not _same(collection[key], value)
        for key, value in {
            "job_id": job_id,
            "bundle_digest": bundle.digest,
            "task_context_digest": bundle.task_context.digest,
            "authority_epoch": authority.epoch,
            "authority_digest": authority.authority_digest,
        }.items()
    ):
        raise OpenWorldInvocationConflictError("diagnostic context changed")
    documents = [
        TypeAdapter(EvidenceDocument).validate_python(session["source_document"])
        for session in collection["sessions"]
    ]
    trajectories = {
        (s["user_id"], s["session_id"]): s["trajectory_json"]
        for s in collection["sessions"]
    }
    data = native_rule_evidence_input(
        bundle=bundle,
        incumbent=incumbent,
        documents=documents,
        trajectory_json_by_session=trajectories,
    )
    observed = iter([*collection["sessions"], content])
    diagnostics: list[Diagnostic] = []
    proposal: Proposal | None = None

    def replay_content(
        system: str, payload: dict[str, object], schema: type[StrictStructuredOutput]
    ) -> StrictStructuredOutput:
        nonlocal proposal
        receipt = next(observed, None)
        if receipt is None:
            raise OpenWorldInvocationConflictError(
                "durable Content response is missing"
            )
        value = store.verify_receipt(
            receipt,
            bundle=bundle,
            claim=claim,
            schema=schema,
            system=system,
            payload=payload,
        )
        if isinstance(value, Diagnostic):
            diagnostics.append(value)
        else:
            proposal = cast(Proposal, value)
        return value

    result = run_rule_evidence(data, replay_content)
    if next(observed, None) is not None:
        raise OpenWorldInvocationConflictError("unused Content response")
    validation = validate_evidence_edit(
        bundle=bundle,
        incumbent=incumbent,
        documents=documents,
        trajectory_json_by_session=trajectories,
        diagnostics=diagnostics,
        proposal=proposal,
    )
    common = _common(
        job_id=job_id,
        bundle=bundle,
        authority=authority,
        diagnostics_artifact_digest=diagnostic_artifact.content_digest,
        population_digest=validation.diagnostic_population_digest,
    )
    receipt = {k: content[k] for k in ("prepared", "invocation", "observed_output")}
    if collection[
        "diagnostic_population_digest"
    ] != validation.diagnostic_population_digest or not _same(
        content, _content(common, incumbent, result, receipt)
    ):
        raise OpenWorldInvocationConflictError("admitted Content result changed")
    frame = _frame(
        bundle, incumbent, content["final_content"], content_artifact.content_digest
    )
    used: dict[str, dict[str, Any]] = {}

    def replay_trigger(
        system: str, payload: dict[str, object], schema: type[StrictStructuredOutput]
    ) -> StrictStructuredOutput:
        key = "gate" if schema is TriggerGate else "builder"
        receipt = trigger.get(key + "_receipt")
        if not isinstance(receipt, dict) or key in used:
            raise OpenWorldInvocationConflictError(
                "Trigger response missing or duplicated"
            )
        value = store.verify_receipt(
            receipt,
            bundle=bundle,
            claim=claim,
            schema=schema,
            system=system,
            payload=payload,
            source_frame_digest=frame.digest,
            content_result_artifact_digest=content_artifact.content_digest,
        )
        used[key] = receipt
        return value

    resolved = run_trigger_maintenance(frame, replay_trigger)
    if not _same(trigger, _trigger(common, frame, resolved, used)):
        raise OpenWorldInvocationConflictError("admitted Trigger result changed")
    if not _same(
        candidate,
        _candidate(common, frame, content, trigger_artifact.content_digest, resolved),
    ):
        raise OpenWorldInvocationConflictError("admitted final candidate changed")
    if not _same(
        decision,
        _decision(
            common,
            content_artifact.content_digest,
            trigger_artifact.content_digest,
            candidate_artifact.content_digest,
        ),
    ):
        raise OpenWorldInvocationConflictError("accepted final decision changed")
    if not job.attempt_key:
        raise OpenWorldInvocationConflictError(
            "publication attempt identity is missing"
        )
    return RuleEvidencePublicationInputs(
        job_id=job_id,
        attempt_key=job.attempt_key,
        incumbent_user_playbook_id=incumbent.user_playbook_id,
        incumbent_full_version_fingerprint=bundle.target.incumbent_full_version_fingerprint,
        evidence_bundle_digest=bundle.digest,
        outcome_contract_digest=bundle.task_context.digest,
        candidate=RuleEvidenceCandidate(
            FinalPlaybookRevision(
                revised_content=candidate["revised_content"],
                revised_trigger=candidate["revised_trigger"],
            ),
            FinalContentDiff(**content["content_diff"]),
        ),
        candidate_digest=candidate_artifact.content_digest,
        decision=RuleEvidenceDecision(
            bundle.digest,
            candidate_artifact.content_digest,
            validation.diagnostic_population_digest,
            AcceptDecision(decision="accept"),
        ),
        decision_digest=decision_artifact.content_digest,
        decision_json=decision_artifact.content_json,
        attempt_context_json=context.context_json,
        attempt_context_digest=context.context_digest,
        subject_epochs=OpenWorldEvidenceBundleStore._complete_subject_epochs(bundle),
        qualification_authority=authority,
        claim=claim,
        content_result_artifact_digest=content_artifact.content_digest,
        trigger_result_artifact_digest=trigger_artifact.content_digest,
    )
