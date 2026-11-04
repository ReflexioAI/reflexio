"""Canonical serialization and stable identities for open-world calls."""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from ..evidence.manifests import canonical_json_bytes
from .models import (
    AbstainDecision,
    AcceptDecision,
    AnalystInputIdentity,
    AnalystQualificationResultRef,
    AttemptAnalystRef,
    AttemptDecisionEnvelope,
    AttemptTerminalFailure,
    CandidateCitationRejectionError,
    CandidateCitationRepairRequest,
    CandidateContentDiff,
    CandidateEnvelope,
    CandidateRequest,
    CandidateRevisionV1,
    CitationRepairRequest,
    ComponentIdentity,
    ConfirmationEnvelope,
    ConfirmationSessionMemo,
    ConfirmationSessionRequest,
    DiscoveryMemo,
    DiscoveryRequest,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSpan,
    HeldOutCandidateBinding,
    HeldOutEnvelope,
    HeldOutSessionMemo,
    HeldOutSessionRequest,
    InvocationIntent,
    InvocationRecord,
    ObservedStructuredCompletion,
    OpenWorldCapabilityUnavailableError,
    OutcomeContractDescriptor,
    PinnedGeneratorAuthorization,
    PreparedInvocation,
    ProvisionalCaseAnalyst,
    QualifiedAnalyst,
    RejectedCitationId,
    VerifiedCitation,
    VerifiedDiscoveryEnvelope,
    _candidate_content_diff,
    _candidate_envelope,
    _confirmation_envelope,
    _held_out_envelope,
    _pinned_generator_authorization,
    _prepared_invocation,
    _verified_discovery_envelope,
    held_out_candidate_binding,
)

_CONTRACT_TYPES = {
    EvidenceCitation: "open_world_evidence_citation",
    DiscoveryMemo: "open_world_discovery_memo",
    CandidateRevisionV1: "open_world_candidate_revision",
    HeldOutSessionMemo: "open_world_held_out_session_memo",
    AcceptDecision: "open_world_accept_decision",
    AbstainDecision: "open_world_abstain_decision",
    AttemptTerminalFailure: "open_world_attempt_terminal_failure",
    OutcomeContractDescriptor: "open_world_outcome_contract_descriptor",
    EvidenceSpan: "open_world_evidence_span",
    EvidenceDocument: "open_world_evidence_document",
    CandidateContentDiff: "open_world_candidate_content_diff",
    PinnedGeneratorAuthorization: "open_world_pinned_generator_authorization",
    VerifiedCitation: "open_world_verified_citation",
    InvocationIntent: "open_world_invocation_intent",
    InvocationRecord: "open_world_invocation_record",
    ObservedStructuredCompletion: "open_world_observed_structured_completion",
    AnalystQualificationResultRef: "open_world_analyst_qualification_result_ref",
    PreparedInvocation: "open_world_prepared_invocation",
    VerifiedDiscoveryEnvelope: "open_world_verified_discovery_envelope",
    CandidateEnvelope: "open_world_candidate_envelope",
    HeldOutEnvelope: "open_world_held_out_envelope",
    AttemptDecisionEnvelope: "open_world_attempt_decision_envelope",
    QualifiedAnalyst: "open_world_qualified_analyst",
    ProvisionalCaseAnalyst: "open_world_provisional_case_analyst",
    DiscoveryRequest: "open_world_discovery_request",
    CandidateRequest: "open_world_candidate_request",
    HeldOutSessionRequest: "open_world_held_out_session_request",
    ConfirmationSessionMemo: "open_world_confirmation_session_memo",
    ConfirmationEnvelope: "open_world_confirmation_envelope",
    ConfirmationSessionRequest: "open_world_confirmation_session_request",
}


def canonical_payload(
    value: object,
) -> dict[str, object] | list[object] | str | int | bool | None:
    """Return the complete canonical JSON payload for an open-world contract."""
    if type(value) is ComponentIdentity:
        return {
            "schema_version": "offline-tuner-open-world-component-identity-v1",
            **_dataclass_payload(value),
        }
    if type(value) is AnalystInputIdentity:
        # Attempt 1 keeps the v1 preimage EXACTLY -- `_OMITTED_WHEN_DEFAULT`
        # drops the `attempt` key at its default, and the schema_version stays
        # v1 -- so every row already written keeps its digest and the SQL
        # authorizer's v1 branch keeps authorizing it. Only a retry takes v2.
        # `attempt` sorts ahead of every other key under RFC 8785, which a
        # hand-ordered SQL concat of the preimage must mirror.
        suffix = "v1" if value.attempt == 1 else "v2"
        return {
            "schema_version": (
                f"offline-tuner-open-world-analyst-input-identity-{suffix}"
            ),
            **_dataclass_payload(value),
        }
    contract_type = _CONTRACT_TYPES.get(type(value))
    if contract_type is not None:
        return {
            "contract_type": contract_type,
            "schema_version": "v2" if type(value) is AttemptDecisionEnvelope else "v1",
            "payload": _json_value(value),
        }
    return _json_value(value)


def canonical_bytes(value: object) -> bytes:
    """Serialize a contract with the repository's RFC 8785-compatible encoder."""
    return canonical_json_bytes(canonical_payload(value))


def component_identity_digest(identity: ComponentIdentity) -> str:
    if type(identity) is not ComponentIdentity:
        raise TypeError("identity must be ComponentIdentity")
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def analyst_input_identity(
    *,
    identity: ComponentIdentity,
    qualification_result_digest: str,
    rendered_input_digest: str,
    outcome_contract_digest: str,
    role: Literal["discovery", "candidate", "held_out", "confirmation"],
    attempt: int = 1,
) -> AnalystInputIdentity:
    if type(identity) is not ComponentIdentity:
        raise TypeError("identity must be ComponentIdentity")
    if identity.component_role != role:
        raise ValueError("analyst input role does not match component identity role")
    return AnalystInputIdentity(
        component_identity_digest=identity.digest,
        qualification_result_digest=qualification_result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract_digest,
        role=role,
        attempt=attempt,
    )


def analyst_input_identity_digest(identity: AnalystInputIdentity) -> str:
    if type(identity) is not AnalystInputIdentity:
        raise TypeError("identity must be AnalystInputIdentity")
    return hashlib.sha256(canonical_bytes(identity)).hexdigest()


def validate_outcome_contract_for_target(
    descriptor: OutcomeContractDescriptor, target_success_definition_digest: str
) -> OutcomeContractDescriptor:
    """Require the persisted outcome descriptor to bind the frozen evidence target."""
    if type(descriptor) is not OutcomeContractDescriptor:
        raise TypeError("descriptor must be OutcomeContractDescriptor")
    if descriptor.digest != target_success_definition_digest:
        raise ValueError("outcome contract digest does not match the Phase 2 target")
    return descriptor


def invocation_record(
    *,
    intent: InvocationIntent,
    completion: ObservedStructuredCompletion,
) -> InvocationRecord:
    """Bind one observed structured completion to its prepared provider intent."""
    if type(intent) is not InvocationIntent:
        raise TypeError("intent must be InvocationIntent")
    if type(completion) is not ObservedStructuredCompletion:
        raise TypeError("completion must be ObservedStructuredCompletion")
    if completion.observed_provider != intent.expected_provider:
        raise OpenWorldCapabilityUnavailableError(
            "observed provider does not match invocation intent"
        )
    if completion.observed_model_revision != intent.expected_model_revision:
        raise OpenWorldCapabilityUnavailableError(
            "observed model revision does not match invocation intent"
        )
    return InvocationRecord(
        analyst_input_identity=intent.analyst_input_identity,
        component_identity_digest=intent.component_identity_digest,
        output_digest=hashlib.sha256(canonical_bytes(completion.value)).hexdigest(),
        observed_provider=completion.observed_provider,
        observed_model_revision=completion.observed_model_revision,
        called_at=completion.called_at,
        latency_ms=completion.latency_ms,
        usage_json=completion.usage_json,
    )


def prepare_discovery_request(
    *,
    job_id: int,
    bundle_digest: str,
    discovery_sessions: tuple[EvidenceDocument, ...],
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
    citation_repair: CitationRepairRequest | None = None,
    attempt: int = 1,
) -> DiscoveryRequest:
    rendered_input_digest = discovery_request_input_digest(
        bundle_digest=bundle_digest,
        discovery_sessions=discovery_sessions,
        outcome_contract=outcome_contract,
        analyst=analyst,
    )
    prepared = _prepare_invocation(
        job_id=job_id,
        bundle_digest=bundle_digest,
        identity=analyst.identity,
        qualification=analyst.qualification,
        qualification_result_digest=analyst.qualification.result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract.digest,
        role="discovery",
        attempt=attempt,
    )
    return DiscoveryRequest(
        job_id=job_id,
        bundle_digest=bundle_digest,
        discovery_sessions=discovery_sessions,
        outcome_contract=outcome_contract,
        analyst=analyst,
        rendered_input_digest=rendered_input_digest,
        prepared_invocation=prepared,
        citation_repair=citation_repair,
    )


def prepare_candidate_request(
    *,
    job_id: int,
    bundle_digest: str,
    incumbent_content: str,
    incumbent_full_version_fingerprint: str,
    discovery: VerifiedDiscoveryEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    generator_authorization: PinnedGeneratorAuthorization,
    attempt: int = 1,
) -> CandidateRequest:
    if type(generator_authorization) is not PinnedGeneratorAuthorization:
        raise TypeError("generator_authorization must be PinnedGeneratorAuthorization")
    generator_identity = generator_authorization.generator_identity
    rendered_input_digest = candidate_request_input_digest(
        bundle_digest=bundle_digest,
        incumbent_content=incumbent_content,
        incumbent_full_version_fingerprint=incumbent_full_version_fingerprint,
        discovery=discovery,
        outcome_contract=outcome_contract,
        generator_authorization=generator_authorization,
    )
    prepared = _prepare_invocation(
        job_id=job_id,
        bundle_digest=bundle_digest,
        identity=generator_identity,
        qualification=generator_authorization,
        qualification_result_digest=generator_authorization.digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract.digest,
        role="candidate",
        attempt=attempt,
    )
    return CandidateRequest(
        job_id=job_id,
        bundle_digest=bundle_digest,
        incumbent_content=incumbent_content,
        incumbent_full_version_fingerprint=incumbent_full_version_fingerprint,
        discovery=discovery,
        outcome_contract=outcome_contract,
        generator_authorization=generator_authorization,
        rendered_input_digest=rendered_input_digest,
        prepared_invocation=prepared,
    )


def prepare_held_out_session_request(
    *,
    job_id: int,
    bundle_digest: str,
    manifest_role: Literal["held_out", "safety_only"],
    evidence: EvidenceDocument,
    discovery: VerifiedDiscoveryEnvelope,
    candidate: CandidateEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
    attempt: int = 1,
) -> HeldOutSessionRequest:
    rendered_input_digest = held_out_request_input_digest(
        bundle_digest=bundle_digest,
        manifest_role=manifest_role,
        evidence=evidence,
        discovery=discovery,
        candidate=candidate,
        outcome_contract=outcome_contract,
        analyst=analyst,
    )
    prepared = _prepare_invocation(
        job_id=job_id,
        bundle_digest=bundle_digest,
        identity=analyst.identity,
        qualification=analyst.qualification,
        qualification_result_digest=analyst.qualification.result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract.digest,
        role="held_out",
        attempt=attempt,
        manifest_role=manifest_role,
        user_id=evidence.user_id,
        session_id=evidence.session_id,
        trajectory_digest=evidence.trajectory_digest,
        candidate_binding=held_out_candidate_binding(candidate),
    )
    return HeldOutSessionRequest(
        job_id=job_id,
        bundle_digest=bundle_digest,
        manifest_role=manifest_role,
        evidence=evidence,
        discovery=discovery,
        candidate=candidate,
        outcome_contract=outcome_contract,
        analyst=analyst,
        rendered_input_digest=rendered_input_digest,
        prepared_invocation=prepared,
    )


def prepare_confirmation_session_request(
    *,
    job_id: int,
    bundle_digest: str,
    evidence: EvidenceDocument,
    discovery: VerifiedDiscoveryEnvelope,
    candidate: CandidateEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
    attempt: int = 1,
) -> ConfirmationSessionRequest:
    """Prepare one online session's confirmation call.

    ``prepare_held_out_session_request`` minus ``manifest_role``: an online
    session belongs to no frozen manifest, so ``manifest_role`` is NULL all the
    way down to the durable invocation row.
    """
    rendered_input_digest = confirmation_request_input_digest(
        bundle_digest=bundle_digest,
        evidence=evidence,
        discovery=discovery,
        candidate=candidate,
        outcome_contract=outcome_contract,
        analyst=analyst,
    )
    prepared = _prepare_invocation(
        job_id=job_id,
        bundle_digest=bundle_digest,
        identity=analyst.identity,
        qualification=analyst.qualification,
        qualification_result_digest=analyst.qualification.result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract.digest,
        role="confirmation",
        attempt=attempt,
        manifest_role=None,
        user_id=evidence.user_id,
        session_id=evidence.session_id,
        trajectory_digest=evidence.trajectory_digest,
        candidate_binding=held_out_candidate_binding(candidate),
    )
    return ConfirmationSessionRequest(
        job_id=job_id,
        bundle_digest=bundle_digest,
        evidence=evidence,
        discovery=discovery,
        candidate=candidate,
        outcome_contract=outcome_contract,
        analyst=analyst,
        rendered_input_digest=rendered_input_digest,
        prepared_invocation=prepared,
    )


def build_discovery_envelope(
    *,
    request: DiscoveryRequest,
    observed_memo: DiscoveryMemo,
    invocation: InvocationRecord,
) -> VerifiedDiscoveryEnvelope:
    return _verified_discovery_envelope(
        request=request,
        observed_memo=observed_memo,
        invocation=invocation,
    )


def build_candidate_envelope(
    *,
    request: CandidateRequest,
    revision: CandidateRevisionV1,
    invocation: InvocationRecord,
) -> CandidateEnvelope:
    if revision.revised_content == request.incumbent_content:
        raise ValueError("candidate content is unchanged")
    available_citation_ids = {
        citation.citation_id for citation in request.discovery.verified_citations
    }
    rejected_citation_ids = tuple(
        citation_id
        for citation_id in revision.supporting_citation_ids
        if citation_id not in available_citation_ids
    )
    if rejected_citation_ids:
        # Membership is unchanged -- exact, byte for byte. What is new is that
        # the offending IDs travel with the error, so a repair turn can name
        # them instead of re-deriving this same subset check at the catch site.
        raise CandidateCitationRejectionError(rejected_citation_ids)
    incumbent_digest = hashlib.sha256(request.incumbent_content.encode()).hexdigest()
    candidate_digest = hashlib.sha256(revision.revised_content.encode()).hexdigest()
    diff = "".join(
        difflib.unified_diff(
            request.incumbent_content.splitlines(keepends=True),
            revision.revised_content.splitlines(keepends=True),
            fromfile="incumbent",
            tofile="candidate",
        )
    )
    return _candidate_envelope(
        bundle_digest=request.bundle_digest,
        incumbent_full_version_fingerprint=(request.incumbent_full_version_fingerprint),
        revision=revision,
        content_diff=_candidate_content_diff(
            incumbent_content_digest=incumbent_digest,
            candidate_content_digest=candidate_digest,
            unified_diff=diff,
        ),
        verified_discovery_citation_ids=revision.supporting_citation_ids,
        generator_identity=request.generator_identity,
        invocation=invocation,
        prepared_invocation=request.prepared_invocation,
    )


def build_held_out_envelope(
    *,
    request: HeldOutSessionRequest,
    observed_memo: HeldOutSessionMemo,
    invocation: InvocationRecord,
) -> HeldOutEnvelope:
    return _held_out_envelope(
        request=request,
        observed_memo=observed_memo,
        invocation=invocation,
    )


def build_confirmation_envelope(
    *,
    request: ConfirmationSessionRequest,
    observed_memo: ConfirmationSessionMemo,
    invocation: InvocationRecord,
) -> ConfirmationEnvelope:
    return _confirmation_envelope(
        request=request,
        observed_memo=observed_memo,
        invocation=invocation,
    )


def _prepare_invocation(
    *,
    job_id: int,
    bundle_digest: str,
    identity: ComponentIdentity,
    qualification: AnalystQualificationResultRef | PinnedGeneratorAuthorization,
    qualification_result_digest: str,
    rendered_input_digest: str,
    outcome_contract_digest: str,
    role: Literal["discovery", "candidate", "held_out", "confirmation"],
    manifest_role: Literal["held_out", "safety_only"] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trajectory_digest: str | None = None,
    candidate_binding: HeldOutCandidateBinding | None = None,
    attempt: int = 1,
) -> PreparedInvocation:
    input_identity = analyst_input_identity(
        identity=identity,
        qualification_result_digest=qualification_result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract_digest,
        role=role,
        attempt=attempt,
    )
    intent = InvocationIntent(
        job_id=job_id,
        analyst_input_identity=input_identity.digest,
        component_identity_digest=identity.digest,
        input_digest=rendered_input_digest,
        role=role,
        expected_provider=identity.provider,
        expected_model_revision=identity.immutable_model_revision,
        manifest_role=manifest_role,
        user_id=user_id,
        session_id=session_id,
    )
    return _prepared_invocation(
        job_id=job_id,
        bundle_digest=bundle_digest,
        component_identity=identity,
        qualification=qualification,
        qualification_result_digest=qualification_result_digest,
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract_digest,
        role=role,
        manifest_role=manifest_role,
        user_id=user_id,
        session_id=session_id,
        trajectory_digest=trajectory_digest,
        candidate_binding=candidate_binding,
        analyst_input_identity=input_identity,
        intent=intent,
    )


def discovery_request_input_digest(
    *,
    bundle_digest: str,
    discovery_sessions: tuple[EvidenceDocument, ...],
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
) -> str:
    return _request_input_digest(
        "open_world_discovery_request_input",
        model_input=discovery_model_input(
            bundle_digest=bundle_digest,
            discovery_sessions=discovery_sessions,
            outcome_contract=outcome_contract,
        ),
        semantic_bindings={
            "analyst": analyst,
        },
    )


def candidate_request_input_digest(
    *,
    bundle_digest: str,
    incumbent_content: str,
    incumbent_full_version_fingerprint: str,
    discovery: VerifiedDiscoveryEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    generator_authorization: PinnedGeneratorAuthorization,
) -> str:
    return _request_input_digest(
        "open_world_candidate_request_input",
        model_input=candidate_model_input(
            bundle_digest=bundle_digest,
            discovery=discovery,
            incumbent_content=incumbent_content,
            outcome_contract=outcome_contract,
        ),
        semantic_bindings={
            "discovery_analyst_input_identity": (
                discovery.prepared_invocation.analyst_input_identity.digest
            ),
            "incumbent_full_version_fingerprint": incumbent_full_version_fingerprint,
            "generator_authorization": generator_authorization,
        },
    )


def held_out_request_input_digest(
    *,
    bundle_digest: str,
    manifest_role: Literal["held_out", "safety_only"],
    evidence: EvidenceDocument,
    discovery: VerifiedDiscoveryEnvelope,
    candidate: CandidateEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
) -> str:
    return _request_input_digest(
        "open_world_held_out_request_input",
        model_input=held_out_model_input(
            bundle_digest=bundle_digest,
            candidate=candidate,
            discovery=discovery,
            evidence=evidence,
            manifest_role=manifest_role,
            outcome_contract=outcome_contract,
        ),
        semantic_bindings={
            "analyst": analyst,
        },
    )


def confirmation_request_input_digest(
    *,
    bundle_digest: str,
    evidence: EvidenceDocument,
    discovery: VerifiedDiscoveryEnvelope,
    candidate: CandidateEnvelope,
    outcome_contract: OutcomeContractDescriptor,
    analyst: AttemptAnalystRef,
) -> str:
    return _request_input_digest(
        "open_world_confirmation_request_input",
        model_input=confirmation_model_input(
            bundle_digest=bundle_digest,
            candidate=candidate,
            discovery=discovery,
            evidence=evidence,
            outcome_contract=outcome_contract,
        ),
        semantic_bindings={
            "analyst": analyst,
        },
    )


def discovery_model_input(
    *,
    bundle_digest: str,
    discovery_sessions: tuple[EvidenceDocument, ...],
    outcome_contract: OutcomeContractDescriptor,
) -> dict[str, object]:
    """Identify the QUESTION put to the discovery analyst.

    A citation repair is deliberately NOT part of this mapping, and therefore
    not part of ``analyst_input_identity``. The durable model already states
    the rule this follows: ``uq_open_world_invocations_manifest_slot`` admits
    exactly one discovery invocation per job, which is the schema saying that a
    job asks discovery ONE question about one frozen bundle. A repair does not
    ask a second question -- it re-asks this one, having told the analyst that
    its transcription of the answer was rejected -- so it shares this identity,
    this durable row, and this evidence.

    What must NOT be shared is the provider idempotency key, or the repair
    would invite the provider to replay the very answer it exists to replace;
    ``repair_attempt_idempotency_key`` derives a distinct key per attempt.

    Use ``discovery_model_payload`` to build what is actually SENT: it is this
    mapping plus the repair. The two are one function calling the other so that
    they cannot drift apart.
    """
    return {
        "bundle_digest": bundle_digest,
        "discovery_sessions": discovery_sessions,
        "outcome_contract": outcome_contract,
    }


def discovery_model_payload(
    *,
    bundle_digest: str,
    discovery_sessions: tuple[EvidenceDocument, ...],
    outcome_contract: OutcomeContractDescriptor,
    citation_repair: CitationRepairRequest | None = None,
) -> dict[str, object]:
    """Build what is actually sent to the discovery analyst.

    This is ``discovery_model_input`` plus the repair, when there is one. The
    repair key is omitted rather than emitted as null when absent, so a first
    attempt serialises to exactly the bytes it did before repair existed and
    the packaged qualification fixture's canonical user messages still match.
    """
    payload = discovery_model_input(
        bundle_digest=bundle_digest,
        discovery_sessions=discovery_sessions,
        outcome_contract=outcome_contract,
    )
    if citation_repair is not None:
        payload["citation_repair"] = citation_repair
    return payload


def repair_attempt_idempotency_key(*, base_key: str, attempt: int) -> str:
    """Derive one attempt's provider idempotency key from the shared identity.

    Shared by the discovery and candidate repair loops. They cannot collide
    despite the ``schema_version`` naming only discovery: ``base_key`` is the
    stage's own ``analyst_input_identity`` digest, which already separates them.

    Every attempt at one question shares an ``analyst_input_identity``
    but must not share an idempotency key: a provider that honours the header is
    entitled to replay the first answer for the second request, which would
    silently turn a repair into a no-op.

    Args:
        base_key (str): The prepared invocation's idempotency key
        attempt (int): The 1-based attempt number

    Returns:
        str: A 64-character hex key, distinct per attempt, matching the width
            the provider clients require
    """
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    if attempt == 1:
        # The first attempt keeps the identity's own key verbatim, so nothing
        # about an un-repaired call changes.
        return base_key
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "open-world-discovery-repair-attempt-key-v1",
                "base_key": base_key,
                "attempt": attempt,
            }
        )
    ).hexdigest()


# NOTE -- the per-attempt DURABLE ROW identity (`AnalystInputIdentity.attempt`).
#
# `analyst_input_identity` is not merely a row key the Python side chooses. A
# storage backend may RECOMPUTE it from the canonical preimage before admitting
# any completed artifact, for example in SQL:
#
#     v_input_identity_preimage := concat(
#         '{"component_identity_digest":', ...
#         ',"schema_version":"offline-tuner-open-world-analyst-input-identity-v1"}'
#     );
#     IF ... encode(sha256(convert_to(v_input_identity_preimage, 'UTF8')), 'hex')
#            <> v_inv.analyst_input_identity
#         THEN RAISE EXCEPTION 'artifact does not bind the exact completed invocation';
#
# Advancing the identity in Python alone cannot work: prepare advances, the
# analyst is CALLED and answers, and persisting the memo is then refused --
# discarding the answer and reporting `infrastructure_failure`. Any backend
# that recomputes the preimage must carry the attempt-aware form: `attempt` is
# an optional sixth key, and a missing key means 1.
#
# Two properties make it safe:
#
#  1. ATTEMPT 1 IS BYTE-IDENTICAL to the pre-attempt form. `attempt` is omitted
#     at its default (`_OMITTED_WHEN_DEFAULT`) and the schema_version stays v1,
#     so every row already written keeps its digest and no backfill is needed.
#  2. THE ENGINES MUST AGREE at attempt 1 AND attempt 2. A backend that
#     recomputes the preimage needs a test that drives prepare -> complete ->
#     persist against its real authorizer rather than a copied preimage.
#
# If you change one side of that preimage, change the other in the same commit,
# and make that test prove it. Python-side alone is a paid-failure generator.


def candidate_model_input(
    *,
    bundle_digest: str,
    discovery: VerifiedDiscoveryEnvelope,
    incumbent_content: str,
    outcome_contract: OutcomeContractDescriptor,
) -> dict[str, object]:
    return {
        "bundle_digest": bundle_digest,
        "discovery_memo": discovery.memo,
        "incumbent_content": incumbent_content,
        "outcome_contract": outcome_contract,
        "verified_discovery_citation_ids": tuple(
            citation.citation_id for citation in discovery.verified_citations
        ),
    }


def candidate_model_payload(
    *,
    bundle_digest: str,
    discovery: VerifiedDiscoveryEnvelope,
    incumbent_content: str,
    outcome_contract: OutcomeContractDescriptor,
    citation_repair: CandidateCitationRepairRequest | None = None,
) -> dict[str, object]:
    """Build what is actually sent to the candidate generator.

    This is ``candidate_model_input`` plus the repair, when there is one. The
    repair key is omitted rather than emitted as null when absent, so a first
    attempt serialises to exactly the bytes it did before repair existed and
    the packaged qualification fixture's canonical user messages still match.

    The two are one function calling the other so that the QUESTION and what is
    SENT cannot drift apart.
    """
    payload = candidate_model_input(
        bundle_digest=bundle_digest,
        discovery=discovery,
        incumbent_content=incumbent_content,
        outcome_contract=outcome_contract,
    )
    if citation_repair is not None:
        payload["citation_repair"] = citation_repair
    return payload


# What a rejected supporting citation ID tells the generator. A fixed string,
# chosen once here and never assembled from evidence, so the repair payload's
# vocabulary cannot vary with the tenant's data.
_CANDIDATE_REPAIR_REASON = (
    "this is not one of the verified_discovery_citation_ids supplied with your "
    "input -- it was not found in that list, character for character. Copy an "
    "ID from that list exactly, or drop the claim that needed it"
)


def candidate_citation_repair_request(
    *,
    attempt: int,
    rejected_citation_ids: Sequence[str],
) -> CandidateCitationRepairRequest | None:
    """Describe the rejected supporting IDs so the generator can repair them.

    Takes **no evidence and no verified IDs** -- only strings the model itself
    emitted. That signature is the guarantee, not a comment about one: there is
    nothing in scope here that a repair payload could leak, however it is
    called.

    Args:
        attempt (int): The 1-based number of the attempt this repair will feed,
            which is always 2 or more
        rejected_citation_ids (Sequence[str]): The IDs the candidate offered
            that were not in the verified set

    Returns:
        CandidateCitationRepairRequest | None: The repair to attach to the next
            attempt, or ``None`` when nothing was rejected
    """
    rejected = tuple(
        RejectedCitationId(citation_id=citation_id, reason=_CANDIDATE_REPAIR_REASON)
        for citation_id in dict.fromkeys(rejected_citation_ids)
    )
    if not rejected:
        return None
    return CandidateCitationRepairRequest(
        attempt=attempt, rejected_citation_ids=rejected
    )


def held_out_model_input(
    *,
    bundle_digest: str,
    candidate: CandidateEnvelope,
    discovery: VerifiedDiscoveryEnvelope,
    evidence: EvidenceDocument,
    manifest_role: Literal["held_out", "safety_only"],
    outcome_contract: OutcomeContractDescriptor,
) -> dict[str, object]:
    return {
        "bundle_digest": bundle_digest,
        "candidate_diff": candidate.content_diff,
        "hypothesis": discovery.memo.hypothesis,
        "evidence": evidence,
        "manifest_role": manifest_role,
        "outcome_contract": outcome_contract,
    }


def confirmation_model_input(
    *,
    bundle_digest: str,
    candidate: CandidateEnvelope,
    discovery: VerifiedDiscoveryEnvelope,
    evidence: EvidenceDocument,
    outcome_contract: OutcomeContractDescriptor,
) -> dict[str, object]:
    """``held_out_model_input`` minus ``manifest_role``.

    ``hypothesis`` and ``candidate_diff`` are here for the same reason they are
    in the held-out input, and they are what makes L2 bite: substituting either
    frozen artifact changes this mapping, therefore the digest, therefore the
    request cannot be constructed against a stale one.
    """
    return {
        "bundle_digest": bundle_digest,
        "candidate_diff": candidate.content_diff,
        "hypothesis": discovery.memo.hypothesis,
        "evidence": evidence,
        "outcome_contract": outcome_contract,
    }


def _request_input_digest(
    contract_type: str,
    *,
    model_input: Mapping[str, object],
    semantic_bindings: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_type": contract_type,
                "schema_version": "v1",
                "payload": {
                    "model_input": _json_value(model_input),
                    "semantic_bindings": _request_json_value(semantic_bindings),
                },
            }
        )
    ).hexdigest()


def pinned_generator_authorization(
    identity: ComponentIdentity,
) -> PinnedGeneratorAuthorization:
    return _pinned_generator_authorization(
        generator_identity=identity,
        digest=pinned_generator_authorization_digest(identity),
    )


def pinned_generator_authorization_digest(identity: ComponentIdentity) -> str:
    if (
        type(identity) is not ComponentIdentity
        or identity.component_role != "candidate"
    ):
        raise TypeError("generator authorization requires candidate ComponentIdentity")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_type": "open_world_pinned_generator_authorization",
                "schema_version": "v1",
                "component_identity": canonical_payload(identity),
            }
        )
    ).hexdigest()


# Fields added to a contract AFTER it was frozen, which are omitted from the
# canonical payload when they hold their absent value rather than serialised as
# an explicit null.
#
# The reason is compatibility of digests, and it is load-bearing rather than
# cosmetic. `DiscoveryRequest.citation_repair` is absent on every first attempt,
# which is every attempt the packaged qualification fixture describes. Emitting
# `"citation_repair": null` there would restate the canonical request bytes of
# four frozen suite cases, changing digests that attest something the change did
# not touch. Omitting the absent value keeps a first attempt byte-identical to
# what it was before repair existed, so only a genuine REPAIR attempt -- which
# really is a different question -- gets a different digest.
#
# This mirrors `discovery_model_input`, which omits the same field on the same
# terms. Keep the two in step: a field listed here must be omitted from the
# model input too, or a request and the payload it renders would disagree.
#
# `CandidateRequest.citation_repair` is listed for exactly that reason.
# `candidate_model_payload` already omits the key when there is no repair --
# "the two are one function calling the other so that the QUESTION and what is
# SENT cannot drift apart" -- but the canonical request was still emitting
# `"citation_repair": null`, so the request and the payload it renders
# disagreed on a first attempt. That is the asymmetry this table exists to
# prevent, in the one place it had reappeared.
#
# Listing it moves no durable digest. `CandidateEnvelope` does not embed the
# request, and `_request_digest` in `qualification.py` is typed
# `DiscoveryRequest | HeldOutSessionRequest`, so no packaged suite fixture and
# no persisted artifact canonicalizes a `CandidateRequest` at all.
_OMITTED_WHEN_ABSENT: dict[str, frozenset[str]] = {
    "DiscoveryRequest": frozenset({"citation_repair"}),
    "CandidateRequest": frozenset({"citation_repair"}),
}

# The same rule as `_OMITTED_WHEN_ABSENT`, for a field whose "absent" value is a
# DEFAULT rather than ``None``.
#
# `AnalystInputIdentity.attempt` has to be readable by PostgreSQL -- the tenant
# authorizer recomputes the row identity from this very payload, so an attempt
# it cannot see is an attempt it cannot verify. But emitting `"attempt":1` on a
# first attempt would change the digest of every request, envelope and suite
# fixture that nests an identity, for a value that carries no information.
# Omitting it at the default keeps a first attempt byte-identical to what it was
# before retries existed, and the SQL side COALESCEs a missing key to 1 to
# match. Only a genuine RETRY -- which really is a different durable row -- gets
# a different digest.
_OMITTED_WHEN_DEFAULT: dict[str, dict[str, object]] = {
    "AnalystInputIdentity": {"attempt": 1},
}


def _canonical_fields(value: Any) -> tuple[Any, ...]:
    """Yield the dataclass fields that belong in a canonical payload."""
    type_name = type(value).__name__
    omitted = _OMITTED_WHEN_ABSENT.get(type_name, frozenset())
    defaults = _OMITTED_WHEN_DEFAULT.get(type_name, {})
    return tuple(
        field
        for field in dataclasses.fields(value)
        if not field.name.startswith("_")
        and not (field.name in omitted and getattr(value, field.name) is None)
        and not (
            field.name in defaults
            and getattr(value, field.name) == defaults[field.name]
        )
    )


def _dataclass_payload(value: Any) -> dict[str, object]:
    return {
        field.name: _json_value(getattr(value, field.name))
        for field in _canonical_fields(value)
    }


# Constant span identity, carried ONCE per document in the rendered payload.
#
# Every span in a document shares its tenant, its manifest arm, its user and
# its session -- `EvidenceDocument.__post_init__` refuses a span that does not
# -- so repeating them on every span cost the analyst one copy per turn and
# told it nothing it could not read off the document. MEASURED on the live
# payload path: 24 846 -> 8 281 tokens for the same session, 3.0x.
#
# WHAT IS NOT HOISTED, AND WHY:
#
#  * `request_id` is per REQUEST, and a session is a TUPLE of them
#    (`OpenWorldEvidenceSession.request_ids`, which `_validate_event` asserts
#    the span set matches exactly). Hoisting one request's id onto spans drawn
#    from another request is wrong for any multi-request session, and silently
#    so -- every span would still render, naming the wrong request.
#  * `field_path` is positional and differs per span.
#  * `span_id` keeps its full width. `trusted_citation_id_block` admits
#    `[0-9a-f]{64}` precisely because a hex digest cannot express an
#    instruction, and `verify_citations` resolves a mistranscribed id back to
#    the server's own copy of the text. An ordinal would be forgeable and would
#    carry no content binding for that repair ladder to resolve against.
#
# This is a PROJECTION and nothing else. `EvidenceSpan` is untouched, so
# `_span_id`, `_trajectory_digest_from_span_id` and the citation verifier -- all
# of which read identity off the OBJECT, never off the payload -- are
# unaffected. A span rendered outside a document (a semantic binding, say)
# still carries its full identity, because there is then no document to read it
# from.
_HOISTED_SPAN_IDENTITY_FIELDS = frozenset(
    {"tenant_id", "manifest_role", "user_id", "session_id"}
)


def _evidence_document_payload(document: EvidenceDocument) -> dict[str, object]:
    """Project one evidence document with its span identity hoisted."""
    tenants = {span.tenant_id for span in document.spans}
    if len(tenants) > 1:
        # One document is one session and a session belongs to one tenant, so
        # this is not a shape to project -- it is two tenants' text in one
        # analyst call. `EvidenceDocument` checks the other three constants but
        # not this one, so the check lives here rather than nowhere.
        raise ValueError("evidence document spans disagree on tenant")
    payload = _dataclass_payload(document)
    if document.compaction_summary is None:
        payload.pop("compaction_summary", None)
    # `EvidenceDocument` carries no `tenant_id` field of its own; the spans are
    # where it was recorded, and dropping it from them without stating it once
    # here would take tenant identity out of the payload entirely.
    payload["tenant_id"] = tenants.pop() if tenants else None
    payload["spans"] = [
        {
            key: value
            for key, value in _dataclass_payload(span).items()
            if key not in _HOISTED_SPAN_IDENTITY_FIELDS
        }
        for span in document.spans
    ]
    return payload


def _json_value(value: object) -> Any:
    if type(value) is EvidenceDocument:
        return _evidence_document_payload(value)
    if isinstance(value, BaseModel):
        return {
            key: _json_value(item)
            for key, item in value.model_dump(mode="python").items()
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _dataclass_payload(value)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"Unsupported canonical contract value: {type(value).__name__}")


def _request_json_value(value: object) -> Any:
    contract_type = _CONTRACT_TYPES.get(type(value))
    if contract_type is not None:
        return {
            "contract_type": contract_type,
            "schema_version": "v1",
            "payload": _request_dataclass_payload(value),
        }
    if isinstance(value, BaseModel):
        return {
            key: _request_json_value(item)
            for key, item in value.model_dump(mode="python").items()
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _request_dataclass_payload(value)
    if isinstance(value, tuple | list):
        return [_request_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _request_json_value(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool):
        return value
    raise TypeError(f"Unsupported canonical contract value: {type(value).__name__}")


def _request_dataclass_payload(value: Any) -> dict[str, object]:
    return {
        field.name: _request_json_value(getattr(value, field.name))
        for field in _canonical_fields(value)
    }
