"""Durable execution of the one promoted rule-evidence method.

Uses the existing job, invocation and artifact stores. A completed Diagnostic is
stored once per source session; the aggregate Proposal has its own invocation.
No completion evaluator, retired tuner, held-out model or fallback is invoked.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol, cast

from pydantic import TypeAdapter

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    UserPlaybook,
)
from reflexio.models.api_schema.service_schemas import OptimizationJobStage
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm.failure import OutageWait, classify_llm_failure
from reflexio.server.llm.litellm_client import (
    LiteLLMClientError,
    StructuredOutputParseError,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    RULE_EVIDENCE_POLICY_ID,
    OpenWorldEvidenceBundle,
    OpenWorldEvidenceSchemaUnsupportedError,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    ChatMessage,
    OpenWorldProviderRateLimitedError,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    AbstainDecision,
    AcceptDecision,
    AnalystInputIdentity,
    AttemptTerminalFailure,
    AttemptTerminalFailureReason,
    CandidateContentDiff,
    EvidenceDocument,
    InvocationRecord,
    _candidate_content_diff,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    RuleEvidenceMethod,
)
from reflexio.server.services.offline_tuner.open_world.runner import AttemptLease
from reflexio.server.services.offline_tuner.open_world_invocations import (
    CompletionRecord,
    ExpectedProvenance,
    OpenWorldInvocationBinding,
    OpenWorldRefusalError,
    terminal_reason_for,
)
from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldAttemptStorage,
    OpenWorldStageStore,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldAttemptContextRow,
    OpenWorldEvidenceStorageRecord,
    OpenWorldInvocationConflictError,
    OpenWorldInvocationIdentityBoundError,
    OpenWorldInvocationLeaseConflictError,
    OpenWorldInvocationRow,
    OpenWorldInvocationSlotExhaustedError,
    OpenWorldQualificationAuthorityRow,
    OpenWorldQualificationAuthorityStaleError,
)
from reflexio.server.services.playbook.publication import QualificationAuthorityRef
from reflexio.server.services.playbook.rule_evidence import run_rule_evidence
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    contract_schema,
    output_kind,
)
from reflexio.server.services.storage.error import StorageError

from ..evidence.open_world_store import (
    FrozenOpenWorldEvidence,
    OpenWorldEvidenceBundleStore,
    _parse_canonical_bundle,
)
from .diagnostics import (
    TERMINAL_DIAGNOSTIC_METADATA_KEY,
    build_terminal_failure_diagnostic,
    diagnostic_log_fragment,
    diagnostic_payload,
)
from .evidence_documents import RehydratedEvidence
from .rule_evidence import (
    EvidenceEditValidation,
    diagnostic_population_digest,
    native_rule_evidence_input,
    observed_output_bytes,
    validate_evidence_edit,
)


def _json(value: object) -> str:
    return canonical_json_bytes(value).decode()


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


#: Bound on the exception-chain walk; the client's wrapping is two levels deep.
_MAX_CHAIN_DEPTH = 8


def rejected_model_output(exc: BaseException) -> bool:
    """Whether the model answered completely and the schema rejected the answer.

    True only for a written, uncut reply that failed validation: a
    ``StructuredOutputParseError`` in the chain carrying validation errors,
    with a non-empty body that did not stop at the output limit. Everything
    else -- an empty body, a truncated or length-cut body, a transport or
    platform fault -- is not the model's answer and stays
    ``infrastructure_failure``.

    The client wraps the final parse failure in a ``LiteLLMClientError``
    (``raise ... from``), so the parse error is reached through ``__cause__``.
    ONLY ``__cause__`` is followed, never the implicit ``__context__``: the
    corrective retry runs inside the first parse error's handler, so a
    transport failure or timeout on that retry carries the earlier, HANDLED
    rejection as its context. That rejection did not end the attempt -- the
    outage did -- and must not relabel it.

    Args:
        exc (BaseException): The exception that ended the attempt.

    Returns:
        bool: True when a complete reply was rejected by the schema.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if len(seen) >= _MAX_CHAIN_DEPTH:
            return False
        seen.add(id(current))
        if isinstance(current, StructuredOutputParseError):
            return (
                current.failure_kind == "parse"
                and current.finish_reason != "length"
                and bool(current.validation_errors)
            )
        current = current.__cause__
    return False


class RuleEvidenceStorage(OpenWorldAttemptStorage, OpenWorldStageStore, Protocol):
    """What the rule-evidence attempt runs on: the attempt and stage stores of
    ``OpenWorldStoragePort``, plus the frozen-bundle reload and tenant id that
    publication rechecks the source bundle against."""

    org_id: str

    def load_open_world_evidence_bundle(
        self, job_id: int
    ) -> OpenWorldEvidenceStorageRecord: ...


@dataclass(frozen=True)
class RuleEvidenceDecision:
    bundle_digest: str
    candidate_digest: str | None
    diagnostic_population_digest: str
    decision: AcceptDecision | AbstainDecision


@dataclass(frozen=True)
class ContentRevision:
    revised_content: str


@dataclass(frozen=True)
class RuleEvidenceCandidate:
    revision: ContentRevision
    content_diff: CandidateContentDiff


@dataclass(frozen=True)
class RuleEvidencePublicationInputs:
    job_id: int
    attempt_key: str
    incumbent_user_playbook_id: int
    incumbent_full_version_fingerprint: str
    evidence_bundle_digest: str
    outcome_contract_digest: str
    candidate: RuleEvidenceCandidate
    candidate_digest: str
    decision: RuleEvidenceDecision
    decision_digest: str
    decision_json: str
    attempt_context_json: str
    attempt_context_digest: str
    subject_epochs: tuple[tuple[str, int], ...]
    qualification_authority: QualificationAuthorityRef
    claim: OptimizationJobClaim


@dataclass
class RuleEvidenceAnalysisStore:
    storage: RuleEvidenceStorage
    method: RuleEvidenceMethod
    transport: Any
    clock: Callable[[], int]

    def reconcile_authority(self) -> QualificationAuthorityRef:
        row = self.storage.reconcile_rule_evidence_authority(
            authority_json=self.method.canonical_json,
            authority_digest=self.method.digest,
            now=self.clock(),
        )
        return self._authority_ref(row)

    def _authority_ref(
        self, row: OpenWorldQualificationAuthorityRow | None
    ) -> QualificationAuthorityRef:
        if (
            row is None
            or row.state != "active"
            or row.authority_digest != self.method.digest
            or row.method_authority_json != self.method.canonical_json
            or row.discovery_component_identity_digest != self.method.diagnostic.digest
            or row.discovery_qualification_suite_digest != self.method.suite_digest
            or row.discovery_qualification_result_digest != self.method.result_digest
            or row.candidate_generator_identity_digest != self.method.proposal.digest
            or row.candidate_generator_authorization_digest
            != self.method.proposal_authorization_digest
            or any(
                value is not None
                for value in (
                    row.held_out_component_identity_digest,
                    row.held_out_qualification_suite_digest,
                    row.held_out_qualification_result_digest,
                )
            )
        ):
            raise OpenWorldQualificationAuthorityStaleError(
                "rule-evidence authority is missing, stale or incomplete"
            )
        return self.method.authority_ref(epoch=row.epoch)

    def bind_context(
        self, bundle: OpenWorldEvidenceBundle, claim: OptimizationJobClaim
    ) -> tuple[OpenWorldAttemptContextRow, QualificationAuthorityRef]:
        authority = self._authority_ref(
            self.storage.load_open_world_qualification_authority()
        )
        context = self.storage.bind_open_world_attempt_context(
            job_id=claim.job_id,
            bundle_digest=bundle.digest,
            outcome_contract_digest=bundle.task_context.digest,
            authority_epoch=authority.epoch,
            authority_digest=authority.authority_digest,
            fence=claim.fence,
            now=self.clock(),
        )
        if (
            context.job_id != claim.job_id
            or context.bundle_digest != bundle.digest
            or context.outcome_contract_digest != bundle.task_context.digest
            or context.qualification_authority_epoch != authority.epoch
            or context.qualification_authority_digest != authority.authority_digest
            or sha256(context.context_json.encode()).hexdigest()
            != context.context_digest
        ):
            raise OpenWorldInvocationConflictError("attempt context binding changed")
        return context, authority

    def write(
        self, kind: str, payload: dict[str, Any], claim: OptimizationJobClaim
    ) -> PlaybookOptimizationArtifact:
        text = _json(payload)
        digest = sha256(text.encode()).hexdigest()
        artifact = self.storage.write_open_world_artifact(
            job_id=claim.job_id,
            artifact_kind=kind,
            content_json=text,
            content_digest=digest,
            fence=claim.fence,
            now=self.clock(),
        )
        if artifact.content_json != text or artifact.content_digest != digest:
            raise OpenWorldInvocationConflictError("persisted artifact changed")
        return artifact

    def read(
        self, kind: str, contract: str, claim: OptimizationJobClaim
    ) -> tuple[PlaybookOptimizationArtifact, dict[str, Any]] | None:
        artifact = self.storage.read_open_world_artifact(
            job_id=claim.job_id, artifact_kind=kind, fence=claim.fence, now=self.clock()
        )
        if artifact is None:
            return None
        raw = json.loads(artifact.content_json)
        if raw.get("contract_type") != contract or raw.get("schema_version") != "v1":
            raise OpenWorldEvidenceSchemaUnsupportedError(
                "unsupported rule-evidence artifact schema"
            )
        if (
            set(raw) != {"contract_type", "schema_version", "payload"}
            or not isinstance(raw["payload"], dict)
            or _json(raw) != artifact.content_json
            or _digest(raw) != artifact.content_digest
            or artifact.job_id != claim.job_id
        ):
            raise OpenWorldInvocationConflictError("artifact canonical binding changed")
        return artifact, raw["payload"]

    @staticmethod
    def _record(row: OpenWorldInvocationRow) -> InvocationRecord:
        if (
            row.state != "completed"
            or row.output_digest is None
            or row.observed_provider is None
            or row.observed_model_revision is None
            or row.called_at is None
            or row.latency_ms is None
        ):
            raise OpenWorldInvocationConflictError("invocation is not complete")
        return InvocationRecord(
            analyst_input_identity=row.analyst_input_identity,
            component_identity_digest=row.component_identity_digest,
            output_digest=row.output_digest,
            observed_provider=row.observed_provider,
            observed_model_revision=row.observed_model_revision,
            called_at=row.called_at,
            latency_ms=row.latency_ms,
            usage_json=row.usage_json,
        )

    @staticmethod
    def _check_row(
        row: OpenWorldInvocationRow,
        binding: OpenWorldInvocationBinding,
        expected: ExpectedProvenance,
    ) -> None:
        for field in (
            "job_id",
            "analyst_input_identity",
            "component_identity_digest",
            "component_role",
            "manifest_role",
            "user_id",
            "session_id",
            "qualification_result_digest",
            "trajectory_digest",
            "candidate_artifact_digest",
            "candidate_analyst_input_identity",
        ):
            if getattr(row, field) != getattr(binding, field):
                raise OpenWorldInvocationConflictError("invocation binding changed")
        if row.input_digest != expected.input_digest:
            raise OpenWorldInvocationConflictError("invocation input changed")
        if row.state == "completed" and (
            row.observed_provider != expected.expected_provider
            or row.observed_model_revision != expected.expected_model_revision
            or row.output_json is None
            or sha256(row.output_json.encode()).hexdigest() != row.output_digest
        ):
            raise OpenWorldInvocationConflictError(
                "invocation provenance/output changed"
            )

    def invoke(
        self,
        *,
        bundle: OpenWorldEvidenceBundle,
        claim: OptimizationJobClaim,
        system: str,
        payload: dict[str, object],
        schema: type[StrictStructuredOutput],
    ) -> tuple[StrictStructuredOutput, dict[str, Any]]:
        self.bind_context(bundle, claim)
        diagnostic = issubclass(schema, Diagnostic)
        output_kind(schema)
        identity = self.method.diagnostic if diagnostic else self.method.proposal
        if (
            sha256(system.encode()).hexdigest() != identity.prompt_template_digest
            or _digest(contract_schema(output_kind(schema)).model_json_schema())
            != identity.output_schema_digest
        ):
            raise OpenWorldQualificationAuthorityStaleError(
                "prompt/schema identity drift"
            )
        request_json = _json(
            {
                "bundle_digest": bundle.digest,
                "task_context_digest": bundle.task_context.digest,
                "component_identity_digest": identity.digest,
                "system": system,
                "payload": payload,
                "output_schema": schema.model_json_schema(),
            }
        )
        input_digest = sha256(request_json.encode()).hexdigest()
        session = None
        if diagnostic:
            session = next(
                (
                    item
                    for item in bundle.discovery.sessions
                    if (item.user_id, item.session_id)
                    == (payload.get("user_id"), payload.get("session_id"))
                ),
                None,
            )
            if session is None:
                raise OpenWorldInvocationConflictError(
                    "diagnostic source is not frozen"
                )
        role: Literal["discovery", "candidate"] = (
            "discovery" if diagnostic else "candidate"
        )
        qualification = (
            self.method.result_digest
            if diagnostic
            else self.method.proposal_authorization_digest
        )
        expected = ExpectedProvenance(
            expected_provider=identity.provider,
            expected_model_revision=identity.immutable_model_revision,
            input_digest=input_digest,
        )
        for attempt in range(1, 4):
            question = AnalystInputIdentity(
                component_identity_digest=identity.digest,
                qualification_result_digest=qualification,
                rendered_input_digest=input_digest,
                outcome_contract_digest=bundle.task_context.digest,
                role=role,
                attempt=attempt,
            )
            binding = OpenWorldInvocationBinding(
                job_id=claim.job_id,
                analyst_input_identity=question.digest,
                component_identity_digest=identity.digest,
                component_role=role,
                manifest_role="discovery" if diagnostic else None,
                user_id=session.user_id if session else None,
                session_id=session.session_id if session else None,
                bundle_digest=bundle.digest,
                outcome_contract_digest=bundle.task_context.digest,
                qualification_suite_digest=self.method.suite_digest
                if diagnostic
                else None,
                qualification_result_digest=qualification,
                trajectory_digest=session.trajectory_digest if session else None,
                candidate_artifact_digest=None,
                candidate_analyst_input_identity=None,
            )
            try:
                row = self.storage.prepare_open_world_invocation(
                    binding=binding,
                    expected=expected,
                    fence=claim.fence,
                    now=self.clock(),
                )
            except OpenWorldInvocationIdentityBoundError:
                continue
            break
        else:
            raise OpenWorldInvocationSlotExhaustedError(
                "all method call slots are bound"
            )
        self._check_row(row, binding, expected)
        if row.state != "completed":
            completion = self.transport.complete(
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(
                        role="user",
                        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                ],
                response_format=schema,
                model_revision=identity.immutable_model_revision,
                idempotency_key=binding.analyst_input_identity,
            )
            value = schema.model_validate_json(completion.value.model_dump_json())
            if not isinstance(value, (Diagnostic, Proposal)):
                raise TypeError("provider output did not match rule-evidence schema")
            output = observed_output_bytes(cast(Diagnostic | Proposal, value)).decode()
            row = self.storage.complete_open_world_invocation(
                binding=binding,
                completion=CompletionRecord(
                    output_json=output,
                    output_digest=sha256(output.encode()).hexdigest(),
                    observed_provider=completion.observed_provider,
                    observed_model_revision=completion.observed_model_revision,
                    called_at=completion.called_at,
                    latency_ms=completion.latency_ms,
                    usage_json=completion.usage_json,
                ),
                fence=claim.fence,
                now=self.clock(),
            )
        self._check_row(row, binding, expected)
        raw = json.loads(cast(str, row.output_json))
        value = schema.model_validate_json(json.dumps(raw["payload"]))
        if (
            observed_output_bytes(cast(Diagnostic | Proposal, value)).decode()
            != row.output_json
        ):
            raise OpenWorldInvocationConflictError("observed output contract changed")
        return value, {
            "prepared": {
                "binding": dataclasses.asdict(binding),
                "expected": dataclasses.asdict(expected),
                "component_identity": dataclasses.asdict(identity),
                "analyst_input_identity": dataclasses.asdict(question),
                "request_json": request_json,
            },
            "invocation": dataclasses.asdict(self._record(row)),
            "observed_output": value.model_dump(mode="json"),
        }

    def verify_receipt(
        self,
        receipt: dict[str, Any],
        *,
        bundle: OpenWorldEvidenceBundle,
        claim: OptimizationJobClaim,
        schema: type[StrictStructuredOutput],
        system: str,
        payload: dict[str, object],
    ) -> StrictStructuredOutput:
        """Recheck canonical request, actual output and exact durable invocation."""
        prepared = receipt["prepared"]
        identity = (
            self.method.diagnostic
            if issubclass(schema, Diagnostic)
            else self.method.proposal
        )
        if prepared["component_identity"] != dataclasses.asdict(identity):
            raise OpenWorldQualificationAuthorityStaleError("receipt component changed")
        question = AnalystInputIdentity(**prepared["analyst_input_identity"])
        binding = OpenWorldInvocationBinding(**prepared["binding"])
        expected = ExpectedProvenance(**prepared["expected"])
        request = {
            "bundle_digest": bundle.digest,
            "task_context_digest": bundle.task_context.digest,
            "component_identity_digest": identity.digest,
            "system": system,
            "payload": payload,
            "output_schema": schema.model_json_schema(),
        }
        qualification = (
            self.method.result_digest
            if issubclass(schema, Diagnostic)
            else self.method.proposal_authorization_digest
        )
        if (
            prepared["request_json"] != _json(request)
            or expected.input_digest != _digest(request)
            or expected.expected_provider != identity.provider
            or expected.expected_model_revision != identity.immutable_model_revision
            or question.rendered_input_digest != expected.input_digest
            or question.component_identity_digest != identity.digest
            or question.qualification_result_digest != qualification
            or question.outcome_contract_digest != bundle.task_context.digest
            or question.role != identity.component_role
            or binding.analyst_input_identity != question.digest
            or binding.job_id != claim.job_id
            or binding.bundle_digest != bundle.digest
            or binding.outcome_contract_digest != bundle.task_context.digest
        ):
            raise OpenWorldInvocationConflictError(
                "receipt does not bind exact request"
            )
        row = self.storage.load_open_world_invocation(
            binding=binding, fence=claim.fence, now=self.clock()
        )
        self._check_row(row, binding, expected)
        if dataclasses.asdict(self._record(row)) != receipt["invocation"]:
            raise OpenWorldInvocationConflictError(
                "receipt differs from durable invocation"
            )
        output = schema.model_validate_json(json.dumps(receipt["observed_output"]))
        if (
            observed_output_bytes(cast(Diagnostic | Proposal, output)).decode()
            != row.output_json
        ):
            raise OpenWorldInvocationConflictError(
                "receipt differs from observed output"
            )
        return output

    def _validated_artifacts(
        self,
        *,
        bundle: OpenWorldEvidenceBundle,
        incumbent: UserPlaybook,
        claim: OptimizationJobClaim,
    ) -> tuple[
        PlaybookOptimizationArtifact,
        dict[str, Any],
        tuple[PlaybookOptimizationArtifact, dict[str, Any]] | None,
        EvidenceEditValidation,
        OpenWorldAttemptContextRow,
        QualificationAuthorityRef,
    ]:
        diagnostics_row = self.read(
            "open_world_discovery_memo", "rule_evidence_diagnostics", claim
        )
        if diagnostics_row is None:
            raise OpenWorldInvocationConflictError("diagnostic artifact is missing")
        diagnostics_artifact, collection = diagnostics_row
        context, authority = self.bind_context(bundle, claim)
        if (
            collection["job_id"] != claim.job_id
            or collection["bundle_digest"] != bundle.digest
            or collection["task_context_digest"] != bundle.task_context.digest
            or collection["authority_epoch"] != authority.epoch
            or collection["authority_digest"] != authority.authority_digest
        ):
            raise OpenWorldInvocationConflictError("diagnostic context changed")
        documents = [
            TypeAdapter(EvidenceDocument).validate_python(session["source_document"])
            for session in collection["sessions"]
        ]
        trajectory_json_by_session = {
            (session["user_id"], session["session_id"]): session["trajectory_json"]
            for session in collection["sessions"]
        }
        data = native_rule_evidence_input(
            bundle=bundle,
            incumbent=incumbent,
            documents=documents,
            trajectory_json_by_session=trajectory_json_by_session,
        )
        candidate_row = self.read(
            "open_world_candidate", "rule_evidence_candidate", claim
        )
        candidate = candidate_row[1] if candidate_row is not None else None
        receipts = list(collection["sessions"])
        if candidate is not None:
            receipts.append(candidate)
        observed = iter(receipts)
        diagnostic_outputs: list[Diagnostic] = []
        proposal: Proposal | None = None

        def replay(
            system: str,
            payload: dict[str, object],
            schema: type[StrictStructuredOutput],
        ) -> StrictStructuredOutput:
            nonlocal proposal
            try:
                receipt = next(observed)
            except StopIteration as exc:
                raise OpenWorldInvocationConflictError(
                    "durable method output is missing"
                ) from exc
            value = self.verify_receipt(
                receipt,
                bundle=bundle,
                claim=claim,
                schema=schema,
                system=system,
                payload=payload,
            )
            if issubclass(schema, Diagnostic):
                diagnostic_outputs.append(cast(Diagnostic, value))
            else:
                proposal = cast(Proposal, value)
            return value

        result = run_rule_evidence(data, replay)
        if next(observed, None) is not None:
            raise OpenWorldInvocationConflictError("unexpected durable method output")
        validation = validate_evidence_edit(
            bundle=bundle,
            incumbent=incumbent,
            documents=documents,
            trajectory_json_by_session=trajectory_json_by_session,
            diagnostics=diagnostic_outputs,
            proposal=proposal,
        )
        if (
            collection["diagnostic_population_digest"]
            != validation.diagnostic_population_digest
        ):
            raise OpenWorldInvocationConflictError("diagnostic population changed")
        if candidate is not None:
            expected = candidate_payload(
                job_id=claim.job_id,
                bundle=bundle,
                incumbent=incumbent,
                diagnostics_artifact_digest=diagnostics_artifact.content_digest,
                diagnostics=diagnostic_outputs,
                result=result,
                citations=validation.verified_citations,
                receipt={
                    key: candidate[key]
                    for key in ("prepared", "invocation", "observed_output")
                },
            )
            if candidate != expected:
                raise OpenWorldInvocationConflictError(
                    "derived exact-edit artifact changed"
                )
        return (
            diagnostics_artifact,
            collection,
            candidate_row,
            validation,
            context,
            authority,
        )

    def load_accepted_publication_inputs(
        self, *, job_id: int, claim: OptimizationJobClaim, now: int
    ) -> RuleEvidencePublicationInputs:
        job = self.storage.get_playbook_optimization_job(job_id)
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
            raise OpenWorldInvocationLeaseConflictError(
                "publication lease/stage is stale"
            )
        frozen = self.storage.load_open_world_evidence_bundle(job_id)
        bundle = _parse_canonical_bundle(frozen.artifact.content_json)
        if (
            frozen.artifact.content_digest != bundle.digest
            or bundle.target.tenant_id != self.storage.org_id
            or job.target_id != bundle.target.user_playbook_id
            or job.expected_population_manifest_digest != bundle.expected.digest
        ):
            raise OpenWorldInvocationConflictError("publication source bundle changed")
        incumbent = self.storage.get_user_playbook_by_id(bundle.target.user_playbook_id)
        if incumbent is None:
            raise OpenWorldInvocationConflictError("publication incumbent is missing")
        (
            diagnostics_artifact,
            collection,
            candidate_row,
            validation,
            context,
            authority,
        ) = self._validated_artifacts(bundle=bundle, incumbent=incumbent, claim=claim)
        decision_row = self.read(
            "open_world_attempt_decision", "rule_evidence_decision", claim
        )
        if (
            candidate_row is None
            or decision_row is None
            or validation.result.decision != "proposal"
        ):
            raise OpenWorldInvocationConflictError(
                "publication lacks an accepted exact edit"
            )
        candidate_artifact, candidate = candidate_row
        decision_artifact, decision = decision_row
        expected_decision = decision_payload(
            bundle=bundle,
            diagnostics_artifact_digest=diagnostics_artifact.content_digest,
            population_digest=collection["diagnostic_population_digest"],
            candidate_digest=candidate_artifact.content_digest,
        )
        if decision != expected_decision:
            raise OpenWorldInvocationConflictError("accepted decision changed")
        if not job.attempt_key:
            raise OpenWorldInvocationConflictError(
                "publication attempt identity is missing"
            )
        return RuleEvidencePublicationInputs(
            job_id=job_id,
            attempt_key=job.attempt_key,
            incumbent_user_playbook_id=bundle.target.user_playbook_id,
            incumbent_full_version_fingerprint=bundle.target.incumbent_full_version_fingerprint,
            evidence_bundle_digest=bundle.digest,
            outcome_contract_digest=bundle.task_context.digest,
            candidate=RuleEvidenceCandidate(
                ContentRevision(candidate["revised_content"]),
                _candidate_content_diff(**candidate["content_diff"]),
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
            subject_epochs=OpenWorldEvidenceBundleStore._complete_subject_epochs(
                bundle
            ),
            qualification_authority=authority,
            claim=claim,
        )


def candidate_payload(
    *,
    job_id: int,
    bundle: OpenWorldEvidenceBundle,
    incumbent: UserPlaybook,
    diagnostics_artifact_digest: str,
    diagnostics: Sequence[Diagnostic],
    result: Any,
    citations: Sequence[Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if (
        result.decision != "proposal"
        or result.revised_content is None
        or not result.rule_audit.edits
    ):
        raise ValueError("candidate requires an admitted exact edit")
    diff = _candidate_content_diff(
        incumbent_content_digest=sha256(incumbent.content.encode()).hexdigest(),
        candidate_content_digest=sha256(result.revised_content.encode()).hexdigest(),
        unified_diff="".join(
            difflib.unified_diff(
                incumbent.content.splitlines(keepends=True),
                result.revised_content.splitlines(keepends=True),
                fromfile="incumbent",
                tofile="candidate",
            )
        ),
    )
    return {
        "job_id": job_id,
        "bundle_digest": bundle.digest,
        "task_context_digest": bundle.task_context.digest,
        "incumbent_full_version_fingerprint": bundle.target.incumbent_full_version_fingerprint,
        "diagnostics_artifact_digest": diagnostics_artifact_digest,
        "diagnostic_population_digest": diagnostic_population_digest(diagnostics),
        **receipt,
        "original_clause": incumbent.content,
        "replacement": result.revised_content,
        "rule_audit": result.rule_audit.model_dump(mode="json"),
        "revised_content": result.revised_content,
        "content_diff": dataclasses.asdict(diff),
        "verified_citations": [dataclasses.asdict(citation) for citation in citations],
    }


def decision_payload(
    *,
    bundle: OpenWorldEvidenceBundle,
    diagnostics_artifact_digest: str,
    population_digest: str,
    candidate_digest: str | None,
    proposal_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "bundle_digest": bundle.digest,
        "policy_id": RULE_EVIDENCE_POLICY_ID,
        "diagnostics_artifact_digest": diagnostics_artifact_digest,
        "diagnostic_population_digest": population_digest,
        "candidate_digest": candidate_digest,
        "proposal_receipt": proposal_receipt,
        "decision": (
            {"decision": "accept"}
            if candidate_digest is not None
            else {"decision": "abstain", "reason": "no_grounded_hypothesis"}
        ),
    }


def artifact_payload(contract: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"contract_type": contract, "schema_version": "v1", "payload": payload}


@dataclass
class RuleEvidenceAttemptRunner:
    """Stages within the existing offline-tuner worker, with no alternate engine."""

    store: RuleEvidenceAnalysisStore
    lease: AttemptLease
    lease_seconds: int

    def _claim(self) -> OptimizationJobClaim:
        claim = self.lease.claim
        now = self.store.clock()
        if claim.expires_at - now < 600:
            claim = self.store.storage.renew_playbook_optimization_job_lease(
                claim.job_id, claim.owner, claim.fence, self.lease_seconds, now=now
            )
            self.lease.claim = claim
        if claim.expires_at <= now:
            raise OpenWorldInvocationLeaseConflictError("method lease expired")
        return claim

    def _advance(self, stage: OptimizationJobStage) -> None:
        claim = self._claim()
        if self.store.storage.advance_playbook_optimization_stage(
            claim.job_id, claim.fence, stage, now=self.store.clock()
        ):
            return
        job = self.store.storage.get_playbook_optimization_job(claim.job_id)
        order = (
            "evidence_frozen",
            "discovery_analyzed",
            "candidate_generated",
            "held_out_analyzed",
            "publishing",
        )
        if (
            job is None
            or job.status != "running"
            or job.lease_owner != claim.owner
            or job.lease_fence != claim.fence
            or job.lease_expires_at != claim.expires_at
            or job.stage not in order
            or stage not in order
            or order.index(job.stage) < order.index(stage)
        ):
            raise OpenWorldInvocationLeaseConflictError("method stage/lease changed")

    def _terminal_claim(self) -> OptimizationJobClaim | None:
        """Recover only this worker's expired lease for its final fenced write."""
        claim = self.lease.claim
        job = self.store.storage.get_playbook_optimization_job(claim.job_id)
        if (
            job is None
            or job.status not in ("pending", "running")
            or job.lease_owner != claim.owner
            or job.lease_fence != claim.fence
        ):
            return None
        if job.lease_expires_at is None or job.lease_expires_at <= self.store.clock():
            try:
                claim = self.store.storage.reclaim_playbook_optimization_job(
                    claim.job_id,
                    claim.owner,
                    self.lease_seconds,
                    now=self.store.clock(),
                )
            except StorageError:
                # A peer can reclaim after the read. Its new live lease wins.
                return None
            self.lease.claim = claim
        return claim

    def run(
        self,
        *,
        frozen: FrozenOpenWorldEvidence,
        incumbent: UserPlaybook,
        evidence: RehydratedEvidence
        | Callable[[Callable[[], None]], RehydratedEvidence],
    ) -> RuleEvidenceDecision | AttemptTerminalFailure:
        """Run prepared evidence or load/compact it inside the same failure boundary.

        A lazy callback receives the live-lease/authority guard for every model
        batch and receipt write. Persisted terminal decisions skip it entirely.
        """
        try:
            return self._run(frozen=frozen, incumbent=incumbent, evidence=evidence)
        except (
            StorageError,
            LiteLLMClientError,
            ValueError,
            TypeError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            reason: AttemptTerminalFailureReason = "infrastructure_failure"
            # Retry-eligible, not day-spending: the analyst's budget ran out on
            # a rate limit, or the provider is down for every caller (the
            # durable-extraction classification). A refusal from an open
            # breaker, e.g. during compaction, never reached the provider.
            rate_limited = isinstance(
                exc, OpenWorldProviderRateLimitedError
            ) or isinstance(classify_llm_failure(exc), OutageWait)
            if isinstance(exc, OpenWorldRefusalError):
                reason = cast(
                    AttemptTerminalFailureReason, terminal_reason_for(exc.refusal)
                )
            diagnostic = build_terminal_failure_diagnostic(reason, exc)
            failure_detail = diagnostic_log_fragment(diagnostic)
            output_rejected = reason == "infrastructure_failure" and (
                rejected_model_output(exc)
            )
            claim = self._terminal_claim()
            if claim is None:
                return AttemptTerminalFailure(
                    reason=reason,
                    rate_limited=rate_limited,
                    output_rejected=output_rejected,
                    failure_detail=failure_detail,
                )
            try:
                self.store.storage.merge_playbook_optimization_job_metadata(
                    claim.job_id,
                    {TERMINAL_DIAGNOSTIC_METADATA_KEY: diagnostic_payload(diagnostic)},
                )
            except StorageError as diagnostic_error:
                logging.getLogger(__name__).warning(
                    "rule-evidence failure diagnostic unavailable error_type=%s",
                    type(diagnostic_error).__name__,
                )
            terminalized = self.store.storage.advance_playbook_optimization_stage(
                claim.job_id,
                claim.fence,
                "failed",
                terminal_outcome=reason,
                now=self.store.clock(),
            )
            if not terminalized:
                logging.getLogger(__name__).warning(
                    "rule-evidence terminal write refused after lease changed"
                )
            return AttemptTerminalFailure(
                reason=reason,
                rate_limited=rate_limited,
                output_rejected=output_rejected,
                failure_detail=failure_detail,
            )

    def _run(
        self,
        *,
        frozen: FrozenOpenWorldEvidence,
        incumbent: UserPlaybook,
        evidence: RehydratedEvidence
        | Callable[[Callable[[], None]], RehydratedEvidence],
    ) -> RuleEvidenceDecision:
        bundle = frozen.bundle
        if not callable(evidence) and evidence.bundle_digest != bundle.digest:
            raise OpenWorldInvocationConflictError("rehydrated evidence bundle changed")
        job = self.store.storage.get_playbook_optimization_job(self.lease.claim.job_id)
        if job is not None and job.stage == "abstained":
            # This protected read independently checks current target, governance,
            # durable source/receipts and terminal-decision authority in SQL.
            # Rebinding or preparing calls here would require an already-ended lease.
            persisted = self.store.read(
                "open_world_attempt_decision",
                "rule_evidence_decision",
                self.lease.claim,
            )
            if persisted is None:
                raise OpenWorldInvocationConflictError("terminal decision is missing")
            payload = persisted[1]
            if (
                job.status != "skipped"
                or job.lease_fence != self.lease.claim.fence
                or job.terminal_outcome != "no_grounded_hypothesis"
                or payload["bundle_digest"] != bundle.digest
                or payload["policy_id"] != RULE_EVIDENCE_POLICY_ID
                or payload["candidate_digest"] is not None
                or payload["decision"]
                != {"decision": "abstain", "reason": "no_grounded_hypothesis"}
            ):
                raise OpenWorldInvocationConflictError("terminal decision changed")
            return RuleEvidenceDecision(
                bundle.digest,
                None,
                payload["diagnostic_population_digest"],
                AbstainDecision(decision="abstain", reason="no_grounded_hypothesis"),
            )

        def before_preparation_call() -> None:
            self.store.bind_context(bundle, self._claim())

        if callable(evidence):
            evidence = evidence(before_preparation_call)
        if evidence.bundle_digest != bundle.digest:
            raise OpenWorldInvocationConflictError("rehydrated evidence bundle changed")
        context, authority = self.store.bind_context(bundle, self._claim())
        documents = [
            evidence.documents_by_session[(session.user_id, session.session_id)]
            for session in bundle.discovery.sessions
        ]
        trajectory_json_by_session = evidence.trajectory_json_by_session
        data = native_rule_evidence_input(
            bundle=bundle,
            incumbent=incumbent,
            documents=documents,
            trajectory_json_by_session=trajectory_json_by_session,
        )
        sessions: list[dict[str, Any]] = []
        diagnostics: list[Diagnostic] = []
        proposal_receipt: dict[str, Any] | None = None
        diagnostics_artifact: PlaybookOptimizationArtifact | None = None

        def persist_diagnostics() -> PlaybookOptimizationArtifact:
            nonlocal diagnostics_artifact
            diagnostics_artifact = self.store.write(
                "open_world_discovery_memo",
                artifact_payload(
                    "rule_evidence_diagnostics",
                    {
                        "job_id": self.lease.claim.job_id,
                        "bundle_digest": bundle.digest,
                        "task_context_digest": bundle.task_context.digest,
                        "authority_epoch": authority.epoch,
                        "authority_digest": authority.authority_digest,
                        "sessions": sessions,
                        "diagnostic_population_digest": diagnostic_population_digest(
                            diagnostics
                        ),
                    },
                ),
                self._claim(),
            )
            self._advance("discovery_analyzed")
            return diagnostics_artifact

        def invoke(
            system: str,
            payload: dict[str, object],
            schema: type[StrictStructuredOutput],
        ) -> StrictStructuredOutput:
            nonlocal proposal_receipt
            if issubclass(schema, Proposal):
                persist_diagnostics()
            from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
                protocol_from_metadata,
            )
            from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
                shared_consumer_admission,
            )

            admission = shared_consumer_admission.set(
                protocol_from_metadata(json.loads(frozen.job.metadata_json or "{}"))
                is not None
            )
            try:
                value, receipt = self.store.invoke(
                    bundle=bundle,
                    claim=self._claim(),
                    system=system,
                    payload=payload,
                    schema=schema,
                )
            finally:
                shared_consumer_admission.reset(admission)
            if issubclass(schema, Diagnostic):
                index = len(sessions)
                document = documents[index]
                source = bundle.discovery.sessions[index]
                sessions.append(
                    {
                        "user_id": source.user_id,
                        "session_id": source.session_id,
                        "trajectory_digest": source.trajectory_digest,
                        "source_document": dataclasses.asdict(document),
                        "trajectory_json": trajectory_json_by_session[
                            (source.user_id, source.session_id)
                        ],
                        **receipt,
                    }
                )
                diagnostics.append(cast(Diagnostic, value))
            else:
                proposal_receipt = receipt
            return value

        result = run_rule_evidence(data, invoke)
        if diagnostics_artifact is None:
            persist_diagnostics()
        diagnostics_artifact = cast(PlaybookOptimizationArtifact, diagnostics_artifact)
        proposal = (
            Proposal.model_validate_json(
                json.dumps(proposal_receipt["observed_output"])
            )
            if proposal_receipt is not None
            else None
        )
        validation = validate_evidence_edit(
            bundle=bundle,
            incumbent=incumbent,
            documents=documents,
            trajectory_json_by_session=trajectory_json_by_session,
            diagnostics=diagnostics,
            proposal=proposal,
        )
        candidate_digest = None
        if result.decision == "proposal":
            candidate = self.store.write(
                "open_world_candidate",
                artifact_payload(
                    "rule_evidence_candidate",
                    candidate_payload(
                        job_id=self.lease.claim.job_id,
                        bundle=bundle,
                        incumbent=incumbent,
                        diagnostics_artifact_digest=diagnostics_artifact.content_digest,
                        diagnostics=diagnostics,
                        result=result,
                        citations=validation.verified_citations,
                        receipt=cast(dict[str, Any], proposal_receipt),
                    ),
                ),
                self._claim(),
            )
            candidate_digest = candidate.content_digest
            self._advance("candidate_generated")
            # Compatibility stage name; no held-out model or population ran.
            self._advance("held_out_analyzed")
        payload = decision_payload(
            bundle=bundle,
            diagnostics_artifact_digest=diagnostics_artifact.content_digest,
            population_digest=validation.diagnostic_population_digest,
            candidate_digest=candidate_digest,
            proposal_receipt=(proposal_receipt if candidate_digest is None else None),
        )
        self.store.write(
            "open_world_attempt_decision",
            artifact_payload("rule_evidence_decision", payload),
            self._claim(),
        )
        if candidate_digest is None:
            claim = self._claim()
            if not self.store.storage.advance_playbook_optimization_stage(
                claim.job_id,
                claim.fence,
                "abstained",
                terminal_outcome="no_grounded_hypothesis",
                now=self.store.clock(),
            ):
                job = self.store.storage.get_playbook_optimization_job(claim.job_id)
                if job is None or job.terminal_outcome != "no_grounded_hypothesis":
                    raise OpenWorldInvocationLeaseConflictError(
                        "abstention was not recorded"
                    )
        return RuleEvidenceDecision(
            bundle.digest,
            candidate_digest,
            validation.diagnostic_population_digest,
            AcceptDecision(decision="accept")
            if candidate_digest is not None
            else AbstainDecision(decision="abstain", reason="no_grounded_hypothesis"),
        )
