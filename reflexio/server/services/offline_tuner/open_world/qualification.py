"""Versioned synthetic qualification fixtures and a pure pass-all reducer.

The analyst adapter owns the execution path. This module only accepts its sealed
invocation records and deterministically evaluates their structured results.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal, Protocol, cast
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from reflexio.models.api_schema.domain import (
    OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION,
    OpenWorldQualificationClassCount,
    OpenWorldQualificationRecord,
)
from reflexio.server.services.governance.config import get_governance_ref_secret
from reflexio.server.services.storage.storage_base import BaseStorage

from ..evidence.manifests import canonical_json_bytes
from .citations import (
    CitationVerificationCategory,
    CitationVerificationError,
    verify_citations,
)
from .identity import (
    build_candidate_envelope,
    build_discovery_envelope,
    canonical_bytes,
    invocation_record,
    pinned_generator_authorization,
    prepare_candidate_request,
    prepare_discovery_request,
    prepare_held_out_session_request,
)
from .models import (
    _QUALIFIED_ANALYST_FACTORY_TOKEN,
    AnalystQualificationResultRef,
    CandidateRevisionV1,
    ComponentIdentity,
    ConfirmationSessionMemo,
    DiscoveryMemo,
    DiscoveryRequest,
    EvidenceCitation,
    EvidenceDocument,
    EvidenceSpan,
    HeldOutSessionMemo,
    HeldOutSessionRequest,
    InvocationRecord,
    ObservedStructuredCompletion,
    OutcomeContractDescriptor,
    ProvisionalCaseAnalyst,
    QualifiedAnalyst,
    QualifiedAnalystSet,
    VerifiedCitation,
)

type QualificationClass = Literal[
    "citation_fidelity",
    "abstention",
    "support",
    "refutation",
    "insufficiency",
    "unsupported_causal_claim_rejection",
    "prompt_injection_resistance",
]
QUALIFICATION_CLASSES: tuple[QualificationClass, ...] = (
    "citation_fidelity",
    "abstention",
    "support",
    "refutation",
    "insufficiency",
    "unsupported_causal_claim_rejection",
    "prompt_injection_resistance",
)
_ROLE_ORDER = {"discovery": 0, "held_out": 1, "confirmation": 2}
_OBSERVATION_FACTORY_TOKEN = object()
_QUALIFICATION_OBSERVATION_TOKEN = object()
_REJECTION_FACTORY_TOKEN = object()
_QUALIFICATION_RESULT_HMAC_DOMAIN = (
    b"reflexio:offline-tuner-open-world-qualification-result-hmac:v1\x00"
)

type CitationRejectionCategory = CitationVerificationCategory


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SyntheticOutcomeContract(_StrictModel):
    digest: str
    kind: Literal["success_definition", "structured_outcome_source"]
    success_definition: str | None
    source_id: str | None
    schema_version: str
    allowed_values: tuple[Literal["success", "failure", "unknown"], ...]

    @field_validator("digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("outcome contract digest", value)

    def to_production(self) -> OutcomeContractDescriptor:
        return OutcomeContractDescriptor(**self.model_dump())


class SyntheticEvidenceSpan(_StrictModel):
    span_id: str
    tenant_id: str
    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    request_id: str
    interaction_id: int
    field_path: str
    immutable_text: str
    # Required, with no default, exactly as on the production contract: a
    # qualification case that omitted the speaker would attest an analyst
    # against a payload shape production never sends.
    speaker: Literal["agent", "user", "other"]

    @field_validator(
        "span_id",
        "tenant_id",
        "user_id",
        "session_id",
        "request_id",
        "field_path",
        "immutable_text",
    )
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text(cls.__name__, value)

    def to_production(self) -> EvidenceSpan:
        return EvidenceSpan(**self.model_dump())


class SyntheticSessionDocument(_StrictModel):
    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str
    spans: tuple[SyntheticEvidenceSpan, ...]
    session_outcome: Literal["success", "failure", "unknown"] | None = None

    @field_validator("user_id", "session_id")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text(cls.__name__, value)

    @field_validator("trajectory_digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("trajectory digest", value)

    @model_validator(mode="after")
    def bind_spans(self) -> SyntheticSessionDocument:
        if not self.spans:
            raise ValueError("session document must contain spans")
        if any(
            (span.manifest_role, span.user_id, span.session_id)
            != (self.manifest_role, self.user_id, self.session_id)
            for span in self.spans
        ):
            raise ValueError("span provenance must match its document")
        return self

    def to_production(self) -> EvidenceDocument:
        return EvidenceDocument(
            manifest_role=self.manifest_role,
            user_id=self.user_id,
            session_id=self.session_id,
            trajectory_digest=self.trajectory_digest,
            spans=tuple(span.to_production() for span in self.spans),
            session_outcome=self.session_outcome,
        )


class CitationExpectation(_StrictModel):
    citation_id: str
    span_id: str
    exact_quote: str
    start_offset: int
    end_offset: int
    manifest_role: Literal["discovery", "held_out", "safety_only"]
    user_id: str
    session_id: str
    trajectory_digest: str

    @field_validator("citation_id", "span_id", "exact_quote", "user_id", "session_id")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text(cls.__name__, value)

    @field_validator("trajectory_digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("citation trajectory digest", value)

    def as_structured(self) -> EvidenceCitation:
        return EvidenceCitation(span_id=self.span_id, exact_quote=self.exact_quote)

    def as_verified(self) -> VerifiedCitation:
        return VerifiedCitation(**self.model_dump())


class SyntheticDiscoveryArtifact(_StrictModel):
    memo: DiscoveryMemo
    verified_citations: tuple[CitationExpectation, ...]
    qualification_result_digest: str

    @field_validator("qualification_result_digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("discovery qualification result digest", value)

    @model_validator(mode="after")
    def bind_citations(self) -> SyntheticDiscoveryArtifact:
        if tuple(
            (item.span_id, item.exact_quote) for item in self.memo.citations
        ) != tuple(
            (item.span_id, item.exact_quote) for item in self.verified_citations
        ):
            raise ValueError("discovery memo must exactly project verified citations")
        return self

    def to_memo(self) -> DiscoveryMemo:
        return self.memo

    def to_verified_citations(
        self, *, allowed_spans: tuple[EvidenceSpan, ...]
    ) -> tuple[VerifiedCitation, ...]:
        seeded = tuple(item.as_verified() for item in self.verified_citations)
        verified = verify_citations(self.memo.citations, allowed_spans=allowed_spans)
        if seeded != verified:
            raise ValueError("discovery artifact has stale citation provenance")
        return seeded


class SyntheticComponentIdentity(_StrictModel):
    component_role: Literal["discovery", "candidate", "held_out", "confirmation"]
    provider: str
    immutable_model_revision: str
    prompt_id: str
    prompt_version: str
    prompt_template_digest: str
    output_schema_id: str
    output_schema_version: str
    output_schema_digest: str
    generation_settings_digest: str
    qualification_suite_version: str

    def to_production(self) -> ComponentIdentity:
        return ComponentIdentity(**self.model_dump())


class SyntheticQualificationRef(_StrictModel):
    suite_digest: str
    result_digest: str

    @field_validator("suite_digest", "result_digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("qualification digest", value)

    def for_identity(
        self, identity: ComponentIdentity
    ) -> AnalystQualificationResultRef:
        return AnalystQualificationResultRef(
            component_identity_digest=identity.digest,
            suite_digest=self.suite_digest,
            result_digest=self.result_digest,
            passed=True,
        )


class SyntheticCompletionMetadata(_StrictModel):
    called_at: int
    latency_ms: int
    usage_json: str | None

    @model_validator(mode="after")
    def check_values(self) -> SyntheticCompletionMetadata:
        if self.called_at < 0 or self.latency_ms < 0:
            raise ValueError("completion metadata must be non-negative")
        if self.usage_json is not None:
            _text("usage json", self.usage_json)
        return self


class SyntheticCandidateArtifact(_StrictModel):
    incumbent_content: str
    incumbent_full_version_fingerprint: str
    revised_content: str
    change_rationale: str
    declared_scope: str
    supporting_citation_ids: tuple[str, ...]
    incumbent_content_digest: str
    candidate_content_digest: str
    unified_diff: str
    generator_identity: SyntheticComponentIdentity

    @field_validator(
        "incumbent_content",
        "revised_content",
        "change_rationale",
        "declared_scope",
        "unified_diff",
    )
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text(cls.__name__, value)

    @field_validator(
        "incumbent_full_version_fingerprint",
        "incumbent_content_digest",
        "candidate_content_digest",
    )
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("candidate artifact digest", value)

    def to_revision(self) -> CandidateRevisionV1:
        return CandidateRevisionV1(
            revised_content=self.revised_content,
            change_rationale=self.change_rationale,
            declared_scope=self.declared_scope,
            supporting_citation_ids=self.supporting_citation_ids,
        )


class SyntheticSuiteSeed(_StrictModel):
    """Deterministic non-production qualification fixture input.

    This models already-observed canonical suite results for qualification
    adapter tests. Production authorization is minted only by
    ``seal_qualified_analyst`` from a result the qualification reducer issued.
    """

    discovery_identity: SyntheticComponentIdentity
    held_out_identity: SyntheticComponentIdentity
    confirmation_identity: SyntheticComponentIdentity
    discovery_qualification: SyntheticQualificationRef
    held_out_qualification: SyntheticQualificationRef
    confirmation_qualification: SyntheticQualificationRef
    discovery_completion: SyntheticCompletionMetadata
    candidate_completion: SyntheticCompletionMetadata
    held_out_completion: SyntheticCompletionMetadata
    confirmation_completion: SyntheticCompletionMetadata

    def case_analyst(
        self, role: Literal["discovery", "held_out", "confirmation"]
    ) -> ProvisionalCaseAnalyst:
        """Render this seed's analyst reference for suite-case rendering only.

        Returns a ``ProvisionalCaseAnalyst``: enough to build the case's
        production request, never enough to authorize an attempt.
        """
        identity = {
            "discovery": self.discovery_identity,
            "held_out": self.held_out_identity,
            "confirmation": self.confirmation_identity,
        }[role].to_production()
        qualification = {
            "discovery": self.discovery_qualification,
            "held_out": self.held_out_qualification,
            "confirmation": self.confirmation_qualification,
        }[role]
        return ProvisionalCaseAnalyst(
            identity=identity, qualification=qualification.for_identity(identity)
        )


class SyntheticHeldOutInput(_StrictModel):
    manifest_role: Literal["held_out", "safety_only"]
    evidence: SyntheticSessionDocument
    discovery: SyntheticDiscoveryArtifact
    candidate: SyntheticCandidateArtifact

    @model_validator(mode="after")
    def bind_evidence(self) -> SyntheticHeldOutInput:
        if self.evidence.manifest_role != self.manifest_role:
            raise ValueError("held-out evidence must bind its manifest role")
        return self


class SyntheticAnalystInput(_StrictModel):
    job_id: int
    bundle_digest: str
    scenario: str
    outcome_contract: SyntheticOutcomeContract
    discovery_sessions: tuple[SyntheticSessionDocument, ...]
    held_out: SyntheticHeldOutInput | None
    untrusted_content: tuple[str, ...]
    label_source: Literal["synthetic_expert"]

    @field_validator("bundle_digest")
    @classmethod
    def check_digest(cls, value: str) -> str:
        return _digest("bundle digest", value)

    @field_validator("scenario")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text("scenario", value)

    @model_validator(mode="after")
    def bind_input(self) -> SyntheticAnalystInput:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("job id must be positive")
        if not self.discovery_sessions or any(
            item.manifest_role != "discovery" for item in self.discovery_sessions
        ):
            raise ValueError("discovery sessions must be non-empty discovery documents")
        if any(not item.strip() for item in self.untrusted_content):
            raise ValueError("untrusted content must be non-empty text")
        return self

    def to_production_request(
        self, seed: SyntheticSuiteSeed
    ) -> DiscoveryRequest | HeldOutSessionRequest:
        outcome = self.outcome_contract.to_production()
        discovery_sessions = tuple(
            document.to_production() for document in self.discovery_sessions
        )
        discovery_analyst = seed.case_analyst("discovery")
        if self.held_out is None:
            return prepare_discovery_request(
                job_id=self.job_id,
                bundle_digest=self.bundle_digest,
                discovery_sessions=discovery_sessions,
                outcome_contract=outcome,
                analyst=discovery_analyst,
            )
        artifact = self.held_out.discovery
        if artifact.qualification_result_digest != (
            discovery_analyst.qualification.result_digest
        ):
            raise ValueError("discovery artifact qualification is stale")
        discovery_request = prepare_discovery_request(
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            discovery_sessions=discovery_sessions,
            outcome_contract=outcome,
            analyst=discovery_analyst,
        )
        discovery_memo = artifact.to_memo()
        discovery_completion = _completion(
            discovery_memo,
            discovery_analyst.identity,
            seed.discovery_completion,
        )
        discovery_verified = artifact.to_verified_citations(
            allowed_spans=tuple(
                span for document in discovery_sessions for span in document.spans
            ),
        )
        discovery = build_discovery_envelope(
            request=discovery_request,
            observed_memo=discovery_memo,
            invocation=invocation_record(
                intent=discovery_request.prepared_invocation.intent,
                completion=discovery_completion,
            ),
        )
        if discovery.verified_citations != discovery_verified:
            raise ValueError(
                "fixture discovery citations failed production verification"
            )
        candidate_artifact = self.held_out.candidate
        generator = candidate_artifact.generator_identity.to_production()
        candidate_request = prepare_candidate_request(
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            incumbent_content=candidate_artifact.incumbent_content,
            incumbent_full_version_fingerprint=(
                candidate_artifact.incumbent_full_version_fingerprint
            ),
            discovery=discovery,
            outcome_contract=outcome,
            generator_authorization=pinned_generator_authorization(generator),
        )
        revision = candidate_artifact.to_revision()
        candidate_completion = _completion(
            revision, generator, seed.candidate_completion
        )
        candidate = build_candidate_envelope(
            request=candidate_request,
            revision=revision,
            invocation=invocation_record(
                intent=candidate_request.prepared_invocation.intent,
                completion=candidate_completion,
            ),
        )
        if (
            candidate.content_diff.incumbent_content_digest
            != candidate_artifact.incumbent_content_digest
            or candidate.content_diff.candidate_content_digest
            != candidate_artifact.candidate_content_digest
            or candidate.content_diff.unified_diff != candidate_artifact.unified_diff
        ):
            raise ValueError("candidate artifact derives stale content diff")
        evidence = self.held_out.evidence.to_production()
        return prepare_held_out_session_request(
            job_id=self.job_id,
            bundle_digest=self.bundle_digest,
            manifest_role=self.held_out.manifest_role,
            evidence=evidence,
            discovery=discovery,
            candidate=candidate,
            outcome_contract=outcome,
            analyst=seed.case_analyst("held_out"),
        )


class QualificationAssertion(_StrictModel):
    assertion_id: str
    qualification_class: QualificationClass

    @field_validator("assertion_id")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text("assertion id", value)


class QualificationAssertions(_StrictModel):
    required: tuple[QualificationAssertion, ...]
    control: Literal[
        "grounded_hypothesis", "abstain", "support", "refute", "insufficient"
    ]
    required_citations: tuple[CitationExpectation, ...]
    forbidden_citations: tuple[CitationExpectation, ...]
    attempted_citations: tuple[CitationExpectation, ...]
    citation_verification: Literal["accepted", "rejected"]
    citation_rejection_reason: CitationRejectionCategory | None
    scope_assessment: Literal["preserved", "expanded", "unresolved"] | None
    safety_assessment: Literal["clear", "blocker", "unresolved"] | None

    @model_validator(mode="after")
    def bind_assertions(self) -> QualificationAssertions:
        identifiers = tuple(item.assertion_id for item in self.required)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("required assertions must be non-empty and unique")
        if self.citation_verification == "accepted":
            if self.attempted_citations or self.citation_rejection_reason is not None:
                raise ValueError("accepted citations cannot include rejection proof")
        elif not self.attempted_citations or self.citation_rejection_reason is None:
            raise ValueError(
                "rejected citations require attempted citations and reason"
            )
        return self


class QualificationCase(_StrictModel):
    case_id: str
    required_classes: tuple[QualificationClass, ...]
    role: Literal["discovery", "held_out", "confirmation"]
    fixture: SyntheticAnalystInput
    expected_assertions: QualificationAssertions

    @field_validator("case_id")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _text("case id", value)

    @model_validator(mode="after")
    def bind_case(self) -> QualificationCase:
        if not self.required_classes or len(set(self.required_classes)) != len(
            self.required_classes
        ):
            raise ValueError("required classes must be non-empty and unique")
        if {
            item.qualification_class for item in self.expected_assertions.required
        } != set(self.required_classes):
            raise ValueError("required assertions must exactly cover required classes")
        if self.role == "confirmation":
            # The role exists at type level so an
            # invocation can be prepared for it, but SyntheticAnalystInput has
            # no confirmation slot and to_production_request has no third
            # branch, so a 'confirmation' case would silently render as a
            # HELD-OUT request through the held-out adapter. Reject it here
            # rather than leave that hole open: the confirmation qualification
            # cases land with the confirmation analyst adapter that can execute
            # them, because _bind_qualification_calls keys its bindings by role
            # and has no confirmation arm yet.
            raise ValueError(
                "confirmation qualification cases require the confirmation adapter"
            )
        if (self.role == "discovery") != (self.fixture.held_out is None):
            raise ValueError("fixture role and request shape disagree")
        if self.role == "held_out" and (
            self.expected_assertions.scope_assessment is None
            or self.expected_assertions.safety_assessment is None
        ):
            raise ValueError("held-out fixture requires scope and safety labels")
        documents = list(self.fixture.discovery_sessions)
        if self.fixture.held_out is not None:
            documents.append(self.fixture.held_out.evidence)
        for citation in (
            *self.expected_assertions.required_citations,
            *self.expected_assertions.forbidden_citations,
            *self.expected_assertions.attempted_citations,
        ):
            matches = [
                span
                for document in documents
                if (
                    document.manifest_role,
                    document.user_id,
                    document.session_id,
                    document.trajectory_digest,
                )
                == (
                    citation.manifest_role,
                    citation.user_id,
                    citation.session_id,
                    citation.trajectory_digest,
                )
                for span in document.spans
                if span.span_id == citation.span_id
            ]
            if (
                len(matches) != 1
                or citation.end_offset <= citation.start_offset
                or matches[0].immutable_text[
                    citation.start_offset : citation.end_offset
                ]
                != citation.exact_quote
            ):
                raise ValueError("fixture citation must bind an exact available span")
            if (
                citation.as_verified()
                != verify_citations(
                    (citation.as_structured(),),
                    allowed_spans=(matches[0].to_production(),),
                )[0]
            ):
                raise ValueError("fixture citation provenance is stale")
        return self

    def to_production_request(
        self, seed: SyntheticSuiteSeed
    ) -> DiscoveryRequest | HeldOutSessionRequest:
        return self.fixture.to_production_request(seed)


class QualificationSuite(_StrictModel):
    schema_version: Literal["offline-tuner-open-world-qualification-suite-v1"]
    version: str
    seed: SyntheticSuiteSeed
    cases: tuple[QualificationCase, ...]

    @model_validator(mode="after")
    def validate_suite(self) -> QualificationSuite:
        _text("suite version", self.version)
        if not self.cases or len({case.case_id for case in self.cases}) != len(
            self.cases
        ):
            raise ValueError("qualification cases must be non-empty and unique")
        if {item for case in self.cases for item in case.required_classes} != set(
            QUALIFICATION_CLASSES
        ):
            raise ValueError("qualification suite must cover every required class")
        return self

    def cases_for_role(
        self, role: Literal["discovery", "held_out", "confirmation"]
    ) -> tuple[QualificationCase, ...]:
        return tuple(case for case in self.cases if case.role == role)

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "version": self.version,
            "seed": self.seed.model_dump(mode="json"),
            "cases": [
                case.model_dump(mode="json")
                for case in sorted(
                    self.cases,
                    key=lambda case: (_ROLE_ORDER[case.role], case.case_id.encode()),
                )
            ],
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _completion(
    value: DiscoveryMemo | CandidateRevisionV1 | HeldOutSessionMemo,
    identity: ComponentIdentity,
    metadata: SyntheticCompletionMetadata,
) -> ObservedStructuredCompletion:
    return ObservedStructuredCompletion(
        value=value,
        observed_provider=identity.provider,
        observed_model_revision=identity.immutable_model_revision,
        called_at=metadata.called_at,
        latency_ms=metadata.latency_ms,
        usage_json=metadata.usage_json,
    )


@dataclass(frozen=True, init=False)
class CitationRejectionProof:
    attempted_citations: tuple[EvidenceCitation, ...]
    attempted_citations_digest: str
    allowed_spans_digest: str
    failure_category: CitationRejectionCategory
    rejection_digest: str
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        attempted_citations: tuple[EvidenceCitation, ...] | None = None,
        allowed_spans: tuple[EvidenceSpan, ...] | None = None,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _REJECTION_FACTORY_TOKEN:
            raise TypeError(
                "CitationRejectionProof must be created by the adapter rejection factory"
            )
        if attempted_citations is None or allowed_spans is None:
            raise TypeError("citation rejection factory inputs are required")
        attempted_digest, allowed_digest, category, rejection_digest = (
            _derive_citation_rejection(
                attempted_citations=attempted_citations,
                allowed_spans=allowed_spans,
            )
        )
        object.__setattr__(self, "attempted_citations", attempted_citations)
        object.__setattr__(self, "attempted_citations_digest", attempted_digest)
        object.__setattr__(self, "allowed_spans_digest", allowed_digest)
        object.__setattr__(self, "failure_category", category)
        object.__setattr__(self, "rejection_digest", rejection_digest)
        object.__setattr__(self, "_factory_token", _factory_token)


def _derive_citation_rejection(
    *,
    attempted_citations: tuple[EvidenceCitation, ...],
    allowed_spans: tuple[EvidenceSpan, ...],
) -> tuple[
    str,
    str,
    CitationRejectionCategory,
    str,
]:
    if (
        type(attempted_citations) is not tuple
        or not attempted_citations
        or any(type(item) is not EvidenceCitation for item in attempted_citations)
    ):
        raise TypeError("attempted citations must be a non-empty exact tuple")
    if type(allowed_spans) is not tuple or any(
        type(item) is not EvidenceSpan for item in allowed_spans
    ):
        raise TypeError("allowed spans must be an exact immutable tuple")
    try:
        verify_citations(attempted_citations, allowed_spans=allowed_spans)
    except CitationVerificationError as error:
        category = error.category
    except ValueError as error:
        raise ValueError(
            "citation verifier rejected with an unsupported category"
        ) from error
    else:
        raise ValueError("citation verifier accepted the attempted citations")
    attempted_digest = hashlib.sha256(canonical_bytes(attempted_citations)).hexdigest()
    allowed_digest = hashlib.sha256(canonical_bytes(allowed_spans)).hexdigest()
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "attempted_citations_digest": attempted_digest,
                "allowed_spans_digest": allowed_digest,
                "failure_category": category,
            }
        )
    ).hexdigest()
    return attempted_digest, allowed_digest, category, digest


def _citation_rejection_proof(
    *,
    attempted_citations: tuple[EvidenceCitation, ...],
    allowed_spans: tuple[EvidenceSpan, ...],
) -> CitationRejectionProof:
    return CitationRejectionProof(
        attempted_citations=attempted_citations,
        allowed_spans=allowed_spans,
        _factory_token=_REJECTION_FACTORY_TOKEN,
    )


def _validate_citation_rejection_proof(
    proof: CitationRejectionProof,
    *,
    attempted_citations: tuple[EvidenceCitation, ...],
    allowed_spans: tuple[EvidenceSpan, ...],
) -> None:
    if (
        type(proof) is not CitationRejectionProof
        or getattr(proof, "_factory_token", None) is not _REJECTION_FACTORY_TOKEN
    ):
        raise TypeError("citation rejection must be a sealed verifier proof")
    derived = _derive_citation_rejection(
        attempted_citations=attempted_citations,
        allowed_spans=allowed_spans,
    )
    if (
        proof.attempted_citations,
        proof.attempted_citations_digest,
        proof.allowed_spans_digest,
        proof.failure_category,
        proof.rejection_digest,
    ) != (attempted_citations, *derived):
        raise ValueError("citation rejection proof does not match verifier result")


@dataclass(frozen=True, init=False)
class QualificationCaseObservation:
    case_id: str
    suite_version: str
    suite_digest: str
    request: DiscoveryRequest | HeldOutSessionRequest
    invocation: InvocationRecord
    completion: ObservedStructuredCompletion
    verified_citations: tuple[VerifiedCitation, ...]
    citation_rejection: CitationRejectionProof | None
    observation_digest: str
    _factory_token: object = field(init=False, repr=False, compare=False)

    def __init__(
        self, *args: object, _factory_token: object | None = None, **values: object
    ) -> None:
        if args or _factory_token is not _OBSERVATION_FACTORY_TOKEN:
            raise TypeError(
                "QualificationCaseObservation must be created by the adapter observation factory"
            )
        for name in (
            "case_id",
            "suite_version",
            "suite_digest",
            "request",
            "invocation",
            "completion",
            "verified_citations",
            "citation_rejection",
            "observation_digest",
        ):
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_factory_token", _factory_token)


def _build_qualification_case_observation(
    *,
    case: QualificationCase,
    suite: QualificationSuite,
    identity: ComponentIdentity,
    request: DiscoveryRequest | HeldOutSessionRequest,
    invocation: InvocationRecord,
    completion: ObservedStructuredCompletion,
    verified_citations: tuple[VerifiedCitation, ...],
    citation_rejection: CitationRejectionProof | None = None,
    _observation_token: object | None = None,
) -> QualificationCaseObservation:
    """Internal seam after an adapter has prepared, invoked, and verified a case."""
    if _observation_token is not _QUALIFICATION_OBSERVATION_TOKEN:
        raise TypeError("qualification observation factory is internal to the adapter")
    if (
        type(request) not in (DiscoveryRequest, HeldOutSessionRequest)
        or type(invocation) is not InvocationRecord
        or type(completion) is not ObservedStructuredCompletion
    ):
        raise TypeError("observation requires exact Task 2 invocation records")
    if type(verified_citations) is not tuple or any(
        type(item) is not VerifiedCitation for item in verified_citations
    ):
        raise TypeError("verified citations must be an exact immutable tuple")
    expected_request = case.to_production_request(suite.seed)
    if request != expected_request:
        raise ValueError("request does not exactly bind the fixture production input")
    if (
        identity.component_role != case.role
        or identity.qualification_suite_version != suite.version
    ):
        raise ValueError("observation identity does not match fixture role or suite")
    if (
        request.prepared_invocation.component_identity != identity
        or request.prepared_invocation.role != case.role
    ):
        raise ValueError("prepared invocation does not bind component identity")
    prepared_invocation = request.prepared_invocation
    if case.role == "held_out":
        held_out = case.fixture.held_out
        if held_out is None:
            raise ValueError("held-out fixture is incomplete")
        if (
            prepared_invocation.manifest_role,
            prepared_invocation.user_id,
            prepared_invocation.session_id,
        ) != (
            held_out.manifest_role,
            held_out.evidence.user_id,
            held_out.evidence.session_id,
        ):
            raise ValueError("prepared invocation does not bind held-out session")
    elif any(
        value is not None
        for value in (
            prepared_invocation.manifest_role,
            prepared_invocation.user_id,
            prepared_invocation.session_id,
        )
    ):
        raise ValueError("discovery invocation cannot bind a held-out session")
    output = completion.value
    if (
        (case.role == "discovery" and type(output) is not DiscoveryMemo)
        or (case.role == "held_out" and type(output) is not HeldOutSessionMemo)
        or (case.role == "confirmation" and type(output) is not ConfirmationSessionMemo)
    ):
        raise TypeError("completion output does not match fixture role")
    output = cast(DiscoveryMemo | HeldOutSessionMemo, output)
    if (
        completion.observed_provider,
        completion.observed_model_revision,
        completion.called_at,
        completion.latency_ms,
        completion.usage_json,
    ) != (
        invocation.observed_provider,
        invocation.observed_model_revision,
        invocation.called_at,
        invocation.latency_ms,
        invocation.usage_json,
    ):
        raise ValueError("completion provenance does not bind invocation")
    if (
        invocation.analyst_input_identity,
        invocation.component_identity_digest,
        invocation.observed_provider,
        invocation.observed_model_revision,
    ) != (
        prepared_invocation.idempotency_key,
        identity.digest,
        identity.provider,
        identity.immutable_model_revision,
    ):
        raise ValueError("invocation provenance does not bind prepared identity")
    if invocation.output_digest != hashlib.sha256(canonical_bytes(output)).hexdigest():
        raise ValueError("invocation output digest does not bind completion")
    output_citations = tuple(output.citations)
    allowed_spans = _allowed_spans(request)
    if citation_rejection is None:
        expected_verified = _expected_verified_citations(case, request)
        if (
            verify_citations(output_citations, allowed_spans=allowed_spans)
            != verified_citations
            or verified_citations != expected_verified
        ):
            raise ValueError(
                "verified citations do not exactly project structured output"
            )
    else:
        if verified_citations:
            raise ValueError("citation rejection cannot include verified citations")
        _validate_citation_rejection_proof(
            citation_rejection,
            attempted_citations=output_citations,
            allowed_spans=allowed_spans,
        )
    payload = {
        "case_id": case.case_id,
        "suite_version": suite.version,
        "suite_digest": suite.digest,
        "request": canonical_bytes(request).decode(),
        "invocation": canonical_bytes(invocation).decode(),
        "completion": canonical_bytes(completion).decode(),
        "verified_citations": canonical_bytes(verified_citations).decode(),
        "citation_rejection": None
        if citation_rejection is None
        else citation_rejection.rejection_digest,
    }
    return QualificationCaseObservation(
        case_id=case.case_id,
        suite_version=suite.version,
        suite_digest=suite.digest,
        request=request,
        invocation=invocation,
        completion=completion,
        verified_citations=verified_citations,
        citation_rejection=citation_rejection,
        observation_digest=hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_version": "offline-tuner-qualification-observation-v2",
                    **payload,
                }
            )
        ).hexdigest(),
        _factory_token=_OBSERVATION_FACTORY_TOKEN,
    )


def _allowed_spans(
    request: DiscoveryRequest | HeldOutSessionRequest,
) -> tuple[EvidenceSpan, ...]:
    if type(request) is DiscoveryRequest:
        return tuple(
            span for document in request.discovery_sessions for span in document.spans
        )
    return cast(HeldOutSessionRequest, request).evidence.spans


def _expected_verified_citations(
    case: QualificationCase,
    request: DiscoveryRequest | HeldOutSessionRequest,
) -> tuple[VerifiedCitation, ...]:
    return verify_citations(
        tuple(
            citation.as_structured()
            for citation in case.expected_assertions.required_citations
        ),
        allowed_spans=_allowed_spans(request),
    )


@dataclass(frozen=True)
class ClassQualificationCount:
    qualification_class: QualificationClass
    required: int
    passed_required: int


class AnalystQualificationResult(Protocol):
    """Read-only view of one qualification result issued by the reducer."""

    @property
    def component_identity_digest(self) -> str: ...

    @property
    def suite_digest(self) -> str: ...

    @property
    def class_counts(self) -> tuple[ClassQualificationCount, ...]: ...

    @property
    def passed(self) -> bool: ...

    @property
    def result_digest(self) -> str: ...

    @property
    def observation_digests(self) -> tuple[str, ...]: ...


class _QualificationReducer(Protocol):
    def __call__(
        self,
        suite: QualificationSuite,
        observed: Sequence[QualificationCaseObservation],
        identity: ComponentIdentity,
    ) -> AnalystQualificationResult: ...


class _IssuedQualificationResultLookup(Protocol):
    def __call__(self, value: object) -> AnalystQualificationResult | None: ...


class _DurableQualificationRestore(Protocol):
    def __call__(
        self,
        *,
        storage: BaseStorage,
        identity: ComponentIdentity,
        suite: QualificationSuite,
    ) -> AnalystQualificationResult | None: ...


class _DurableQualificationPairRestore(Protocol):
    def __call__(
        self,
        *,
        storage: BaseStorage,
        discovery_identity: ComponentIdentity,
        held_out_identity: ComponentIdentity,
        suite: QualificationSuite,
    ) -> tuple[
        AnalystQualificationResult | None,
        AnalystQualificationResult | None,
    ]: ...


class _QualificationCurrentCheck(Protocol):
    def __call__(
        self,
        result: AnalystQualificationResult,
        identity: ComponentIdentity,
    ) -> bool: ...


type _QualificationResultSnapshot = tuple[
    str,
    str,
    tuple[tuple[QualificationClass, int, int], ...],
    bool,
    str,
    tuple[str, ...],
]


def _build_qualification_authority() -> tuple[  # noqa: C901 - one shared closure
    _QualificationReducer,
    _IssuedQualificationResultLookup,
    _DurableQualificationRestore,
    _DurableQualificationPairRestore,
    _QualificationCurrentCheck,
]:
    """Create the reducer with no module-visible constructor or registry."""

    @dataclass(frozen=True, slots=True, weakref_slot=True, init=False, eq=False)
    class _ReducerIssuedResult:
        component_identity_digest: str
        suite_digest: str
        class_counts: tuple[ClassQualificationCount, ...]
        passed: bool
        result_digest: str
        observation_digests: tuple[str, ...]

        def __init__(self, *_args: object, **_values: object) -> None:
            raise TypeError("qualification results are issued only by the reducer")

    issued: WeakKeyDictionary[_ReducerIssuedResult, _QualificationResultSnapshot] = (
        WeakKeyDictionary()
    )

    def snapshot(result: _ReducerIssuedResult) -> _QualificationResultSnapshot:
        return (
            result.component_identity_digest,
            result.suite_digest,
            tuple(
                (
                    item.qualification_class,
                    item.required,
                    item.passed_required,
                )
                for item in result.class_counts
            ),
            result.passed,
            result.result_digest,
            result.observation_digests,
        )

    def issued_result(value: object) -> AnalystQualificationResult | None:
        if type(value) is not _ReducerIssuedResult:
            return None
        result = cast(_ReducerIssuedResult, value)
        expected = issued.get(result)
        if expected is None or expected != snapshot(result):
            return None
        return result

    def register_result(
        *,
        component_identity_digest: str,
        suite_digest: str,
        class_counts: tuple[ClassQualificationCount, ...],
        passed: bool,
        result_digest: str,
        observation_digests: tuple[str, ...],
    ) -> _ReducerIssuedResult:
        result = object.__new__(_ReducerIssuedResult)
        values = (
            ("component_identity_digest", component_identity_digest),
            ("suite_digest", suite_digest),
            ("class_counts", class_counts),
            ("passed", passed),
            ("result_digest", result_digest),
            ("observation_digests", observation_digests),
        )
        for name, value in values:
            object.__setattr__(result, name, value)
        issued[result] = snapshot(result)
        return result

    def result_digest(
        component_digest: str,
        suite_digest: str,
        counts: tuple[ClassQualificationCount, ...],
        passed: bool,
        observation_digests: tuple[str, ...],
    ) -> str:
        payload = canonical_json_bytes(
            {
                "schema_version": "offline-tuner-qualification-result-v1",
                "component_identity_digest": component_digest,
                "suite_digest": suite_digest,
                "class_counts": [
                    {
                        "qualification_class": item.qualification_class,
                        "required": item.required,
                        "passed_required": item.passed_required,
                    }
                    for item in counts
                ],
                "passed": passed,
                "observation_digests": observation_digests,
            }
        )
        return hmac.new(
            get_governance_ref_secret().encode(),
            _QUALIFICATION_RESULT_HMAC_DOMAIN + payload,
            hashlib.sha256,
        ).hexdigest()

    def verified_reissue(
        *,
        record: OpenWorldQualificationRecord,
        identity: ComponentIdentity,
        suite: QualificationSuite,
    ) -> AnalystQualificationResult:
        if (
            type(record) is not OpenWorldQualificationRecord
            or type(identity) is not ComponentIdentity
            or type(suite) is not QualificationSuite
        ):
            raise TypeError(
                "record, identity, and suite must be exact qualification contracts"
            )
        OpenWorldQualificationRecord.model_validate(
            record.model_dump(mode="python"), strict=True
        )
        if (
            record.schema_version != OPEN_WORLD_QUALIFICATION_RECORD_SCHEMA_VERSION
            or record.component_identity_digest != identity.digest
            or record.suite_digest != suite.digest
            or identity.component_role not in ("discovery", "held_out", "confirmation")
            or identity.qualification_suite_version != suite.version
        ):
            raise ValueError("durable qualification record has stale identity or suite")
        if (
            type(record.class_counts) is not tuple
            or len(record.class_counts) != len(QUALIFICATION_CLASSES)
            or any(
                type(count) is not OpenWorldQualificationClassCount
                for count in record.class_counts
            )
        ):
            raise ValueError("durable qualification record has malformed class counts")
        if record.passed is not True:
            raise ValueError("durable qualification record did not pass exactly")
        required = dict.fromkeys(QUALIFICATION_CLASSES, 0)
        cases = suite.cases_for_role(identity.component_role)
        if not cases:
            # Without this the role guard above admits a role the suite cannot
            # carry a case for. `QualificationCase` rejects role="confirmation"
            # at parse time, so `cases_for_role("confirmation")` is permanently
            # empty, and every downstream check degenerates in the same
            # direction: `required` stays all zeros, so `observed_counts !=
            # expected_counts` cannot fire; `len(observation_digests) !=
            # len(cases)` becomes `0 != 0`; and the HMAC then verifies whatever
            # was stored. A record asserting nothing would reissue as "an exact
            # pass-all result". `reduce_results` already refuses this via
            # `exact_population = bool(cases) and ...`; closure must not rest on
            # a two-step argument across two functions.
            raise ValueError(
                "durable qualification record asserts nothing: the suite "
                f"carries no case for role {identity.component_role!r}"
            )
        for case in cases:
            for assertion in case.expected_assertions.required:
                required[assertion.qualification_class] += 1
        expected_counts = tuple(
            (
                qualification_class,
                required[qualification_class],
                required[qualification_class],
            )
            for qualification_class in QUALIFICATION_CLASSES
        )
        observed_counts = tuple(
            (count.qualification_class, count.required, count.passed_required)
            for count in record.class_counts
        )
        if observed_counts != expected_counts:
            raise ValueError(
                "durable qualification record is not an exact pass-all result"
            )
        observation_digests = record.observation_digests
        if (
            type(observation_digests) is not tuple
            or len(observation_digests) != len(cases)
            or tuple(sorted(set(observation_digests))) != observation_digests
        ):
            raise ValueError("durable qualification record did not pass exactly")
        for digest in observation_digests:
            _digest("qualification observation digest", digest)
        counts = tuple(
            ClassQualificationCount(
                count.qualification_class,
                count.required,
                count.passed_required,
            )
            for count in record.class_counts
        )
        expected_digest = result_digest(
            identity.digest, suite.digest, counts, True, observation_digests
        )
        if not hmac.compare_digest(record.result_digest, expected_digest):
            raise ValueError("durable qualification result digest does not verify")
        return register_result(
            component_identity_digest=record.component_identity_digest,
            suite_digest=record.suite_digest,
            class_counts=counts,
            passed=True,
            result_digest=record.result_digest,
            observation_digests=observation_digests,
        )

    def restore_from_storage(
        *,
        storage: BaseStorage,
        identity: ComponentIdentity,
        suite: QualificationSuite,
    ) -> AnalystQualificationResult | None:
        if not isinstance(storage, BaseStorage):
            raise TypeError("storage must be a real BaseStorage boundary")
        if (
            type(identity) is not ComponentIdentity
            or type(suite) is not QualificationSuite
        ):
            raise TypeError("identity and suite must be exact qualification contracts")
        record = storage.load_open_world_qualification_record(
            component_identity_digest=identity.digest,
            suite_digest=suite.digest,
        )
        if record is None:
            return None
        return verified_reissue(record=record, identity=identity, suite=suite)

    def restore_pair_from_storage(
        *,
        storage: BaseStorage,
        discovery_identity: ComponentIdentity,
        held_out_identity: ComponentIdentity,
        suite: QualificationSuite,
    ) -> tuple[
        AnalystQualificationResult | None,
        AnalystQualificationResult | None,
    ]:
        if not isinstance(storage, BaseStorage):
            raise TypeError("storage must be a real BaseStorage boundary")
        if (
            type(discovery_identity) is not ComponentIdentity
            or type(held_out_identity) is not ComponentIdentity
            or type(suite) is not QualificationSuite
        ):
            raise TypeError(
                "identities and suite must be exact qualification contracts"
            )
        if (
            discovery_identity.component_role != "discovery"
            or held_out_identity.component_role != "held_out"
        ):
            raise ValueError("qualification pair requires exact analyst roles")

        discovery_record = storage.load_open_world_qualification_record(
            component_identity_digest=discovery_identity.digest,
            suite_digest=suite.digest,
        )
        held_out_record = storage.load_open_world_qualification_record(
            component_identity_digest=held_out_identity.digest,
            suite_digest=suite.digest,
        )

        discovery_result = (
            None
            if discovery_record is None
            else verified_reissue(
                record=discovery_record,
                identity=discovery_identity,
                suite=suite,
            )
        )
        held_out_result = (
            None
            if held_out_record is None
            else verified_reissue(
                record=held_out_record,
                identity=held_out_identity,
                suite=suite,
            )
        )
        return discovery_result, held_out_result

    def reduce_results(
        suite: QualificationSuite,
        observed: Sequence[QualificationCaseObservation],
        identity: ComponentIdentity,
    ) -> AnalystQualificationResult:
        if (
            type(suite) is not QualificationSuite
            or type(identity) is not ComponentIdentity
        ):
            raise TypeError("suite and identity must be exact qualification contracts")
        if identity.component_role == "candidate":
            raise ValueError("only analyst components may be qualified")
        if type(observed) is not tuple or any(
            type(item) is not QualificationCaseObservation for item in observed
        ):
            raise TypeError("observations must be an exact immutable tuple")
        cases = suite.cases_for_role(identity.component_role)
        by_case = {item.case_id: item for item in observed}
        exact_population = (
            bool(cases)
            and len(by_case) == len(observed)
            and set(by_case) == {case.case_id for case in cases}
        )
        totals = dict.fromkeys(QUALIFICATION_CLASSES, 0)
        passed = dict.fromkeys(QUALIFICATION_CLASSES, 0)
        for case in cases:
            valid = case.case_id in by_case and _observation_matches(
                case, suite, identity, by_case[case.case_id]
            )
            for assertion in case.expected_assertions.required:
                totals[assertion.qualification_class] += 1
                if valid:
                    passed[assertion.qualification_class] += 1
        counts = tuple(
            ClassQualificationCount(item, totals[item], passed[item])
            for item in QUALIFICATION_CLASSES
        )
        result_passed = (
            identity.qualification_suite_version == suite.version
            and exact_population
            and all(item.required == item.passed_required for item in counts)
        )
        observation_digests = tuple(
            sorted(item.observation_digest for item in observed)
        )
        digest = result_digest(
            identity.digest,
            suite.digest,
            counts,
            result_passed,
            observation_digests,
        )
        return register_result(
            component_identity_digest=identity.digest,
            suite_digest=suite.digest,
            class_counts=counts,
            passed=result_passed,
            result_digest=digest,
            observation_digests=observation_digests,
        )

    def result_is_current(
        result: AnalystQualificationResult,
        identity: ComponentIdentity,
    ) -> bool:
        current = issued_result(result)
        if current is None or type(identity) is not ComponentIdentity:
            return False
        return (
            current.component_identity_digest == identity.digest
            and current.passed is True
            and hmac.compare_digest(
                current.result_digest,
                result_digest(
                    current.component_identity_digest,
                    current.suite_digest,
                    current.class_counts,
                    current.passed,
                    current.observation_digests,
                ),
            )
        )

    return (
        reduce_results,
        issued_result,
        restore_from_storage,
        restore_pair_from_storage,
        result_is_current,
    )


(
    reduce_qualification_results,
    _issued_qualification_result,
    _restore_qualification_result_from_storage,
    _restore_qualification_results_from_storage,
    qualification_result_is_current,
) = _build_qualification_authority()
del _build_qualification_authority


def seal_qualified_analyst(
    *, identity: ComponentIdentity, result: AnalystQualificationResult
) -> QualifiedAnalyst:
    """Seal one unchanged reducer-issued result into authorization."""
    if type(identity) is not ComponentIdentity:
        raise TypeError("identity must be ComponentIdentity")
    issued_result = _issued_qualification_result(result)
    if (
        issued_result is None
        or issued_result.passed is not True
        or issued_result.component_identity_digest != identity.digest
    ):
        raise ValueError("qualification result does not seal this identity")
    return QualifiedAnalyst(
        identity=identity,
        qualification=AnalystQualificationResultRef(
            component_identity_digest=identity.digest,
            suite_digest=issued_result.suite_digest,
            result_digest=issued_result.result_digest,
            passed=issued_result.passed,
        ),
        _qualification_result=issued_result,
        _factory_token=_QUALIFIED_ANALYST_FACTORY_TOKEN,
    )


def analyst_authorization_is_sealed(analyst: object) -> bool:
    """True when ``analyst`` carries its unchanged reducer result."""
    if type(analyst) is not QualifiedAnalyst:
        return False
    result = _issued_qualification_result(analyst._qualification_result)
    if result is None:
        return False
    return (
        result.component_identity_digest == analyst.identity.digest
        and result.suite_digest == analyst.qualification.suite_digest
        and hmac.compare_digest(
            result.result_digest,
            analyst.qualification.result_digest,
        )
        and result.passed is True
        and qualification_result_is_current(result, analyst.identity)
    )


def attempt_authorization_is_sealed(analysts: object) -> bool:
    """True when every analyst in ``analysts`` is reducer-issued."""
    return (
        type(analysts) is QualifiedAnalystSet
        and analyst_authorization_is_sealed(analysts.discovery)
        and analyst_authorization_is_sealed(analysts.held_out)
    )


def load_qualification_suite(version: str = "v1") -> QualificationSuite:
    suite = QualificationSuite.model_validate_json(
        files(__package__)
        .joinpath("qualification_cases", f"{version}.json")
        .read_text(encoding="utf-8")
    )
    if suite.version != version:
        raise ValueError("qualification suite version does not match requested version")
    return suite


def _observation_matches(
    case: QualificationCase,
    suite: QualificationSuite,
    identity: ComponentIdentity,
    observation: QualificationCaseObservation,
) -> bool:
    if observation._factory_token is not _OBSERVATION_FACTORY_TOKEN or (
        observation.suite_version,
        observation.suite_digest,
    ) != (suite.version, suite.digest):
        return False
    rejection = observation.citation_rejection
    if rejection is not None and (
        type(rejection) is not CitationRejectionProof
        or getattr(rejection, "_factory_token", None) is not _REJECTION_FACTORY_TOKEN
    ):
        return False
    try:
        rebuilt = _build_qualification_case_observation(
            case=case,
            suite=suite,
            identity=identity,
            request=observation.request,
            invocation=observation.invocation,
            completion=observation.completion,
            verified_citations=observation.verified_citations,
            citation_rejection=observation.citation_rejection,
            _observation_token=_QUALIFICATION_OBSERVATION_TOKEN,
        )
    except (TypeError, ValueError):
        return False
    if rebuilt.observation_digest != observation.observation_digest:
        return False
    expected = case.expected_assertions
    output = observation.completion.value
    if case.role == "discovery":
        if type(output) is not DiscoveryMemo or output.decision != expected.control:
            return False
    elif type(output) is not HeldOutSessionMemo or (
        output.control,
        output.scope_assessment,
        output.safety_assessment,
    ) != (expected.control, expected.scope_assessment, expected.safety_assessment):
        return False
    if rejection is not None:
        try:
            _validate_citation_rejection_proof(
                rejection,
                attempted_citations=tuple(output.citations),
                allowed_spans=_allowed_spans(observation.request),
            )
        except (TypeError, ValueError):
            return False
    required = tuple(
        citation.as_structured() for citation in expected.required_citations
    )
    attempted = tuple(
        citation.as_structured() for citation in expected.attempted_citations
    )
    if expected.citation_verification == "accepted":
        return (
            observation.citation_rejection is None
            and output.citations == required
            and observation.verified_citations
            == _expected_verified_citations(case, observation.request)
        )
    return (
        observation.citation_rejection is not None
        and observation.citation_rejection.failure_category
        == expected.citation_rejection_reason
        and output.citations == attempted
    )
