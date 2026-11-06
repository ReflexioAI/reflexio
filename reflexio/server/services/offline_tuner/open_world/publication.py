"""Provisional publication of one durable accepted open-world attempt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, cast

from reflexio.models.api_schema.domain import OptimizationJobClaim, UserPlaybook
from reflexio.server.services.offline_tuner.open_world.identity import canonical_bytes
from reflexio.server.services.offline_tuner.open_world.models import (
    OpenWorldAcceptedPublicationInputs,
)
from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldProvisionalPublicationStore,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldEvidenceTargetStaleError,
    OpenWorldPublicationIncumbentChangedError,
    OpenWorldPublicationRetryIdentityMismatchError,
)
from reflexio.server.services.offline_tuner.search_projection import (
    CandidateSearchProjection,
    CandidateSearchProjectionRequest,
    CandidateSearchProjector,
    publication_search_projection,
)
from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    ProvisionalPublicationRequest,
    ProvisionalPublicationResult,
    PublicationClaim,
    QualificationAuthorityRef,
    canonical_json_bytes,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)

from .rule_evidence_runtime import (
    RuleEvidenceAnalysisStore,
    RuleEvidencePublicationInputs,
)

PublicationInputs = OpenWorldAcceptedPublicationInputs | RuleEvidencePublicationInputs

_PROOF_SCHEMA_VERSION = "offline-tuner-open-world-publication-proof-v1"


class OpenWorldPublicationError(ValueError):
    """A durable attempt is not eligible for provisional publication."""


class UserPlaybookLoader(Protocol):
    def get_user_playbook_by_id(
        self, user_playbook_id: int, *, include_tombstones: bool = False
    ) -> UserPlaybook | None: ...


def _authority_payload(authority: QualificationAuthorityRef) -> dict[str, object]:
    payload: dict[str, object] = {
        "epoch": authority.epoch,
        "authority_digest": authority.authority_digest,
        "discovery_component_identity_digest": (
            authority.discovery_component_identity_digest
        ),
        "discovery_qualification_suite_digest": (
            authority.discovery_qualification_suite_digest
        ),
        "discovery_qualification_result_digest": (
            authority.discovery_qualification_result_digest
        ),
        "held_out_component_identity_digest": (
            authority.held_out_component_identity_digest
        ),
        "held_out_qualification_suite_digest": (
            authority.held_out_qualification_suite_digest
        ),
        "held_out_qualification_result_digest": (
            authority.held_out_qualification_result_digest
        ),
        "candidate_generator_identity_digest": (
            authority.candidate_generator_identity_digest
        ),
        "candidate_generator_authorization_digest": (
            authority.candidate_generator_authorization_digest
        ),
    }
    if authority.policy_id == "evidence-grounded-exact-edit-v1":
        payload["policy_id"] = authority.policy_id
    return payload


def open_world_decision_proof_envelope(
    inputs: PublicationInputs,
) -> DecisionProofEnvelope:
    """Bind the accepted decision to every durable publication authority."""
    if type(inputs) not in {
        OpenWorldAcceptedPublicationInputs,
        RuleEvidencePublicationInputs,
    }:
        raise TypeError("inputs must be validated native publication inputs")
    new_method = isinstance(inputs, RuleEvidencePublicationInputs)
    payload = {
        "schema_version": (
            "offline-tuner-rule-evidence-publication-proof-v1"
            if new_method
            else _PROOF_SCHEMA_VERSION
        ),
        "optimizer_kind": "offline_tuner_open_world",
        "decision": "apply",
        "job_id": inputs.job_id,
        "attempt_key": inputs.attempt_key,
        "attempt_decision_digest": inputs.decision_digest,
        "attempt_decision_json": (
            inputs.decision_json
            if new_method
            else canonical_bytes(inputs.decision).decode("utf-8")
        ),
        "evidence_bundle_digest": inputs.evidence_bundle_digest,
        "outcome_contract_digest": inputs.outcome_contract_digest,
        "candidate_digest": inputs.candidate_digest,
        "candidate_content_digest": inputs.candidate.content_diff.candidate_content_digest,
        "incumbent_full_version_fingerprint": inputs.incumbent_full_version_fingerprint,
        "attempt_context_digest": inputs.attempt_context_digest,
        "qualification_authority": _authority_payload(inputs.qualification_authority),
        "governance_subjects": [
            {"ref": subject_ref, "epoch": epoch}
            for subject_ref, epoch in inputs.subject_epochs
        ],
    }
    if isinstance(inputs, RuleEvidencePublicationInputs):
        payload["policy_id"] = inputs.qualification_authority.policy_id
        payload["diagnostic_population_digest"] = (
            inputs.decision.diagnostic_population_digest
        )
    else:
        payload["held_out_population_digest"] = (
            inputs.decision.held_out_population_digest
        )
    canonical_json = canonical_json_bytes(payload).decode("utf-8")
    return DecisionProofEnvelope(
        optimizer_kind="offline_tuner_open_world",
        schema_version=str(payload["schema_version"]),
        canonical_json=canonical_json,
        digest=sha256(canonical_json.encode("utf-8")).hexdigest(),
        decision="apply",
    )


def _incumbent_snapshot_json(incumbent: UserPlaybook) -> str:
    return canonical_json_bytes(
        {
            "schema_version": "user-playbook-full-version-v1",
            "user_playbook": incumbent.model_dump(mode="json", exclude={"embedding"})
            | {
                "governance_subject_ref": incumbent.governance_subject_ref,
                "retired_at": incumbent.retired_at,
            },
        }
    ).decode("utf-8")


def _subject_epochs_json(inputs: PublicationInputs) -> str:
    return canonical_json_bytes(
        {
            "subjects": [
                {"ref": subject_ref, "epoch": epoch}
                for subject_ref, epoch in inputs.subject_epochs
            ]
        }
    ).decode("utf-8")


@dataclass(frozen=True)
class OpenWorldProvisionalPublisher:
    """Build and commit publication only from current durable attempt state."""

    analysis_store: RuleEvidenceAnalysisStore
    publication_store: OpenWorldProvisionalPublicationStore
    playbook_loader: UserPlaybookLoader
    projector: CandidateSearchProjector
    clock: Callable[[], int]

    def publish(
        self,
        *,
        job_id: int,
        claim: OptimizationJobClaim,
    ) -> ProvisionalPublicationResult:
        if type(job_id) is not int or job_id <= 0:
            raise OpenWorldPublicationError("job_id must be positive")
        if type(claim) is not OptimizationJobClaim or claim.job_id != job_id:
            raise OpenWorldPublicationError("claim does not authorize this job")

        committed = self._load_committed(
            job_id=job_id,
            claim=claim,
        )
        if committed is not None:
            return committed

        now = self.clock()
        if type(now) is not int or now < 0:
            raise OpenWorldPublicationError("publisher clock must return an integer")
        if claim.expires_at <= now:
            raise OpenWorldPublicationError("publication worker lease has expired")
        loaded = self._load_inputs(job_id=job_id, claim=claim, now=now)
        if isinstance(loaded, ProvisionalPublicationResult):
            return loaded
        inputs = loaded
        if inputs.job_id != job_id or inputs.claim != claim:
            raise OpenWorldPublicationError(
                "durable publication inputs changed retry identity"
            )

        incumbent = self.playbook_loader.get_user_playbook_by_id(
            inputs.incumbent_user_playbook_id
        )
        if (
            incumbent is not None
            and not (incumbent.governance_subject_ref or "").strip()
        ):
            raise OpenWorldPublicationError(
                "current incumbent has no governance subject"
            )
        if self._incumbent_drifted(incumbent=incumbent, inputs=inputs):
            return self._resolve_stale(
                job_id=job_id,
                claim=claim,
            )
        incumbent = cast(UserPlaybook, incumbent)

        incumbent_snapshot_json = _incumbent_snapshot_json(incumbent)
        if sha256(incumbent_snapshot_json.encode("utf-8")).hexdigest() != (
            inputs.incumbent_full_version_fingerprint
        ):
            raise OpenWorldPublicationError("incumbent snapshot fingerprint changed")

        projection_request = CandidateSearchProjectionRequest(
            incumbent_user_playbook_id=incumbent.user_playbook_id,
            incumbent_content_digest=sha256(
                incumbent.content.encode("utf-8")
            ).hexdigest(),
            candidate_content=inputs.candidate.revision.revised_content,
            preserved_trigger=incumbent.trigger,
        )
        projection = self.projector.build(projection_request)
        self._validate_projection(projection, projection_request)

        try:
            publication_claim = (
                self.publication_store.claim_user_playbook_provisional_publication(
                    job_id=job_id,
                    owner=claim.owner,
                    worker_fence=claim.fence,
                    incumbent_user_playbook_id=inputs.incumbent_user_playbook_id,
                    incumbent_full_version_fingerprint=(
                        inputs.incumbent_full_version_fingerprint
                    ),
                    incumbent_snapshot_json=incumbent_snapshot_json,
                )
            )
        except OpenWorldPublicationIncumbentChangedError as exc:
            return self._resolve_stale(job_id=job_id, claim=claim, cause=exc)

        if (
            type(publication_claim) is not PublicationClaim
            or publication_claim.job_id != job_id
            or publication_claim.owner != claim.owner
        ):
            raise OpenWorldPublicationError(
                "durable publication claim changed worker identity"
            )
        request = ProvisionalPublicationRequest(
            optimizer_kind="offline_tuner_open_world",
            job_id=job_id,
            attempt_key=inputs.attempt_key,
            publication_claim=publication_claim,
            worker_fence=claim.fence,
            incumbent_user_playbook_id=inputs.incumbent_user_playbook_id,
            incumbent_full_version_fingerprint=(
                inputs.incumbent_full_version_fingerprint
            ),
            incumbent_snapshot_json=incumbent_snapshot_json,
            revised_content=inputs.candidate.revision.revised_content,
            projection=publication_search_projection(projection),
            decision_proof=open_world_decision_proof_envelope(inputs),
            subject_epochs_json=_subject_epochs_json(inputs),
            qualification_authority=inputs.qualification_authority,
            evidence_bundle_digest=inputs.evidence_bundle_digest,
            candidate_digest=inputs.candidate_digest,
        )
        try:
            staged = self.publication_store.stage_user_playbook_provisional_publication(
                request
            )
        except Exception as exc:
            if self._incumbent_drifted(
                incumbent=self.playbook_loader.get_user_playbook_by_id(
                    inputs.incumbent_user_playbook_id
                ),
                inputs=inputs,
            ):
                return self._resolve_stale(job_id=job_id, claim=claim, cause=exc)
            raise
        if not staged:
            # A refused stage carries no information about whether the attempt
            # was actually cleaned up: the stage RPC calls the SQL stale cleanup
            # and discards its boolean. Every other stale outcome resolves
            # through _resolve_stale, which consults the cleanup result and
            # falls back to the durable committed row; raising straight from
            # here reported "incumbent does not match" for a publication that a
            # racing worker had already committed.
            return self._resolve_stale(job_id=job_id, claim=claim)
        try:
            result = (
                self.publication_store.commit_user_playbook_provisional_publication(
                    request
                )
            )
            if result.outcome == "incumbent_changed":
                raise OpenWorldPublicationError(
                    "current incumbent does not match the frozen full version"
                )
            return result
        except Exception as exc:
            committed = self._load_committed(
                job_id=job_id,
                claim=claim,
            )
            if committed is not None:
                return committed
            if self._incumbent_drifted(
                incumbent=self.playbook_loader.get_user_playbook_by_id(
                    inputs.incumbent_user_playbook_id
                ),
                inputs=inputs,
            ):
                return self._resolve_stale(job_id=job_id, claim=claim, cause=exc)
            raise

    def _validate_projection(
        self,
        projection: CandidateSearchProjection,
        request: CandidateSearchProjectionRequest,
    ) -> None:
        if type(projection) is not CandidateSearchProjection:
            raise OpenWorldPublicationError(
                "candidate projector returned an invalid projection"
            )
        if projection.key_identity != self.projector.key_identity:
            raise OpenWorldPublicationError("candidate projector identity changed")
        if projection.candidate_content_digest != request.candidate_content_digest:
            raise OpenWorldPublicationError("candidate projection content changed")
        if projection.preserved_trigger != request.preserved_trigger:
            raise OpenWorldPublicationError("candidate projection trigger changed")

    def _load_inputs(
        self, *, job_id: int, claim: OptimizationJobClaim, now: int
    ) -> PublicationInputs | ProvisionalPublicationResult:
        try:
            return self.analysis_store.load_accepted_publication_inputs(
                job_id=job_id,
                claim=claim,
                now=now,
            )
        except OpenWorldEvidenceTargetStaleError as exc:
            return self._resolve_stale(job_id=job_id, claim=claim, cause=exc)

    @staticmethod
    def _incumbent_drifted(
        *,
        incumbent: UserPlaybook | None,
        inputs: PublicationInputs,
    ) -> bool:
        return (
            incumbent is None
            or incumbent.user_playbook_id != inputs.incumbent_user_playbook_id
            or incumbent.status is not None
            or user_playbook_full_version_fingerprint(incumbent)
            != inputs.incumbent_full_version_fingerprint
            or sha256(incumbent.content.encode("utf-8")).hexdigest()
            != inputs.candidate.content_diff.incumbent_content_digest
        )

    def _resolve_stale(
        self,
        *,
        job_id: int,
        claim: OptimizationJobClaim,
        cause: Exception | None = None,
    ) -> ProvisionalPublicationResult:
        cleaned = (
            self.publication_store.cleanup_user_playbook_provisional_stale_attempt(
                job_id=job_id,
                owner=claim.owner,
                worker_fence=claim.fence,
            )
        )
        if not cleaned:
            committed = self._load_committed(
                job_id=job_id,
                claim=claim,
            )
            if committed is not None:
                return committed
        error = OpenWorldPublicationError(
            "current incumbent does not match the frozen full version"
        )
        if cause is not None:
            raise error from cause
        raise error

    def _load_committed(
        self,
        *,
        job_id: int,
        claim: OptimizationJobClaim,
    ) -> ProvisionalPublicationResult | None:
        try:
            result = (
                self.publication_store.load_open_world_provisional_publication_retry(
                    job_id=job_id,
                    claim=claim,
                )
            )
        except OpenWorldPublicationRetryIdentityMismatchError as exc:
            raise OpenWorldPublicationError(
                "committed provisional retry identity changed"
            ) from exc
        return result


__all__ = [
    "OpenWorldProvisionalPublisher",
    "OpenWorldPublicationError",
    "UserPlaybookLoader",
    "open_world_decision_proof_envelope",
]
