"""Strict shared contracts for atomic user-playbook publication."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

from reflexio.models.api_schema.domain.entities import OptimizerKind

PublicationOutcome = Literal["applied", "incumbent_changed"]
PublishableOptimizerKind = Literal["gepa", "offline_tuner_replay"]

_PUBLISHABLE_OPTIMIZERS = frozenset({"gepa", "offline_tuner_replay"})


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _canonical_payload(name: str, value: str) -> object:
    try:
        payload = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if canonical != value:
        raise ValueError(f"{name} must use canonical JSON bytes")
    return payload


def _validate_optimizer(value: object) -> None:
    if value not in _PUBLISHABLE_OPTIMIZERS:
        raise ValueError("optimizer_kind is not publishable")


@dataclass(frozen=True)
class PublicationClaim:
    job_id: int
    owner: str
    fence: int

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("publication claim job_id must be positive")
        _require_text("publication claim owner", self.owner)
        if type(self.fence) is not int or self.fence <= 0:
            raise ValueError("publication claim fence must be positive")


@dataclass(frozen=True)
class DecisionProofEnvelope:
    optimizer_kind: OptimizerKind
    schema_version: str
    canonical_json: str
    digest: str
    decision: Literal["apply"]

    def __post_init__(self) -> None:
        _validate_optimizer(self.optimizer_kind)
        _require_text("decision proof schema_version", self.schema_version)
        _require_digest("decision proof digest", self.digest)
        if self.decision != "apply":
            raise ValueError("publication decision must be apply")
        payload = _canonical_payload(
            "decision proof canonical_json", self.canonical_json
        )
        if sha256(self.canonical_json.encode("utf-8")).hexdigest() != self.digest:
            raise ValueError("decision proof digest does not match canonical JSON")
        if not isinstance(payload, dict):
            raise ValueError("decision proof canonical JSON must be an object")
        if payload.get("optimizer_kind") != self.optimizer_kind:
            raise ValueError("decision proof optimizer_kind does not match envelope")
        if payload.get("schema_version") != self.schema_version:
            raise ValueError("decision proof schema_version does not match envelope")
        if payload.get("decision") != self.decision:
            raise ValueError("decision proof decision does not match envelope")


@dataclass(frozen=True)
class PublicationSearchProjection:
    schema_version: str
    canonical_json: str
    digest: str
    content_digest: str
    trigger: str | None
    embedding_model_id: str
    embedding: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    lexical_document: str

    def __post_init__(self) -> None:
        _require_text("search projection schema_version", self.schema_version)
        _require_digest("search projection digest", self.digest)
        _require_digest("search projection content_digest", self.content_digest)
        _require_text("search projection embedding_model_id", self.embedding_model_id)
        _require_text("search projection lexical_document", self.lexical_document)
        if self.trigger is not None and not isinstance(self.trigger, str):
            raise ValueError("search projection trigger must be text or None")
        if not isinstance(self.embedding, tuple):
            raise ValueError("search projection embedding must be a tuple")
        if not isinstance(self.expanded_terms, tuple) or any(
            not isinstance(term, str) or not term.strip()
            for term in self.expanded_terms
        ):
            raise ValueError(
                "search projection expanded_terms must contain non-empty text"
            )
        for coordinate in self.embedding:
            if not isinstance(coordinate, str) or not coordinate:
                raise ValueError(
                    "search projection embedding must contain decimal strings"
                )
            try:
                numeric = float(coordinate)
            except ValueError as exc:
                raise ValueError(
                    "search projection embedding must contain decimal strings"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError("search projection embedding must be finite")
        payload = _canonical_payload(
            "search projection canonical_json", self.canonical_json
        )
        if sha256(self.canonical_json.encode("utf-8")).hexdigest() != self.digest:
            raise ValueError("search projection digest does not match canonical JSON")
        expected = {
            "content_digest": self.content_digest,
            "embedding": list(self.embedding),
            "embedding_model_id": self.embedding_model_id,
            "expanded_terms": list(self.expanded_terms),
            "lexical_document": self.lexical_document,
            "schema_version": self.schema_version,
            "trigger": self.trigger,
        }
        if payload != expected:
            raise ValueError("search projection fields do not match canonical JSON")


@dataclass(frozen=True)
class PublicationRequest:
    optimizer_kind: OptimizerKind
    job_id: int
    attempt_key: str
    publication_claim: PublicationClaim
    worker_fence: int
    incumbent_user_playbook_id: int
    revised_content: str
    projection: PublicationSearchProjection
    decision_proof: DecisionProofEnvelope
    subject_epochs_json: str
    request_id: str

    def __post_init__(self) -> None:
        _validate_optimizer(self.optimizer_kind)
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("publication job_id must be positive")
        _require_text("publication attempt_key", self.attempt_key)
        if self.publication_claim.job_id != self.job_id:
            raise ValueError("publication claim job_id must match request job_id")
        if type(self.worker_fence) is not int or self.worker_fence <= 0:
            raise ValueError("worker_fence must be positive")
        if (
            type(self.incumbent_user_playbook_id) is not int
            or self.incumbent_user_playbook_id <= 0
        ):
            raise ValueError("incumbent_user_playbook_id must be positive")
        _require_text("revised_content", self.revised_content)
        _require_text("publication request_id", self.request_id)
        if self.decision_proof.optimizer_kind != self.optimizer_kind:
            raise ValueError("decision proof optimizer_kind must match request")
        if sha256(self.revised_content.encode("utf-8")).hexdigest() != (
            self.projection.content_digest
        ):
            raise ValueError("revised content digest must match search projection")
        epochs = _canonical_payload("subject_epochs_json", self.subject_epochs_json)
        if not isinstance(epochs, dict) or not isinstance(epochs.get("subjects"), list):
            raise ValueError("subject_epochs_json must contain a subjects list")
        subject_refs: set[str] = set()
        for item in epochs["subjects"]:
            if not isinstance(item, dict):
                raise ValueError("subject epochs must contain objects")
            subject_ref = item.get("ref", item.get("subject_ref"))
            epoch = item.get("epoch", item.get("erasure_epoch"))
            if (
                not isinstance(subject_ref, str)
                or not subject_ref
                or type(epoch) is not int
                or epoch < 0
            ):
                raise ValueError("subject epochs contain an invalid identity or epoch")
            if subject_ref in subject_refs:
                raise ValueError("subject epochs must contain unique subject refs")
            subject_refs.add(subject_ref)


@dataclass(frozen=True)
class PublicationResult:
    job_id: int
    outcome: PublicationOutcome
    successor_user_playbook_id: int | None

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id <= 0:
            raise ValueError("publication result job_id must be positive")
        if self.outcome not in {"applied", "incumbent_changed"}:
            raise ValueError("publication result outcome is invalid")
        if self.outcome == "applied":
            if (
                type(self.successor_user_playbook_id) is not int
                or self.successor_user_playbook_id <= 0
            ):
                raise ValueError("applied publication requires a successor id")
        elif self.successor_user_playbook_id is not None:
            raise ValueError("incumbent_changed publication cannot have a successor id")


class PublicationDecisionVerifier(Protocol):
    """Optimizer-specific proof verification performed before any storage write."""

    def verify(self, request: PublicationRequest) -> None: ...


class UserPlaybookPublicationStore(Protocol):
    """Backend-neutral durable publication operations."""

    def claim_user_playbook_publication(
        self, *, job_id: int, owner: str, worker_fence: int
    ) -> PublicationClaim: ...

    def stage_user_playbook_publication(self, request: PublicationRequest) -> None: ...

    def commit_user_playbook_publication(
        self, request: PublicationRequest
    ) -> PublicationResult: ...

    def load_user_playbook_publication_result(
        self, job_id: int
    ) -> PublicationResult | None: ...


class _EnvelopeVerifier:
    def verify(self, request: PublicationRequest) -> None:
        # Dataclass construction already validates canonical bytes and field binding.
        request.__post_init__()
        request.decision_proof.__post_init__()
        request.projection.__post_init__()


class UserPlaybookPublicationService:
    """Coordinates proof verification with durable staging and atomic commit."""

    def __init__(
        self,
        storage: UserPlaybookPublicationStore,
        verifier: PublicationDecisionVerifier | None = None,
    ) -> None:
        self._storage = storage
        self._verifier = verifier or _EnvelopeVerifier()

    def claim(self, *, job_id: int, owner: str, worker_fence: int) -> PublicationClaim:
        return self._storage.claim_user_playbook_publication(
            job_id=job_id,
            owner=owner,
            worker_fence=worker_fence,
        )

    def stage(self, request: PublicationRequest) -> None:
        self._verifier.verify(request)
        self._storage.stage_user_playbook_publication(request)

    def publish(self, request: PublicationRequest) -> PublicationResult:
        self._verifier.verify(request)
        self._storage.stage_user_playbook_publication(request)
        return self._storage.commit_user_playbook_publication(request)

    def load_committed(self, job_id: int) -> PublicationResult | None:
        return self._storage.load_user_playbook_publication_result(job_id)


def publish_user_playbook_successor(
    service: UserPlaybookPublicationService,
    request: PublicationRequest,
) -> PublicationResult:
    """Publish one revised user playbook through the shared service."""
    return service.publish(request)


__all__ = [
    "DecisionProofEnvelope",
    "PublicationClaim",
    "PublicationDecisionVerifier",
    "PublicationOutcome",
    "PublicationRequest",
    "PublicationResult",
    "PublicationSearchProjection",
    "UserPlaybookPublicationService",
    "UserPlaybookPublicationStore",
    "publish_user_playbook_successor",
]
