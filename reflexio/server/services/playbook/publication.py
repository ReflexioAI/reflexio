"""Strict shared contracts for atomic user-playbook publication."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol

from reflexio.models.api_schema.domain.entities import OptimizerKind

PublicationOutcome = Literal["applied", "incumbent_changed"]
PublishableOptimizerKind = Literal["gepa", "offline_tuner_replay"]

_PUBLISHABLE_OPTIMIZERS = frozenset({"gepa", "offline_tuner_replay"})
_PROJECTION_SCHEMA_VERSION = "offline-tuner-candidate-search-projection-v1"
_CANONICAL_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY = "publication_subject_epochs"
PUBLICATION_PROOF_JSON_METADATA_KEY = "publication_proof_json"
PUBLICATION_PROJECTION_JSON_METADATA_KEY = "publication_projection_json"
PUBLICATION_INCUMBENT_CONTENT_DIGEST_METADATA_KEY = (
    "publication_incumbent_content_digest"
)
PUBLICATION_INCUMBENT_TRIGGER_METADATA_KEY = "publication_incumbent_trigger"
PUBLICATION_INCUMBENT_SEMANTIC_DIGEST_METADATA_KEY = (
    "publication_incumbent_semantic_digest"
)


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


def canonical_json_bytes(payload: object) -> bytes:
    """Encode integer/string-only JSON values using RFC 8785 ordering."""
    return _canonical_json(payload).encode("utf-8")


def _canonical_json(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        _reject_surrogates(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        if not -(2**53) < value < 2**53:
            raise ValueError(
                "RFC 8785 integers must be exactly representable by IEEE 754"
            )
        return str(value)
    if isinstance(value, tuple | list):
        return "[" + ",".join(_canonical_json(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("RFC 8785 object keys must be strings")
        for key in value:
            _reject_surrogates(key)
        keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        return (
            "{"
            + ",".join(
                f"{_canonical_json(key)}:{_canonical_json(value[key])}" for key in keys
            )
            + "}"
        )
    raise TypeError(f"Unsupported RFC 8785 value: {type(value).__name__}")


def _reject_surrogates(value: str) -> None:
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("RFC 8785 strings cannot contain surrogate code points")


def _canonical_payload(name: str, value: str) -> object:
    try:
        payload = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
        canonical = canonical_json_bytes(payload).decode("utf-8")
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    if canonical != value:
        raise ValueError(f"{name} must use canonical JSON bytes")
    return payload


def _validate_optimizer(value: object) -> None:
    if value not in _PUBLISHABLE_OPTIMIZERS:
        raise ValueError("optimizer_kind is not publishable")


def incumbent_user_playbook_semantic_digest(
    *, content_digest: str, trigger: str | None
) -> str:
    """Bind the behaviorally mutable incumbent fields used by publication."""
    _require_digest("incumbent content digest", content_digest)
    if trigger is not None and not isinstance(trigger, str):
        raise TypeError("incumbent trigger must be a string or null")
    payload = {
        "content_digest": content_digest,
        "schema_version": "user-playbook-incumbent-semantic-v1",
        "trigger": trigger,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


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
    projector_id: str
    projector_version: str
    projector_code_digest: str
    candidate_content_digest: str
    preserved_trigger: str | None
    embedding_model_id: str
    embedding: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    lexical_document: str

    def __post_init__(self) -> None:
        if self.schema_version != _PROJECTION_SCHEMA_VERSION:
            raise ValueError("search projection schema is unsupported")
        _require_digest("search projection digest", self.digest)
        _require_text("search projection projector_id", self.projector_id)
        _require_text("search projection projector_version", self.projector_version)
        _require_digest(
            "search projection projector_code_digest", self.projector_code_digest
        )
        _require_digest(
            "search projection candidate_content_digest",
            self.candidate_content_digest,
        )
        _require_text("search projection embedding_model_id", self.embedding_model_id)
        _require_text("search projection lexical_document", self.lexical_document)
        if self.preserved_trigger is not None:
            _require_text("search projection preserved_trigger", self.preserved_trigger)
        if not isinstance(self.embedding, tuple) or not self.embedding:
            raise ValueError("search projection embedding must be a non-empty tuple")
        if not isinstance(self.expanded_terms, tuple) or any(
            not isinstance(term, str) or not term.strip()
            for term in self.expanded_terms
        ):
            raise ValueError(
                "search projection expanded_terms must contain non-empty text"
            )
        for coordinate in self.embedding:
            if (
                not isinstance(coordinate, str)
                or coordinate == "-0"
                or _CANONICAL_DECIMAL.fullmatch(coordinate) is None
            ):
                raise ValueError(
                    "search projection embedding must contain canonical decimals"
                )
        payload = _canonical_payload(
            "search projection canonical_json", self.canonical_json
        )
        if sha256(self.canonical_json.encode("utf-8")).hexdigest() != self.digest:
            raise ValueError("search projection digest does not match canonical JSON")
        expected = {
            "candidate_content_digest": self.candidate_content_digest,
            "embedding": list(self.embedding),
            "embedding_model_id": self.embedding_model_id,
            "expanded_terms": list(self.expanded_terms),
            "lexical_document": self.lexical_document,
            "preserved_trigger": self.preserved_trigger,
            "projector_code_digest": self.projector_code_digest,
            "projector_id": self.projector_id,
            "projector_version": self.projector_version,
            "schema_version": self.schema_version,
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
    incumbent_content_digest: str
    incumbent_trigger: str | None
    incumbent_semantic_digest: str
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
        _require_digest("incumbent content digest", self.incumbent_content_digest)
        _require_digest("incumbent semantic digest", self.incumbent_semantic_digest)
        expected_semantic_digest = incumbent_user_playbook_semantic_digest(
            content_digest=self.incumbent_content_digest,
            trigger=self.incumbent_trigger,
        )
        if self.incumbent_semantic_digest != expected_semantic_digest:
            raise ValueError("incumbent semantic digest does not match frozen fields")
        if self.projection.preserved_trigger != self.incumbent_trigger:
            raise ValueError("search projection must preserve incumbent trigger")
        _require_text("revised_content", self.revised_content)
        _require_text("publication request_id", self.request_id)
        if self.decision_proof.optimizer_kind != self.optimizer_kind:
            raise ValueError("decision proof optimizer_kind must match request")
        if sha256(self.revised_content.encode("utf-8")).hexdigest() != (
            self.projection.candidate_content_digest
        ):
            raise ValueError("revised content digest must match search projection")
        epochs = _canonical_payload("subject_epochs_json", self.subject_epochs_json)
        if (
            not isinstance(epochs, dict)
            or set(epochs) != {"subjects"}
            or not isinstance(epochs.get("subjects"), list)
            or not epochs["subjects"]
        ):
            raise ValueError("subject epochs must contain a non-empty subjects list")
        subject_refs: set[str] = set()
        for item in epochs["subjects"]:
            if not isinstance(item, dict):
                raise ValueError("subject epochs must contain objects")
            if set(item) != {"ref", "epoch"}:
                raise ValueError("subject epochs must use ref and epoch fields")
            subject_ref = item["ref"]
            epoch = item["epoch"]
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


class UserPlaybookPublicationService:
    """Coordinates proof verification with durable staging and atomic commit."""

    def __init__(
        self,
        storage: UserPlaybookPublicationStore,
        verifier: PublicationDecisionVerifier,
    ) -> None:
        if not callable(getattr(verifier, "verify", None)):
            raise TypeError("verifier must implement PublicationDecisionVerifier")
        self._storage = storage
        self._verifier = verifier

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
    "PUBLICATION_INCUMBENT_CONTENT_DIGEST_METADATA_KEY",
    "PUBLICATION_INCUMBENT_SEMANTIC_DIGEST_METADATA_KEY",
    "PUBLICATION_INCUMBENT_TRIGGER_METADATA_KEY",
    "PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY",
    "UserPlaybookPublicationService",
    "UserPlaybookPublicationStore",
    "canonical_json_bytes",
    "incumbent_user_playbook_semantic_digest",
    "publish_user_playbook_successor",
]
