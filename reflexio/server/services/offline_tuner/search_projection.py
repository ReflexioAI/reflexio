"""Immutable candidate search projection shared by replay and publication."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, Self

from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.playbook.publication import PublicationSearchProjection

_CANONICAL_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")


def _require_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex")


@dataclass(frozen=True)
class SearchProjectorKeyIdentity:
    """Identity of the projector that produced a candidate search projection.

    Field names, validation and error strings are fixed, because the payload
    is digested and those digests are persisted.
    """

    projector_id: str
    projector_version: str
    code_digest: str

    def __post_init__(self) -> None:
        for name in ("projector_id", "projector_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        _require_sha256(self.code_digest, field_name="search projector code_digest")

    def canonical_payload(self) -> dict[str, str]:
        return {
            "projector_id": self.projector_id,
            "projector_version": self.projector_version,
            "code_digest": self.code_digest,
        }


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_canonical_decimal(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != "-0"
        and _CANONICAL_DECIMAL.fullmatch(value) is not None
    )


@dataclass(frozen=True)
class CandidateSearchProjectionRequest:
    """Replay-neutral inputs to the pinned candidate search projector."""

    incumbent_user_playbook_id: int
    incumbent_content_digest: str
    candidate_content: str
    preserved_trigger: str | None

    def __post_init__(self) -> None:
        if (
            type(self.incumbent_user_playbook_id) is not int
            or self.incumbent_user_playbook_id <= 0
        ):
            raise ValueError("incumbent_user_playbook_id must be positive")
        _require_sha256(
            self.incumbent_content_digest,
            field_name="incumbent_content_digest",
        )
        if not _is_nonempty_string(self.candidate_content):
            raise ValueError("candidate_content must be non-empty")
        if self.preserved_trigger is not None and not _is_nonempty_string(
            self.preserved_trigger
        ):
            raise ValueError("preserved_trigger must be null or non-empty")

    @property
    def candidate_content_digest(self) -> str:
        return hashlib.sha256(self.candidate_content.encode("utf-8")).hexdigest()


class CandidateSearchProjector(Protocol):
    @property
    def key_identity(self) -> SearchProjectorKeyIdentity: ...

    def build(
        self, request: CandidateSearchProjectionRequest
    ) -> CandidateSearchProjection: ...


@dataclass(frozen=True)
class CandidateSearchProjection:
    """One byte-stable search artifact built once for a candidate."""

    projector_id: str
    projector_version: str
    projector_code_digest: str
    candidate_content_digest: str
    preserved_trigger: str | None
    embedding_model_id: str
    embedding: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    lexical_document: str
    schema_version: Literal["offline-tuner-candidate-search-projection-v1"] = (
        "offline-tuner-candidate-search-projection-v1"
    )
    _artifact_bytes: bytes = field(init=False, repr=False, compare=False)

    SCHEMA_VERSION: ClassVar[str] = "offline-tuner-candidate-search-projection-v1"

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("candidate search projection schema is unsupported")
        _require_sha256(self.projector_code_digest, field_name="projector_code_digest")
        _require_sha256(
            self.candidate_content_digest, field_name="candidate_content_digest"
        )
        if not _is_nonempty_string(self.projector_id) or not _is_nonempty_string(
            self.projector_version
        ):
            raise ValueError("projector identity must be non-empty")
        if self.preserved_trigger is not None and not _is_nonempty_string(
            self.preserved_trigger
        ):
            raise ValueError("preserved_trigger must be null or a non-empty string")
        if not _is_nonempty_string(self.embedding_model_id):
            raise ValueError("embedding_model_id must be non-empty")
        if (
            not isinstance(self.embedding, tuple)
            or not self.embedding
            or any(not _is_canonical_decimal(value) for value in self.embedding)
        ):
            raise ValueError(
                "embedding must be a non-empty vector of canonical finite decimals"
            )
        if not isinstance(self.expanded_terms, tuple) or any(
            not _is_nonempty_string(term) for term in self.expanded_terms
        ):
            raise ValueError("expanded_terms must contain non-empty strings")
        if not _is_nonempty_string(self.lexical_document):
            raise ValueError("lexical_document must be a non-empty string")
        object.__setattr__(self, "_artifact_bytes", self._encode())

    def _encode(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": self.schema_version,
                "projector_id": self.projector_id,
                "projector_version": self.projector_version,
                "projector_code_digest": self.projector_code_digest,
                "candidate_content_digest": self.candidate_content_digest,
                "preserved_trigger": self.preserved_trigger,
                "embedding_model_id": self.embedding_model_id,
                "embedding": list(self.embedding),
                "expanded_terms": list(self.expanded_terms),
                "lexical_document": self.lexical_document,
            }
        )

    def canonical_bytes(self) -> bytes:
        """Return the exact artifact bytes used by replay and publication."""
        return self._artifact_bytes

    @property
    def digest(self) -> str:
        return self.digest_bytes(self._artifact_bytes)

    @property
    def key_identity(self) -> SearchProjectorKeyIdentity:
        return SearchProjectorKeyIdentity(
            projector_id=self.projector_id,
            projector_version=self.projector_version,
            code_digest=self.projector_code_digest,
        )

    @staticmethod
    def digest_bytes(artifact_bytes: bytes) -> str:
        return hashlib.sha256(artifact_bytes).hexdigest()

    @classmethod
    def from_canonical_bytes(cls, artifact_bytes: bytes) -> Self:
        try:
            if not isinstance(artifact_bytes, bytes):
                raise TypeError("candidate search projection artifact must be bytes")
            payload = json.loads(artifact_bytes)
            projection = cls(
                schema_version=payload["schema_version"],
                projector_id=payload["projector_id"],
                projector_version=payload["projector_version"],
                projector_code_digest=payload["projector_code_digest"],
                candidate_content_digest=payload["candidate_content_digest"],
                preserved_trigger=payload["preserved_trigger"],
                embedding_model_id=payload["embedding_model_id"],
                embedding=tuple(payload["embedding"]),
                expanded_terms=tuple(payload["expanded_terms"]),
                lexical_document=payload["lexical_document"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid canonical candidate search projection") from exc
        if projection.canonical_bytes() != artifact_bytes:
            raise ValueError("candidate search projection bytes are not canonical")
        return projection


def publication_search_projection(
    projection: CandidateSearchProjection,
) -> PublicationSearchProjection:
    """Expose the exact frozen projection through the shared publication contract."""
    if type(projection) is not CandidateSearchProjection:
        raise TypeError("projection must be a CandidateSearchProjection")
    return PublicationSearchProjection(
        schema_version=projection.schema_version,
        canonical_json=projection.canonical_bytes().decode("utf-8"),
        digest=projection.digest,
        projector_id=projection.projector_id,
        projector_version=projection.projector_version,
        projector_code_digest=projection.projector_code_digest,
        candidate_content_digest=projection.candidate_content_digest,
        preserved_trigger=projection.preserved_trigger,
        embedding_model_id=projection.embedding_model_id,
        embedding=projection.embedding,
        expanded_terms=projection.expanded_terms,
        lexical_document=projection.lexical_document,
    )


__all__ = [
    "CandidateSearchProjection",
    "CandidateSearchProjectionRequest",
    "CandidateSearchProjector",
    "publication_search_projection",
]
