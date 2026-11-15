"""Immutable candidate search projection shared by replay and publication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol, Self, overload

from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.playbook.publication import (
    FinalVersionPublicationSearchProjection,
    PublicationSearchProjection,
)

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


@dataclass(frozen=True)
class FinalVersionSearchProjectionRequest:
    """Exact final pair for the v2 projector; no preservation alias."""

    incumbent_user_playbook_id: int
    incumbent_content_digest: str
    candidate_content: str
    candidate_trigger: str | None

    def __post_init__(self) -> None:
        if (
            type(self.incumbent_user_playbook_id) is not int
            or self.incumbent_user_playbook_id <= 0
        ):
            raise ValueError("incumbent_user_playbook_id must be positive")
        _require_sha256(
            self.incumbent_content_digest, field_name="incumbent_content_digest"
        )
        if not _is_nonempty_string(self.candidate_content):
            raise ValueError("candidate_content must be non-empty")
        if self.candidate_trigger is not None and not isinstance(
            self.candidate_trigger, str
        ):
            raise ValueError("candidate_trigger must be text or null")

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
        _validate_projection_values(
            self,
            schema_version=self.SCHEMA_VERSION,
            trigger_key="preserved_trigger",
            trigger=self.preserved_trigger,
        )
        object.__setattr__(self, "_artifact_bytes", self._encode())

    def _encode(self) -> bytes:
        return _projection_bytes(
            self, trigger_key="preserved_trigger", trigger=self.preserved_trigger
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
        return _parse_projection_bytes(
            cls, artifact_bytes, trigger_key="preserved_trigger"
        )


@dataclass(frozen=True)
class FinalVersionSearchProjection:
    """Byte-stable v2 artifact for the explicit final Content and Trigger."""

    projector_id: str
    projector_version: str
    projector_code_digest: str
    candidate_content_digest: str
    candidate_trigger: str | None
    embedding_model_id: str
    embedding: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    lexical_document: str
    schema_version: Literal["offline-tuner-candidate-search-projection-v2"] = (
        "offline-tuner-candidate-search-projection-v2"
    )
    _artifact_bytes: bytes = field(init=False, repr=False, compare=False)

    SCHEMA_VERSION: ClassVar[str] = "offline-tuner-candidate-search-projection-v2"

    def __post_init__(self) -> None:
        _validate_projection_values(
            self,
            schema_version=self.SCHEMA_VERSION,
            trigger_key="candidate_trigger",
            trigger=self.candidate_trigger,
        )
        object.__setattr__(self, "_artifact_bytes", self._encode())

    def _encode(self) -> bytes:
        return _projection_bytes(
            self, trigger_key="candidate_trigger", trigger=self.candidate_trigger
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
        return _parse_projection_bytes(
            cls, artifact_bytes, trigger_key="candidate_trigger"
        )


def _validate_projection_values(
    projection: CandidateSearchProjection | FinalVersionSearchProjection,
    *,
    schema_version: str,
    trigger_key: str,
    trigger: str | None,
) -> None:
    if projection.schema_version != schema_version:
        raise ValueError("candidate search projection schema is unsupported")
    _require_sha256(
        projection.projector_code_digest, field_name="projector_code_digest"
    )
    _require_sha256(
        projection.candidate_content_digest, field_name="candidate_content_digest"
    )
    if not _is_nonempty_string(projection.projector_id) or not _is_nonempty_string(
        projection.projector_version
    ):
        raise ValueError("projector identity must be non-empty")
    if trigger_key == "preserved_trigger":
        if trigger is not None and not _is_nonempty_string(trigger):
            raise ValueError("preserved_trigger must be null or a non-empty string")
    elif trigger is not None and not isinstance(trigger, str):
        raise ValueError("candidate_trigger must be text or null")
    if not _is_nonempty_string(projection.embedding_model_id):
        raise ValueError("embedding_model_id must be non-empty")
    if (
        not isinstance(projection.embedding, tuple)
        or not projection.embedding
        or any(not _is_canonical_decimal(value) for value in projection.embedding)
    ):
        raise ValueError(
            "embedding must be a non-empty vector of canonical finite decimals"
        )
    if not isinstance(projection.expanded_terms, tuple) or any(
        not _is_nonempty_string(term) for term in projection.expanded_terms
    ):
        raise ValueError("expanded_terms must contain non-empty strings")
    if not _is_nonempty_string(projection.lexical_document):
        raise ValueError("lexical_document must be a non-empty string")


def _projection_bytes(
    projection: CandidateSearchProjection | FinalVersionSearchProjection,
    *,
    trigger_key: str,
    trigger: str | None,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": projection.schema_version,
            "projector_id": projection.projector_id,
            "projector_version": projection.projector_version,
            "projector_code_digest": projection.projector_code_digest,
            "candidate_content_digest": projection.candidate_content_digest,
            trigger_key: trigger,
            "embedding_model_id": projection.embedding_model_id,
            "embedding": list(projection.embedding),
            "expanded_terms": list(projection.expanded_terms),
            "lexical_document": projection.lexical_document,
        }
    )


class _CanonicalProjection(Protocol):
    def canonical_bytes(self) -> bytes: ...


def _parse_projection_bytes[Projection: _CanonicalProjection](
    factory: Callable[..., Projection],
    artifact_bytes: bytes,
    *,
    trigger_key: str,
) -> Projection:
    try:
        if not isinstance(artifact_bytes, bytes):
            raise TypeError("candidate search projection artifact must be bytes")
        payload = json.loads(artifact_bytes)
        projection = factory(
            schema_version=payload["schema_version"],
            projector_id=payload["projector_id"],
            projector_version=payload["projector_version"],
            projector_code_digest=payload["projector_code_digest"],
            candidate_content_digest=payload["candidate_content_digest"],
            embedding_model_id=payload["embedding_model_id"],
            embedding=tuple(payload["embedding"]),
            expanded_terms=tuple(payload["expanded_terms"]),
            lexical_document=payload["lexical_document"],
            **{trigger_key: payload[trigger_key]},
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid canonical candidate search projection") from exc
    if projection.canonical_bytes() != artifact_bytes:
        raise ValueError("candidate search projection bytes are not canonical")
    return projection


class FinalVersionSearchProjector(Protocol):
    @property
    def key_identity(self) -> SearchProjectorKeyIdentity: ...

    @property
    def final_version_key_identity(self) -> SearchProjectorKeyIdentity: ...

    def build(
        self,
        request: CandidateSearchProjectionRequest | FinalVersionSearchProjectionRequest,
    ) -> CandidateSearchProjection | FinalVersionSearchProjection: ...


@overload
def publication_search_projection(
    projection: CandidateSearchProjection,
) -> PublicationSearchProjection: ...


@overload
def publication_search_projection(
    projection: FinalVersionSearchProjection,
) -> FinalVersionPublicationSearchProjection: ...


def publication_search_projection(
    projection: CandidateSearchProjection | FinalVersionSearchProjection,
) -> PublicationSearchProjection | FinalVersionPublicationSearchProjection:
    """Expose the exact frozen projection through the shared publication contract."""
    if type(projection) is FinalVersionSearchProjection:
        return FinalVersionPublicationSearchProjection(
            schema_version=projection.schema_version,
            canonical_json=projection.canonical_bytes().decode("utf-8"),
            digest=projection.digest,
            projector_id=projection.projector_id,
            projector_version=projection.projector_version,
            projector_code_digest=projection.projector_code_digest,
            candidate_content_digest=projection.candidate_content_digest,
            candidate_trigger=projection.candidate_trigger,
            embedding_model_id=projection.embedding_model_id,
            embedding=projection.embedding,
            expanded_terms=projection.expanded_terms,
            lexical_document=projection.lexical_document,
        )
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
    "FinalVersionSearchProjectionRequest",
    "FinalVersionSearchProjection",
    "FinalVersionSearchProjector",
    "publication_search_projection",
]
