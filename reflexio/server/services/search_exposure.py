"""Optional synchronous recording boundary for served user playbooks."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from secrets import token_hex
from typing import Protocol

from reflexio.models.api_schema.domain import UserPlaybook
from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.playbook.publication import (
    canonical_json_bytes,
    incumbent_user_playbook_semantic_digest,
)

MAX_EXPOSURE_EVENTS_PER_BATCH = 100


@dataclass(frozen=True)
class SearchExposureBatch:
    """The final user-playbook set returned by an authenticated search."""

    org_id: str
    request_id: str | None
    session_id: str | None
    interaction_id: int | None
    user_id: str | None
    user_playbooks: tuple[UserPlaybook, ...]
    invocation_id: str = field(default_factory=lambda: token_hex(16))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _normalize_correlation_id(self.request_id)
        )
        object.__setattr__(
            self, "session_id", _normalize_correlation_id(self.session_id)
        )
        object.__setattr__(self, "user_id", _normalize_correlation_id(self.user_id))
        if self.interaction_id is not None and self.interaction_id <= 0:
            object.__setattr__(self, "interaction_id", None)


@dataclass(frozen=True)
class UserPlaybookExposureEvent:
    """Immutable durable envelope for one served user playbook."""

    exposure_event_id: str
    request_id: str | None
    session_id: str | None
    user_id: str | None
    playbook_owner_user_id: str | None
    user_playbook_id: int | None
    served_semantic_digest: str | None
    served_full_version_fingerprint: str | None
    exposed_at: int | None
    ingested_at: int
    governance_subject_ref: str | None
    playbook_owner_governance_subject_ref: str | None


@dataclass(frozen=True)
class ExposureEventWriteResult:
    """Durable write result returned by the enterprise ledger store."""

    recorded: bool
    integrity_state: str
    integrity_reasons: tuple[str, ...]


class SearchExposureRecorder(Protocol):
    """Durably record a final search result set before response release."""

    def record(self, batch: SearchExposureBatch) -> None: ...


SEARCH_EXPOSURE_RECORDER = ServiceKey[SearchExposureRecorder](
    "search_exposure_recorder"
)


class SearchExposureOutcome(StrEnum):
    """Whether a batch actually reached a durable recorder."""

    RECORDED = "recorded"
    NO_RECORDER = "no_recorder"
    UNCORRELATED = "uncorrelated"


def _normalize_correlation_id(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def batch_is_uncorrelated(batch: SearchExposureBatch) -> bool:
    """Return whether the batch carries no persisted correlation handle.

    The two correlation columns the ledger actually stores are ``request_id``
    and ``session_id``; ``interaction_id`` is *not* a column -- it only seasons
    the deterministic event id -- so a batch carrying nothing but an
    ``interaction_id`` still lands on disk with both correlation columns NULL.
    This predicate is therefore deliberately the schema's own definition of
    ``missing_correlation`` (see the ``integrity_reasons`` generated column on
    ``tenant.user_playbook_exposure_events``), not a second, subtly different
    notion of "correlated" invented here.
    """
    return batch.request_id is None and batch.session_id is None


def record_search_exposures(batch: SearchExposureBatch) -> SearchExposureOutcome:
    """Synchronously invoke the optional enterprise exposure recorder.

    A batch with neither ``request_id`` nor ``session_id`` is refused before any
    recorder is consulted. Such a row can never bind a served playbook to a
    session -- ``request_id`` is the only key the evidence loader can resolve a
    session and user from, and there is no reverse lookup from a session -- so
    no reader can ever draw value from it. It is not inert, though: every
    reconstructability gate counts it in the denominator and never in the
    numerator, so a single uncorrelated retrieval permanently lowers the org's
    ratio, and exposure is append-only (``reject_exposure_event_mutation``
    blocks DELETE), so it can never be taken back.

    An absent recorder is likewise a supported configuration -- shared
    ``create_app()`` installs no default, so local/no-auth OSS deployments
    legitimately persist nothing. It is therefore reported, not raised. Callers
    that *depend* on the batch reaching the ledger (corpus reconstruction,
    replay tooling) must inspect the outcome; exposure is append-only, so a
    batch dropped here can never be backfilled.

    A registered recorder that raises still propagates unchanged: enterprise
    search routes fail closed on recorder failure.

    Args:
        batch (SearchExposureBatch): The final served user-playbook set.

    Returns:
        SearchExposureOutcome: ``RECORDED`` when a registered recorder accepted
        the batch, ``UNCORRELATED`` when the batch carried no correlation and
        was refused, and ``NO_RECORDER`` when none was registered and nothing
        was persisted.
    """
    if batch_is_uncorrelated(batch):
        return SearchExposureOutcome.UNCORRELATED
    recorder = get_service(SEARCH_EXPOSURE_RECORDER)
    if recorder is None:
        return SearchExposureOutcome.NO_RECORDER
    recorder.record(batch)
    return SearchExposureOutcome.RECORDED


def validate_exposure_batch_size(events: Sized) -> None:
    """Reject exposure batches that exceed the fixed storage safety bound."""
    if len(events) > MAX_EXPOSURE_EVENTS_PER_BATCH:
        raise ValueError(
            f"exposure batch must contain at most {MAX_EXPOSURE_EVENTS_PER_BATCH} events"
        )


def user_playbook_full_version_fingerprint(playbook: UserPlaybook) -> str:
    """Bind every persisted playbook field except its derived embedding vector.

    Adding or changing persisted ``UserPlaybook`` fields requires bumping
    ``user-playbook-full-version-v1``; cross-version fingerprint comparisons are
    undefined.
    """
    payload = {
        "schema_version": "user-playbook-full-version-v1",
        "user_playbook": playbook.model_dump(mode="json", exclude={"embedding"})
        | {
            "governance_subject_ref": playbook.governance_subject_ref,
            "retired_at": playbook.retired_at,
        },
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def build_user_playbook_exposure_event(
    batch: SearchExposureBatch,
    playbook: UserPlaybook,
    *,
    exposed_at: int,
    ingested_at: int,
    governance_subject_ref: str | None,
    playbook_owner_governance_subject_ref: str | None,
) -> UserPlaybookExposureEvent:
    """Build one deterministic event identity from retrieval-owned correlation."""
    if batch.user_id is not None and playbook.user_id != batch.user_id:
        raise ValueError(
            "served playbook owner does not match retrieval subject: "
            f"user_playbook_id={playbook.user_playbook_id}"
        )
    identity: dict[str, object] = {
        "schema_version": "user-playbook-exposure-event-v1",
        "org_id": batch.org_id,
        "request_id": batch.request_id,
        "session_id": batch.session_id,
        "interaction_id": batch.interaction_id,
        "user_playbook_id": playbook.user_playbook_id,
    }
    if (
        batch.request_id is None
        and batch.session_id is None
        and batch.interaction_id is None
    ):
        identity["invocation_id"] = batch.invocation_id
    content_digest = sha256(playbook.content.encode("utf-8")).hexdigest()
    return UserPlaybookExposureEvent(
        exposure_event_id=sha256(canonical_json_bytes(identity)).hexdigest(),
        request_id=batch.request_id,
        session_id=batch.session_id,
        user_id=batch.user_id,
        playbook_owner_user_id=playbook.user_id,
        user_playbook_id=playbook.user_playbook_id,
        served_semantic_digest=incumbent_user_playbook_semantic_digest(
            content_digest=content_digest,
            trigger=playbook.trigger,
        ),
        served_full_version_fingerprint=user_playbook_full_version_fingerprint(
            playbook
        ),
        exposed_at=exposed_at,
        ingested_at=ingested_at,
        governance_subject_ref=governance_subject_ref,
        playbook_owner_governance_subject_ref=(playbook_owner_governance_subject_ref),
    )


__all__ = [
    "MAX_EXPOSURE_EVENTS_PER_BATCH",
    "SEARCH_EXPOSURE_RECORDER",
    "ExposureEventWriteResult",
    "SearchExposureBatch",
    "SearchExposureOutcome",
    "SearchExposureRecorder",
    "UserPlaybookExposureEvent",
    "batch_is_uncorrelated",
    "build_user_playbook_exposure_event",
    "record_search_exposures",
    "user_playbook_full_version_fingerprint",
    "validate_exposure_batch_size",
]
