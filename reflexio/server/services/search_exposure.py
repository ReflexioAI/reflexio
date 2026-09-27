"""Optional synchronous recording boundary for served user playbooks."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sized
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from secrets import token_hex
from typing import Protocol

from reflexio.models.api_schema.domain import UserPlaybook
from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.extensions import ServiceKey, get_service
from reflexio.server.services.playbook.publication import (
    canonical_json_bytes,
    incumbent_user_playbook_semantic_digest,
)

logger = logging.getLogger(__name__)

MAX_EXPOSURE_EVENTS_PER_BATCH = 100

# An uncorrelated exposure is per-search, so at fleet scale it is a firehose:
# the deployment that motivated this signal stored roughly 95k of them. The
# actionable fact is "this deployment is not correlating its exposures at all",
# not each individual serve, so they are counted and reported on a throttle
# that carries the count since the last report. The FIRST one always reports,
# because the whole point is to find out on day one rather than day thirteen.
_UNCORRELATED_ANOMALY_THROTTLE_SECONDS = 3600.0


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
    RECORDED_UNCORRELATED = "recorded_uncorrelated"


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


_uncorrelated_lock = threading.Lock()
_uncorrelated_since_report = 0
_uncorrelated_last_report: float | None = None


def _report_uncorrelated_batch(batch: SearchExposureBatch) -> None:
    """Make an uncorrelated exposure visible, at a bounded rate.

    Called only once a registered recorder has ACCEPTED the batch. The caller
    enforces both halves of that -- it returns ``NO_RECORDER`` before reaching
    here and calls ``recorder.record`` first -- so this function performs no
    recorder lookup of its own; one invariant, one place.

    Both halves matter. A deployment with no recorder is a supported
    configuration (shared ``create_app()`` installs none, so local and no-auth
    OSS persist nothing by design), and telling it that its exposures are
    uncorrelated would be reporting a non-problem on every search. And
    reporting before the write would count a row that a raising recorder never
    stored, while burning the throttle window that the next hour of genuine
    misses needed.

    This is deliberately NOT raised, and deliberately not returned as a new
    outcome. The routes ignore the outcome on purpose -- they cannot distinguish
    a supported recorder-less deployment from an enterprise misconfiguration,
    and asserting there would turn a supported deployment's search into a 500.
    That reasoning is still right; the gap it left was that nobody was told
    anything at all. So the signal goes here, where the distinction IS
    available, and stays out of the response path entirely.
    """
    global _uncorrelated_since_report, _uncorrelated_last_report
    now = time.monotonic()
    with _uncorrelated_lock:
        _uncorrelated_since_report += 1
        due = (
            _uncorrelated_last_report is None
            or now - _uncorrelated_last_report >= _UNCORRELATED_ANOMALY_THROTTLE_SECONDS
        )
        if not due:
            return
        uncorrelated = _uncorrelated_since_report
        first = _uncorrelated_last_report is None
        _uncorrelated_since_report = 0
        _uncorrelated_last_report = now
    # Emitted outside the lock: a reporter is caller code and may be slow.
    logger.warning(
        "event=search_exposure_uncorrelated org_id=%s uncorrelated=%d first=%s"
        " -- the caller sent neither request_id nor session_id, so the exposure"
        " is recorded but flagged integrity_state=incomplete"
        " (missing_correlation) and can never be reconstructed to a session",
        batch.org_id,
        uncorrelated,
        first,
    )
    capture_anomaly(
        "search_exposure.uncorrelated",
        org_id=batch.org_id,
        uncorrelated_since_last_report=uncorrelated,
        first_report=first,
    )


def reset_uncorrelated_reporting_state() -> None:
    """Clear the throttle. For tests, which must not inherit each other's state."""
    global _uncorrelated_since_report, _uncorrelated_last_report
    with _uncorrelated_lock:
        _uncorrelated_since_report = 0
        _uncorrelated_last_report = None


def record_search_exposures(batch: SearchExposureBatch) -> SearchExposureOutcome:
    """Synchronously invoke the optional enterprise exposure recorder.

    A batch with neither ``request_id`` nor ``session_id`` is RECORDED and
    flagged, not refused. It is reported as an anomaly once the recorder has
    accepted it (bounded -- see ``_report_uncorrelated_batch``) and the stored
    row labels itself: the schema's ``integrity_state`` generated column reads
    ``incomplete`` with ``missing_correlation`` in ``integrity_reasons``.

    This reverses the refusal that shipped in #488, and the reason is a
    measurement rather than an argument. #488 reasoned that such a row is
    worthless to a reader and lowers the org's reconstructability ratio without
    ever raising it, so refusing it protects the ratio. That is true for an org
    that correlates. Measured against production on 2026-09-27, no org does:
    across all 93 tenant schemas on the shared data plane, rows carrying a
    ``request_id`` numbered **zero**, and every one of the 95,481 stored rows
    had BOTH correlation columns NULL. So #488's predicate matched 100% of live
    traffic, the refusal protected no ratio that existed, and exposure intake
    went to zero fleet-wide two days after it deployed -- three unrelated orgs
    stopped on the same day and nothing was written for the following 16 days.

    The trade being made, stated plainly because it is real: an org that DOES
    correlate and occasionally omits an id will take a proportional dent in its
    ratio that it can never undo, exposure being append-only
    (``reject_exposure_event_mutation`` blocks DELETE). That is strictly better
    than the alternative this replaces, which was losing the entire audit trail
    -- which playbooks were served, under which fingerprints -- for every
    uncorrelated search. A proportional dent beats total loss.

    What must NOT be done in response is filter these rows out of the coverage
    denominator. ``coverage_for_reasons`` keeps every entry on purpose, "so an
    evidence pipeline that has stopped joining sessions reads as poor coverage
    instead of disappearing". Excluding them would let one correlated row beside
    a thousand uncorrelated ones compute as 100% coverage and pass a 75%
    publication gate.

    An absent recorder is likewise a supported configuration -- shared
    ``create_app()`` installs no default, so local/no-auth OSS deployments
    legitimately persist nothing. It is therefore reported, not raised. Callers
    that *depend* on the batch reaching the ledger (corpus reconstruction,
    replay tooling) must inspect the outcome; exposure is append-only, so a
    batch dropped here can never be backfilled.

    A registered recorder that raises still propagates unchanged: enterprise
    search routes fail closed on recorder failure.

    The anomaly report stays exactly as it was, and is now the only mechanism
    telling an operator that a caller is not correlating -- the condition is no
    longer visible as an absence of rows, because the rows are there. The signal
    is what made this diagnosable at all: intake went to zero on 2026-09-11 and
    stayed there for 16 days, and the bounded report is what eventually named
    it. Do not remove it on the grounds that the rows now land.

    Args:
        batch (SearchExposureBatch): The final served user-playbook set.

    Returns:
        SearchExposureOutcome: ``RECORDED`` when a registered recorder accepted
        a correlated batch, ``RECORDED_UNCORRELATED`` when it accepted one
        carrying no correlation (persisted, and flagged ``incomplete`` by the
        schema), and ``NO_RECORDER`` when none was registered and nothing was
        persisted.
    """
    uncorrelated = batch_is_uncorrelated(batch)
    recorder = get_service(SEARCH_EXPOSURE_RECORDER)
    if recorder is None:
        return SearchExposureOutcome.NO_RECORDER
    recorder.record(batch)
    if uncorrelated:
        # Reported only now, because the report asserts the row LANDED. A
        # recorder that raises propagates (fail-closed) having stored nothing,
        # and counting that serve would both overstate the ledger and burn the
        # hour-long throttle window the next genuine miss needs.
        _report_uncorrelated_batch(batch)
        return SearchExposureOutcome.RECORDED_UNCORRELATED
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
    "reset_uncorrelated_reporting_state",
    "build_user_playbook_exposure_event",
    "record_search_exposures",
    "user_playbook_full_version_fingerprint",
    "validate_exposure_batch_size",
]
