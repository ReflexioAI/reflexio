# ruff: noqa: S101 -- Test-only assertions.
"""Shared contract for freezing open-world evidence, part of the port contract.

:class:`~reflexio.test_support.open_world_port_contract.OpenWorldPortContract`
inherits :class:`OpenWorldEvidenceContract`, so every backend binding of the
port contract runs these cases too. They drive the four freeze-time members
of ``OpenWorldStoragePort`` -- the bundle freeze and load, the evidence
snapshot, the governance epoch vector and the lifecycle read -- over evidence
written through storage's own publish, outcome and playbook methods.

All data is synthetic.

Invariants covered
------------------
I1 -- the freeze is atomic and replay-safe: the job is created or found,
leased and given its bundle together or not at all; a stored bundle is never
rewritten and other bytes for the same job refuse; a replay under another
owner's live lease returns the stored record with ``claim=None``; a terminal
job loads with ``claim=None`` and a new freeze opens a new job; an expired job
of another identity is terminalized as ``abandoned`` and its target retaken,
while a live one refuses.

I2 -- freeze and load refuse with ``OpenWorldEvidenceTargetStaleError`` when the
incumbent's current full-version fingerprint is not the frozen one.

I9 -- the snapshot is one consistent read: a row another connection commits
while it is being read is invisible to it. Every source is read whole (no
joined-row cap), under a byte budget and an elapsed deadline that refuse the
snapshot when exhausted.

I19 -- a target or job naming another tenant is refused.

D4 -- a stored playbook carries its owner's governance subject ref, which the
full-version fingerprint covers (I12 is in the exposure contract).

Backend differences, each asserted per backend through a harness flag:

* ``governance_registry`` -- Postgres keeps a registry of governance subjects
  with erasure epochs and refuses a subject it does not list; SQLite keeps
  none, so every subject is at epoch 0 and only an erasure barrier refuses it.

Not asserted: the deployment lifecycle read is exercised only for a playbook
the tuner never published (``()`` and ``incumbent_published_at`` ``None`` on
both). Only the open-world provisional commit writes a lifecycle, and SQLite
has no such commit yet.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, NamedTuple, Protocol
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    PlaybookOptimizationJob,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
    UserPlaybook,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    PRODUCTION_OPEN_WORLD_READ_LIMITS,
    OpenWorldEvidenceReadLimits,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    OpenWorldEvidenceIntegrityError,
    incumbent_published_at,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldEvidenceGovernanceError,
    OpenWorldEvidenceStorageRecord,
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.playbook.publication import (
    incumbent_user_playbook_semantic_digest,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
    StorageError,
)
from reflexio.test_support.open_world_exposure_contract import (
    ExposureHarness,
    _interaction,
    _owned_playbook,
    _publish,
    _receipt,
    _ref,
    _served,
    _uid,
)

_DAY = 86_400
_LEASE = 600
#: More joined rows than the pipeline-health sampling cap (1,024), which must
#: never bound an evidence source.
_LONG_SOURCE_ROWS = 1_030
_LIMITS = OpenWorldEvidenceReadLimits(
    historical_window_seconds=14 * _DAY,
    maximum_population=100,
    max_trajectory_bytes=PRODUCTION_OPEN_WORLD_READ_LIMITS.max_trajectory_bytes,
)
_SUCCESS_DEFINITION = "e" * 64
_BUDGET_EXCEEDED = "evidence_source_read_budget_exceeded"
_DEADLINE_EXCEEDED = "evidence_source_read_deadline_exceeded"


class JobRow(NamedTuple):
    """One stored ``playbook_optimization_jobs`` row, as the harness reads it."""

    job_id: int
    status: str
    stage: str | None
    terminal_outcome: str | None


class EvidenceHarness(ExposureHarness, Protocol):
    """The freeze-time half of the port harness."""

    @property
    def governance_registry(self) -> bool:
        """True when the store keeps a governance-subject registry with epochs."""
        ...

    def register_subject(self, user_id: str, *, epoch: int = 0) -> str:
        """Make ``user_id``'s subject writable at ``epoch``; return its ref.

        A store without a registry only computes the ref, and holds every
        subject at epoch 0.
        """
        ...

    def optimization_jobs(self, target_id: int) -> list[JobRow]:
        """Every optimizer job on ``target_id``, oldest first."""
        ...

    def fail_artifact_writes(self) -> AbstractContextManager[None]:
        """While open, inserting an optimizer artifact raises inside the store."""
        ...

    def second_storage(self) -> AbstractContextManager[Any]:
        """A second store on the same database, over its own connection."""
        ...

    def during_snapshot(
        self, write: Callable[[], None]
    ) -> AbstractContextManager[None]:
        """Run ``write`` once, mid-way through the next snapshot read.

        After the snapshot's first reads, before it reads outcomes and sources.
        """
        ...

    def source_read_budget(self, limit_bytes: int) -> AbstractContextManager[None]:
        """Bound the snapshot's serialized source-read budget to ``limit_bytes``."""
        ...

    def source_read_clock_jumps(self) -> AbstractContextManager[None]:
        """Make each reading of the snapshot's elapsed clock 1,000 s later."""
        ...


@contextmanager
def call_once_before(
    module: object, attribute: str, write: Callable[[], None]
) -> Iterator[None]:
    """Run ``write`` once, just before the first call of ``module.attribute``.

    A binding's ``during_snapshot``: it names the function its snapshot read
    calls between its first reads and its later ones.
    """
    real = getattr(module, attribute)
    calls: list[None] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not calls:
            calls.append(None)
            write()
        return real(*args, **kwargs)

    with patch.object(module, attribute, wrapped):
        yield
    assert calls, f"{attribute} was never called; the write never ran"


@contextmanager
def jumping_clock(module: object, attribute: str = "monotonic") -> Iterator[None]:
    """Make each reading of ``module.attribute`` 1,000 s after the previous one."""
    readings: list[float] = [0.0]

    def clock() -> float:
        readings.append(readings[-1] + 1_000.0)
        return readings[-1]

    with patch.object(module, attribute, clock):
        yield


@dataclass(frozen=True)
class _World:
    """One published, judged session exposing one owned playbook."""

    owner: str
    viewer: str
    session_id: str
    request_id: str
    playbook: UserPlaybook
    frozen_at: int


def _subject_epochs(*refs: str) -> tuple[tuple[str, int], ...]:
    return tuple((ref, 0) for ref in sorted(set(refs), key=str.encode))


def _record_outcome(storage: Any, session_id: str, *, at: int) -> None:
    result = storage.record_session_outcome(
        SetSessionOutcomeRequest(
            session_id=session_id, outcome=SessionOutcomeKind.FAILURE, occurred_at=at
        ),
        created_at=at,
        expected_context=storage.get_session_outcome_context(session_id),
    )
    assert result.recorded, result


def _published_world(
    harness: EvidenceHarness, *, extra_interactions: int = 0, judged: bool = True
) -> _World:
    """The owner's session citing their own playbook with a valid receipt.

    A user playbook is searched and served to its owner, so the viewer is the
    owner; ``viewer`` names the role.
    """
    owner = _uid("owner")
    viewer = owner
    session_id, request_id = _uid("session"), _uid("request")
    playbook = _owned_playbook(harness, owner)
    served = _served(harness, playbook)
    receipt = _receipt(harness, served, session_id=session_id, user_id=viewer)
    interactions: list[Interaction] = [
        _interaction(viewer, request_id, [_ref(served.user_playbook_id, receipt)])
    ]
    interactions += [
        _interaction(viewer, request_id, []) for _ in range(extra_interactions)
    ]
    _publish(harness, user_id=viewer, session_id=session_id, interactions=interactions)
    (exposure,) = harness.exposure_rows(request_id)
    frozen_at = max(exposure.ingested_at, exposure.exposed_at or 0) + 1
    if judged:
        _record_outcome(harness.storage, session_id, at=frozen_at - 1)
    harness.register_subject(owner)
    harness.register_subject(viewer)
    stored = harness.storage.get_user_playbook_by_id(playbook.user_playbook_id)
    assert stored is not None
    return _World(owner, viewer, session_id, request_id, stored, frozen_at)


def _target(
    harness: EvidenceHarness, world: _World, *, tenant_id: str | None = None
) -> OpenWorldTargetIdentity:
    playbook = world.playbook
    return OpenWorldTargetIdentity(
        tenant_id=tenant_id or harness.storage.org_id,
        user_playbook_id=playbook.user_playbook_id,
        incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
            content_digest=sha256(playbook.content.encode()).hexdigest(),
            trigger=playbook.trigger,
        ),
        incumbent_full_version_fingerprint=user_playbook_full_version_fingerprint(
            playbook
        ),
        success_definition_digest=_SUCCESS_DEFINITION,
        governance_epoch_vector=harness.storage.list_open_world_governance_epochs(),
        frozen_at=world.frozen_at,
    )


def _snapshot(
    harness: EvidenceHarness,
    target: OpenWorldTargetIdentity,
    *,
    limits: OpenWorldEvidenceReadLimits = _LIMITS,
) -> OpenWorldEvidenceSnapshot:
    return harness.storage.load_open_world_evidence_snapshot(
        target=target, frozen_at=target.frozen_at, limits=limits
    )


# ---------------------------------------------------------------------------
# Freeze requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Freeze:
    """One freeze request, as the bundle store would build it."""

    job: PlaybookOptimizationJob
    content_json: str
    content_digest: str
    fingerprint: str
    subject_epochs: tuple[tuple[str, int], ...]


def _freeze_request(
    harness: EvidenceHarness,
    playbook: UserPlaybook,
    subject_epochs: tuple[tuple[str, int], ...],
    *,
    attempt: str = "first",
    bundle: object = None,
    tenant_id: str | None = None,
    now: int,
) -> _Freeze:
    fingerprint = user_playbook_full_version_fingerprint(playbook)
    content = canonical_json_bytes(
        bundle if bundle is not None else {"synthetic_bundle": attempt}
    ).decode()
    digest = sha256(content.encode()).hexdigest()
    metadata = {
        "open_world_evidence": {
            "tenant_id": tenant_id or harness.storage.org_id,
            "user_playbook_id": playbook.user_playbook_id,
            "incumbent_full_version_fingerprint": fingerprint,
            "bundle_digest": digest,
        },
        "publication_subject_epochs": {
            "subjects": [
                {"subject_ref": ref, "erasure_epoch": epoch}
                for ref, epoch in subject_epochs
            ]
        },
    }
    job = PlaybookOptimizationJob(
        optimizer_kind=OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
        target_kind="user_playbook",
        target_id=playbook.user_playbook_id,
        metadata_json=canonical_json_bytes(metadata).decode(),
        discovery_key=sha256(f"discovery:{attempt}".encode()).hexdigest(),
        attempt_key=sha256(f"attempt:{attempt}:{digest}".encode()).hexdigest(),
        stage="evidence_frozen",
        expected_population_manifest_digest=sha256(attempt.encode()).hexdigest(),
        created_at=now,
        updated_at=now,
    )
    return _Freeze(job, content, digest, fingerprint, subject_epochs)


def _freeze(
    harness: EvidenceHarness,
    request: _Freeze,
    *,
    owner: str = "worker-a",
    now: int,
    content_json: str | None = None,
    content_digest: str | None = None,
    fingerprint: str | None = None,
) -> OpenWorldEvidenceStorageRecord:
    return harness.storage.freeze_open_world_evidence_bundle(
        job=request.job,
        artifact_content_json=content_json or request.content_json,
        artifact_content_digest=content_digest or request.content_digest,
        target_full_version_fingerprint=fingerprint or request.fingerprint,
        subject_epochs=request.subject_epochs,
        owner=owner,
        lease_seconds=_LEASE,
        now=now,
    )


def _frozen_target(
    harness: EvidenceHarness, *, attempt: str = "first"
) -> tuple[UserPlaybook, _Freeze, int]:
    """An owned, registered playbook and a freeze request for it."""
    owner = _uid("owner")
    playbook = _owned_playbook(harness, owner)
    stored = harness.storage.get_user_playbook_by_id(playbook.user_playbook_id)
    assert stored is not None
    now = harness.now()
    request = _freeze_request(
        harness,
        stored,
        _subject_epochs(harness.register_subject(owner)),
        attempt=attempt,
        now=now,
    )
    return stored, request, now


def _trajectory_rows(snapshot: OpenWorldEvidenceSnapshot) -> int:
    (event,) = snapshot.events
    assert event.trajectory is not None
    return sum(
        max(1, len(request["interactions"])) for request in event.trajectory["requests"]
    )


class OpenWorldEvidenceContract:
    """Freezing open-world evidence, asserted identically on every backend."""

    # ------------------------------------------------------------------ I1

    def test_a_freeze_creates_leases_and_stores_the_bundle_together(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)

        record = _freeze(port_harness, request, now=now)

        assert record.claim is not None
        assert (record.claim.owner, record.claim.fence) == ("worker-a", 1)
        assert record.claim.expires_at == now + _LEASE
        assert record.job.job_id == record.claim.job_id
        assert (record.job.status, record.job.stage) == ("running", "evidence_frozen")
        assert record.job.lease_owner == "worker-a"
        assert record.artifact.job_id == record.job.job_id
        assert record.artifact.artifact_kind == OPEN_WORLD_EVIDENCE_ARTIFACT_KIND
        assert record.artifact.content_json == request.content_json
        assert record.artifact.content_digest == request.content_digest
        assert json.loads(record.job.metadata_json) == json.loads(
            request.job.metadata_json
        )
        assert port_harness.optimization_jobs(playbook.user_playbook_id) == [
            JobRow(record.job.job_id, "running", "evidence_frozen", None)
        ]

    def test_the_owners_replay_renews_and_never_rewrites(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        first = _freeze(port_harness, request, now=now)

        replay = _freeze(port_harness, request, now=now + 60)

        assert replay.job.job_id == first.job.job_id
        assert replay.artifact == first.artifact, "the stored bundle is final"
        assert replay.claim is not None and first.claim is not None
        assert replay.claim.fence == first.claim.fence, "a renewal keeps the fence"
        assert replay.claim.expires_at == now + 60 + _LEASE
        assert len(port_harness.optimization_jobs(playbook.user_playbook_id)) == 1

    def test_a_replay_under_another_owners_live_lease_reads_without_a_claim(
        self, port_harness: EvidenceHarness
    ) -> None:
        _playbook, request, now = _frozen_target(port_harness)
        first = _freeze(port_harness, request, now=now)

        replay = _freeze(port_harness, request, owner="worker-b", now=now + 60)

        assert replay.claim is None
        assert replay.job.job_id == first.job.job_id
        assert replay.job.lease_owner == "worker-a"
        assert replay.artifact == first.artifact

    def test_an_expired_lease_is_reclaimed_under_a_new_fence(
        self, port_harness: EvidenceHarness
    ) -> None:
        _playbook, request, now = _frozen_target(port_harness)
        first = _freeze(port_harness, request, now=now)

        later = now + _LEASE + 1
        replay = _freeze(port_harness, request, owner="worker-b", now=later)

        assert replay.claim is not None and first.claim is not None
        assert replay.claim.owner == "worker-b"
        assert replay.claim.fence == first.claim.fence + 1
        assert replay.artifact == first.artifact

    def test_other_bytes_for_a_frozen_job_are_refused_and_the_bundle_kept(
        self, port_harness: EvidenceHarness
    ) -> None:
        _playbook, request, now = _frozen_target(port_harness)
        first = _freeze(port_harness, request, now=now)
        other = canonical_json_bytes({"synthetic_bundle": "rewritten"}).decode()

        with pytest.raises(OptimizationArtifactIntegrityError):
            _freeze(
                port_harness,
                request,
                now=now + 60,
                content_json=other,
                content_digest=sha256(other.encode()).hexdigest(),
            )

        assert (
            port_harness.storage.load_open_world_evidence_bundle(
                first.job.job_id
            ).artifact
            == first.artifact
        )

    def test_a_failed_bundle_write_leaves_no_job_behind(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)

        with port_harness.fail_artifact_writes(), pytest.raises(Exception):  # noqa: B017
            _freeze(port_harness, request, now=now)

        assert port_harness.optimization_jobs(playbook.user_playbook_id) == [], (
            "the job and its lease commit with the bundle or not at all"
        )
        record = _freeze(port_harness, request, now=now)
        assert record.claim is not None and record.claim.fence == 1

    def test_a_request_disagreeing_with_its_metadata_writes_nothing(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)

        with pytest.raises(OptimizationArtifactIntegrityError):
            _freeze(port_harness, request, now=now, fingerprint="f" * 64)
        with pytest.raises(OptimizationArtifactIntegrityError):
            _freeze(port_harness, request, now=now, content_digest="0" * 64)

        assert port_harness.optimization_jobs(playbook.user_playbook_id) == []

    def test_a_terminal_job_loads_without_a_claim_and_a_new_freeze_opens_a_job(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        first = _freeze(port_harness, request, now=now)
        assert first.claim is not None
        assert port_harness.storage.advance_playbook_optimization_stage(
            first.job.job_id,
            first.claim.fence,
            "failed",
            terminal_outcome="infrastructure_failure",
            now=now + 1,
        )

        loaded = port_harness.storage.load_open_world_evidence_bundle(first.job.job_id)

        assert loaded.claim is None
        assert (loaded.job.status, loaded.job.stage) == ("failed", "failed")
        assert loaded.artifact == first.artifact
        again = _freeze(port_harness, request, now=now + 2)
        assert again.job.job_id != first.job.job_id
        assert again.claim is not None and again.claim.fence == 1
        assert [
            row.status
            for row in port_harness.optimization_jobs(playbook.user_playbook_id)
        ] == ["failed", "running"]

    def test_a_second_identity_on_a_live_target_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        _freeze(port_harness, request, now=now)
        second = _freeze_request(
            port_harness,
            playbook,
            request.subject_epochs,
            attempt="second",
            now=now,
        )

        with pytest.raises(OptimizationJobIdentityConflictError):
            _freeze(port_harness, second, owner="worker-b", now=now + 60)

        assert [
            row.status
            for row in port_harness.optimization_jobs(playbook.user_playbook_id)
        ] == ["running"]

    def test_an_expired_job_of_another_identity_is_abandoned_and_retaken(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        stranded = _freeze(port_harness, request, now=now)
        second = _freeze_request(
            port_harness,
            playbook,
            request.subject_epochs,
            attempt="second",
            now=now,
        )
        later = now + _LEASE + 1

        record = _freeze(port_harness, second, owner="worker-b", now=later)

        assert record.job.job_id != stranded.job.job_id
        assert record.claim is not None and record.claim.fence == 1
        assert port_harness.optimization_jobs(playbook.user_playbook_id) == [
            JobRow(stranded.job.job_id, "failed", "failed", "abandoned"),
            JobRow(record.job.job_id, "running", "evidence_frozen", None),
        ]

    # ------------------------------------------------------------------ I2

    def test_a_freeze_of_a_stale_target_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        port_harness.storage.update_user_playbook(
            playbook.user_playbook_id, tags=["synthetic-edit"]
        )

        with pytest.raises(OpenWorldEvidenceTargetStaleError):
            _freeze(port_harness, request, now=now)

        assert port_harness.optimization_jobs(playbook.user_playbook_id) == []

    @pytest.mark.parametrize("change", ["edited", "archived"])
    def test_a_load_after_the_target_moved_is_refused(
        self, port_harness: EvidenceHarness, change: str
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        record = _freeze(port_harness, request, now=now)
        if change == "edited":
            port_harness.storage.update_user_playbook(
                playbook.user_playbook_id, tags=["synthetic-edit"]
            )
        else:
            assert playbook.user_id is not None
            assert port_harness.storage.archive_user_playbook_by_id(
                playbook.user_id, playbook.user_playbook_id
            )

        with pytest.raises(OpenWorldEvidenceTargetStaleError):
            port_harness.storage.load_open_world_evidence_bundle(record.job.job_id)

    # ---------------------------------------------------------- governance

    def test_a_subject_under_erasure_refuses_freeze_and_load(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        record = _freeze(port_harness, request, now=now)
        assert playbook.user_id is not None
        port_harness.block_subject(playbook.user_id)

        with pytest.raises(OpenWorldEvidenceGovernanceError):
            port_harness.storage.load_open_world_evidence_bundle(record.job.job_id)
        with pytest.raises(OpenWorldEvidenceGovernanceError):
            _freeze(port_harness, request, now=now + 60)

    def test_an_epoch_the_store_does_not_hold_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        ((ref, _epoch),) = request.subject_epochs
        stale = _freeze_request(port_harness, playbook, ((ref, 1),), now=now)

        with pytest.raises(OpenWorldEvidenceGovernanceError):
            _freeze(port_harness, stale, now=now)

        assert port_harness.optimization_jobs(playbook.user_playbook_id) == []

    def test_an_unregistered_subject(self, port_harness: EvidenceHarness) -> None:
        owner = _uid("owner")
        playbook = _owned_playbook(port_harness, owner)
        stored = port_harness.storage.get_user_playbook_by_id(playbook.user_playbook_id)
        assert stored is not None
        now = port_harness.now()
        ref = port_harness.storage._subject_ref_for_user_id(owner)  # noqa: SLF001
        request = _freeze_request(port_harness, stored, _subject_epochs(ref), now=now)

        if port_harness.governance_registry:
            with pytest.raises(OpenWorldEvidenceGovernanceError):
                _freeze(port_harness, request, now=now)
        else:
            assert _freeze(port_harness, request, now=now).claim is not None

    def test_the_epoch_vector_covers_every_subject_the_evidence_carries(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        storage = port_harness.storage
        owner_ref = storage._subject_ref_for_user_id(world.owner)  # noqa: SLF001
        viewer_ref = storage._subject_ref_for_user_id(world.viewer)  # noqa: SLF001

        epochs = dict(storage.list_open_world_governance_epochs())

        assert (epochs.get(owner_ref), epochs.get(viewer_ref)) == (0, 0)
        refs = [ref for ref, _epoch in storage.list_open_world_governance_epochs()]
        assert refs == sorted(refs, key=str.encode), "ordered by UTF-8 bytes"
        if port_harness.governance_registry:
            port_harness.register_subject(world.viewer, epoch=2)
            assert dict(storage.list_open_world_governance_epochs())[viewer_ref] == 2

    # ----------------------------------------------------------------- I19

    def test_a_job_naming_another_tenant_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, request, now = _frozen_target(port_harness)
        foreign = _freeze_request(
            port_harness,
            playbook,
            request.subject_epochs,
            tenant_id="another-tenant",
            now=now,
        )

        with pytest.raises(OpenWorldEvidenceGovernanceError):
            _freeze(port_harness, foreign, now=now)

        assert port_harness.optimization_jobs(playbook.user_playbook_id) == []

    def test_a_snapshot_for_another_tenant_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)

        with pytest.raises(OpenWorldEvidenceIntegrityError, match="tenant"):
            _snapshot(
                port_harness, _target(port_harness, world, tenant_id="another-tenant")
            )

    # ------------------------------------------------------------- snapshot

    def test_the_snapshot_reads_the_published_window(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)

        snapshot = _snapshot(port_harness, target)

        assert snapshot.tenant_id == port_harness.storage.org_id
        assert snapshot.population_limit_exceeded is False
        (event,) = snapshot.events
        assert (event.request_id, event.user_id, event.session_id) == (
            world.request_id,
            world.viewer,
            world.session_id,
        )
        assert event.user_playbook_id == world.playbook.user_playbook_id
        assert event.served_semantic_digest == target.incumbent_semantic_digest
        assert event.served_full_version_fingerprint == (
            target.incumbent_full_version_fingerprint
        )
        assert event.correlation_reason is None
        assert event.outcome == "failure"
        assert event.outcome_id is not None and event.outcome_revision == 1
        assert event.outcome_contract_digest is not None, "the contract is known"
        assert event.trajectory is not None
        assert event.trajectory_digest_matches is True
        assert event.governance_valid is True
        assert event.awaiting_reaction is False
        assert event.outcome_is_inferred is False

    def test_a_target_that_moved_is_refused(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)
        port_harness.storage.update_user_playbook(
            world.playbook.user_playbook_id, tags=["synthetic-edit"]
        )

        with pytest.raises(OpenWorldEvidenceTargetStaleError):
            _snapshot(port_harness, target)

    def test_a_subject_under_erasure_invalidates_its_events(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)
        port_harness.block_subject(world.viewer)

        (event,) = _snapshot(port_harness, target).events

        assert event.governance_valid is False

    def test_the_population_bound_refuses_the_whole_window(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        limits = OpenWorldEvidenceReadLimits(
            historical_window_seconds=_LIMITS.historical_window_seconds,
            maximum_population=0,
            max_trajectory_bytes=_LIMITS.max_trajectory_bytes,
        )

        snapshot = _snapshot(port_harness, _target(port_harness, world), limits=limits)

        assert snapshot.population_limit_exceeded is True
        assert snapshot.events == ()

    def test_the_snapshot_needs_its_own_transaction(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)

        with (
            pytest.raises(ValueError, match="its own transaction"),
            port_harness.storage.commit_scope(),
        ):
            _snapshot(port_harness, target)

    # ------------------------------------------------------------------ I9

    def test_the_snapshot_is_one_consistent_read(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness, judged=False)
        target = _target(port_harness, world)

        with port_harness.second_storage() as other:

            def judge_elsewhere() -> None:
                _record_outcome(other, world.session_id, at=world.frozen_at - 1)

            with port_harness.during_snapshot(judge_elsewhere):
                (during,) = _snapshot(port_harness, target).events

        assert during.outcome is None, (
            "an outcome committed by another connection mid-read is invisible"
        )
        (after,) = _snapshot(port_harness, target).events
        assert after.outcome == "failure", "the same outcome is read afterwards"

    def test_a_source_above_the_sampling_cap_is_read_whole(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness, extra_interactions=_LONG_SOURCE_ROWS - 1)

        snapshot = _snapshot(port_harness, _target(port_harness, world))

        assert _trajectory_rows(snapshot) == _LONG_SOURCE_ROWS
        (event,) = snapshot.events
        assert event.trajectory_digest_matches is True

    def test_an_exhausted_source_budget_refuses_the_snapshot(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)

        with (
            port_harness.source_read_budget(100),
            pytest.raises(StorageError, match=_BUDGET_EXCEEDED),
        ):
            _snapshot(port_harness, target)

        assert _snapshot(port_harness, target).events, "the budget, not the data"

    def test_an_exhausted_source_deadline_refuses_the_snapshot(
        self, port_harness: EvidenceHarness
    ) -> None:
        world = _published_world(port_harness)
        target = _target(port_harness, world)

        with (
            port_harness.source_read_clock_jumps(),
            pytest.raises(StorageError, match=_DEADLINE_EXCEEDED),
        ):
            _snapshot(port_harness, target)

    # ----------------------------------------------------------- lifecycles

    def test_a_playbook_the_tuner_never_published_has_no_lifecycle(
        self, port_harness: EvidenceHarness
    ) -> None:
        playbook, _request, _now = _frozen_target(port_harness)

        assert (
            tuple(
                port_harness.storage.list_playbook_deployment_lifecycles(
                    user_playbook_ids=(playbook.user_playbook_id,)
                )
            )
            == ()
        )
        assert (
            incumbent_published_at(
                port_harness.storage, user_playbook_id=playbook.user_playbook_id
            )
            is None
        )

    # ------------------------------------------------------------------ D4

    def test_a_stored_playbook_carries_its_owners_subject_ref(
        self, port_harness: EvidenceHarness
    ) -> None:
        owner = _uid("owner")
        playbook = _owned_playbook(port_harness, owner)

        stored = port_harness.storage.get_user_playbook_by_id(playbook.user_playbook_id)
        served = _served(port_harness, playbook)

        assert stored is not None
        expected = port_harness.storage._subject_ref_for_user_id(owner)  # noqa: SLF001
        assert stored.governance_subject_ref == expected
        assert served.governance_subject_ref == expected
