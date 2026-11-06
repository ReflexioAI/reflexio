"""Durable freeze boundary for open-world evidence bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from pydantic import TypeAdapter, ValidationError

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA,
    OpenWorldEvidenceBundle,
    OpenWorldEvidenceSchemaUnsupportedError,
    OpenWorldEvidenceSelection,
    OpenWorldTargetIdentity,
    validate_expected_manifest_attribution_bound,
    validate_open_world_evidence_bundle,
    validate_open_world_evidence_selection,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
    OpenWorldRefusalError,
)
from reflexio.server.services.offline_tuner.open_world_job_metadata import (
    identity_bearing_metadata,
)
from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldBundleStorage,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldEvidenceGovernanceError,
    OpenWorldEvidenceLeaseConflictError,
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
    StorageError,
)

_METADATA_KEY = "open_world_evidence"
_METADATA_SCHEMA = "offline-tuner-open-world-store-v3"
_BUNDLE_ADAPTER = TypeAdapter(OpenWorldEvidenceBundle)


class OpenWorldEvidenceConflictError(StorageError):
    """Raised when a durable open-world identity is reused for other bytes."""


class OpenWorldEvidenceIntegrityError(StorageError, OpenWorldRefusalError):
    """Raised when frozen bytes, digests, or metadata no longer agree."""

    refusal = OpenWorldRefusal.EVIDENCE_INTEGRITY


class OpenWorldEvidenceGovernanceInvalidError(StorageError, OpenWorldRefusalError):
    """Raised when frozen governance epochs no longer authorize later work."""

    refusal = OpenWorldRefusal.GOVERNANCE_INVALID


@dataclass(frozen=True)
class FrozenOpenWorldEvidence:
    job: PlaybookOptimizationJob
    artifact: PlaybookOptimizationArtifact
    bundle: OpenWorldEvidenceBundle
    claim: OptimizationJobClaim | None = None


def incumbent_published_at(storage: Any, *, user_playbook_id: int) -> int | None:
    """When the tuner published THIS version, or ``None`` if it never did.

    The evidence-attribution boundary of INV-2, and deliberately the ONLY
    expression of it: the composition root's freeze-time bound delegates
    here rather than restating the query, so the freeze-time bound and
    the reload-time check cannot disagree about what "published" means.

    A lifecycle row names two versions and
    ``list_playbook_deployment_lifecycles`` matches either end, so the
    ``successor_id`` filter is what makes this "when this version STARTED
    serving" rather than "when it stopped". A hand-authored playbook appears at
    neither end and correctly yields ``None``.

    ``max`` over the matches rather than the first: a version can in principle
    be the successor of more than one lifecycle (a retried publication writes
    its own row), and the boundary that keeps another version's sessions out is
    the LATEST one.

    Args:
        storage (Any): Storage exposing ``list_playbook_deployment_lifecycles``.
        user_playbook_id (int): The version being asked about.

    Returns:
        int | None: The publication epoch, or ``None``.
    """
    lifecycles = storage.list_playbook_deployment_lifecycles(
        user_playbook_ids=(user_playbook_id,)
    )
    published = [
        int(row.published_at)
        for row in lifecycles
        if int(row.successor_id) == user_playbook_id
    ]
    return max(published) if published else None


class OpenWorldEvidenceBundleStore:
    """Freeze and reload one canonical open-world evidence bundle."""

    def __init__(
        self,
        storage: OpenWorldBundleStorage,
    ) -> None:
        self.storage = storage

    def freeze(
        self,
        selection: OpenWorldEvidenceSelection,
        owner: str,
        lease_seconds: int,
        now: int,
    ) -> FrozenOpenWorldEvidence:
        if type(selection) is not OpenWorldEvidenceSelection:
            raise TypeError("selection must be an OpenWorldEvidenceSelection")
        if not owner.strip():
            raise ValueError("owner must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        bundle = selection.bundle
        self._validate_storage_tenant(bundle.target)
        self._validate_selection_shape(selection)
        subject_epochs = self._complete_subject_epochs(bundle)

        discovery_key = self._discovery_key(bundle)
        attempt_key = self._attempt_key(
            discovery_key=discovery_key, bundle_digest=bundle.digest
        )
        job_request = PlaybookOptimizationJob(
            optimizer_kind=OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
            target_kind="user_playbook",
            target_id=bundle.target.user_playbook_id,
            metadata_json=self._metadata_json(
                bundle, subject_epochs, shared_compaction=True
            ),
            discovery_key=discovery_key,
            attempt_key=attempt_key,
            stage="evidence_frozen",
            expected_population_manifest_digest=bundle.expected.digest,
            created_at=now,
            updated_at=now,
        )
        try:
            freeze = self.storage.freeze_open_world_evidence_bundle
        except AttributeError as exc:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "storage adapter has no PostgreSQL open-world governance boundary"
            ) from exc
        try:
            record = freeze(
                job=job_request,
                artifact_content_json=bundle.canonical_bytes().decode(),
                artifact_content_digest=bundle.digest,
                target_full_version_fingerprint=(
                    bundle.target.incumbent_full_version_fingerprint
                ),
                subject_epochs=subject_epochs,
                owner=owner,
                lease_seconds=lease_seconds,
                now=now,
            )
        except (
            OptimizationJobIdentityConflictError,
            OpenWorldEvidenceLeaseConflictError,
        ) as exc:
            raise OpenWorldEvidenceConflictError(
                "open-world attempt conflict or active target has a different "
                "evidence manifest"
            ) from exc
        except OptimizationArtifactIntegrityError as exc:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence artifact content/digest mismatch"
            ) from exc
        except OpenWorldEvidenceTargetStaleError as exc:
            raise OpenWorldEvidenceIntegrityError(
                "current target user playbook is stale"
            ) from exc
        except OpenWorldEvidenceGovernanceError as exc:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "open-world evidence governance invalidated"
            ) from exc
        job = record.job
        self._validate_job_identity(
            job,
            expected=job_request,
            bundle=bundle,
        )
        return self._frozen_from_artifact(
            job,
            record.artifact,
            claim=record.claim,
        )

    def load(self, job_id: int) -> FrozenOpenWorldEvidence:
        try:
            load = self.storage.load_open_world_evidence_bundle
        except AttributeError as exc:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "storage adapter has no PostgreSQL open-world governance boundary"
            ) from exc
        try:
            record = load(job_id)
        except OptimizationArtifactIntegrityError as exc:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence artifact content/digest mismatch"
            ) from exc
        except OpenWorldEvidenceTargetStaleError as exc:
            raise OpenWorldEvidenceIntegrityError(
                "current target user playbook is stale"
            ) from exc
        except OpenWorldEvidenceGovernanceError as exc:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "open-world evidence governance invalidated"
            ) from exc
        frozen = self._frozen_from_artifact(record.job, record.artifact, claim=None)
        self._validate_attribution_bound(frozen.bundle)
        return frozen

    def _validate_attribution_bound(self, bundle: OpenWorldEvidenceBundle) -> None:
        """Re-check the reloaded window's lower bound against the lifecycle row.

        ``_validate_expected_manifest_identity`` runs on ``(target, expected)``
        alone, so all it can enforce is that the window lies somewhere inside
        the fixed policy scan -- a 14-day band. Here the lifecycle row IS
        reachable, and it turns that range back into an equality: a manifest
        re-derived against the wrong ``published_at`` is rejected instead of
        being accepted anywhere in the band.

        Skipped, not failed, when the adapter exposes no lifecycle read. The
        only such adapters are test doubles for the freeze contract; every
        storage that can hold a lifecycle row implements it, and the storage
        backends' integration tests exercise the wiring against a real
        database. Failing here instead would turn "this fake
        does not model publications" into an integrity error.
        """
        reader = getattr(self.storage, "list_playbook_deployment_lifecycles", None)
        if reader is None:
            return
        published_at = incumbent_published_at(
            self.storage, user_playbook_id=bundle.target.user_playbook_id
        )
        try:
            validate_expected_manifest_attribution_bound(bundle, published_at)
        except ValueError as exc:
            raise OpenWorldEvidenceIntegrityError(
                f"open-world evidence attribution bound is stale: {exc}"
            ) from exc

    @staticmethod
    def _discovery_key(bundle: OpenWorldEvidenceBundle) -> str:
        return sha256(
            canonical_json_bytes(
                {
                    "schema_version": "offline-tuner-open-world-discovery-key-v1",
                    "target": bundle.target.canonical_payload(),
                    "expected_manifest_digest": bundle.expected.digest,
                }
            )
        ).hexdigest()

    @staticmethod
    def _attempt_key(*, discovery_key: str, bundle_digest: str) -> str:
        return sha256(
            canonical_json_bytes(
                {
                    "schema_version": "offline-tuner-open-world-attempt-key-v1",
                    "discovery_key": discovery_key,
                    "bundle_digest": bundle_digest,
                }
            )
        ).hexdigest()

    @staticmethod
    def _metadata_payload(bundle: OpenWorldEvidenceBundle) -> dict[str, object]:
        return {
            "schema_version": _METADATA_SCHEMA,
            "tenant_id": bundle.target.tenant_id,
            "user_playbook_id": bundle.target.user_playbook_id,
            "incumbent_full_version_fingerprint": (
                bundle.target.incumbent_full_version_fingerprint
            ),
            "target_digest": bundle.target.digest,
            "bundle_digest": bundle.digest,
            "expected_manifest_digest": bundle.expected.digest,
            "frozen_at": bundle.target.frozen_at,
            "governance_epoch_vector": [
                {"subject": subject, "epoch": epoch}
                for subject, epoch in bundle.target.governance_epoch_vector
            ],
            "global_coverage": bundle.global_coverage.canonical_payload(),
            "target_coverage": bundle.target_coverage.canonical_payload(),
            "failed_sessions": bundle.failed_sessions,
            "successful_sessions": bundle.successful_sessions,
            "unknown_sessions": bundle.unknown_sessions,
            "session_exclusion_reasons": [
                {"reason": reason, "count": count}
                for reason, count in bundle.session_exclusion_reasons
            ],
            "discovery_exclusion_reasons": [
                {"reason": reason, "count": count}
                for reason, count in bundle.discovery_exclusion_reasons
            ],
            "eligible": bundle.eligible,
            "reason": bundle.reason,
        }

    @classmethod
    def _metadata_json(
        cls,
        bundle: OpenWorldEvidenceBundle,
        subject_epochs: tuple[tuple[str, int], ...] | None = None,
        *,
        shared_compaction: bool = False,
    ) -> str:
        complete_epochs = subject_epochs or cls._complete_subject_epochs(bundle)
        metadata = {
            _METADATA_KEY: cls._metadata_payload(bundle),
            "publication_subject_epochs": {
                "subjects": [
                    {"subject_ref": subject, "erasure_epoch": epoch}
                    for subject, epoch in complete_epochs
                ]
            },
        }
        if shared_compaction:
            from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
                PROTOCOL,
                PROTOCOL_KEY,
            )

            metadata[PROTOCOL_KEY] = PROTOCOL
        return canonical_json_bytes(metadata).decode()

    def _frozen_from_artifact(
        self,
        job: PlaybookOptimizationJob,
        artifact: PlaybookOptimizationArtifact,
        *,
        claim: OptimizationJobClaim | None,
    ) -> FrozenOpenWorldEvidence:
        if job.optimizer_kind != OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND:
            raise OpenWorldEvidenceIntegrityError("optimizer kind is not open-world")
        if job.stage != "evidence_frozen":
            raise OpenWorldEvidenceIntegrityError("open-world evidence is not frozen")
        if (
            artifact.content_digest
            != sha256(artifact.content_json.encode()).hexdigest()
        ):
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence artifact content/digest mismatch"
            )
        bundle = _parse_canonical_bundle(artifact.content_json)
        self._validate_storage_tenant(bundle.target)
        self._validate_bundle_shape(bundle)
        if artifact.content_digest != bundle.digest:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence artifact is not canonical bundle bytes"
            )
        metadata = self._metadata(job)
        open_world_metadata = metadata[_METADATA_KEY]
        if open_world_metadata.get("bundle_digest") != bundle.digest:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence bundle digest changed"
            )
        if (
            open_world_metadata.get("expected_manifest_digest")
            != bundle.expected.digest
        ):
            raise OpenWorldEvidenceIntegrityError(
                "open-world expected manifest digest changed"
            )
        # The advisory diagnostics the runner writes under their reserved keys
        # are NOT part of the frozen evidence's identity -- see
        # ``open_world_job_metadata``. Comparing the column whole would reject
        # a bundle whose only difference is a record of why an earlier attempt
        # failed. Everything else still has to match byte for byte.
        #
        # BOTH sides go through the reduction, not just the durable one. The
        # expected side is freshly built and carries no diagnostic today, so
        # reducing it is a no-op -- which is exactly why it is safe, and why it
        # belongs here: an asymmetric comparison is one advisory key away from
        # being wrong, and nothing about `_metadata_json` promises it will stay
        # diagnostic-free. The sibling validators in the attempt store and the
        # storage-side freeze already reduce both sides; this is the third.
        try:
            durable_metadata_bytes = canonical_json_bytes(
                identity_bearing_metadata(metadata)
            )
            expected_metadata_bytes = canonical_json_bytes(
                identity_bearing_metadata(
                    json.loads(
                        self._metadata_json(
                            bundle, shared_compaction=self._shared_compaction(metadata)
                        )
                    )
                )
            )
        except (TypeError, ValueError) as exc:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence metadata is not canonicalizable"
            ) from exc
        if durable_metadata_bytes != expected_metadata_bytes:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence metadata changed"
            )
        expected_discovery_key = self._discovery_key(bundle)
        expected_attempt_key = self._attempt_key(
            discovery_key=expected_discovery_key,
            bundle_digest=bundle.digest,
        )
        if (
            job.discovery_key != expected_discovery_key
            or job.attempt_key != expected_attempt_key
            or job.target_kind != "user_playbook"
            or job.target_id != bundle.target.user_playbook_id
            or job.expected_population_manifest_digest != bundle.expected.digest
        ):
            raise OpenWorldEvidenceIntegrityError(
                "open-world durable job identity changed"
            )
        return FrozenOpenWorldEvidence(
            job=job if claim is not None else self._lease_redacted_job(job),
            artifact=artifact,
            bundle=bundle,
            claim=claim,
        )

    @staticmethod
    def _lease_redacted_job(job: PlaybookOptimizationJob) -> PlaybookOptimizationJob:
        return job.model_copy(
            update={
                "lease_owner": None,
                "lease_fence": 0,
                "lease_expires_at": None,
            }
        )

    def _validate_job_identity(
        self,
        job: PlaybookOptimizationJob,
        *,
        expected: PlaybookOptimizationJob,
        bundle: OpenWorldEvidenceBundle,
    ) -> None:
        if (
            job.optimizer_kind != OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND
            or job.target_kind != "user_playbook"
            or job.target_id != bundle.target.user_playbook_id
        ):
            raise OpenWorldEvidenceConflictError(
                "active target has a different open-world optimizer identity"
            )
        if job.discovery_key == expected.discovery_key:
            if job.attempt_key != expected.attempt_key:
                raise OpenWorldEvidenceConflictError(
                    "open-world discovery attempt changed"
                )
            return
        if job.attempt_key == expected.attempt_key:
            raise OpenWorldEvidenceConflictError("open-world attempt discovery changed")
        raise OpenWorldEvidenceConflictError(
            "active target has a different open-world evidence manifest"
        )

    @staticmethod
    def _shared_compaction(metadata: dict[str, Any]) -> bool:
        from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
            protocol_from_metadata,
        )

        return protocol_from_metadata(metadata) is not None

    def _metadata(self, job: PlaybookOptimizationJob) -> dict[str, Any]:
        try:
            metadata = json.loads(job.metadata_json or "{}")
        except json.JSONDecodeError as exc:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence metadata is not valid JSON"
            ) from exc
        if not isinstance(metadata, dict):
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence metadata is missing"
            )
        value = metadata.get(_METADATA_KEY)
        if not isinstance(value, dict) or value.get("schema_version") != (
            _METADATA_SCHEMA
        ):
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence metadata is missing"
            )
        return metadata

    @staticmethod
    def _validate_selection_shape(selection: OpenWorldEvidenceSelection) -> None:
        parsed = _parse_canonical_bundle(selection.canonical_bytes().decode())
        if parsed != selection.bundle:
            raise OpenWorldEvidenceIntegrityError(
                "open-world evidence selection failed strict schema validation"
            )
        try:
            validate_open_world_evidence_selection(selection)
        except OpenWorldEvidenceSchemaUnsupportedError:
            # ABOVE the widening handler below -- see `_validate_bundle_shape`.
            raise
        except (TypeError, ValueError) as exc:
            raise OpenWorldEvidenceIntegrityError(
                f"open-world evidence selection semantic validation failed: {exc}"
            ) from exc
        OpenWorldEvidenceBundleStore._validate_bundle_digests(selection.bundle)

    @staticmethod
    def _validate_bundle_shape(bundle: OpenWorldEvidenceBundle) -> None:
        try:
            validate_open_world_evidence_bundle(bundle)
        except OpenWorldEvidenceSchemaUnsupportedError:
            # ABOVE the widening handler below, which it would otherwise match
            # -- it is a ``ValueError``.
            #
            # `rehydrate_frozen_evidence` already keeps this class intact, and
            # says why: "the class has to survive to the runner intact for the
            # refusal to keep its name". The STORAGE boundary was the sibling
            # path that did not. Every production load of a frozen bundle goes
            # through `_frozen_from_artifact` -> here BEFORE
            # `OpenWorldPhase3Runner.run` is ever entered, so a bundle frozen
            # by an older image was widened to
            # `OpenWorldEvidenceIntegrityError` here and could only ever be
            # reported as the retryable `infrastructure_failure` -- the exact
            # operator instruction `bundle_schema_unsupported` exists to
            # replace, and the runner's handler for it was unreachable through
            # this path.
            #
            # The `ValueError` base is retained (see the class docstring), so
            # nothing that already caught this at a storage boundary stops
            # catching it.
            raise
        except (TypeError, ValueError) as exc:
            raise OpenWorldEvidenceIntegrityError(
                f"open-world evidence bundle semantic validation failed: {exc}"
            ) from exc
        OpenWorldEvidenceBundleStore._validate_bundle_digests(bundle)

    @staticmethod
    def _validate_bundle_digests(bundle: OpenWorldEvidenceBundle) -> None:
        if (
            bundle.expected.digest
            != sha256(bundle.expected.canonical_bytes()).hexdigest()
        ):
            raise OpenWorldEvidenceIntegrityError("expected manifest digest is invalid")
        if bundle.digest != sha256(bundle.canonical_bytes()).hexdigest():
            raise OpenWorldEvidenceIntegrityError("bundle digest is invalid")

    def _validate_storage_tenant(self, target: OpenWorldTargetIdentity) -> None:
        storage_tenant = self.storage.org_id
        if not isinstance(storage_tenant, str) or not storage_tenant:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "storage tenant is unavailable for open-world evidence"
            )
        if target.tenant_id != storage_tenant:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "target tenant does not match storage tenant"
            )

    @staticmethod
    def _complete_subject_epochs(
        bundle: OpenWorldEvidenceBundle,
    ) -> tuple[tuple[str, int], ...]:
        epochs = dict(bundle.target.governance_epoch_vector)
        refs = set(epochs)
        refs.update(
            subject
            for event in bundle.expected.events
            for subject in event.governance_subject_refs
        )
        refs.update(
            subject
            for manifest in (bundle.discovery, bundle.held_out, bundle.safety_only)
            for session in manifest.sessions
            for subject in session.governance_subject_refs
        )
        missing = refs - epochs.keys()
        if missing:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "durable evidence contains a subject absent from the target vector"
            )
        if not refs:
            raise OpenWorldEvidenceGovernanceInvalidError(
                "governance epoch vector is empty"
            )
        return tuple(sorted((subject, epochs[subject]) for subject in refs))


__all__ = [
    "FrozenOpenWorldEvidence",
    "OpenWorldEvidenceBundleStore",
    "OpenWorldEvidenceConflictError",
    "OpenWorldEvidenceGovernanceInvalidError",
    "OpenWorldEvidenceIntegrityError",
]


def _normalize_bundle_payload(raw: object) -> object:
    if not isinstance(raw, dict):
        return raw
    payload = dict(raw)
    target = payload.get("target")
    if isinstance(target, dict):
        payload["target"] = {
            **target,
            "governance_epoch_vector": tuple(
                (item["subject"], item["epoch"])
                for item in target.get("governance_epoch_vector", ())
            ),
        }
    expected = payload.get("expected")
    if isinstance(expected, dict):
        payload["expected"] = {
            **expected,
            "events": tuple(
                {
                    **event,
                    "trajectory_request_ids": tuple(
                        event.get("trajectory_request_ids", ())
                    ),
                    "governance_subject_refs": tuple(
                        event.get("governance_subject_refs", ())
                    ),
                }
                for event in expected.get("events", ())
            ),
        }
    for coverage_key in ("global_coverage", "target_coverage"):
        coverage = payload.get(coverage_key)
        if isinstance(coverage, dict):
            payload[coverage_key] = {
                **coverage,
                "reasons": tuple(
                    (item["reason"], item["count"])
                    for item in coverage.get("reasons", ())
                ),
            }
    payload["session_exclusion_reasons"] = tuple(
        (item["reason"], item["count"])
        for item in payload.get("session_exclusion_reasons", ())
    )
    payload["discovery_exclusion_reasons"] = tuple(
        (item["reason"], item["count"])
        for item in payload.get("discovery_exclusion_reasons", ())
    )
    for role in ("discovery", "held_out", "safety_only"):
        manifest = payload.get(role)
        if isinstance(manifest, dict):
            payload[role] = {
                **manifest,
                "sessions": tuple(
                    {
                        **session,
                        "exposure_event_ids": tuple(
                            session.get("exposure_event_ids", ())
                        ),
                        "request_ids": tuple(session.get("request_ids", ())),
                        "governance_subject_refs": tuple(
                            session.get("governance_subject_refs", ())
                        ),
                    }
                    for session in manifest.get("sessions", ())
                ),
            }
    return payload


def _require_supported_bundle_schema(raw: object) -> None:
    """Name the one parse failure no retry, on any host, can fix.

    ``OpenWorldEvidenceBundle.schema_version`` is a ``Literal`` pinned to the
    version THIS image derives, so an artifact frozen by an OLDER image does
    not reach ``validate_open_world_evidence_bundle``'s named refusal at all --
    it fails the ``TypeAdapter`` first and arrives indistinguishable from
    corrupt bytes. That is how ``bundle_schema_unsupported`` came to have no
    production writer: every real stale artifact died here, as the retryable
    ``infrastructure_failure``, while the runner's handler for it could only be
    reached by a bundle built in memory.

    Only a DECLARED, differing version is named. A payload with no
    ``schema_version``, or a non-string one, is malformed rather than stale and
    keeps the generic integrity refusal.
    """
    if not isinstance(raw, dict):
        return
    declared = raw.get("schema_version")
    if isinstance(declared, str) and declared != OPEN_WORLD_EVIDENCE_BUNDLE_SCHEMA:
        raise OpenWorldEvidenceSchemaUnsupportedError("bundle schema is unsupported")


def _parse_canonical_bundle(content_json: str) -> OpenWorldEvidenceBundle:
    try:
        raw = json.loads(content_json)
        _require_supported_bundle_schema(raw)
        bundle = _BUNDLE_ADAPTER.validate_python(_normalize_bundle_payload(raw))
        canonical_raw = canonical_json_bytes(raw).decode()
    except OpenWorldEvidenceSchemaUnsupportedError:
        # ABOVE the widening handler below, which it would otherwise match --
        # it is a ``ValueError``. See `_require_supported_bundle_schema`.
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        raise OpenWorldEvidenceIntegrityError(
            "open-world evidence artifact is not a valid bundle schema"
        ) from exc
    if (
        canonical_raw != content_json
        or bundle.canonical_bytes().decode() != content_json
    ):
        raise OpenWorldEvidenceIntegrityError(
            "open-world evidence artifact failed strict canonical schema validation"
        )
    return bundle
