"""Pure checks a storage backend runs before freezing an open-world bundle.

``freeze_open_world_evidence_bundle`` and ``load_open_world_evidence_bundle``
refuse on the same conditions in every backend. The conditions that need no
database live here as plain functions, so a backend calls them instead of
restating them. The PostgreSQL freeze in the host backend still carries
private copies of the same four checks; the shared port contract
(``reflexio.test_support.open_world_evidence_contract``) holds both to the same
answers until it calls these instead.

What stays in the backend: the transaction, the target row lock, the
governance read and the job, lease and artifact writes.
"""

from __future__ import annotations

import json

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.open_world_job_metadata import (
    identity_bearing_metadata_json,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldEvidenceGovernanceError,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
)

#: The job fields a replayed freeze must repeat exactly.
FREEZE_IMMUTABLE_JOB_FIELDS = (
    "optimizer_kind",
    "target_kind",
    "target_id",
    "discovery_key",
    "attempt_key",
    "stage",
    "expected_population_manifest_digest",
)


def canonical_artifact_payload(content_json: str) -> tuple[object, str]:
    """Decode a bundle artifact and return it with its canonical text.

    Args:
        content_json (str): The artifact JSON the caller froze.

    Returns:
        tuple[object, str]: The decoded payload and its canonical JSON text.

    Raises:
        OptimizationArtifactIntegrityError: The text is not canonicalizable JSON.
    """
    try:
        payload = json.loads(content_json)
        canonical_content = canonical_json_bytes(payload).decode()
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeError) as exc:
        raise OptimizationArtifactIntegrityError(
            "open-world artifact is not valid JSON"
        ) from exc
    return payload, canonical_content


def validate_subject_epoch_shape(
    subject_epochs: tuple[tuple[str, int], ...],
) -> None:
    """Refuse a governance subject vector that is empty or not canonical.

    Canonical means: non-empty, ``(str, int)`` pairs, ordered by the UTF-8
    bytes of the subject, no duplicate subject, no blank subject and no
    negative epoch.

    Raises:
        OpenWorldEvidenceGovernanceError: The vector breaks any of those rules.
    """
    if not subject_epochs:
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence subject vector is empty"
        )
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not int
        for item in subject_epochs
    ):
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence subject vector has invalid entries"
        )
    if (
        tuple(sorted(subject_epochs, key=lambda item: item[0].encode("utf-8")))
        != subject_epochs
    ):
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence subject vector is not canonical"
        )
    if len({subject for subject, _epoch in subject_epochs}) != len(subject_epochs):
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence subject vector contains duplicates"
        )
    if any(not subject.strip() or epoch < 0 for subject, epoch in subject_epochs):
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence subject vector is invalid"
        )


def freeze_validation_identity(
    job: PlaybookOptimizationJob, *, org_id: str
) -> tuple[str, tuple[tuple[str, int], ...]]:
    """The fingerprint and subject vector a frozen job's metadata declares.

    Args:
        job (PlaybookOptimizationJob): The requested or stored job.
        org_id (str): The storage's tenant.

    Returns:
        tuple[str, tuple[tuple[str, int], ...]]: The incumbent's full-version
        fingerprint and the governance subject vector.

    Raises:
        OpenWorldEvidenceGovernanceError: The metadata names another tenant.
        OptimizationArtifactIntegrityError: The metadata is malformed or does
            not describe this job's target.
    """
    try:
        metadata = json.loads(job.metadata_json)
        open_world = metadata["open_world_evidence"]
        if not isinstance(open_world, dict):
            raise TypeError("open-world metadata is not an object")
        tenant_id = open_world["tenant_id"]
        if not isinstance(tenant_id, str):
            raise TypeError("open-world tenant is not text")
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizationArtifactIntegrityError(
            "open-world job metadata is malformed"
        ) from exc
    if tenant_id != org_id:
        raise OpenWorldEvidenceGovernanceError(
            "open-world evidence tenant does not match storage tenant"
        )
    try:
        if (
            job.optimizer_kind != OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND
            or job.target_kind != "user_playbook"
            or open_world["user_playbook_id"] != job.target_id
        ):
            raise OptimizationArtifactIntegrityError(
                "open-world job target metadata is malformed"
            )
        fingerprint = open_world["incumbent_full_version_fingerprint"]
        subjects = metadata["publication_subject_epochs"]["subjects"]
        subject_epochs = tuple(
            (item["subject_ref"], item["erasure_epoch"]) for item in subjects
        )
        if not isinstance(fingerprint, str):
            raise TypeError("fingerprint is not text")
        validate_subject_epoch_shape(subject_epochs)
        return fingerprint, subject_epochs
    except (KeyError, TypeError, ValueError, OpenWorldEvidenceGovernanceError) as exc:
        raise OptimizationArtifactIntegrityError(
            "open-world job metadata is malformed"
        ) from exc


def validate_freeze_job_request(
    persisted: PlaybookOptimizationJob, request: PlaybookOptimizationJob
) -> None:
    """Refuse a freeze whose immutable open-world identity has moved.

    ``metadata_json`` is compared through ``identity_bearing_metadata_json``,
    so the advisory diagnostics the runner writes are not identity. A job
    admitted before the shared-compaction protocol keeps its original
    protocol: when the stored job names none, the request's is ignored.

    Raises:
        OptimizationJobIdentityConflictError: An immutable field differs, or
            the metadata cannot be canonicalized.
    """
    if any(
        getattr(persisted, field) != getattr(request, field)
        for field in FREEZE_IMMUTABLE_JOB_FIELDS
    ):
        raise OptimizationJobIdentityConflictError(
            "active target has a different immutable open-world identity"
        )
    # BOTH exception types: the canonical encoder raises TypeError for a
    # non-string key and ValueError for a surrogate or an inexact integer.
    try:
        from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
            PROTOCOL_KEY,
            protocol_from_metadata,
        )

        persisted_identity = identity_bearing_metadata_json(persisted.metadata_json)
        persisted_metadata = json.loads(persisted.metadata_json or "{}")
        requested_metadata = json.loads(request.metadata_json or "{}")
        known_protocol = protocol_from_metadata(persisted_metadata)
        protocol_from_metadata(requested_metadata)
        if known_protocol is None:
            requested_metadata.pop(PROTOCOL_KEY, None)
        requested_identity = identity_bearing_metadata_json(
            canonical_json_bytes(requested_metadata).decode()
        )
    except (TypeError, ValueError) as exc:
        raise OptimizationJobIdentityConflictError(
            "open-world optimizer job metadata is not canonicalizable"
        ) from exc
    if persisted_identity != requested_identity:
        raise OptimizationJobIdentityConflictError(
            "active target has a different immutable open-world identity"
        )


def validate_artifact_request(
    artifact: PlaybookOptimizationArtifact,
    *,
    content_json: str,
    content_digest: str,
) -> None:
    """Refuse a replay whose bundle bytes differ from the stored artifact.

    A stored artifact is never rewritten: the same job asked to freeze other
    bytes is an integrity failure, not an update.

    Raises:
        OptimizationArtifactIntegrityError: The content or digest differs.
    """
    if (
        artifact.content_json != content_json
        or artifact.content_digest != content_digest
    ):
        raise OptimizationArtifactIntegrityError(
            "open-world optimizer artifact conflicts with exact bytes"
        )


__all__ = [
    "FREEZE_IMMUTABLE_JOB_FIELDS",
    "canonical_artifact_payload",
    "freeze_validation_identity",
    "validate_artifact_request",
    "validate_freeze_job_request",
    "validate_subject_epoch_shape",
]
