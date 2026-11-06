"""``OpenWorldStoragePort``: what the open-world tuner asks of one storage.

The port is the union of consumer-defined Protocols. Each one names exactly
the calls one part of the active open-world path makes, so a consumer that
grows a call grows its Protocol, and the port grows with it. A backend that
implements the port can run the whole attempt: freeze the evidence bundle,
run the analysts under a job lease, and publish a provisional successor.

The port covers the active path only. Methods only the retired V1 attempt and
qualification stores call are not members.

Records and errors the members return or raise live in ``open_world_records``;
this module holds only the Protocols. It imports no ``open_world`` module, so a
storage backend may name it without inverting the dependency arrow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    OptimizationJobStage,
    OptimizationTerminalOutcome,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
    UserPlaybook,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    OpenWorldEvidenceReadLimits,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    CompletionRecord,
    ExpectedProvenance,
    OpenWorldInvocationBinding,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldAttemptContextRow,
    OpenWorldEvidenceStorageRecord,
    OpenWorldInvocationRow,
    OpenWorldQualificationAuthorityRow,
    SweepProgressRecord,
    SweepQueueEntry,
)
from reflexio.server.services.playbook.publication import (
    ProvisionalPublicationResult,
    UserPlaybookProvisionalPublicationStore,
)


class OpenWorldStageStore(Protocol):
    """The job-lifecycle surface the runner advances an attempt through.

    A member of :class:`OpenWorldStoragePort`: a method added here is a
    method every port implementation must provide.
    """

    def advance_playbook_optimization_stage(
        self,
        job_id: int,
        fence: int,
        stage: OptimizationJobStage,
        *,
        terminal_outcome: OptimizationTerminalOutcome | None = None,
        now: int | None = None,
    ) -> bool: ...

    def get_playbook_optimization_job(
        self, job_id: int
    ) -> PlaybookOptimizationJob | None: ...

    def update_playbook_optimization_job(
        self, job_id: int, *, metadata_json: str | None = None
    ) -> None: ...

    def merge_playbook_optimization_job_metadata(
        self, job_id: int, patch: dict[str, Any]
    ) -> None: ...

    def renew_playbook_optimization_job_lease(
        self,
        job_id: int,
        owner: str,
        fence: int,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim: ...

    def reclaim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int = 60,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim: ...


class PublishedLifecycle(Protocol):
    """The two fields of a deployment lifecycle the evidence window reads.

    ``published_at`` of the lifecycle whose ``successor_id`` is the incumbent
    is when the incumbent started serving: the evidence-attribution boundary.
    Read-only properties, so a frozen dataclass row satisfies it.
    """

    @property
    def successor_id(self) -> int: ...

    @property
    def published_at(self) -> int: ...


class OpenWorldBundleStorage(Protocol):
    """Freeze and reload the attempt's evidence bundle, atomically with its job."""

    org_id: str

    def freeze_open_world_evidence_bundle(
        self,
        *,
        job: PlaybookOptimizationJob,
        artifact_content_json: str,
        artifact_content_digest: str,
        target_full_version_fingerprint: str,
        subject_epochs: tuple[tuple[str, int], ...],
        owner: str,
        lease_seconds: int,
        now: int,
    ) -> OpenWorldEvidenceStorageRecord: ...

    def load_open_world_evidence_bundle(
        self, job_id: int
    ) -> OpenWorldEvidenceStorageRecord: ...


class OpenWorldAttemptStorage(Protocol):
    """Durable attempt state: authority, invocations, context and artifacts.

    Every fenced call refuses when ``fence`` is no longer the job's live lease.
    """

    def rule_evidence_method_ready(self) -> bool: ...

    def reconcile_rule_evidence_authority(
        self, *, authority_json: str, authority_digest: str, now: int
    ) -> OpenWorldQualificationAuthorityRow: ...

    def load_open_world_qualification_authority(
        self,
    ) -> OpenWorldQualificationAuthorityRow | None: ...

    def prepare_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        expected: ExpectedProvenance,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow: ...

    def complete_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        completion: CompletionRecord,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow: ...

    def load_open_world_invocation(
        self,
        *,
        binding: OpenWorldInvocationBinding,
        fence: int,
        now: int,
    ) -> OpenWorldInvocationRow: ...

    def bind_open_world_attempt_context(
        self,
        *,
        job_id: int,
        bundle_digest: str,
        outcome_contract_digest: str,
        authority_epoch: int,
        authority_digest: str,
        fence: int,
        now: int,
    ) -> OpenWorldAttemptContextRow: ...

    def write_open_world_artifact(
        self,
        *,
        job_id: int,
        artifact_kind: str,
        content_json: str,
        content_digest: str,
        fence: int,
        now: int,
    ) -> PlaybookOptimizationArtifact: ...

    def read_open_world_artifact(
        self, *, job_id: int, artifact_kind: str, fence: int, now: int
    ) -> PlaybookOptimizationArtifact | None: ...

    def get_user_playbook_by_id(
        self, user_playbook_id: int, *, include_tombstones: bool = False
    ) -> UserPlaybook | None: ...


class OpenWorldFreezeReads(Protocol):
    """The reads that select the evidence window an attempt is frozen from."""

    #: The candidate projector's embedding client.
    llm_client: Any

    def list_open_world_governance_epochs(self) -> tuple[tuple[str, int], ...]:
        """The live epoch vector a fresh target must declare.

        An empty vector is not a neutral default -- it marks every event
        governance-invalid -- so this read is a prerequisite.
        """
        ...

    def load_open_world_evidence_snapshot(
        self,
        *,
        target: OpenWorldTargetIdentity,
        frozen_at: int,
        limits: OpenWorldEvidenceReadLimits,
    ) -> OpenWorldEvidenceSnapshot:
        """The one consistent window read the attempt is selected from.

        ``limits`` carries the policy bounds; storage owns none of them.
        """
        ...

    def list_playbook_deployment_lifecycles(
        self, *, user_playbook_ids: Sequence[int]
    ) -> Sequence[PublishedLifecycle]:
        """Lifecycles touching these playbooks, at either end.

        Required rather than probed: a backend missing it would read the
        incumbent as never published and silently widen the evidence window
        back to the predecessor's sessions.
        """
        ...

    def session_compaction_checkpoint(
        self,
        *,
        job_id: int,
        identity: str,
        fence: int,
        now: int,
        receipt: dict[str, Any] | None = None,
        protocol: str | None = None,
        expected_digest: str | None = None,
    ) -> dict[str, Any] | None:
        """Read or advance session compaction under the live attempt fence."""
        ...


class OpenWorldProvisionalPublicationStore(
    UserPlaybookProvisionalPublicationStore, Protocol
):
    """Provisional publication plus recovery of a commit whose response was lost."""

    def load_open_world_provisional_publication_retry(
        self,
        *,
        job_id: int,
        claim: OptimizationJobClaim,
    ) -> ProvisionalPublicationResult | None: ...


class OpenWorldSweepProgressStore(Protocol):
    """The sweep's durable progress: one record per tick, and the queue it drains.

    The admission rules both methods apply, and their constants, are in
    ``open_world_records`` (the "Sweep progress" section). The sweep reaches
    storage only through these two calls.
    """

    def record_open_world_sweep_progress(
        self, *, user_playbook_id: int, outcome: str, job_id: int | None
    ) -> SweepProgressRecord:
        """Record that a playbook had its tick today, whatever the outcome.

        One row per playbook per UTC day, chosen with ``attempted_at`` from one
        clock read. A re-record the same day replaces ``outcome`` and
        ``attempted_at``, keeps the earlier ``job_id`` when given none, and
        increments ``attempt_count``. Raises when ``user_playbook_id`` or
        ``job_id`` is not positive or ``outcome`` is blank, writing nothing.
        """
        ...

    def list_open_world_playbooks_without_progress(
        self, *, limit: int, window_seconds: int | None
    ) -> tuple[SweepQueueEntry, ...]:
        """Current playbooks whose day is not spent, oldest attempt first.

        ``window_seconds`` admits only playbooks with a published exposure no
        older than that many seconds before the start of today (UTC); ``None``
        admits every current playbook. A limit below 1 reads as 1.
        """
        ...


class OpenWorldStoragePort(
    OpenWorldStageStore,
    OpenWorldBundleStorage,
    OpenWorldAttemptStorage,
    OpenWorldFreezeReads,
    OpenWorldProvisionalPublicationStore,
    OpenWorldSweepProgressStore,
    Protocol,
):
    """Everything the active open-world path asks of one storage."""


__all__ = [
    "OpenWorldAttemptStorage",
    "OpenWorldBundleStorage",
    "OpenWorldFreezeReads",
    "OpenWorldProvisionalPublicationStore",
    "OpenWorldStageStore",
    "OpenWorldStoragePort",
    "OpenWorldSweepProgressStore",
    "PublishedLifecycle",
]
