"""Abstract playbook optimization job store declarations."""

from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    OpenWorldQualificationRecord,
    OptimizationArtifactKind,
    OptimizationJobClaim,
    OptimizationJobStage,
    OptimizationTerminalOutcome,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)


class OptimizationJobStoreMixin:
    """Abstract playbook optimization job/candidate/evaluation store methods."""

    # ==============================
    # Playbook optimization methods
    # ==============================

    @abstractmethod
    def create_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        """Persist a playbook optimization job and return it with id populated."""
        raise NotImplementedError

    def get_playbook_optimization_job(
        self, job_id: int
    ) -> PlaybookOptimizationJob | None:
        """Load one optimizer job by its durable identity."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def prepare_gepa_user_playbook_publication(
        self,
        *,
        job_id: int,
        owner: str,
        lease_seconds: int,
        winner_candidate_id: int,
        candidate_content_digest: str,
        search_projection_digest: str,
        publication_proof_digest: str,
        projection_json: str,
        decision_proof_json: str,
        subject_epochs_json: str,
        metadata_json: str,
    ) -> PlaybookOptimizationJob:
        """Fence and persist GEPA publication authority before staging."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def get_unconsumed_gepa_user_playbook_publishing_job(
        self, target_id: int
    ) -> PlaybookOptimizationJob | None:
        """Load an active GEPA user publication job with no terminal result."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def reclaim_gepa_user_playbook_publishing_job(
        self,
        target_id: int,
        owner: str,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> PlaybookOptimizationJob | None:
        """Recover one unconsumed GEPA user publication job.

        Outcomes are part of the storage contract:
        return the reclaimed job when an expired publishing lease is fenced to
        the new owner, return ``None`` when no unconsumed publishing job exists,
        and raise ``OptimizationJobLeaseLiveError`` when a matching job exists
        but its optimizer lease is still live.
        """
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def create_or_get_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        """Atomically insert or return the active job with the same replay identity."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def claim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        """Claim an unleased active optimizer job and issue a new fence."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def reclaim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int = 60,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        """Reclaim an expired active optimizer job and issue a newer fence."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def renew_playbook_optimization_job_lease(
        self,
        job_id: int,
        owner: str,
        fence: int,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        """Extend a current, unexpired optimizer lease without changing its fence."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def advance_playbook_optimization_stage(
        self,
        job_id: int,
        fence: int,
        stage: OptimizationJobStage,
        *,
        terminal_outcome: OptimizationTerminalOutcome | None = None,
        now: int | None = None,
    ) -> bool:
        """Advance the linear replay stage only for the current unexpired fence."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def upsert_playbook_optimization_artifact(
        self,
        artifact: PlaybookOptimizationArtifact,
        fence: int,
        *,
        now: int | None = None,
    ) -> PlaybookOptimizationArtifact:
        """Write a singleton artifact only under the current lease fence."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def get_playbook_optimization_artifact(
        self,
        job_id: int,
        artifact_kind: OptimizationArtifactKind,
    ) -> PlaybookOptimizationArtifact | None:
        """Return one typed singleton artifact when present."""
        raise NotImplementedError(
            "Storage backend does not support replay-gated playbook optimization"
        )

    def persist_open_world_qualification_record(
        self, record: OpenWorldQualificationRecord
    ) -> OpenWorldQualificationRecord:
        """Persist one immutable qualification result and return the stored row.

        The cache key is ``(component_identity_digest, suite_digest)``. The
        first insert controls ``created_at``; replaying a semantically
        identical record is idempotent and returns the stored row unchanged.

        Args:
            record (OpenWorldQualificationRecord): The result to persist.

        Returns:
            OpenWorldQualificationRecord: The durable record for this key.

        Raises:
            OpenWorldQualificationConflictError: If a record already exists for
                the key and differs in any field other than ``created_at``.
        """
        raise NotImplementedError(
            "Storage backend does not support open-world analyst qualification"
        )

    def load_open_world_qualification_record(
        self,
        *,
        component_identity_digest: str,
        suite_digest: str,
    ) -> OpenWorldQualificationRecord | None:
        """Load the cached qualification result for one exact identity/suite pair.

        Args:
            component_identity_digest (str): Pinned analyst component identity.
            suite_digest (str): Canonical qualification-suite digest.

        Returns:
            OpenWorldQualificationRecord | None: The stored record, or ``None``
                when this key has never been qualified.
        """
        raise NotImplementedError(
            "Storage backend does not support open-world analyst qualification"
        )

    @abstractmethod
    def update_playbook_optimization_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        best_candidate_id: int | None = None,
        successor_target_id: int | None = None,
        decision_reason: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        """Update mutable fields on a playbook optimization job."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_candidate(
        self, candidate: PlaybookOptimizationCandidate
    ) -> PlaybookOptimizationCandidate:
        """Persist an optimizer candidate and return it with id populated."""
        raise NotImplementedError

    @abstractmethod
    def list_playbook_optimization_candidates(
        self, job_id: int
    ) -> list[PlaybookOptimizationCandidate]:
        """List optimizer candidates for a job."""
        raise NotImplementedError

    @abstractmethod
    def update_playbook_optimization_candidate(
        self,
        candidate_id: int,
        *,
        aggregate_score: float | None = None,
        is_winner: bool | None = None,
    ) -> None:
        """Update mutable optimizer candidate result fields."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_evaluation(
        self, evaluation: PlaybookOptimizationEvaluation
    ) -> PlaybookOptimizationEvaluation:
        """Persist an optimizer evaluation and return it with id populated."""
        raise NotImplementedError

    @abstractmethod
    def list_playbook_optimization_evaluations(
        self, job_id: int
    ) -> list[PlaybookOptimizationEvaluation]:
        """List optimizer evaluations for a job."""
        raise NotImplementedError

    @abstractmethod
    def insert_playbook_optimization_event(
        self, event: PlaybookOptimizationEvent
    ) -> PlaybookOptimizationEvent:
        """Persist an optimizer callback/event and return it with id populated."""
        raise NotImplementedError
