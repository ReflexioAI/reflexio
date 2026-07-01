"""Abstract playbook optimization job store declarations."""

from abc import abstractmethod

from reflexio.models.api_schema.domain import (
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
