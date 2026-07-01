import logging
from abc import abstractmethod

from reflexio.models.api_schema.domain import (
    AgentSuccessEvaluationResult,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)

logger = logging.getLogger(__name__)

# Shared prefix for aggregate lineage event reasons.
# Consumers: storage_base (here), sqlite_storage/_playbook.py, and lib/_agent_playbook.py
# (which imports this constant to keep the parser and producers in sync).
AGGREGATE_REASON_PREFIX = "aggregate:"


class PlaybookMixin:
    """Mixin for playbook and agent success evaluation methods."""

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

    # ==============================
    # Agent Success Evaluation methods
    # ==============================

    @abstractmethod
    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        """Save agent success evaluation results to storage.

        Args:
            results (list[AgentSuccessEvaluationResult]): List of agent success evaluation results to save
        """
        raise NotImplementedError

    @abstractmethod
    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        """Get agent success evaluation results from storage.

        Args:
            limit (int): Maximum number of results to return
            agent_version (str, optional): The agent version to filter by. If None, returns all results.

        Returns:
            list[AgentSuccessEvaluationResult]: List of agent success evaluation result objects
        """
        raise NotImplementedError

    def get_agent_success_evaluation_results_in_window(
        self,
        from_ts: int,
        to_ts: int,
        agent_version: str | None = None,
        limit: int | None = None,
    ) -> list[AgentSuccessEvaluationResult]:
        """Return eval results in ``[from_ts, to_ts]``.

        Default implementation filters the existing latest-results method.
        SQL backends should override so callers do not depend on an arbitrary
        latest-row cap.
        """
        rows = self.get_agent_success_evaluation_results(
            limit=limit or 10_000,
            agent_version=agent_version,
        )
        return [r for r in rows if from_ts <= r.created_at <= to_ts]

    def get_agent_success_evaluation_result_ids(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> list[int]:
        """Return result ids for one eval identity tuple."""
        rows = self.get_agent_success_evaluation_results(
            limit=10_000,
            agent_version=agent_version,
        )
        return [
            r.result_id
            for r in rows
            if r.user_id == user_id
            and r.session_id == session_id
            and r.evaluation_name == evaluation_name
        ]

    @abstractmethod
    def delete_all_agent_success_evaluation_results(self) -> None:
        """Delete all agent success evaluation results from storage."""
        raise NotImplementedError

    @abstractmethod
    def delete_agent_success_evaluation_results_for_session(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> int:
        """Delete stored results for (user_id, session_id, evaluation_name, agent_version).

        Args:
            user_id (str): User whose session results to clear.
            session_id (str): Session whose results to clear.
            evaluation_name (str): Which evaluator's results to clear.
            agent_version (str): Agent version scope.

        Returns:
            int: Number of rows deleted.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_agent_success_evaluation_results_by_ids(
        self, result_ids: list[int]
    ) -> int:
        """Delete agent success eval results matching specific result_ids.

        Used by the regenerate flow to remove only the prior-run rows after the
        new rows have been saved durably (so an LLM/save failure cannot leave
        the session with zero rows).

        Args:
            result_ids (list[int]): Primary-key result_ids to delete.

        Returns:
            int: Number of rows deleted.
        """
        raise NotImplementedError
