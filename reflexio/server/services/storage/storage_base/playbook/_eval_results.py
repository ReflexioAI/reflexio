"""Abstract agent success evaluation result store declarations."""

from abc import abstractmethod
from typing import Any

from reflexio.models.api_schema.domain import (
    AgentSuccessEvaluationResult,
    RetrievedLearningEvaluationResult,
)

from ..retrieved_learning_state import (
    DEFAULT_TRANSCRIPT_CHAR_LIMIT,
    BoundedRetrievedLearningSnapshot,
    RetrievedLearningCommitResult,
)


class AgentEvaluationResultStoreMixin:
    """Abstract agent success evaluation result store methods."""

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

    # ==============================
    # Retrieved-learning evaluation methods
    # ==============================
    #
    # Concurrency contract (see storage_base/retrieved_learning_state.py):
    # generation fencing + session-fingerprint CAS recomputed under the
    # replacement transaction's lock. No interaction write/delete path is
    # instrumented.

    @abstractmethod
    def begin_retrieved_learning_evaluation_run(
        self, user_id: str, session_id: str
    ) -> int:
        """Allocate the next evaluation generation for one session.

        Atomically increments (creating on first use) the ``generation``
        counter in the session's ``_operation_state`` row after asserting the
        governance subject is writable. Concurrent callers receive distinct
        generations.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.

        Returns:
            int: The allocated generation (monotonically increasing per
            session).
        """
        raise NotImplementedError

    @abstractmethod
    def load_bounded_retrieved_learning_snapshot(
        self,
        user_id: str,
        session_id: str,
        raw_ref_limit: int = 5_000,
        transcript_char_limit: int = DEFAULT_TRANSCRIPT_CHAR_LIMIT,
    ) -> BoundedRetrievedLearningSnapshot:
        """Load the bounded session projection for evaluation.

        Streams interaction rows (no ``Interaction`` objects, no embeddings or
        image encodings) and counts raw attachment occurrences before
        appending; on occurrence ``raw_ref_limit + 1`` the scan aborts with
        ``attachment_limit_exceeded=True``.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.
            raw_ref_limit (int): Maximum raw attachment occurrences to parse.
            transcript_char_limit (int): Maximum transcript characters retained.

        Returns:
            BoundedRetrievedLearningSnapshot: The bounded projection.
        """
        raise NotImplementedError

    @abstractmethod
    def get_matching_retrieved_learning_terminal_state(
        self, user_id: str, session_id: str, session_fingerprint: str
    ) -> dict[str, Any] | None:
        """Return the session's terminal evaluation state if still fresh.

        Implementations must use a writer transaction and recompute the live
        session fingerprint in that transaction. A match requires terminal
        status and equality among the supplied, persisted, and live
        fingerprints.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.
            session_fingerprint (str): Fingerprint of the current session.

        Returns:
            dict | None: The persisted state JSON on a match, else None.
        """
        raise NotImplementedError

    @abstractmethod
    def replace_retrieved_learning_evaluation_results(
        self,
        user_id: str,
        session_id: str,
        generation: int,
        session_fingerprint: str,
        proposed_status: str,
        diagnostics: dict[str, Any],
        results: list[RetrievedLearningEvaluationResult],
    ) -> RetrievedLearningCommitResult:
        """Atomically replace one session's evaluation result set.

        In one transaction: locks the session's state row, requires the
        current generation to equal ``generation`` (else ``superseded``),
        recomputes the session fingerprint from live rows and requires it to
        equal ``session_fingerprint`` (else ``stale``), rechecks every
        result's source row for retrieval eligibility (ineligible rows are
        dropped), then deletes the prior session set, inserts the filtered
        set, and persists completion state — all or nothing.

        When commit-time eligibility removes every candidate, prior rows are
        cleared and the final status is ``not_applicable`` regardless of
        ``proposed_status``.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.
            generation (int): Generation allocated by
                :meth:`begin_retrieved_learning_evaluation_run`.
            session_fingerprint (str): Fingerprint of the judged snapshot.
            proposed_status (str): ``"complete"`` or ``"degraded"``.
            diagnostics (dict): Sanitized counters/timestamps persisted in
                operation state (never prompt content or raw exception text).
            results (list[RetrievedLearningEvaluationResult]): Proposed rows.

        Returns:
            RetrievedLearningCommitResult: Disposition plus authoritative
            final status/count when applied.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_retrieved_learning_evaluation_run(
        self,
        user_id: str,
        session_id: str,
        generation: int,
        status: str,
        diagnostics: dict[str, Any],
    ) -> None:
        """Record a non-applied run outcome (``failed`` or ``pending``).

        Generation-guarded CAS: updates state only while ``generation`` is
        still current, so an older failure cannot overwrite a newer run.
        Result rows are never touched.

        Args:
            user_id (str): Session owner.
            session_id (str): Evaluated session.
            generation (int): Generation of the finishing run.
            status (str): ``"failed"`` or ``"pending"``.
            diagnostics (dict): Sanitized counters and error type.
        """
        raise NotImplementedError

    @abstractmethod
    def get_retrieved_learning_evaluation_results(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int = 100,
    ) -> list[RetrievedLearningEvaluationResult]:
        """Read persisted per-learning verdicts.

        Args:
            user_id (str, optional): Filter by session owner.
            session_id (str, optional): Filter by session.
            from_ts (int, optional): Target-interaction lower timestamp.
            to_ts (int, optional): Target-interaction upper timestamp.
            limit (int): Maximum rows, ordered by
                target interaction recency when a time filter is supplied,
                otherwise ``created_at DESC, result_id DESC``.

        Returns:
            list[RetrievedLearningEvaluationResult]: Matching rows.
        """
        raise NotImplementedError
