from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
)
from reflexio.server.services.storage.storage_base import BaseStorage

from .assistant_webhook import AssistantFailedError
from .judge import PairwiseJudge
from .models import (
    CandidateEvaluationOutput,
    EvaluationTrajectory,
    ScenarioWindow,
)
from .rollout import MultiTurnRollout

if TYPE_CHECKING:
    from gepa.core.adapter import EvaluationBatch, ProposalFn

logger = logging.getLogger(__name__)

PLAYBOOK_CONTENT_COMPONENT = "playbook_content"
_EvaluationCacheKey = tuple[str, int | None, tuple[int, ...]]


def _evaluation_cache_key(
    candidate_content: str, window: ScenarioWindow
) -> _EvaluationCacheKey:
    return (
        candidate_content,
        window.user_playbook_id,
        tuple(window.source_interaction_ids),
    )


class ReflexioPlaybookGEPAAdapter:
    """GEPA adapter for paired real-assistant playbook rollouts.

    GEPA proposes candidate playbook content; this adapter scores each
    candidate by running the *same* user turns twice — once with the
    incumbent playbook injected and once with the candidate — then asking
    an LLM judge which conversation is better.

    Per ``(candidate, window)`` evaluation it:

    1. Persists (or reuses) a ``PlaybookOptimizationCandidate`` row keyed by
       content, so duplicate proposals share an id.
    2. Runs two ``MultiTurnRollout`` invocations against the configured
       assistant backend.
    3. Calls ``PairwiseJudge.judge`` to produce a verdict / score / Likert /
       ASI rationale.
    4. Inserts a ``PlaybookOptimizationEvaluation`` row for offline review.

    Backend faults (any ``AssistantFailedError``) are caught in
    ``_evaluate_one`` and recorded as ``verdict='aborted'`` so the GEPA
    search loop can keep going. The optimizer reads those rows back to
    decide whether the run as a whole should fail.
    """

    def __init__(
        self,
        *,
        storage: BaseStorage,
        job_id: int,
        target_kind: str,
        target_id: int,
        incumbent: AgentPlaybook,
        rollout: MultiTurnRollout,
        judge: PairwiseJudge,
        max_turns: int,
    ) -> None:
        self.storage = storage
        self.job_id = job_id
        self.target_kind = target_kind
        self.target_id = target_id
        self.incumbent = incumbent
        self.rollout = rollout
        self.judge = judge
        self.max_turns = max_turns
        self.propose_new_texts: ProposalFn | None = None
        self._candidate_ids_by_content: dict[str, int] = {}
        self._evaluation_cache: dict[
            _EvaluationCacheKey, CandidateEvaluationOutput
        ] = {}

    def evaluate(
        self,
        batch: list[ScenarioWindow],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EvaluationTrajectory, CandidateEvaluationOutput]:
        from gepa.core.adapter import EvaluationBatch

        candidate_content = candidate.get(PLAYBOOK_CONTENT_COMPONENT, "")
        persisted_candidate = self._ensure_candidate(candidate_content)
        outputs: list[CandidateEvaluationOutput] = []
        scores: list[float] = []
        trajectories: list[EvaluationTrajectory] = []

        for window in batch:
            cache_key = _evaluation_cache_key(candidate_content, window)
            cached = self._evaluation_cache.get(cache_key)
            if cached is not None:
                output = cached
                logger.info(
                    "event=playbook_optimization_evaluation_cache_hit job_id=%d "
                    "candidate_id=%d scenario_user_playbook_id=%s verdict=%s",
                    self.job_id,
                    persisted_candidate.candidate_id,
                    window.user_playbook_id,
                    output.verdict,
                )
            else:
                output = self._evaluate_one(window, candidate_content)
                self.storage.insert_playbook_optimization_evaluation(
                    PlaybookOptimizationEvaluation(
                        job_id=self.job_id,
                        candidate_id=persisted_candidate.candidate_id,
                        target_kind=self.target_kind,  # type: ignore[arg-type]
                        target_id=self.target_id,
                        scenario_user_playbook_id=window.user_playbook_id,
                        source_interaction_ids=window.source_interaction_ids,
                        score=output.score,
                        verdict=output.verdict,
                        likert=output.likert,
                        rationale=output.rationale,
                        asi_json=output.asi.model_dump_json(),
                        incumbent_rollout_json=output.incumbent_rollout.model_dump_json(),
                        candidate_rollout_json=output.candidate_rollout.model_dump_json(),
                    )
                )
                # Persist before caching so a failed insert does not leave a
                # phantom result behind; skip caching transient assistant-
                # backend aborts so the next iteration retries the rollout.
                if output.verdict != "aborted":
                    self._evaluation_cache[cache_key] = output
                logger.info(
                    "event=playbook_optimization_evaluation job_id=%d candidate_id=%d "
                    "scenario_user_playbook_id=%s verdict=%s score=%.3f likert=%d",
                    self.job_id,
                    persisted_candidate.candidate_id,
                    window.user_playbook_id,
                    output.verdict,
                    output.score,
                    output.likert,
                )
            outputs.append(output)
            scores.append(output.score)
            if capture_traces:
                trajectories.append(
                    EvaluationTrajectory(
                        scenario=window,
                        candidate_content=candidate_content,
                        output=output,
                    )
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EvaluationTrajectory, CandidateEvaluationOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        records: list[dict[str, Any]] = [
            {
                "Inputs": {
                    "source_interaction_ids": trajectory.scenario.source_interaction_ids,
                    "current_playbook_content": candidate.get(
                        PLAYBOOK_CONTENT_COMPONENT, ""
                    ),
                },
                "Generated Outputs": {
                    "incumbent_rollout": [
                        message.model_dump()
                        for message in trajectory.output.incumbent_rollout.messages
                    ],
                    "candidate_rollout": [
                        message.model_dump()
                        for message in trajectory.output.candidate_rollout.messages
                    ],
                },
                "Feedback": json.dumps(
                    {
                        "score": trajectory.output.score,
                        "verdict": trajectory.output.verdict,
                        "likert": trajectory.output.likert,
                        "rationale": trajectory.output.rationale,
                        "asi": trajectory.output.asi.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            }
            for trajectory in eval_batch.trajectories or []
        ]
        return dict.fromkeys(components_to_update, records)

    def _evaluate_one(
        self, window: ScenarioWindow, candidate_content: str
    ) -> CandidateEvaluationOutput:
        candidate_playbook = self.incumbent.model_copy(
            update={"content": candidate_content}
        )
        try:
            incumbent_rollout = self.rollout.run(
                window=window,
                playbook=self.incumbent,
                max_turns=self.max_turns,
            )
            candidate_rollout = self.rollout.run(
                window=window,
                playbook=candidate_playbook,
                max_turns=self.max_turns,
            )
            judged = self.judge.judge(
                window=window,
                incumbent=self.incumbent,
                candidate=candidate_playbook,
                incumbent_rollout=incumbent_rollout,
                candidate_rollout=candidate_rollout,
            )
            return CandidateEvaluationOutput(
                score=judged.score,
                verdict=judged.verdict,
                likert=judged.likert,
                rationale=judged.rationale,
                asi=judged.asi,
                incumbent_rollout=incumbent_rollout,
                candidate_rollout=candidate_rollout,
            )
        except AssistantFailedError as exc:
            logger.exception("Playbook optimizer assistant failed")
            return CandidateEvaluationOutput(
                score=0.0,
                verdict="aborted",
                likert=0,
                rationale=f"Assistant failed: {exc}",
            )
        except Exception as exc:
            logger.exception("Playbook optimizer evaluation failed")
            return CandidateEvaluationOutput(
                score=0.0,
                verdict="aborted",
                likert=0,
                rationale=f"Evaluation failed: {exc}",
            )

    def _ensure_candidate(self, content: str) -> PlaybookOptimizationCandidate:
        candidate_id = self._candidate_ids_by_content.get(content)
        if candidate_id is not None:
            existing = self.storage.list_playbook_optimization_candidates(self.job_id)
            for candidate in existing:
                if candidate.candidate_id == candidate_id:
                    return candidate
        candidate = self.storage.insert_playbook_optimization_candidate(
            PlaybookOptimizationCandidate(
                job_id=self.job_id,
                candidate_index=len(self._candidate_ids_by_content),
                content=content,
            )
        )
        self._candidate_ids_by_content[content] = candidate.candidate_id
        return candidate
