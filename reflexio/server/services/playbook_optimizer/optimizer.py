from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.aggregation_trigger import (
    maybe_trigger_user_playbook_aggregation,
)
from reflexio.server.services.playbook.publication import (
    PUBLICATION_PROJECTION_JSON_METADATA_KEY,
    PUBLICATION_PROOF_JSON_METADATA_KEY,
    PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY,
    PublicationRequest,
    PublicationResult,
    UserPlaybookPublicationService,
)
from reflexio.server.services.storage.error import OptimizationJobLeaseLiveError
from reflexio.server.tracing import sentry_tags

from .assistant_webhook import AssistantCallable, LocalScriptAssistant, WebhookAssistant
from .gepa_adapter import PLAYBOOK_CONTENT_COMPONENT, ReflexioPlaybookGEPAAdapter
from .gepa_publication import (
    GEPA_PUBLICATION_AUTHORITY_METADATA_KEY,
    GEPAUserPlaybookDecisionVerifier,
    build_gepa_decision_proof,
    build_gepa_search_projection,
    parse_gepa_decision_proof,
    parse_gepa_search_projection,
)
from .judge import PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID, PairwiseJudge
from .models import ScenarioWindow
from .rollout import MultiTurnRollout
from .scenario_resolver import ScenarioResolver

logger = logging.getLogger(__name__)

# Prefix used when constructing the run-scoped lineage request_id from a job id.
# Shared with tests so the format is defined in one place.
_OPTIMIZER_RUN_ID_PREFIX = "optjob_"

_GEPA_PUBLICATION_LEASE_SECONDS = 60


def optimizer_run_request_id(job_id: int) -> str:
    """Return the lineage request_id for a playbook optimizer run.

    Args:
        job_id (int): The ``PlaybookOptimizationJob.job_id`` for this run.

    Returns:
        str: A non-empty, job-scoped request id of the form ``optjob_<job_id>``.
    """
    return f"{_OPTIMIZER_RUN_ID_PREFIX}{job_id}"


class PlaybookOptimizationTarget(BaseModel):
    """A single playbook (agent or user) the optimizer should try to improve."""

    kind: Literal["agent_playbook", "user_playbook"]
    target_id: int


# Outcome of one ``optimize()`` invocation. Used by the scheduler to drive its
# abort-cooldown logic — only ``aborted`` (assistant-backend faults) trips the
# cooldown; ``failed`` (config / GEPA bugs) does not.
PlaybookOptimizationRunStatus = Literal["skipped", "completed", "failed", "aborted"]


class PlaybookOptimizer:
    """Orchestrates one GEPA-driven optimization run for a single playbook.

    The optimizer:

    1. Picks an assistant backend from config (``_create_assistant``).
    2. Loads the incumbent playbook and the source interaction windows that
       produced it.
    3. Runs ``gepa.optimize`` with a ``ReflexioPlaybookGEPAAdapter`` — the
       adapter is what actually calls the assistant and the LLM judge.
    4. Persists every candidate, evaluation, and GEPA event for offline
       inspection.
    5. Optionally commits a successor playbook if the winner clears the
       configured score / Likert / per-window thresholds.
    """

    def __init__(self, request_context: RequestContext, llm_client: LiteLLMClient):
        self.request_context = request_context
        if request_context.storage is None:
            raise ValueError("Playbook optimizer requires storage")
        self.storage = request_context.storage
        self.llm_client = llm_client
        self.resolver = ScenarioResolver(self.storage)

    def optimize(
        self, target: PlaybookOptimizationTarget
    ) -> PlaybookOptimizationRunStatus:
        """Run a full optimization pass for ``target``.

        Returns the run status so the scheduler can react (e.g. enter abort
        cooldown when the assistant backend repeatedly fails). Side-effects:
        creates one ``playbook_optimization_jobs`` row and possibly archives
        the incumbent in favour of a successor playbook.
        """
        if target.kind == "user_playbook":
            recovered = self._recover_gepa_user_playbook_publication(target)
            if recovered is not None:
                return recovered
        config = self._config()
        if not self._enabled_for_target(config, target):
            return "skipped"
        # Backend selection happens before any storage work so an unconfigured
        # optimizer short-circuits cheaply — useful in tests and dev setups.
        assistant = self._create_assistant(config)
        if assistant is None:
            logger.info(
                "Skipping playbook optimization: no assistant backend configured"
            )
            return "skipped"

        incumbent = self._load_incumbent(target)
        if incumbent is None:
            return "skipped"
        windows = self._resolve_windows(target, config)
        if not windows:
            return "skipped"
        train_windows, validation_windows = _split_train_validation_windows(
            windows, config
        )
        if len(validation_windows) < config.min_commit_windows:
            logger.info(
                "Skipping playbook optimization: validation windows below "
                "min_commit_windows target_kind=%s target_id=%d "
                "validation_windows=%d min_commit_windows=%d",
                target.kind,
                target.target_id,
                len(validation_windows),
                config.min_commit_windows,
            )
            return "skipped"
        if not _can_adopt_winner(target, config):
            logger.info(
                "Skipping playbook optimization: no configured adoption path "
                "target_kind=%s target_id=%d",
                target.kind,
                target.target_id,
            )
            return "skipped"
        split_metadata = _split_metadata(
            windows,
            train_windows,
            validation_windows,
            config=config,
            assistant=assistant,
            llm_client=self.llm_client,
        )

        job = self.storage.create_playbook_optimization_job(
            PlaybookOptimizationJob(
                optimizer_kind="gepa",
                target_kind=target.kind,
                target_id=target.target_id,
                status="running",
                metadata_json=json.dumps(split_metadata, ensure_ascii=False),
                attempt_key=(
                    f"gepa-user-{uuid.uuid4().hex}"
                    if target.kind == "user_playbook"
                    else None
                ),
            )
        )
        run_request_id = optimizer_run_request_id(job.job_id)
        logger.info(
            "event=playbook_optimization_start job_id=%d candidate_id=none "
            "target_kind=%s target_id=%d "
            "windows=%d backend=%s max_metric_calls=%d max_turns=%d",
            job.job_id,
            target.kind,
            target.target_id,
            len(windows),
            type(assistant).__name__,
            config.max_metric_calls,
            config.max_turns,
        )

        adapter = ReflexioPlaybookGEPAAdapter(
            storage=self.storage,
            job_id=job.job_id,
            target_kind=target.kind,
            target_id=target.target_id,
            incumbent=incumbent,
            rollout=MultiTurnRollout(assistant),
            judge=PairwiseJudge(
                self.request_context,
                self.llm_client,
                config.reflection_model,
            ),
            max_turns=config.max_turns,
        )
        try:
            result = self._run_gepa(
                config,
                incumbent.content,
                train_windows,
                validation_windows,
                adapter,
            )
        except Exception as exc:
            with sentry_tags(
                subsystem="playbook_optimizer",
                op="run",
                org_id=self.request_context.org_id,
                job_id=job.job_id,
                target_kind=target.kind,
                target_id=target.target_id,
                error_type=type(exc).__name__,
            ):
                logger.exception("Playbook optimization failed")
            self.storage.update_playbook_optimization_job(
                job.job_id, status="failed", decision_reason=str(exc)
            )
            return "failed"

        best = result.best_candidate
        best_content = (
            best.get(PLAYBOOK_CONTENT_COMPONENT, incumbent.content)
            if isinstance(best, dict)
            else str(best)
        )
        best_score = float(result.val_aggregate_scores[result.best_idx])
        winner_candidate = adapter._ensure_candidate(best_content)
        self.storage.update_playbook_optimization_candidate(
            winner_candidate.candidate_id,
            aggregate_score=best_score,
            is_winner=True,
        )
        logger.info(
            "event=gepa_run_end job_id=%d candidate_id=%d "
            "best_idx=%d best_score=%.3f num_candidates=%d",
            job.job_id,
            winner_candidate.candidate_id,
            result.best_idx,
            best_score,
            len(result.val_aggregate_scores),
        )

        if self._has_aborted_evaluations(job.job_id):
            logger.warning(
                "event=playbook_optimization_aborted job_id=%d candidate_id=%d "
                "reason='assistant backend aborted one or more evaluations' best_score=%.3f",
                job.job_id,
                winner_candidate.candidate_id,
                best_score,
            )
            self.storage.update_playbook_optimization_job(
                job.job_id,
                status="failed",
                best_candidate_id=winner_candidate.candidate_id,
                decision_reason="assistant backend aborted one or more evaluations",
                metadata_json=json.dumps(
                    _result_metadata(result, split_metadata),
                    ensure_ascii=False,
                    default=str,
                ),
            )
            return "aborted"

        if not self._passes_commit_thresholds(
            job.job_id,
            winner_candidate.candidate_id,
            validation_windows,
            best_score,
            config,
        ):
            logger.info(
                "event=playbook_optimization_no_commit job_id=%d candidate_id=%d "
                "best_score=%.3f reason='did not pass commit thresholds'",
                job.job_id,
                winner_candidate.candidate_id,
                best_score,
            )
            self.storage.update_playbook_optimization_job(
                job.job_id,
                status="completed",
                best_candidate_id=winner_candidate.candidate_id,
                decision_reason="best candidate did not pass commit thresholds",
                metadata_json=json.dumps(
                    _result_metadata(result, split_metadata),
                    ensure_ascii=False,
                    default=str,
                ),
            )
            return "completed"

        result_metadata = _result_metadata(result, split_metadata)
        if target.kind == "user_playbook":
            publication_result = self._publish_user_playbook_winner(
                job=job,
                incumbent_id=target.target_id,
                winner_candidate_id=winner_candidate.candidate_id,
                result_metadata=result_metadata,
                run_request_id=run_request_id,
            )
            successor_id = publication_result.successor_user_playbook_id
        else:
            successor_id = self._commit_if_allowed(
                target, incumbent, best_content, config, run_request_id
            )
        logger.info(
            "event=playbook_optimization_committed job_id=%d candidate_id=%d "
            "successor_target_id=%s best_score=%.3f",
            job.job_id,
            winner_candidate.candidate_id,
            successor_id if successor_id is not None else "none",
            best_score,
        )
        if target.kind == "agent_playbook":
            self.storage.update_playbook_optimization_job(
                job.job_id,
                status="completed",
                best_candidate_id=winner_candidate.candidate_id,
                successor_target_id=successor_id,
                decision_reason=(
                    "committed" if successor_id else "winner persisted only"
                ),
                metadata_json=json.dumps(
                    result_metadata,
                    ensure_ascii=False,
                    default=str,
                ),
            )
        return "completed"

    def _run_gepa(
        self,
        config: PlaybookOptimizerConfig,
        seed_content: str,
        train_windows: list[ScenarioWindow],
        validation_windows: list[ScenarioWindow],
        adapter: ReflexioPlaybookGEPAAdapter,
    ) -> Any:
        from gepa.api import optimize as gepa_optimize
        from gepa.utils.stop_condition import ScoreThresholdStopper

        reflection_lm = config.reflection_model or self.llm_client.config.model
        return gepa_optimize(
            seed_candidate={PLAYBOOK_CONTENT_COMPONENT: seed_content},
            trainset=train_windows,
            valset=None
            if _same_window_sequence(train_windows, validation_windows)
            else validation_windows,
            adapter=adapter,
            reflection_lm=reflection_lm,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=config.reflection_minibatch_size,
            use_merge=config.use_merge,
            max_merge_invocations=config.max_merge_invocations,
            max_metric_calls=config.max_metric_calls,
            stop_callbacks=[ScoreThresholdStopper(config.early_stop_score)],
            raise_on_exception=False,
            display_progress_bar=False,
            cache_evaluation=True,
            callbacks=cast(Any, [_GEPAStorageCallback(self.storage, adapter.job_id)]),
        )

    def _config(self) -> PlaybookOptimizerConfig:
        config = self.request_context.configurator.get_config()
        return config.playbook_optimizer_config

    def _create_assistant(
        self, config: PlaybookOptimizerConfig
    ) -> AssistantCallable | None:
        """Pick the assistant backend implied by config.

        ``webhook_url`` and ``assistant_script_path`` are mutually exclusive
        (validated in ``PlaybookOptimizerConfig``), so this is a simple two-way
        dispatch. Returning ``None`` means the optimizer is enabled but has no
        backend configured — ``optimize()`` treats that as a no-op.

        The ``webhook_*`` retry/timeout fields govern *both* backends; the
        prefix is preserved only for config-schema compatibility.
        """
        if config.webhook_url:
            return WebhookAssistant(
                url=config.webhook_url,
                auth_header=config.webhook_auth_header,
                timeout_s=config.webhook_timeout_seconds,
                max_retries=config.webhook_max_retries,
                backoff_base_s=config.webhook_backoff_base_seconds,
            )
        if config.assistant_script_path:
            if not _is_executable_file(config.assistant_script_path):
                logger.warning(
                    "Skipping playbook optimization: assistant_script_path is not "
                    "an executable file path=%s",
                    config.assistant_script_path,
                )
                return None
            return LocalScriptAssistant(
                script_path=config.assistant_script_path,
                script_args=config.assistant_script_args,
                timeout_s=config.webhook_timeout_seconds,
                max_retries=config.webhook_max_retries,
                backoff_base_s=config.webhook_backoff_base_seconds,
            )
        return None

    def _enabled_for_target(
        self, config: PlaybookOptimizerConfig, target: PlaybookOptimizationTarget
    ) -> bool:
        if not config.enabled:
            return False
        if target.kind == "agent_playbook":
            return config.optimize_agent_playbooks
        return config.optimize_user_playbooks

    def _load_incumbent(
        self, target: PlaybookOptimizationTarget
    ) -> AgentPlaybook | None:
        if target.kind == "agent_playbook":
            playbook = self.storage.get_agent_playbook_by_id(target.target_id)
            if (
                playbook is None
                or playbook.status is not None
                or playbook.playbook_status != PlaybookStatus.PENDING
            ):
                return None
            return playbook
        user_playbook = self.storage.get_user_playbook_by_id(target.target_id)
        if user_playbook is None or user_playbook.status is not None:
            return None
        return _agent_like_playbook(user_playbook)

    def _resolve_windows(
        self, target: PlaybookOptimizationTarget, config: PlaybookOptimizerConfig
    ) -> list:
        if target.kind == "agent_playbook":
            return self.resolver.for_agent_playbook(target.target_id)
        if not config.optimize_user_playbooks:
            return []
        return self.resolver.for_user_playbook(target.target_id)

    def _passes_commit_thresholds(
        self,
        job_id: int,
        candidate_id: int,
        validation_windows: list[ScenarioWindow],
        best_score: float,
        config: PlaybookOptimizerConfig,
    ) -> bool:
        if best_score < config.min_commit_score:
            return False
        evaluations = self.storage.list_playbook_optimization_evaluations(job_id)
        validation_keys = {_window_eval_key(window) for window in validation_windows}
        winning_windows = {
            _evaluation_key(
                evaluation.scenario_user_playbook_id,
                evaluation.source_interaction_ids,
            )
            for evaluation in evaluations
            if evaluation.candidate_id == candidate_id
            and _evaluation_key(
                evaluation.scenario_user_playbook_id,
                evaluation.source_interaction_ids,
            )
            in validation_keys
            and evaluation.verdict == "candidate"
            and evaluation.score >= config.min_commit_score
            and evaluation.likert >= config.min_commit_likert
        }
        return len(winning_windows) >= config.min_commit_windows

    def _has_aborted_evaluations(self, job_id: int) -> bool:
        evaluations = self.storage.list_playbook_optimization_evaluations(job_id)
        return any(evaluation.verdict == "aborted" for evaluation in evaluations)

    def _publish_user_playbook_winner(
        self,
        *,
        job: PlaybookOptimizationJob,
        incumbent_id: int,
        winner_candidate_id: int,
        result_metadata: dict[str, Any],
        run_request_id: str,
    ) -> PublicationResult:
        incumbent = self.storage.get_user_playbook_by_id(incumbent_id)
        if incumbent is None:
            raise ValueError("GEPA publication incumbent no longer exists")
        winner = next(
            (
                candidate
                for candidate in self.storage.list_playbook_optimization_candidates(
                    job.job_id
                )
                if candidate.candidate_id == winner_candidate_id
            ),
            None,
        )
        if winner is None:
            raise ValueError("GEPA publication winner no longer exists")
        projection = build_gepa_search_projection(
            self.storage, incumbent, winner.content
        )
        subject_epochs_json = self.storage.get_user_playbook_publication_subject_epochs(
            incumbent_id
        )
        evaluations = self.storage.list_playbook_optimization_evaluations(job.job_id)
        proof = build_gepa_decision_proof(
            job=job,
            winner=winner,
            evaluations=evaluations,
            metadata=result_metadata,
            subject_epochs_json=subject_epochs_json,
            projection_digest=projection.digest,
        )
        owner = f"gepa-publication-{job.job_id}"
        durable_job = self.storage.prepare_gepa_user_playbook_publication(
            job_id=job.job_id,
            owner=owner,
            lease_seconds=_GEPA_PUBLICATION_LEASE_SECONDS,
            winner_candidate_id=winner.candidate_id,
            candidate_content_digest=projection.candidate_content_digest,
            search_projection_digest=projection.digest,
            publication_proof_digest=proof.digest,
            projection_json=projection.canonical_json,
            decision_proof_json=proof.canonical_json,
            subject_epochs_json=subject_epochs_json,
            metadata_json=json.dumps(
                result_metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ),
        )
        if durable_job.attempt_key is None:
            raise ValueError("GEPA publication attempt identity is missing")
        if durable_job.lease_owner is None or durable_job.lease_fence <= 0:
            raise ValueError("GEPA publication worker claim is missing")
        service = UserPlaybookPublicationService(
            self.storage,
            GEPAUserPlaybookDecisionVerifier(self.storage),
        )
        publication_claim = service.claim(
            job_id=job.job_id,
            owner=durable_job.lease_owner,
            worker_fence=durable_job.lease_fence,
        )
        request = PublicationRequest(
            optimizer_kind="gepa",
            job_id=job.job_id,
            attempt_key=durable_job.attempt_key,
            publication_claim=publication_claim,
            worker_fence=durable_job.lease_fence,
            incumbent_user_playbook_id=incumbent_id,
            revised_content=winner.content,
            projection=projection,
            decision_proof=proof,
            subject_epochs_json=subject_epochs_json,
            request_id=run_request_id,
        )
        return self._publish_prepared_user_playbook_request(service, request)

    def _recover_gepa_user_playbook_publication(
        self, target: PlaybookOptimizationTarget
    ) -> PlaybookOptimizationRunStatus | None:
        owner = f"gepa-publication-recovery-{target.target_id}"
        try:
            durable_job = self.storage.reclaim_gepa_user_playbook_publishing_job(
                target.target_id,
                owner,
                _GEPA_PUBLICATION_LEASE_SECONDS,
            )
        except OptimizationJobLeaseLiveError:
            job = self.storage.get_unconsumed_gepa_user_playbook_publishing_job(
                target.target_id
            )
            logger.info(
                "Skipping playbook optimization: GEPA publication lease is live "
                "job_id=%s target_id=%d",
                job.job_id if job is not None else "unknown",
                target.target_id,
            )
            return "skipped"
        if durable_job is None:
            return None
        if durable_job.attempt_key is None:
            raise ValueError("GEPA publication job disappeared during recovery")
        if durable_job.lease_owner is None or durable_job.lease_fence <= 0:
            raise ValueError("GEPA publication recovery claim is missing")
        metadata = json.loads(durable_job.metadata_json)
        projection_json = metadata.get(PUBLICATION_PROJECTION_JSON_METADATA_KEY)
        proof_json = metadata.get(PUBLICATION_PROOF_JSON_METADATA_KEY)
        subject_epochs = metadata.get(PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY)
        if (
            not isinstance(projection_json, str)
            or not isinstance(proof_json, str)
            or not isinstance(subject_epochs, dict)
        ):
            raise ValueError("GEPA publication durable bytes are missing")
        winner = next(
            (
                candidate
                for candidate in self.storage.list_playbook_optimization_candidates(
                    durable_job.job_id
                )
                if candidate.candidate_id == durable_job.best_candidate_id
            ),
            None,
        )
        if winner is None:
            raise ValueError("GEPA publication winner is missing during recovery")
        projection = parse_gepa_search_projection(projection_json)
        proof = parse_gepa_decision_proof(proof_json)
        service = UserPlaybookPublicationService(
            self.storage,
            GEPAUserPlaybookDecisionVerifier(self.storage),
        )
        publication_claim = service.claim(
            job_id=durable_job.job_id,
            owner=durable_job.lease_owner,
            worker_fence=durable_job.lease_fence,
        )
        request = PublicationRequest(
            optimizer_kind="gepa",
            job_id=durable_job.job_id,
            attempt_key=durable_job.attempt_key,
            publication_claim=publication_claim,
            worker_fence=durable_job.lease_fence,
            incumbent_user_playbook_id=target.target_id,
            revised_content=winner.content,
            projection=projection,
            decision_proof=proof,
            subject_epochs_json=json.dumps(
                subject_epochs,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            request_id=optimizer_run_request_id(durable_job.job_id),
        )
        self._publish_prepared_user_playbook_request(service, request)
        return "completed"

    def _publish_prepared_user_playbook_request(
        self,
        service: UserPlaybookPublicationService,
        request: PublicationRequest,
    ) -> PublicationResult:
        try:
            publication_result = service.publish(request)
        except Exception:
            committed = service.load_committed(request.job_id)
            if committed is None:
                raise
            publication_result = committed

        if publication_result.outcome == "applied":
            successor_id = publication_result.successor_user_playbook_id
            if successor_id is None:
                raise RuntimeError("applied GEPA publication has no successor")
            try:
                successor = self.storage.get_user_playbook_by_id(successor_id)
                if successor is not None and successor.agent_version:
                    maybe_trigger_user_playbook_aggregation(
                        request_context=self.request_context,
                        llm_client=self.llm_client,
                        agent_version=successor.agent_version,
                        reason="playbook_optimizer",
                    )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "playbook_optimizer aggregation trigger failed after successor commit",
                    extra={"successor_id": successor_id},
                )
        return publication_result

    def _commit_if_allowed(
        self,
        target: PlaybookOptimizationTarget,
        incumbent: AgentPlaybook,
        best_content: str,
        config: PlaybookOptimizerConfig,
        run_request_id: str,
    ) -> int | None:
        # The optimize() entrypoint already gates on _can_adopt_winner before
        # creating the job, but check again so this stays correct if called
        # from elsewhere.
        if not _can_adopt_winner(target, config):
            return None
        if target.kind == "agent_playbook":
            source_windows = _source_windows_with_backfill(
                self.storage, target.target_id
            )
            current = self.storage.get_agent_playbook_by_id(target.target_id)
            if (
                current is None
                or current.status is not None
                or current.playbook_status != PlaybookStatus.PENDING
            ):
                return None
            metadata = _append_optimizer_metadata(
                incumbent.playbook_metadata, target.target_id
            )
            successor_id = _supersede_agent_playbook(
                self.storage,
                incumbent,
                best_content,
                "playbook_optimizer",
                playbook_metadata=metadata,
                request_id=run_request_id,
            )
            if successor_id is not None:
                self.storage.set_source_windows_for_agent_playbook(
                    successor_id, source_windows
                )
            return successor_id
        raise ValueError("GEPA user playbooks require durable atomic publication")


def _agent_like_playbook(playbook: UserPlaybook) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=playbook.user_playbook_id,
        playbook_name=playbook.playbook_name,
        agent_version=playbook.agent_version,
        content=playbook.content,
        trigger=playbook.trigger,
        rationale=playbook.rationale,
        playbook_status=PlaybookStatus.PENDING,
        status=playbook.status,
    )


def _can_adopt_winner(
    target: PlaybookOptimizationTarget,
    config: PlaybookOptimizerConfig,
) -> bool:
    if target.kind == "agent_playbook":
        return config.auto_update_pending_agent_playbooks
    return config.auto_update_user_playbooks


def _append_optimizer_metadata(existing: str, predecessor_id: int) -> str:
    suffix = f"optimized_from_agent_playbook_id={predecessor_id}"
    if not existing:
        return suffix
    return f"{existing}; {suffix}"


class _LostAgentSupersedeRaceError(Exception):
    """Internal rollback signal for an agent-playbook successor race."""


def _supersede_agent_playbook(
    storage: Any,
    incumbent: AgentPlaybook,
    best_content: str,
    source: str,
    playbook_metadata: str = "",
    *,
    request_id: str,
) -> int | None:
    """Insert an agent-playbook successor then atomically supersede the incumbent.

    Returns the new ``agent_playbook_id`` on success, or ``None`` when the
    incumbent is no longer CURRENT.

    Args:
        storage: A storage instance implementing ``save_agent_playbooks``,
            ``supersede_record``, and ``commit_scope``.
        incumbent: The current agent playbook to replace.
        best_content: Content for the successor playbook.
        source: Provenance label written to the lineage event actor field.
        playbook_metadata: Optional metadata string for the successor row.
        request_id: Run-scoped correlation id for the lineage event. Must be
            non-empty; use the job-derived id from the calling optimizer run.

    Returns:
        int | None: ``agent_playbook_id`` of the successor, or ``None`` if the
            incumbent was not CURRENT.

    Raises:
        ValueError: If ``request_id`` is empty or None.
    """
    if not request_id:
        raise ValueError(
            "_supersede_agent_playbook: request_id must be non-empty (run-correlation id)"
        )
    successor = incumbent.model_copy(
        update={
            "agent_playbook_id": 0,
            "content": best_content,
            "status": None,
            "playbook_status": PlaybookStatus.PENDING,
            "playbook_metadata": playbook_metadata,
        }
    )
    ctx = LineageContext(
        op_kind="revise",
        actor=source,
        request_id=request_id,
    )
    try:
        with storage.commit_scope():
            saved = storage.save_agent_playbooks([successor])
            if not saved or not saved[0].agent_playbook_id:
                raise _LostAgentSupersedeRaceError
            successor_id = saved[0].agent_playbook_id
            if not storage.supersede_record(
                entity_type="agent_playbook",
                incumbent_id=str(incumbent.agent_playbook_id),
                successor_id=str(successor_id),
                context=ctx,
            ):
                raise _LostAgentSupersedeRaceError
    except _LostAgentSupersedeRaceError:
        successor.agent_playbook_id = 0
        return None
    return successor_id


class _GEPAStorageCallback:
    def __init__(self, storage: Any, job_id: int) -> None:
        self.storage = storage
        self.job_id = job_id

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("on_"):
            raise AttributeError(name)

        def _record(event: dict[str, Any]) -> None:
            self.storage.insert_playbook_optimization_event(
                PlaybookOptimizationEvent(
                    job_id=self.job_id,
                    event_type=name.removeprefix("on_"),
                    payload_json=json.dumps(
                        _safe_event_payload(event), ensure_ascii=False, default=str
                    ),
                )
            )

        return _record


def _safe_event_payload(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _safe_event_payload(v, depth + 1)
            for k, v in value.items()
            if k != "final_state"
        }
    if isinstance(value, list | tuple | set):
        return [_safe_event_payload(v, depth + 1) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _safe_event_payload(value.model_dump(), depth + 1)
    return str(value)


def _split_train_validation_windows(
    windows: list[ScenarioWindow],
    config: PlaybookOptimizerConfig,
) -> tuple[list[ScenarioWindow], list[ScenarioWindow]]:
    if len(windows) <= 1:
        return windows, windows

    ordered_windows = sorted(
        windows,
        key=lambda window: (
            window.user_playbook_id if window.user_playbook_id is not None else -1,
            tuple(window.source_interaction_ids),
        ),
    )
    validation_count = min(config.max_validation_windows, len(ordered_windows) - 1)
    validation_windows = ordered_windows[:validation_count]
    train_windows = ordered_windows[validation_count:]
    return train_windows, validation_windows


def _same_window_sequence(
    left: list[ScenarioWindow],
    right: list[ScenarioWindow],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        _window_eval_key(left_window) == _window_eval_key(right_window)
        for left_window, right_window in zip(left, right, strict=True)
    )


def _source_windows_with_backfill(
    storage: Any, agent_playbook_id: int
) -> list[AgentPlaybookSourceWindow]:
    source_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    missing_ids = [
        window.user_playbook_id
        for window in source_windows
        if not window.source_interaction_ids
    ]
    if not missing_ids:
        return source_windows

    playbooks = storage.get_user_playbooks_by_ids_any_user(
        missing_ids, status_filter=None
    )
    source_ids_by_playbook_id = {
        playbook.user_playbook_id: playbook.source_interaction_ids
        for playbook in playbooks
        if playbook.source_interaction_ids
    }
    return [
        window
        if window.source_interaction_ids
        else AgentPlaybookSourceWindow(
            user_playbook_id=window.user_playbook_id,
            source_interaction_ids=list(
                source_ids_by_playbook_id.get(window.user_playbook_id, [])
            ),
        )
        for window in source_windows
    ]


def _split_metadata(
    windows: list[ScenarioWindow],
    train_windows: list[ScenarioWindow],
    validation_windows: list[ScenarioWindow],
    *,
    config: PlaybookOptimizerConfig,
    assistant: AssistantCallable,
    llm_client: LiteLLMClient,
) -> dict[str, Any]:
    return {
        GEPA_PUBLICATION_AUTHORITY_METADATA_KEY: _gepa_publication_authority(
            config=config,
            train_windows=train_windows,
            validation_windows=validation_windows,
            assistant=assistant,
            llm_client=llm_client,
        ),
        "source_window_count": len(windows),
        "train_window_count": len(train_windows),
        "train_windows": [
            {
                "scenario_user_playbook_id": window.user_playbook_id,
                "source_interaction_ids": list(window.source_interaction_ids),
            }
            for window in train_windows
        ],
        "validation_window_count": len(validation_windows),
        "validation_scenario_user_playbook_ids": [
            window.user_playbook_id for window in validation_windows
        ],
        "validation_windows": [
            {
                "scenario_user_playbook_id": window.user_playbook_id,
                "source_interaction_ids": list(window.source_interaction_ids),
            }
            for window in validation_windows
        ],
    }


def _gepa_publication_authority(
    *,
    config: PlaybookOptimizerConfig,
    train_windows: list[ScenarioWindow],
    validation_windows: list[ScenarioWindow],
    assistant: AssistantCallable,
    llm_client: LiteLLMClient,
) -> dict[str, Any]:
    judge_model_id = config.reflection_model or llm_client.config.model
    return {
        "adoption_policy": {
            "auto_update_user_playbooks": config.auto_update_user_playbooks,
            "min_commit_likert": config.min_commit_likert,
            "min_commit_score": _decimal_string(config.min_commit_score),
            "min_commit_windows": config.min_commit_windows,
        },
        "backend_identity": _assistant_backend_identity(config, assistant),
        "evaluator_identity": {
            "judge_class": _code_identity(PairwiseJudge),
            "judge_code_digest": _code_digest(PairwiseJudge),
            "judge_model_id": judge_model_id,
            "judge_prompt_id": PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID,
        },
        "optimizer_identity": {
            "adapter_class": _code_identity(ReflexioPlaybookGEPAAdapter),
            "adapter_code_digest": _code_digest(ReflexioPlaybookGEPAAdapter),
            "rollout_class": _code_identity(MultiTurnRollout),
            "rollout_code_digest": _code_digest(MultiTurnRollout),
        },
        "train_manifest": {
            "digest": _window_manifest_digest(train_windows),
            "windows": [_window_manifest_item(window) for window in train_windows],
        },
        "validation_manifest": {
            "digest": _window_manifest_digest(validation_windows),
            "windows": [
                {
                    **_window_manifest_item(window),
                    "min_commit_likert": config.min_commit_likert,
                    "min_commit_score": _decimal_string(config.min_commit_score),
                }
                for window in validation_windows
            ],
        },
    }


def _assistant_backend_identity(
    config: PlaybookOptimizerConfig,
    assistant: AssistantCallable,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "backoff_base_seconds": _decimal_string(config.webhook_backoff_base_seconds),
        "backend_class": _code_identity(type(assistant)),
        "backend_class_code_digest": _code_digest(type(assistant)),
        "max_retries": config.webhook_max_retries,
        "timeout_seconds": config.webhook_timeout_seconds,
    }
    if config.webhook_url:
        identity["backend_kind"] = "webhook"
        identity["webhook_auth_configured"] = bool(config.webhook_auth_header)
        identity["webhook_url_digest"] = _text_digest(config.webhook_url)
    elif config.assistant_script_path:
        identity["backend_kind"] = "local_script"
        identity["script_args_digest"] = _json_digest(config.assistant_script_args)
        identity["script_content_digest"] = sha256(
            Path(config.assistant_script_path).read_bytes()
        ).hexdigest()
    else:
        identity["backend_kind"] = "none"
    return identity


def _window_manifest_item(window: ScenarioWindow) -> dict[str, Any]:
    return {
        "scenario_user_playbook_id": window.user_playbook_id,
        "source_interaction_ids": list(window.source_interaction_ids),
    }


def _window_manifest_digest(windows: list[ScenarioWindow]) -> str:
    return _json_digest([_window_manifest_item(window) for window in windows])


def _json_digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _code_identity(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _code_digest(value: type[Any]) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = _code_identity(value)
    return _text_digest(source)


def _decimal_string(value: float) -> str:
    return format(float(value), ".15g")


def _is_executable_file(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_file() and os.access(candidate, os.X_OK)


def _result_metadata(result: Any, split_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = result.to_dict()
    if not isinstance(metadata, dict):
        metadata = {"gepa_result": metadata}
    metadata.update(split_metadata)
    return metadata


def _window_eval_key(window: ScenarioWindow) -> tuple[int | None, tuple[int, ...]]:
    return window.user_playbook_id, tuple(window.source_interaction_ids)


def _evaluation_key(
    scenario_user_playbook_id: int | None, source_interaction_ids: list[int]
) -> tuple[int | None, tuple[int, ...]]:
    return scenario_user_playbook_id, tuple(source_interaction_ids)
