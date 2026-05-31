from __future__ import annotations

import json
import logging
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
from reflexio.models.config_schema import PlaybookOptimizerConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.polarity_utils import (
    looks_negative,
    warn_if_polarity_content_mismatch,
)

from .assistant_webhook import AssistantCallable, LocalScriptAssistant, WebhookAssistant
from .gepa_adapter import PLAYBOOK_CONTENT_COMPONENT, ReflexioPlaybookGEPAAdapter
from .judge import PairwiseJudge
from .models import ScenarioWindow
from .rollout import MultiTurnRollout
from .scenario_resolver import ScenarioResolver

logger = logging.getLogger(__name__)


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
        split_metadata = _split_metadata(windows, train_windows, validation_windows)

        job = self.storage.create_playbook_optimization_job(
            PlaybookOptimizationJob(
                target_kind=target.kind,
                target_id=target.target_id,
                status="running",
                metadata_json=json.dumps(split_metadata, ensure_ascii=False),
            )
        )
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

        successor_id = self._commit_if_allowed(target, incumbent, best_content, config)
        logger.info(
            "event=playbook_optimization_committed job_id=%d candidate_id=%d "
            "successor_target_id=%s best_score=%.3f",
            job.job_id,
            winner_candidate.candidate_id,
            successor_id if successor_id is not None else "none",
            best_score,
        )
        self.storage.update_playbook_optimization_job(
            job.job_id,
            status="completed",
            best_candidate_id=winner_candidate.candidate_id,
            successor_target_id=successor_id,
            decision_reason="committed" if successor_id else "winner persisted only",
            metadata_json=json.dumps(
                _result_metadata(result, split_metadata),
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

    def _commit_if_allowed(
        self,
        target: PlaybookOptimizationTarget,
        incumbent: AgentPlaybook,
        best_content: str,
        config: PlaybookOptimizerConfig,
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
            self.storage.archive_agent_playbooks_by_ids([target.target_id])
            successor = incumbent.model_copy(
                update={
                    "agent_playbook_id": 0,
                    "content": best_content,
                    "status": None,
                    "playbook_status": PlaybookStatus.PENDING,
                    "playbook_metadata": _append_optimizer_metadata(
                        incumbent.playbook_metadata, target.target_id
                    ),
                }
            )
            saved = self.storage.save_agent_playbooks([successor])
            if saved and saved[0].agent_playbook_id:
                self.storage.set_source_windows_for_agent_playbook(
                    saved[0].agent_playbook_id, source_windows
                )
                return saved[0].agent_playbook_id
            return None
        current_user = self.storage.get_user_playbook_by_id(target.target_id)
        if current_user is None or current_user.status is not None:
            return None
        if current_user.user_id is None:
            return None
        archived = self.storage.archive_user_playbook_by_id(
            current_user.user_id, current_user.user_playbook_id
        )
        if not archived:
            return None
        # Recompute polarity from the optimized content. The optimizer can
        # legitimately flip framing (positive guidance -> negative
        # anti-pattern or vice versa), in which case keeping
        # ``current_user.polarity`` would corrupt the polarity-based
        # clustering and consolidation rules. Use the same heuristic the
        # consistency checker uses so the stored polarity matches the
        # content shape.
        resolved_polarity = "negative" if looks_negative(best_content) else "positive"
        successor_user = current_user.model_copy(
            update={
                "user_playbook_id": 0,
                "content": best_content,
                "polarity": resolved_polarity,
                "status": None,
            }
        )
        warn_if_polarity_content_mismatch(successor_user)
        self.storage.save_user_playbooks([successor_user])
        return successor_user.user_playbook_id or None


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
) -> dict[str, Any]:
    return {
        "source_window_count": len(windows),
        "train_window_count": len(train_windows),
        "validation_window_count": len(validation_windows),
        "validation_scenario_user_playbook_ids": [
            window.user_playbook_id for window in validation_windows
        ],
    }


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
