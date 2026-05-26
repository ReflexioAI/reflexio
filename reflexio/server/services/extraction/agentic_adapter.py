"""Adapter wiring ``ExtractionAgent`` into the classic publish flow.

The classic ``GenerationService.run`` expects a pair of generation services
(profile + playbook) it can fan out in parallel.  The agentic-v2 runner is
a single service that calls ``ExtractionAgent`` for the configured profile and
playbook extractors, committing directly to storage via ``commit_plan``.

This module provides ``AgenticExtractionRunner`` — a thin wrapper that:

1. Applies the same ``_cheap_should_run_reject`` pre-filter the classic
   path uses (honouring ``force_extraction``).
2. Renders the scoped interactions into a transcript string.
3. Runs the configured ``ProfileExtractorConfig`` and
   ``UserPlaybookExtractorConfig``. The agent itself handles search, create, delete, and
   commit (supersession / merge / expansion).
4. Triggers ``PlaybookAggregator`` when the configured playbook has an
   ``aggregation_config``, unless ``skip_aggregation`` was set on the publish request.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Request
from reflexio.server.services.base_generation_service import _cheap_should_run_reject
from reflexio.server.services.extraction.extraction_agent import (
    DeferredExtractionRun,
    ExtractionAgent,
)
from reflexio.server.services.extraction.self_critique import SelfCritiqueAgent
from reflexio.server.services.extraction.tools import (
    PLAYBOOK_EXTRACTION_TOOLS,
    PROFILE_EXTRACTION_TOOLS,
)
from reflexio.server.services.extraction.unify import UnifyAgent
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.service_utils import format_sessions_to_history_string

# Matches a 4-digit year in 20xx, optionally followed by an ISO month/day.
# Used by the post-extraction wall-clock sanitizer (L2 — defense against
# the Codex backend's silent runtime "current_date" injection that the
# extraction prompt cannot fully suppress).
_YEAR_RE = re.compile(r"\b(20\d{2})(?:-\d{2}-\d{2})?\b")
# Slack above session_year_max for legitimate near-future planned events
# (weddings, trips, due dates). Threshold = session_year_max + this slack.
_FUTURE_YEAR_SLACK = 2

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import Interaction
    from reflexio.models.api_schema.service_schemas import PublishUserInteractionRequest
    from reflexio.models.config_schema import Config
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class AgenticExtractionRunner:
    """Wrap ``ExtractionAgent`` so it mirrors the classic publish contract.

    Runs the configured profile and playbook extractors. The agent handles its own
    search-then-mutate loop and commits the plan directly to storage.

    Args:
        llm_client (LiteLLMClient): Configured LLM client.
        request_context (RequestContext): Provides ``storage``, ``prompt_manager``,
            and ``configurator``.
    """

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage

    def run(
        self,
        *,
        publish_request: PublishUserInteractionRequest,
        request_id: str,
        new_interactions: list[Interaction],
        new_request: Request,
        config: Config,
    ) -> list[str]:
        """Run agentic extraction + aggregation and persist.

        Args:
            publish_request (PublishUserInteractionRequest): The original
                publish request — ``source``, ``agent_version``,
                ``force_extraction``, ``skip_aggregation`` are read from it.
            request_id (str): Per-publish UUID assigned by ``GenerationService.run``.
            new_interactions (list[Interaction]): Interactions persisted for
                this publish, used for both the pre-filter and transcript.
            new_request (Request): The ``Request`` row just persisted; used
                to synthesise the precheck ``RequestInteractionDataModel``.
            config (Config): Resolved top-level config. ``profile_extractor_config``
                and ``user_playbook_extractor_config`` drive extraction; the playbook
                config also drives aggregation.

        Returns:
            list[str]: Non-fatal warnings to surface back to the caller.
        """
        warnings: list[str] = []
        session_data_models = self._build_session_data_models(
            new_interactions=new_interactions, new_request=new_request
        )

        # Phase 1 — pre-filter: cheap reject for sessions with no learnable signal.
        if not publish_request.force_extraction:
            reason = _cheap_should_run_reject(session_data_models)
            if reason is not None:
                logger.info(
                    "agentic pre-filter rejected: reason=%s identifier=%s",
                    reason,
                    publish_request.user_id,
                )
                return warnings

        # Phase 2 — render transcript once; all agent calls share the same text.
        sessions_str = format_sessions_to_history_string(session_data_models)

        # Phase 3 — build typed extractor config list (profile then playbook).
        # Each tuple carries: (entity_kind, extractor_config, tool_registry).
        # Profile extraction now runs as TWO parallel passes per config:
        #   1. UserProfile  — user-side facts only (extraction_user_profile v1.2.0+)
        #   2. UserProfileAgentRec — agent-named-answer axis (extraction_user_profile_agent_rec)
        # The split addresses the agentic-loop variance where a single combined
        # prompt would stochastically crowd out one axis or the other.
        #
        # ``config.skip_extraction_axes`` may suppress any subset of axes by
        # name. Default is an empty set, so all three axes run unchanged.
        profile_configs = (
            [config.profile_extractor_config] if config.profile_extractor_config else []
        )
        playbook_configs = (
            [config.user_playbook_extractor_config]
            if config.user_playbook_extractor_config
            else []
        )
        typed_configs = self._build_typed_configs(
            profile_configs=profile_configs,
            playbook_configs=playbook_configs,
            skip_axes=set(config.skip_extraction_axes or []),
        )

        # Phase 4 — run all enabled extractor configs IN PARALLEL with
        # commit deferred. Three axes (UserProfile, UserProfileAgentRec,
        # UserPlaybook) examine the same transcript without seeing each
        # other's in-flight writes; this removes the cross-axis dedup
        # contention the sequential pipeline had.
        agents_by_kind: dict[str, ExtractionAgent] = {}
        deferred_runs, run_warnings = self._run_passes_in_parallel(
            typed_configs=typed_configs,
            sessions_str=sessions_str,
            publish_request=publish_request,
            request_id=request_id,
            agents_by_kind=agents_by_kind,
        )
        warnings.extend(run_warnings)

        # Phase 4a' — self-critique pass (default ON, opt-out via
        # EXTRACTION_SELF_CRITIQUE=0). r119 validated +1pt heldout macro
        # vs r116 baseline; primarily K-U +25pt via supersession of
        # operand-bearing facts; cost ~+30% latency and LLM spend.
        if (
            os.getenv("EXTRACTION_SELF_CRITIQUE", "1") == "1"
            and len(deferred_runs) >= 1
        ):
            try:
                added = SelfCritiqueAgent(
                    client=self.client,
                    prompt_manager=self.request_context.prompt_manager,
                ).run(deferred_runs, transcript=sessions_str)
                logger.info("self_critique added=%d", added)
            except Exception as e:  # noqa: BLE001 — keep current ops on failure
                logger.warning(
                    "self_critique failed: %s: %s — proceeding without",
                    type(e).__name__,
                    e,
                )
                warnings.append(f"self_critique failed: {e}")

        # Phase 4b — single LLM call adjudicates cross-axis duplicates.
        # Mutates each deferred run's plan in place; never invents ops.
        if len(deferred_runs) >= 2:
            try:
                dropped = UnifyAgent(
                    client=self.client,
                    prompt_manager=self.request_context.prompt_manager,
                ).run(deferred_runs)
                logger.info("unify_agent dropped=%d", dropped)
            except Exception as e:  # noqa: BLE001 — keep all ops on failure
                logger.warning(
                    "unify_agent failed: %s: %s — committing without unify",
                    type(e).__name__,
                    e,
                )
                warnings.append(f"unify_agent failed: {e}")

        # Phase 4b'' — speaker attribution filter. The extraction prompt
        # tells the LLM to only extract User-role first-person facts and to
        # tag source_span with the role prefix ("User: ..." or "Assistant: ...").
        # When the LLM honors the prefix, this filter deterministically drops
        # any op whose source_span starts with "Assistant:" — preventing the
        # ~18% measured misattribution rate where Assistant first-person
        # statements were stored as user facts in legacy LoCoMo storage.
        try:
            n_dropped_speaker = self._filter_assistant_attributed_ops(deferred_runs)
            if n_dropped_speaker:
                logger.warning(
                    "speaker_attr_filter dropped=%d (ops attributed to Assistant turns)",
                    n_dropped_speaker,
                )
        except Exception as e:  # noqa: BLE001 — filter is best-effort
            logger.warning(
                "speaker_attr_filter failed: %s: %s — proceeding without",
                type(e).__name__,
                e,
            )

        # Phase 4b' — wall-clock sanitizer (L2 defense against Codex
        # backend's "current_date" injection). Even with prompt rules to
        # ignore the runtime-injected today, gpt-5/gpt-5-mini occasionally
        # leak today's wall-clock year into content (~0.7% of profiles in
        # measured LoCoMo namespaces). Replace any year > session_year+slack
        # with [date_unknown] in both content and source_span.
        try:
            session_year_max = self._compute_session_year_max(publish_request)
            if session_year_max is not None:
                threshold = session_year_max + _FUTURE_YEAR_SLACK
                n_modified = self._sanitize_wallclock_years(
                    deferred_runs, year_threshold=threshold
                )
                if n_modified:
                    logger.warning(
                        "wallclock_sanitize ops_modified=%d threshold_year=%d "
                        "session_year_max=%d",
                        n_modified,
                        threshold,
                        session_year_max,
                    )
        except Exception as e:  # noqa: BLE001 — sanitize is best-effort
            logger.warning(
                "wallclock_sanitize failed: %s: %s — proceeding without",
                type(e).__name__,
                e,
            )

        # Phase 4c — commit each axis's (possibly pruned) plan.
        for run in deferred_runs:
            try:
                commit = agents_by_kind[run.kind].commit_deferred(run)
                warnings.extend(
                    f"extraction_agent[{run.ctx.extractor_name}] violation {v.code}: {v.msg}"
                    for v in commit.violations
                    if v.severity == "hard"
                )
            except Exception as e:  # noqa: BLE001 — one axis failing must not block others
                logger.warning(
                    "extraction_agent[%s] commit failed: %s: %s",
                    run.ctx.extractor_name,
                    type(e).__name__,
                    e,
                )
                warnings.append(
                    f"extraction_agent[{run.ctx.extractor_name}] commit failed: {e}"
                )

        # Phase 5 — playbook aggregation: mirrors classic per-config loop.
        if not publish_request.skip_aggregation:
            self._run_aggregation(
                config=config, publish_request=publish_request, warnings=warnings
            )

        return warnings

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_typed_configs(
        *,
        profile_configs: Sequence[object],
        playbook_configs: Sequence[object],
        skip_axes: set[str],
    ) -> list[tuple[str, object, object]]:
        """Materialise the (axis, config, tool_registry) triples to run.

        Each axis declared in ``skip_axes`` is dropped from the result; if no
        axes are skipped, the output matches the historical hardcoded layout
        (UserProfile then UserProfileAgentRec then UserPlaybook).

        Args:
            profile_configs (Sequence[object]): Resolved profile extractor configs.
            playbook_configs (Sequence[object]): Resolved playbook extractor configs.
            skip_axes (set[str]): Axis names to suppress. Unknown axis names
                are silently no-ops (the axis simply won't appear in the
                axis_specs table to filter against), so callers can rename
                or retire axes without breaking persisted Configs.

        Returns:
            list[tuple[str, object, object]]: One tuple per (axis, config) pair
                that survived the filter, in axis-then-config order.
        """
        # Each entry: (axis_name, configs_for_this_axis, tools_for_this_axis).
        axis_specs: list[tuple[str, Sequence[object], object]] = [
            ("UserProfile", profile_configs, PROFILE_EXTRACTION_TOOLS),
            ("UserProfileAgentRec", profile_configs, PROFILE_EXTRACTION_TOOLS),
            ("UserPlaybook", playbook_configs, PLAYBOOK_EXTRACTION_TOOLS),
        ]
        typed_configs: list[tuple[str, object, object]] = []
        for axis_name, configs_for_axis, tools in axis_specs:
            if axis_name in skip_axes:
                logger.info(
                    "Skipping extraction axis %s (config.skip_extraction_axes)",
                    axis_name,
                )
                continue
            typed_configs.extend((axis_name, cfg, tools) for cfg in configs_for_axis)
        return typed_configs

    def _run_passes_in_parallel(
        self,
        *,
        typed_configs: list[tuple[str, object, object]],
        sessions_str: str,
        publish_request: PublishUserInteractionRequest,
        request_id: str,
        agents_by_kind: dict[str, ExtractionAgent],
    ) -> tuple[list[DeferredExtractionRun], list[str]]:
        """Drive every typed_config through ``ExtractionAgent.run_no_commit`` in parallel.

        Returns the deferred runs (one per pass that succeeded) plus any
        per-extractor warnings. Failures are logged + appended as warnings;
        they do not block the surviving passes from committing.
        """
        deferred: list[DeferredExtractionRun] = []
        warnings: list[str] = []
        # max_workers tracks the number of axes; len(typed_configs) is
        # already small (3 with the default config).
        with ThreadPoolExecutor(
            max_workers=max(1, len(typed_configs)),
            thread_name_prefix="extraction-pass",
        ) as executor:
            future_to_meta = {}
            for kind, cfg, registry in typed_configs:
                extractor_name: str = cfg.extractor_name  # type: ignore[union-attr]
                extraction_criteria: str = cfg.extraction_definition_prompt  # type: ignore[union-attr]
                agent = ExtractionAgent(
                    client=self.client,
                    storage=self.storage,
                    prompt_manager=self.request_context.prompt_manager,
                    registry=registry,  # type: ignore[arg-type]
                    # max_steps=4 (v1.4.11): cost-efficient baseline reverted
                    # from v1.4.9's 12. v1.4.11 drops the phased pipeline
                    # narration overhead, so 4 is sufficient for the simpler
                    # single-loop extraction. Matches the original v1.4.5
                    # locked baseline.
                    max_steps=4,
                )
                # Stash the agent so commit_deferred can reuse the same
                # storage handle / prompt manager bindings later.
                agents_by_kind[kind] = agent
                future = executor.submit(
                    agent.run_no_commit,
                    user_id=publish_request.user_id,
                    agent_version=publish_request.agent_version,
                    extractor_name=extractor_name,
                    extraction_criteria=extraction_criteria,
                    sessions_text=sessions_str,
                    extraction_kind=kind,  # type: ignore[arg-type]
                    request_id=request_id,
                )
                future_to_meta[future] = (kind, extractor_name)

            for future in as_completed(future_to_meta):
                kind, extractor_name = future_to_meta[future]
                try:
                    deferred.append(future.result())
                except Exception as e:  # noqa: BLE001 — degrade gracefully per pass
                    logger.warning(
                        "extraction_agent[%s] kind=%s failed: %s: %s",
                        extractor_name,
                        kind,
                        type(e).__name__,
                        e,
                    )
                    warnings.append(f"extraction_agent[{extractor_name}] failed: {e}")
        return deferred, warnings

    @staticmethod
    def _build_session_data_models(
        *, new_interactions: list[Interaction], new_request: Request
    ) -> list[RequestInteractionDataModel]:
        """Wrap this publish's interactions in a single-element batch for the precheck.

        Args:
            new_interactions (list[Interaction]): The interactions for this publish.
            new_request (Request): The request row just persisted.

        Returns:
            list[RequestInteractionDataModel]: Single-element list for the precheck.
        """
        return [
            RequestInteractionDataModel(
                session_id=new_request.session_id or "",
                request=new_request,
                interactions=list(new_interactions),
            )
        ]

    # Detects "Assistant:" prefix at the start of a source_span (with optional
    # whitespace, code-fence backticks, or quotes wrapping). The extraction
    # prompt instructs the LLM to preserve the role prefix verbatim, so any
    # op whose source_span starts this way describes the Assistant's content,
    # not the user's, and must not be stored under the user's namespace.
    _ASSISTANT_PREFIX_RE = re.compile(r'^\s*[`"\']*\s*assistant\s*:', re.IGNORECASE)

    @classmethod
    def _filter_assistant_attributed_ops(
        cls,
        deferred_runs: Sequence[DeferredExtractionRun],
    ) -> int:
        """Drop CreateUser{Profile,Playbook}Op whose source_span is from an Assistant turn.

        Mutates each run's ``ctx.plan`` in place by filtering. Delete ops are
        always preserved (no source_span). Ops with empty / missing source_span
        are kept (we have no way to attribute, so default to keeping).

        Returns:
            int: Number of create ops dropped.
        """
        n_dropped = 0
        for run in deferred_runs:
            kept: list = []
            for op in run.ctx.plan:
                # Delete ops have no source_span — always keep
                span = getattr(op, "source_span", None)
                if not isinstance(span, str) or not span.strip():
                    kept.append(op)
                    continue
                if cls._ASSISTANT_PREFIX_RE.match(span):
                    n_dropped += 1
                    continue
                kept.append(op)
            run.ctx.plan = kept
        return n_dropped

    @staticmethod
    def _compute_session_year_max(
        publish_request: PublishUserInteractionRequest,
    ) -> int | None:
        """Max year across all interaction timestamps in the publish request.

        Returns the latest UTC year for which any interaction has a
        ``created_at``, or None if no interaction is timestamped (in which
        case wall-clock sanitization is skipped — we have no anchor).
        """
        timestamps = [
            i.created_at for i in publish_request.interaction_data_list if i.created_at
        ]
        if not timestamps:
            return None
        return max(datetime.fromtimestamp(t, tz=UTC).year for t in timestamps)

    @classmethod
    def _sanitize_wallclock_years(
        cls,
        deferred_runs: Sequence[DeferredExtractionRun],
        *,
        year_threshold: int,
    ) -> int:
        """Replace 4-digit years > ``year_threshold`` with [date_unknown].

        Mutates each run's ``ctx.plan`` in place. Touches both ``content``
        and ``source_span`` of CreateUserProfileOp / CreateUserPlaybookOp.
        Delete ops have no content and are unaffected.

        Returns:
            int: Number of ops whose content or source_span was modified.
        """
        n_modified = 0
        for run in deferred_runs:
            for op in run.ctx.plan:
                old_content = getattr(op, "content", None)
                old_span = getattr(op, "source_span", None)
                new_content = (
                    cls._strip_future_years(old_content, year_threshold)
                    if isinstance(old_content, str)
                    else old_content
                )
                new_span = (
                    cls._strip_future_years(old_span, year_threshold)
                    if isinstance(old_span, str)
                    else old_span
                )
                changed = False
                if new_content != old_content:
                    op.content = new_content
                    changed = True
                if new_span != old_span:
                    op.source_span = new_span
                    changed = True
                if changed:
                    n_modified += 1
        return n_modified

    @staticmethod
    def _strip_future_years(text: str, threshold: int) -> str:
        """Replace 4-digit years > threshold (and any ISO -MM-DD suffix) with [date_unknown]."""

        def _sub(m: re.Match[str]) -> str:
            year = int(m.group(1))
            return "[date_unknown]" if year > threshold else m.group(0)

        return _YEAR_RE.sub(_sub, text)

    def _run_aggregation(
        self,
        *,
        config: Config,
        publish_request: PublishUserInteractionRequest,
        warnings: list[str],
    ) -> None:
        """Run ``PlaybookAggregator`` for every configured playbook with an ``aggregation_config``.

        Args:
            config (Config): Resolved top-level config with the playbook extractor config.
            publish_request (PublishUserInteractionRequest): Provides ``agent_version``.
            warnings (list[str]): Mutable list; aggregation failures are appended.
        """
        aggregator = PlaybookAggregator(
            llm_client=self.client,
            request_context=self.request_context,
            agent_version=publish_request.agent_version,
        )
        pb_cfg = config.user_playbook_extractor_config
        if not pb_cfg or not getattr(pb_cfg, "aggregation_config", None):
            return
        try:
            aggregator.run(
                PlaybookAggregatorRequest(
                    agent_version=publish_request.agent_version,
                    playbook_name=pb_cfg.extractor_name,
                )
            )
        except Exception as e:  # noqa: BLE001 - degrade gracefully
            logger.warning(
                "agentic aggregation failed for %s: %s: %s",
                pb_cfg.extractor_name,
                type(e).__name__,
                e,
            )
            warnings.append(f"aggregation failed for {pb_cfg.extractor_name}: {e}")
