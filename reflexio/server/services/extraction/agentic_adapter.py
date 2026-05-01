"""Adapter wiring ``ExtractionAgent`` into the classic publish flow.

The classic ``GenerationService.run`` expects a pair of generation services
(profile + playbook) it can fan out in parallel.  The agentic-v2 runner is
a single service that iterates extractor configs and calls ``ExtractionAgent``
once per config, committing directly to storage via ``commit_plan``.

This module provides ``AgenticExtractionRunner`` — a thin wrapper that:

1. Applies the same ``_cheap_should_run_reject`` pre-filter the classic
   path uses (honouring ``force_extraction``).
2. Renders the scoped interactions into a transcript string.
3. Iterates all enabled ``ProfileExtractorConfig`` and
   ``UserPlaybookExtractorConfig`` entries and calls ``ExtractionAgent.run``
   once per config.  The agent itself handles search, create, delete, and
   commit (supersession / merge / expansion).
4. Triggers ``PlaybookAggregator`` for every configured playbook with an
   ``aggregation_config``, unless ``skip_aggregation`` was set on the
   publish request.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import Interaction
    from reflexio.models.api_schema.service_schemas import PublishUserInteractionRequest
    from reflexio.models.config_schema import Config
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class AgenticExtractionRunner:
    """Wrap ``ExtractionAgent`` so it mirrors the classic publish contract.

    Iterates each enabled extractor config (profile + playbook) and calls
    ``ExtractionAgent.run`` once per config.  The agent handles its own
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
            config (Config): Resolved top-level config.  ``profile_extractor_configs``
                and ``user_playbook_extractor_configs`` each drive one agent loop;
                ``user_playbook_extractor_configs`` also drives the aggregator loop.

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
        profile_configs = list(config.profile_extractor_configs or [])
        playbook_configs = list(config.user_playbook_extractor_configs or [])
        typed_configs: list[tuple[str, object, object]] = [
            *[
                ("UserProfile", cfg, PROFILE_EXTRACTION_TOOLS)
                for cfg in profile_configs
            ],
            *[
                ("UserProfileAgentRec", cfg, PROFILE_EXTRACTION_TOOLS)
                for cfg in profile_configs
            ],
            *[
                ("UserPlaybook", cfg, PLAYBOOK_EXTRACTION_TOOLS)
                for cfg in playbook_configs
            ],
        ]

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
                    # max_steps=4 is the locked baseline. r118 tested 8 and
                    # was net-negative (over-elaboration drops operands).
                    # 4 keeps tight extraction and lets self-critique recover
                    # missed operands.
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
                    warnings.append(
                        f"extraction_agent[{extractor_name}] failed: {e}"
                    )
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

    def _run_aggregation(
        self,
        *,
        config: Config,
        publish_request: PublishUserInteractionRequest,
        warnings: list[str],
    ) -> None:
        """Run ``PlaybookAggregator`` for every configured playbook with an ``aggregation_config``.

        Args:
            config (Config): Resolved top-level config with playbook extractor configs.
            publish_request (PublishUserInteractionRequest): Provides ``agent_version``.
            warnings (list[str]): Mutable list; aggregation failures are appended.
        """
        aggregator = PlaybookAggregator(
            llm_client=self.client,
            request_context=self.request_context,
            agent_version=publish_request.agent_version,
        )
        for pb_cfg in config.user_playbook_extractor_configs or []:
            if not getattr(pb_cfg, "aggregation_config", None):
                continue
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
