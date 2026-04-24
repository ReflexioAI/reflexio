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
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Request
from reflexio.server.services.base_generation_service import _cheap_should_run_reject
from reflexio.server.services.extraction.extraction_agent import ExtractionAgent
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
        org_id (str): Organisation ID, used for downstream aggregator wiring.
        output_pending_status (bool): Legacy flag — v2 runner does not support
            setting ``Status.PENDING`` after commit.  A warning is emitted when
            ``True`` and the agent applied any mutations.
    """

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        org_id: str,
        output_pending_status: bool = False,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage
        self.org_id = org_id
        self.output_pending_status = output_pending_status

    def run(
        self,
        *,
        publish_request: PublishUserInteractionRequest,
        request_id: str,  # noqa: ARG002 — kept for GenerationService.run contract parity
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

        # Phase 3 — build combined extractor config list (profile then playbook).
        extractor_configs = list(config.profile_extractor_configs or []) + list(
            config.user_playbook_extractor_configs or []
        )

        # Phase 4 — run ExtractionAgent once per enabled extractor config.
        agent = ExtractionAgent(
            client=self.client,
            storage=self.storage,
            prompt_manager=self.request_context.prompt_manager,
        )
        total_applied = 0
        for cfg in extractor_configs:
            extractor_name: str = cfg.extractor_name
            extraction_criteria: str = cfg.extraction_definition_prompt
            try:
                result = agent.run(
                    user_id=publish_request.user_id,
                    agent_version=publish_request.agent_version,
                    extractor_name=extractor_name,
                    extraction_criteria=extraction_criteria,
                    sessions_text=sessions_str,
                )
                total_applied += len(result.applied)
                logger.info(
                    "extraction_agent[%s] outcome=%s applied=%d violations=%d",
                    extractor_name,
                    result.outcome,
                    len(result.applied),
                    len(result.violations),
                )
                warnings.extend(
                    f"extraction_agent[{extractor_name}] violation {v.code}: {v.msg}"
                    for v in result.violations
                    if v.severity == "hard"
                )
            except Exception as e:  # noqa: BLE001 - degrade gracefully per extractor
                logger.warning(
                    "extraction_agent[%s] failed: %s: %s",
                    extractor_name,
                    type(e).__name__,
                    e,
                )
                warnings.append(f"extraction_agent[{extractor_name}] failed: {e}")

        # Phase 5 — playbook aggregation: mirrors classic per-config loop.
        if not publish_request.skip_aggregation:
            self._run_aggregation(
                config=config, publish_request=publish_request, warnings=warnings
            )

        # Phase 6 — output_pending_status compatibility notice.
        # TODO: bolt on status-patching in a follow-up once the v2 commit path
        #       exposes a post-commit hook or returns created entity IDs.
        if self.output_pending_status and total_applied > 0:
            warnings.append("output_pending_status not supported by agentic-v2 runner")

        return warnings

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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
        for pb_cfg in config.user_playbook_extractor_configs or []:
            if not getattr(pb_cfg, "aggregation_config", None):
                continue
            try:
                aggregator = PlaybookAggregator(
                    llm_client=self.client,
                    request_context=self.request_context,
                    agent_version=publish_request.agent_version,
                )
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
