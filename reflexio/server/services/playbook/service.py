from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.deferred_learning_plan import GenerationComputePlan
    from reflexio.server.services.storage.storage_base import BaseStorage

from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    DowngradeUserPlaybooksResponse,
    ManualPlaybookGenerationRequest,
    ManualPlaybookGenerationResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    Status,
    UpgradeUserPlaybooksResponse,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    StatusChangeOperation,
)
from reflexio.server.services.deferred_learning_plan import PlaybookWritePlan
from reflexio.server.services.playbook.aggregation_trigger import (
    maybe_trigger_user_playbook_aggregation,
)
from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
    format_expert_comparison_pairs,
    has_expert_content,
    is_evidence_validated,
    uses_evidence_grounded_extraction,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
)

logger = logging.getLogger(__name__)


def read_user_playbook_as_of_for_learning(
    storage: BaseStorage, *, user_playbook_id: int, served_at: int
) -> UserPlaybook | None:
    """Return the exact cited playbook row iff its body is still attributable.

    This is the strict learning-time read path: it dereferences ONLY the cited
    id, never follows merged/superseded pointers to a current survivor. When the
    exact row body is unavailable or was revised in-place after ``served_at``,
    the row is unattributable and the helper returns ``None``.
    """
    playbook = storage.get_user_playbook_by_id(
        user_playbook_id, include_tombstones=True
    )
    if playbook is None:
        return None
    if playbook.created_at >= served_at:
        return None
    if not playbook.content.strip():
        return None
    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(user_playbook_id)
    )
    if any(event.op == "revise" and event.created_at >= served_at for event in events):
        return None
    return playbook


@dataclass
class PlaybookGenerationServiceConfig:
    """Runtime configuration for playbook generation service shared across all extractors.

    Attributes:
        request_id: The request ID
        agent_version: The agent version
        user_id: The user ID for per-user playbook extraction
        source: Source of the interactions
        allow_manual_trigger: Whether to allow extractors with manual_trigger=True
        rerun_start_time: Optional start time filter for rerun flows (Unix timestamp)
        rerun_end_time: Optional end time filter for rerun flows (Unix timestamp)
        auto_run: True for regular flow (checks stride_size), False for rerun/manual (skips stride_size)
    """

    request_id: str
    agent_version: str
    user_id: str | None = None
    source: str | None = None
    allow_manual_trigger: bool = False
    rerun_start_time: int | None = None
    rerun_end_time: int | None = None
    auto_run: bool = True
    force_extraction: bool = False


def _consolidation_search_keys(playbooks: list[UserPlaybook]) -> set[str]:
    """Return the strings consolidation would use as its hybrid-search queries.

    Mirrors the query selection in ``retrieve_existing_playbooks`` so callers can
    tell whether a cached retrieval is still valid for a changed candidate set.
    """
    return {(playbook.trigger or playbook.content).strip() for playbook in playbooks}


class PlaybookGenerationService(
    BaseGenerationService[
        PlaybookConfig,
        PlaybookExtractor,
        PlaybookGenerationServiceConfig,
        PlaybookGenerationRequest,
    ]
):
    """
    Service for generating playbook entries from user interactions.
    Runs the configured PlaybookExtractor for each generation request.
    """

    # Playbook generation produces learnings — opt in to ② Learning billing.
    EMITS_LEARNING_BILLING: bool = True

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        allow_manual_trigger: bool = False,
        output_pending_status: bool = False,
        skip_aggregation: bool = False,
    ) -> None:
        """
        Initialize the playbook generation service.

        Args:
            llm_client: Unified LLM client supporting both OpenAI and Claude
            request_context: Request context with storage, configurator, and org_id
            allow_manual_trigger: Whether to allow extractors with manual_trigger=True
            output_pending_status: Whether to output entries with PENDING status (for rerun)
            skip_aggregation: Whether to skip playbook aggregation (extract only, no agent playbooks)
        """
        super().__init__(llm_client=llm_client, request_context=request_context)
        self.allow_manual_trigger = allow_manual_trigger
        self.output_pending_status = output_pending_status
        self.skip_aggregation = skip_aggregation
        self._review_window_cache: list[RequestInteractionDataModel] | None = None

    def _load_generation_service_config(
        self, request: PlaybookGenerationRequest
    ) -> PlaybookGenerationServiceConfig:
        """
        Extract request parameters from PlaybookGenerationRequest.

        Args:
            request: PlaybookGenerationRequest containing evaluation parameters

        Returns:
            PlaybookGenerationServiceConfig object
        """
        # One service instance can process multiple users in a batch. Never let
        # a review window loaded for the previous request cross that boundary.
        self._review_window_cache = None
        generation_request_id = request.request_id
        return PlaybookGenerationServiceConfig(
            request_id=generation_request_id,
            agent_version=request.agent_version,
            user_id=request.user_id,
            source=request.source,
            allow_manual_trigger=self.allow_manual_trigger,
            rerun_start_time=request.rerun_start_time,
            rerun_end_time=request.rerun_end_time,
            auto_run=request.auto_run,
            force_extraction=request.force_extraction,
        )

    def _configured_playbook_config(self) -> PlaybookConfig | None:
        root_config = self.configurator.get_config()
        return getattr(root_config, "user_playbook_extractor_config", None)

    def _review_interaction_window(
        self, playbook_config: PlaybookConfig
    ) -> list[RequestInteractionDataModel]:
        """Load the interaction window the reviewer reasons over, once per run.

        The reviewer needs the same chronology extraction saw. Extraction runs in
        a separate step that does not hand its window back, so this reloads it —
        memoized because ``_resolve_write_plan`` may be reached more than once for
        one generation.

        Args:
            playbook_config (PlaybookConfig): Config whose window/source filters
                define the interaction window.

        Returns:
            list[RequestInteractionDataModel]: The reloaded interaction window.

        Raises:
            RuntimeError: If the configured source filter excludes the reviewer
                reload after extraction already produced candidates.
        """
        if self._review_window_cache is None:
            if self.service_config is None:
                raise RuntimeError("service_config must be set before review")
            loaded_window = self._create_extractor(
                playbook_config, self.service_config
            )._get_interactions()
            if loaded_window is None:
                raise RuntimeError(
                    "Normal playbook review source filter excluded its interaction window"
                )
            self._review_window_cache = loaded_window
        return self._review_window_cache

    def _load_extractor_config(self) -> PlaybookConfig | None:
        """
        Load the configured user playbook extractor from configurator.

        Returns:
            PlaybookConfig | None: The configured user playbook extractor, if enabled.
        """
        return self._configured_playbook_config()

    def _create_extractor(
        self,
        extractor_config: PlaybookConfig,
        service_config: PlaybookGenerationServiceConfig,
    ) -> PlaybookExtractor:
        """
        Create a PlaybookExtractor instance from configuration.

        Args:
            extractor_config: PlaybookConfig configuration object from YAML
            service_config: PlaybookGenerationServiceConfig containing runtime parameters

        Returns:
            PlaybookExtractor instance
        """
        return PlaybookExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=self.configurator.get_agent_context(),
        )

    def _build_should_run_prompt(
        self,
        scoped_config: PlaybookConfig,
        session_data_models: list[RequestInteractionDataModel],
    ) -> str | None:
        """
        Build prompt for consolidated should_generate check.

        Renders the configured playbook definition for one LLM call.

        Args:
            scoped_config: Playbook extractor config that had scoped interactions
            session_data_models: Deduplicated request interaction data models

        Returns:
            str | None: The rendered prompt, or None if no definitions to check
        """
        # Check for expert content — use expert-specific should-generate prompt
        all_interactions = extract_interactions_from_request_interaction_data_models(
            session_data_models
        )
        if has_expert_content(all_interactions):
            agent_context = self.configurator.get_agent_context()
            comparison_pairs = format_expert_comparison_pairs(session_data_models)
            return self.request_context.prompt_manager.render_prompt(
                PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_EXPERT_PROMPT_ID,
                {
                    "agent_context_prompt": agent_context,
                    "comparison_pairs": comparison_pairs,
                },
            )

        new_interactions = format_sessions_to_history_string(session_data_models)

        combined_definition = (
            scoped_config.extraction_definition_prompt.strip()
            if scoped_config.extraction_definition_prompt
            else ""
        )

        if not combined_definition:
            return None

        # Get tool_can_use from root config
        root_config = self.request_context.configurator.get_config()
        tool_can_use_str = ""
        if root_config and root_config.tool_can_use:
            tool_can_use_str = "\n".join(
                f"{tool.tool_name}: {tool.tool_description}"
                for tool in root_config.tool_can_use
            )

        agent_context = self.configurator.get_agent_context()
        prompt_manager = self.request_context.prompt_manager

        return prompt_manager.render_prompt(
            PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_PROMPT_ID,
            {
                "agent_context_prompt": agent_context,
                "extraction_definition_prompt": combined_definition,
                "new_interactions": new_interactions,
                "tool_can_use": tool_can_use_str,
            },
        )

    def _get_precheck_interaction_query_kwargs(self) -> dict:
        """Return agent_version filter for non-auto runs."""
        return {
            "agent_version": (
                self.service_config.agent_version  # type: ignore[reportOptionalMemberAccess]
                if not self.service_config.auto_run  # type: ignore[reportOptionalMemberAccess]
                else None
            ),
        }

    def _process_results(self, results: list[list[UserPlaybook]]) -> None:
        """
        Process, deduplicate, and save all results. Called once after all extractors complete.

        Args:
            results: List of UserPlaybook results from extractors (one list per extractor)
        """
        all_playbooks = []
        for result in results:
            if isinstance(result, list):
                all_playbooks.extend(result)
        self._finalize_extracted_items(all_playbooks)

    def _resolve_write_plan(
        self, results: list[list[UserPlaybook]]
    ) -> PlaybookWritePlan | None:
        """Compute-half of playbook finalization — NO learning DB write.

        Flattens + ``dedupe_and_drop_empty``, runs the deduplicator (its 2nd
        LLM call + reads of existing rows), assigns ``source``/``status``, and
        **precomputes embeddings** on the survivors. Returns a
        :class:`PlaybookWritePlan` for the persist half, or ``None`` when there
        is nothing to write (matching the pre-split ``if all_playbooks:`` gate,
        which suppressed save/lineage/schedulers alike). Issues no
        ``save_user_playbooks``/``merge_records``/``supersede_*`` — the write is
        the persist half's job (compute is write-free).
        """
        from reflexio.server.services.playbook.playbook_service_utils import (
            dedupe_and_drop_empty,
        )

        generation_request_id = self.service_config.request_id  # type: ignore[reportOptionalMemberAccess]

        all_playbooks: list[UserPlaybook] = []
        for result in results:
            if isinstance(result, list):
                all_playbooks.extend(result)
        all_playbooks = dedupe_and_drop_empty(all_playbooks)

        # Review complete evidence-grounded normal candidates before
        # consolidation. Legacy or expert paths remain unchanged in this
        # iteration; every new normal candidate is expected to carry validated
        # evidence and therefore enters this fail-closed review path.
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )
        from reflexio.server.services.playbook.components.reviewer import (
            PlaybookCandidateReviewer,
        )

        playbook_config = self._configured_playbook_config()
        dedup_config = playbook_config.deduplication_config if playbook_config else None

        consolidator = PlaybookConsolidator(
            request_context=self.request_context,
            llm_client=self.client,
            dedup_config=dedup_config,
        )
        reviewer = PlaybookCandidateReviewer(
            request_context=self.request_context,
            llm_client=self.client,
        )
        # Retrieved once and shared: the reviewer and the consolidator need the
        # same existing rows, and the hybrid search costs one embedding query per
        # candidate.
        existing_playbooks: list[UserPlaybook] | None = None
        # MOCK_LLM_RESPONSE is the repository's explicit no-LLM pipeline mode:
        # extraction creates a deterministic fixture and consolidation also skips
        # its model call. Keep reviewer behavior consistent instead of making mock
        # mode unexpectedly contact a provider.
        if (
            all_playbooks
            and reviewer.is_enabled()
            and playbook_config is not None
            and self.service_config is not None
            and os.getenv("MOCK_LLM_RESPONSE", "").lower() != "true"
        ):
            review_interactions = self._review_interaction_window(playbook_config)
            if not review_interactions:
                raise RuntimeError(
                    "Normal playbook review could not reload its interaction window"
                )
            flat_review_interactions = (
                extract_interactions_from_request_interaction_data_models(
                    review_interactions
                )
            )
            expert_batch = has_expert_content(flat_review_interactions)
            if expert_batch:
                logger.info(
                    "Skipping normal candidate reviewer for expert-content batch"
                )
            else:
                strict_normal = uses_evidence_grounded_extraction(
                    self.request_context.prompt_manager,
                    expert=False,
                )
                if not strict_normal:
                    logger.info(
                        "Skipping normal candidate reviewer for legacy extraction prompt"
                    )
                else:
                    invalid_candidate_indexes = [
                        index
                        for index, playbook in enumerate(all_playbooks, start=1)
                        if not is_evidence_validated(playbook)
                    ]
                    if invalid_candidate_indexes:
                        raise RuntimeError(
                            "Strict normal playbook candidates are missing validated "
                            "evidence metadata: "
                            + ",".join(
                                f"C{index}" for index in invalid_candidate_indexes
                            )
                        )

                    existing_playbooks = consolidator.retrieve_existing_playbooks(
                        all_playbooks,
                        user_id=self.service_config.user_id,
                        agent_version=self.service_config.agent_version,
                    )
                    root_config = self.request_context.configurator.get_config()
                    tool_context = ""
                    if root_config and root_config.tool_can_use:
                        tool_context = "\n".join(
                            f"{tool.tool_name}: {tool.tool_description}"
                            for tool in root_config.tool_can_use
                        )
                    search_keys_before = _consolidation_search_keys(all_playbooks)
                    all_playbooks = dedupe_and_drop_empty(
                        reviewer.review(
                            candidates=all_playbooks,
                            request_interaction_data_models=review_interactions,
                            existing_playbooks=existing_playbooks,
                            agent_context=self.configurator.get_agent_context(),
                            playbook_definition=(
                                playbook_config.extraction_definition_prompt or ""
                            ).strip(),
                            tool_context=tool_context,
                        )
                    )
                    # Consolidation searches by trigger. Reusing this retrieval is
                    # only sound while the survivors' search keys are unchanged; a
                    # revised trigger would match different stored rows, so drop the
                    # cache and let consolidation search again.
                    if not (
                        _consolidation_search_keys(all_playbooks) <= search_keys_before
                    ):
                        existing_playbooks = None

        # Deduplicate reviewed survivors against existing entries in DB.
        (
            deduplicated_playbooks,
            existing_ids_to_delete,
            merge_groups,
        ) = consolidator.deduplicate(
            [all_playbooks],
            generation_request_id,
            self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
            user_id=self.service_config.user_id,  # type: ignore[reportOptionalMemberAccess]
            existing_playbooks=existing_playbooks,
        )
        consolidation_provenance = consolidator.model_provenance
        consolidated_output_indices = consolidator.consolidated_output_indices
        logger.info(
            "User playbook entries after deduplication: %d",
            len(deduplicated_playbooks),
        )
        # An empty result is meaningful: every candidate may have been rejected
        # as redundant. Never retain the pre-consolidation rows on reject-all.
        all_playbooks = deduplicated_playbooks

        # Set status and source for all entries
        for playbook in all_playbooks:
            playbook.status = Status.PENDING if self.output_pending_status else None
            playbook.source = self.service_config.source  # type: ignore[reportOptionalMemberAccess]

        logger.info("All user playbook entries: %s", all_playbooks)

        if not all_playbooks:
            return None

        logger.info(
            "Successfully completed %d %s playbook generation for request id: %s",
            len(all_playbooks),
            self._get_service_name(),
            generation_request_id,
        )

        # Precompute embeddings on the survivors (compute-side, NO DB write). The
        # persist half passes skip_embedding=True so no embedding runs in the fence.
        self.storage.precompute_user_playbook_embeddings(all_playbooks)  # type: ignore[reportOptionalMemberAccess]

        lineage_contexts: list[LineageContext] = []
        for index, _playbook in enumerate(all_playbooks):
            provenance = (
                consolidation_provenance
                if index in consolidated_output_indices
                else self._last_model_provenance
            )
            lineage_contexts.append(
                LineageContext(
                    op_kind="create",
                    actor=(
                        "consolidator"
                        if index in consolidated_output_indices
                        else "extractor"
                    ),
                    request_id=generation_request_id,
                    model_name=provenance.model_name if provenance else None,
                    provider=provenance.provider if provenance else None,
                )
            )

        return PlaybookWritePlan(
            request_id=generation_request_id,
            output_pending_status=self.output_pending_status,
            skip_aggregation=self.skip_aggregation,
            new_playbooks=all_playbooks,
            superseded_ids=existing_ids_to_delete,
            merge_groups=merge_groups,
            lineage_contexts=lineage_contexts,
            consolidation_provenance=consolidation_provenance,
        )

    def _persist_write_plan(self, plan: PlaybookWritePlan) -> None:
        """Persist-half of playbook finalization — apply the resolved write-plan.

        Issues only the fence-critical row writes: saves the new playbooks
        (``skip_embedding=True`` — embeddings were precomputed in compute; the
        save also assigns survivor ids) then materializes the consolidation
        lineage, which MUST see those survivor ids and so runs AFTER the save.
        NO LLM / embedding / dedup. The off-thread optimization/aggregation
        schedulers are NOT here — they fire post-commit in
        ``emit_generation_side_effects`` (durable / ``.run()``) or right after
        persist in ``_finalize_extracted_items`` (resume/manual).
        """
        if not plan.new_playbooks:
            return
        try:
            self.storage.save_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
                plan.new_playbooks,
                skip_embedding=True,
                lineage_contexts=plan.lineage_contexts,
            )
            self._apply_consolidation_lineage(
                plan.new_playbooks,
                plan.merge_groups,
                plan.superseded_ids,
                request_id=plan.request_id,
                model_provenance=plan.consolidation_provenance,
            )
        except Exception as e:
            logger.error(
                "Failed to save %s results for request id: %s due to %s, exception type: %s",
                self._get_service_name(),
                plan.request_id,
                str(e),
                type(e).__name__,
            )
            raise

    def _dispatch_playbook_schedulers(self, plan: PlaybookWritePlan) -> None:
        """Fire the off-thread optimization + aggregation schedulers post-persist.

        Phantom-billing gate: on the durable / ``.run()`` path this is invoked
        from ``emit_generation_side_effects`` (post-commit), so a fence-lost
        (superseded) job never enqueues optimization or triggers aggregation. On
        the synchronous resume/manual path the permanent
        ``_finalize_extracted_items`` wrapper invokes it right after persist,
        keeping that path identical to the pre-split monolith. The two callers
        are mutually exclusive, so the schedulers fire exactly once per run.
        """
        self._enqueue_user_playbook_optimization(plan.new_playbooks)
        if not plan.output_pending_status and not plan.skip_aggregation:
            logger.info("Trigger playbook aggregation")
            self._trigger_playbook_aggregation()

    def emit_generation_side_effects(self, plan: GenerationComputePlan) -> None:
        """Post-commit side-effects — base telemetry/billing + playbook schedulers.

        Extends the base emit (``generation_succeeded`` + ② Learning billing)
        with the off-thread optimization/aggregation schedulers, which move here
        so they fire only for a fence-winning durable job (never for a
        superseded one) — the phantom-billing gate.
        """
        super().emit_generation_side_effects(plan)
        write_plan = plan.write_plan
        if write_plan is not None:
            self._dispatch_playbook_schedulers(write_plan)

    def _finalize_extracted_items(
        self,
        all_playbooks: list[UserPlaybook],
        *,
        model_provenance: ModelProvenance | None = None,
        finalization_run_id: str | None = None,
    ) -> list[str]:
        """Permanent V3 wrapper: compute→persist→schedulers together (no fence).

        Kept for the synchronous resume/manual callers
        (``ExtractionResumeWorker`` calls this directly). Routes them through the
        same ``_resolve_write_plan`` (compute) + ``_persist_write_plan``
        (persist) split the durable worker uses — with no external
        ``commit_scope`` — then dispatches the same off-thread schedulers, so the
        result is identical to the pre-split monolith.
        """
        entity_type = "user_playbook"
        if finalization_run_id is not None:
            receipt = self.storage.get_agent_run_finalization_receipt(  # type: ignore[reportOptionalMemberAccess]
                run_id=finalization_run_id,
                entity_type=entity_type,
            )
            if receipt is not None:
                return receipt
        if model_provenance is not None:
            self._last_model_provenance = model_provenance
        plan = self._resolve_write_plan([all_playbooks])
        learning_ids: list[str] = []
        if finalization_run_id is None:
            if plan is not None:
                self._persist_write_plan(plan)
                self._dispatch_playbook_schedulers(plan)
            return (
                [
                    str(playbook.user_playbook_id)
                    for playbook in plan.new_playbooks
                    if playbook.user_playbook_id
                ]
                if plan is not None
                else []
            )

        with self.storage.commit_scope():  # type: ignore[reportOptionalMemberAccess]
            receipt = self.storage.get_agent_run_finalization_receipt(  # type: ignore[reportOptionalMemberAccess]
                run_id=finalization_run_id,
                entity_type=entity_type,
            )
            if receipt is not None:
                return receipt
            if plan is not None:
                self._persist_write_plan(plan)
                learning_ids = [
                    str(playbook.user_playbook_id)
                    for playbook in plan.new_playbooks
                    if playbook.user_playbook_id
                ]
            self.storage.save_agent_run_finalization_receipt(  # type: ignore[reportOptionalMemberAccess]
                run_id=finalization_run_id,
                entity_type=entity_type,
                learning_ids=learning_ids,
            )
        if plan is not None:
            self._dispatch_playbook_schedulers(plan)
        return learning_ids

    def _apply_consolidation_lineage(
        self,
        saved_playbooks: list[UserPlaybook],
        merge_groups: list[tuple[int, list[int]]],
        existing_ids_to_delete: list[int],
        *,
        request_id: str,
        model_provenance: ModelProvenance | None = None,
    ) -> None:
        """Materialize consolidation merges as lineage tombstones.

        Runs AFTER ``save_user_playbooks`` has assigned survivor ids. For each
        ``unify`` merge group, routes the sources atomically through
        ``storage.merge_records`` so each becomes a MERGED tombstone pointing at
        its survivor with a ``merge`` lineage event (``resolve_current`` then
        follows the pointer). Any archived id NOT covered by a merge group
        (e.g. a ``differentiate`` split source, which has no single survivor) is
        soft-superseded in place so the dead source content stays auditable for
        point-in-time attribution reads.

        Args:
            saved_playbooks: The just-persisted entries (survivor ids assigned).
            merge_groups: ``(survivor_index, source_existing_ids)`` per merge.
            existing_ids_to_delete: ALL archived ids (merge sources + leftovers).
            request_id: Generation request id — the lineage key recorded on the
                merge/supersede events. Passed explicitly (from the write-plan)
                rather than read off ``self.service_config`` so persist stays
                decoupled from the mutable service config on the fenced path.
        """
        generation_request_id = request_id
        merged_source_ids: set[int] = set()
        for survivor_idx, source_ids in merge_groups:
            survivor_id = saved_playbooks[survivor_idx].user_playbook_id
            merged_source_ids.update(source_ids)
            self.storage.merge_records(  # type: ignore[reportOptionalMemberAccess]
                entity_type="user_playbook",
                survivor_id=str(survivor_id),
                source_ids=[str(s) for s in source_ids],
                context=LineageContext(
                    op_kind="merge",
                    actor="consolidator",
                    source_ids=[str(s) for s in source_ids],
                    reason="dedup-merge",
                    request_id=generation_request_id,
                    model_name=(
                        model_provenance.model_name if model_provenance else None
                    ),
                    provider=model_provenance.provider if model_provenance else None,
                ),
            )

        # Leftover archives not covered by any merge group (e.g. differentiate
        # split sources) have no single survivor, so preserve them as dead-source
        # tombstones instead of hard-deleting them.
        leftover_ids = [
            pid for pid in existing_ids_to_delete if pid not in merged_source_ids
        ]
        if leftover_ids:
            try:
                superseded_count = self.storage.supersede_user_playbooks_by_ids(  # type: ignore[reportOptionalMemberAccess]
                    leftover_ids,
                    generation_request_id,
                )
                logger.info(
                    "Superseded %d split-source existing entries", superseded_count
                )
            except Exception:
                logger.exception("Failed to supersede split-source existing entries")
                raise

    def _enqueue_user_playbook_optimization(
        self, saved_playbooks: list[UserPlaybook]
    ) -> None:
        config = self.configurator.get_config().playbook_optimizer_config
        if (
            not config.enabled
            or not config.optimize_user_playbooks
            or not saved_playbooks
        ):
            return
        from reflexio.server.services.playbook_optimizer import (
            PlaybookOptimizationScheduler,
            PlaybookOptimizationTarget,
            PlaybookOptimizer,
        )

        scheduler = PlaybookOptimizationScheduler.get_instance()
        for playbook in saved_playbooks:
            if (
                not playbook.user_playbook_id
                or playbook.status is not None
                or not playbook.source_interaction_ids
            ):
                continue
            target = PlaybookOptimizationTarget(
                kind="user_playbook", target_id=playbook.user_playbook_id
            )
            scheduler.enqueue(
                org_id=self.request_context.org_id,
                target=target,
                callback=lambda target=target: PlaybookOptimizer(
                    self.request_context, self.client
                ).optimize(target),
                jitter_seconds=config.scheduler_jitter_seconds,
                abort_cooldown_threshold=config.abort_cooldown_threshold,
                cooldown_after_aborts_seconds=config.cooldown_after_aborts_seconds,
            )

    def _get_extractor_state_service_name(self) -> str:
        """
        Get the service name for stride_size bookmark lookups.

        Returns:
            str: "playbook_extractor" for OperationStateManager stride_size checks
        """
        return "playbook_extractor"

    def _get_service_name(self) -> str:
        """
        Get the name of the service for logging and operation state tracking.

        Returns:
            Service name string - "rerun_playbook_generation" for rerun operations,
            "playbook_generation" for regular operations
        """
        if self.output_pending_status:
            return "rerun_playbook_generation"
        return "playbook_generation"

    def _get_base_service_name(self) -> str:
        """
        Get the base service name for OperationStateManager keys.

        Returns:
            str: "playbook_generation"
        """
        return "playbook_generation"

    def _should_track_in_progress(self) -> bool:
        """
        Playbook generation should track in-progress state to prevent duplicates.

        Returns:
            bool: True - playbook generation tracks in-progress state
        """
        return True

    def _get_lock_scope_id(self, request: PlaybookGenerationRequest) -> str | None:  # noqa: ARG002
        """
        Get the scope ID for lock key construction.

        Playbook generation is org-scoped, so returns None (no user scope).

        Args:
            request: The PlaybookGenerationRequest

        Returns:
            None: Playbook uses org-level scope only
        """
        return None

    def _trigger_playbook_aggregation(self) -> None:
        """
        Trigger playbook aggregation for playbook types that have aggregator config.
        This is called after raw user playbook entries are saved to check if aggregation should run.
        """
        maybe_trigger_user_playbook_aggregation(
            request_context=self.request_context,
            llm_client=self.client,
            agent_version=self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
            reason="playbook_generation",
        )

    # ===============================
    # Rerun hook implementations (override base class methods)
    # ===============================

    def _pre_process_rerun(self, request: RerunPlaybookGenerationRequest) -> None:
        """Delete existing pending raw entries before generating new ones.

        This ensures that each rerun starts fresh without accumulating pending entries
        from previous reruns.

        Args:
            request: RerunPlaybookGenerationRequest with optional agent_version filter
        """
        deleted_count = self.storage.delete_all_user_playbooks_by_status(  # type: ignore[reportOptionalMemberAccess]
            status=Status.PENDING,
            agent_version=request.agent_version,
        )
        logger.info(
            "Deleted %d existing pending raw entries before rerun (agent_version=%s)",
            deleted_count,
            request.agent_version,
        )

    def _get_rerun_user_ids(self, request: RerunPlaybookGenerationRequest) -> list[str]:
        """Get user IDs to process. Extractors collect their own data.

        Identifies unique user_ids with matching requests via storage-level filtering.

        Args:
            request: RerunPlaybookGenerationRequest with optional time and source filters

        Returns:
            List of user IDs to process
        """
        return self.storage.get_rerun_user_ids(  # type: ignore[reportOptionalMemberAccess]
            user_id=None,
            start_time=(
                int(request.start_time.timestamp()) if request.start_time else None
            ),
            end_time=(int(request.end_time.timestamp()) if request.end_time else None),
            source=request.source,
            agent_version=request.agent_version,
        )

    def _build_rerun_request_params(
        self, request: RerunPlaybookGenerationRequest
    ) -> dict:
        """Build request params dict for operation state tracking.

        Args:
            request: Original rerun request

        Returns:
            Dictionary of request parameters
        """
        return {
            "agent_version": request.agent_version,
            "start_time": (
                request.start_time.isoformat() if request.start_time else None
            ),
            "end_time": request.end_time.isoformat() if request.end_time else None,
        }

    def _create_run_request_for_item(
        self,
        user_id: str,
        request: RerunPlaybookGenerationRequest | ManualPlaybookGenerationRequest,
    ) -> PlaybookGenerationRequest:
        """Create PlaybookGenerationRequest for a single user.

        Handles both rerun and manual request types.

        Args:
            user_id: The user ID to process
            request: The original rerun or manual request

        Returns:
            PlaybookGenerationRequest for this user with filter constraints
        """
        # Handle rerun requests (have start_time/end_time datetime objects)
        if isinstance(request, RerunPlaybookGenerationRequest):
            operation_request_id = f"rerun_playbook_{uuid.uuid4().hex[:8]}"
            return PlaybookGenerationRequest(
                request_id=operation_request_id,
                agent_version=request.agent_version,
                user_id=user_id,
                source=request.source,
                rerun_start_time=(
                    int(request.start_time.timestamp()) if request.start_time else None
                ),
                rerun_end_time=(
                    int(request.end_time.timestamp()) if request.end_time else None
                ),
                auto_run=False,
            )
        # Handle manual requests (ManualPlaybookGenerationRequest)
        operation_request_id = f"manual_{uuid.uuid4().hex[:8]}"
        return PlaybookGenerationRequest(
            request_id=operation_request_id,
            agent_version=request.agent_version,
            user_id=user_id,
            source=request.source,
            auto_run=False,
        )

    def _create_rerun_response(
        self, success: bool, msg: str, count: int
    ) -> RerunPlaybookGenerationResponse:
        """Create RerunPlaybookGenerationResponse.

        Args:
            success: Whether the operation succeeded
            msg: Status message
            count: Number of entries generated

        Returns:
            RerunPlaybookGenerationResponse
        """
        return RerunPlaybookGenerationResponse(
            success=success,
            msg=msg,
            playbooks_generated=count,
        )

    def _get_generated_count(
        self,
        request: RerunPlaybookGenerationRequest,
        processed_user_ids: list[str] | None = None,  # noqa: ARG002
    ) -> int:
        """Get the count of entries generated during rerun.

        Counts entries with pending status, filtered by agent_version.

        Args:
            request: The rerun request object

        Returns:
            Number of entries generated
        """
        playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            agent_version=request.agent_version,
            status_filter=[Status.PENDING],
            limit=10000,
        )
        return len(playbooks)

    # ===============================
    # Manual Regular Generation (window-sized, CURRENT output)
    # ===============================

    def run_manual_regular(
        self, request: ManualPlaybookGenerationRequest
    ) -> ManualPlaybookGenerationResponse:
        """
        Run playbook generation with window-sized interactions and CURRENT output.

        Processes entries per-user. Each extractor collects its own data
        using its configured window_size.
        Uses progress tracking via OperationStateManager.

        Args:
            request: ManualPlaybookGenerationRequest with agent_version and optional source

        Returns:
            ManualPlaybookGenerationResponse with success status and count
        """
        state_manager = self._create_state_manager()

        try:
            # Check for existing in-progress operation
            error = state_manager.check_in_progress()
            if error:
                return ManualPlaybookGenerationResponse(
                    success=False, msg=error, playbooks_generated=0
                )

            # 1. Get user_ids with recent interactions
            requests_dict = self.storage.get_sessions(  # type: ignore[reportOptionalMemberAccess]
                user_id=None,  # All users
                top_k=1000,  # Get recent sessions to find users
            )

            # Get unique user_ids
            user_ids_set: set[str] = set()
            for session_requests in requests_dict.values():
                for rig in session_requests:
                    # Apply source filter if provided
                    if request.source and rig.request.source != request.source:
                        continue
                    user_ids_set.add(rig.request.user_id)

            user_ids = list(user_ids_set)

            if not user_ids:
                return ManualPlaybookGenerationResponse(
                    success=True,
                    msg="No interactions found to process",
                    playbooks_generated=0,
                )

            # 2. Run batch with progress tracking
            request_params = {
                "agent_version": request.agent_version,
                "source": request.source,
                "mode": "manual_regular",
            }
            self._run_batch_with_progress(
                user_ids=user_ids,
                request=request,  # type: ignore[reportArgumentType]
                request_params=request_params,
                state_manager=state_manager,
            )

            # 3. Count generated entries (CURRENT status = None)
            total_playbooks = self._count_manual_generated(request)

            return ManualPlaybookGenerationResponse(
                success=True,
                msg=f"Generated {total_playbooks} playbook entries",
                playbooks_generated=total_playbooks,
            )

        except Exception as e:
            state_manager.mark_progress_failed(str(e))
            return ManualPlaybookGenerationResponse(
                success=False,
                msg=f"Failed to generate playbook entries: {str(e)}",
                playbooks_generated=0,
            )

    def _count_manual_generated(self, request: ManualPlaybookGenerationRequest) -> int:
        """
        Count entries generated during manual regular generation.

        Counts entries with CURRENT status (None), filtered by agent_version.

        Args:
            request: The manual generation request object

        Returns:
            Number of entries with CURRENT status
        """
        playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            agent_version=request.agent_version,
            status_filter=[None],  # CURRENT entries
            limit=10000,
        )
        return len(playbooks)

    # ===============================
    # Upgrade/Downgrade hook implementations (override base class methods)
    # ===============================

    def _has_items_with_status(
        self, status: Status | None, request: PlaybookGenerationRequest
    ) -> bool:
        """Check if raw entries exist with given status.

        Args:
            status: The status to check for (None for CURRENT)
            request: The upgrade/downgrade request object

        Returns:
            bool: True if any matching raw entries exist
        """
        return self.storage.has_user_playbooks_with_status(  # type: ignore[reportOptionalMemberAccess]
            status=status,
            agent_version=getattr(request, "agent_version", None),
        )

    def _delete_items_by_status(
        self, status: Status, request: PlaybookGenerationRequest
    ) -> int:
        """Delete raw entries with given status.

        Args:
            status: The status of raw entries to delete
            request: The upgrade/downgrade request object

        Returns:
            int: Number of raw entries deleted
        """
        return self.storage.delete_all_user_playbooks_by_status(  # type: ignore[reportOptionalMemberAccess]
            status=status,
            agent_version=getattr(request, "agent_version", None),
        )

    def _update_items_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        request: PlaybookGenerationRequest,
        user_ids: list[str] | None = None,  # noqa: ARG002
    ) -> int:
        """Update raw entries from old_status to new_status with request filters.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            request: The upgrade/downgrade request object with filters
            user_ids: Optional pre-computed list of user IDs (not used for playbook service)

        Returns:
            int: Number of raw entries updated
        """
        # Note: user_ids is ignored for playbook service as it uses agent_version filters
        return self.storage.update_all_user_playbooks_status(  # type: ignore[reportOptionalMemberAccess]
            old_status=old_status,
            new_status=new_status,
            agent_version=getattr(request, "agent_version", None),
        )

    def _create_status_change_response(
        self,
        operation: StatusChangeOperation,
        success: bool,
        counts: dict,
        msg: str,
    ) -> UpgradeUserPlaybooksResponse | DowngradeUserPlaybooksResponse:
        """Create upgrade or downgrade response object for raw entries.

        Args:
            operation: The operation type (UPGRADE or DOWNGRADE)
            success: Whether the operation succeeded
            counts: Dictionary of counts
            msg: Status message

        Returns:
            UpgradeUserPlaybooksResponse or DowngradeUserPlaybooksResponse
        """
        if operation == StatusChangeOperation.UPGRADE:
            return UpgradeUserPlaybooksResponse(
                success=success,
                user_playbooks_deleted=counts.get("deleted", 0),
                user_playbooks_archived=counts.get("archived", 0),
                user_playbooks_promoted=counts.get("promoted", 0),
                message=msg,
            )
        # DOWNGRADE
        return DowngradeUserPlaybooksResponse(
            success=success,
            user_playbooks_demoted=counts.get("demoted", 0),
            user_playbooks_restored=counts.get("restored", 0),
            message=msg,
        )
