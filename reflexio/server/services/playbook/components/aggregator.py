from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    SINGLETON_USER_PLAYBOOK_NAME,
    PlaybookAggregatorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AggregationPromptProcessingContext,
    AggregationPromptProcessor,
)
from reflexio.server.services.playbook.components import (
    aggregator_clustering,
    aggregator_prompt_formatting,
)
from reflexio.server.services.playbook.components.aggregator_clustering import (
    CLUSTERING_ALGORITHM_THRESHOLD,
)
from reflexio.server.services.playbook.components.aggregator_postprocessing import (
    AggregationPostProcessing,
)
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
    PlaybookAggregatorRequest,
    StructuredPlaybookContent,
    ensure_playbook_content,
)
from reflexio.server.services.service_utils import log_model_response
from reflexio.server.services.storage.storage_base import AGGREGATE_REASON_PREFIX
from reflexio.server.tracing import capture_anomaly, sentry_tags
from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)


class AggregationEffectCoordinator(Protocol):
    """Optional managed boundary for one atomic aggregation effect."""

    def prepare(self, playbooks: list[AgentPlaybook]) -> None: ...

    def apply_scope(self) -> AbstractContextManager[None]: ...

    def save_agent_playbook(
        self,
        playbook: AgentPlaybook,
        *,
        source_ids: list[str],
        request_id: str,
        run_mode: str,
        provenance: ModelProvenance | None,
    ) -> AgentPlaybook: ...

    def complete(self, result: dict[str, Any]) -> None: ...


class PlaybookAggregator:
    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        agent_version: str,
        aggregation_prompt_processor: AggregationPromptProcessor | None = None,
        effect_coordinator: AggregationEffectCoordinator | None = None,
    ) -> None:
        self.client = llm_client
        self.storage = request_context.storage
        self.configurator = request_context.configurator
        self.request_context = request_context
        self.agent_version = agent_version
        self.aggregation_prompt_processor = aggregation_prompt_processor
        self.effect_coordinator = effect_coordinator
        # Cohesive pre/post-processing component (the enterprise redaction
        # Protocol seam). Constructed from the SAME injected instance stored
        # above — do NOT re-resolve the AGGREGATION_PROMPT_PROCESSOR ServiceKey.
        self._postproc = AggregationPostProcessing(aggregation_prompt_processor)

    # ===============================
    # private methods - operation state
    # ===============================

    def _create_state_manager(self) -> OperationStateManager:
        """
        Create an OperationStateManager for the playbook aggregator.

        Returns:
            OperationStateManager configured for playbook_aggregator
        """
        return OperationStateManager(
            self.storage,  # type: ignore[reportArgumentType]
            self.request_context.org_id,
            "playbook_aggregator",
        )

    def _get_new_user_playbooks_count(
        self, playbook_name: str, rerun: bool = False
    ) -> int:
        """
        Count how many new user playbooks exist since last aggregation.
        Uses efficient SQL COUNT query instead of fetching all user playbooks.

        Args:
            playbook_name: Name of the playbook type
            rerun: If True, count all user playbooks (use last_processed_id=0)

        Returns:
            int: Count of new user playbooks
        """
        # For rerun, use 0 to process all user playbooks
        if rerun:
            last_processed_id = 0
        else:
            mgr = self._create_state_manager()
            bookmark = mgr.get_aggregator_bookmark(
                name=playbook_name, version=self.agent_version
            )
            last_processed_id = bookmark if bookmark is not None else 0

        # Count user playbooks with ID greater than last processed using efficient count query
        # Only count current user playbooks (status=None), not archived or pending ones.
        # Singleton aggregation operates on the user's whole playbook set — no name filter.
        new_count = self.storage.count_user_playbooks(  # pyright: ignore[reportOptionalMemberAccess]
            min_user_playbook_id=last_processed_id,
            agent_version=self.agent_version,
            status_filter=[None],
        )

        logger.info(
            "Found %d new user playbooks for '%s' (agent_version=%s, last processed ID: %d)",
            new_count,
            playbook_name,
            self.agent_version,
            last_processed_id,
        )

        return new_count

    def _should_run_aggregation(
        self,
        playbook_name: str,
        playbook_aggregator_config: PlaybookAggregatorConfig,
        rerun: bool = False,
    ) -> bool:
        """
        Check if aggregation should run based on new user playbooks count.

        Args:
            playbook_name: Name of the playbook type
            playbook_aggregator_config: Configuration for playbook aggregator
            rerun: If True, count all user playbooks to determine if aggregation is needed

        Returns:
            bool: True if aggregation should run, False otherwise
        """
        # Get reaggregation_trigger_count, default to 2 if not set or 0
        trigger_count = playbook_aggregator_config.reaggregation_trigger_count
        if trigger_count <= 0:
            trigger_count = 2

        # Check new user playbooks count (uses all playbooks if rerun=True)
        new_count = self._get_new_user_playbooks_count(playbook_name, rerun=rerun)

        return new_count >= trigger_count

    def _update_operation_state(
        self, playbook_name: str, user_playbooks: list[UserPlaybook]
    ) -> None:
        """
        Update operation state with the highest user_playbook_id processed.

        Args:
            playbook_name: Name of the playbook type
            user_playbooks: List of user playbooks that were processed
        """
        if not user_playbooks:
            return

        # Find max user_playbook_id
        max_id = max(playbook.user_playbook_id for playbook in user_playbooks)

        mgr = self._create_state_manager()
        mgr.update_aggregator_bookmark(
            name=playbook_name,
            version=self.agent_version,
            last_processed_id=max_id,
        )

    # ===============================
    # private methods - aggregation pre/post-processing
    # ===============================
    # Bodies live on the AggregationPostProcessing component (self._postproc).
    # These thin delegators are kept for OSS test call-sites that invoke them by
    # name (test_playbook_aggregator.py). Internal callers use self._postproc.

    def _postprocess_aggregation_output(
        self,
        value: object,
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> tuple[object, int]:
        return self._postproc._postprocess_aggregation_output(value, processing_context)

    def _aggregation_prompt_extra_instructions_for_context(
        self,
        processing_context: AggregationPromptProcessingContext | None,
    ) -> str:
        return self._postproc._aggregation_prompt_extra_instructions_for_context(
            processing_context
        )

    def _record_postprocessing_artifacts(self, artifact_count: int) -> None:
        self._postproc._record_postprocessing_artifacts(artifact_count)

    @staticmethod
    def _get_direction_key(fb: UserPlaybook) -> str:
        return aggregator_prompt_formatting.get_direction_key(fb)

    @staticmethod
    def _token_overlap(str1: str, str2: str, threshold: float = 0.6) -> bool:
        return aggregator_prompt_formatting.token_overlap(str1, str2, threshold)

    @staticmethod
    def _group_playbooks_by_direction(
        cluster_playbooks: list[UserPlaybook],
        threshold: float = 0.6,
    ) -> list[list[UserPlaybook]]:
        return aggregator_prompt_formatting.group_playbooks_by_direction(
            cluster_playbooks, threshold
        )

    def _format_structured_cluster_input(
        self,
        cluster_playbooks: list[UserPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> str:
        return aggregator_prompt_formatting.format_structured_cluster_input(
            cluster_playbooks,
            direction_overlap_threshold=direction_overlap_threshold,
        )

    # ===============================
    # private methods - cluster change detection
    # ===============================

    @staticmethod
    def _compute_cluster_fingerprint(cluster_playbooks: list[UserPlaybook]) -> str:
        return aggregator_clustering.compute_cluster_fingerprint(cluster_playbooks)

    def _determine_cluster_changes(
        self,
        clusters: dict[int, list[UserPlaybook]],
        prev_fingerprints: dict,
    ) -> tuple[dict[int, list[UserPlaybook]], list[int]]:
        return aggregator_clustering.determine_cluster_changes(
            clusters, prev_fingerprints
        )

    # ===============================
    # public methods
    # ===============================

    def run(self, playbook_aggregator_request: PlaybookAggregatorRequest) -> dict:  # noqa: C901
        """Run playbook aggregation.

        Returns:
            dict: Aggregation stats with keys: clusters_found, user_playbooks_processed, playbooks_generated, skipped (optional)
        """
        aggregation_start = time.perf_counter()
        # Stable id for this aggregation run — groups all lineage events produced below.
        _run_id = playbook_aggregator_request.operation_key or str(uuid.uuid4())
        _empty_stats = {
            "clusters_found": 0,
            "user_playbooks_processed": 0,
            "playbooks_generated": 0,
        }

        if (
            self.effect_coordinator is None
            and playbook_aggregator_request.operation_key
            and any(
                event.op == "aggregate"
                for event in self.storage.get_lineage_events(  # type: ignore[reportOptionalMemberAccess]
                    request_id=playbook_aggregator_request.operation_key
                )
            )
        ):
            logger.info(
                "Skipping aggregation operation %s because its effects already exist",
                playbook_aggregator_request.operation_key,
            )
            return {
                **_empty_stats,
                "skipped": "operation already applied",
            }

        # Singleton aggregation: one playbook kind per org. The name is a fixed
        # constant used only for bookmark/archive scoping and telemetry — it is
        # never a selection filter on the read queries below.
        playbook_name = SINGLETON_USER_PLAYBOOK_NAME

        # get playbook aggregator config
        playbook_aggregator_config = self._get_playbook_aggregator_config()
        if (
            not playbook_aggregator_config
            or playbook_aggregator_config.min_cluster_size < 2
        ):
            skip_reason = "no aggregator config or min_cluster_size < 2"
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_gate_evaluated",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="should_skip",
                metadata={"skip_reason": skip_reason},
            )
            logger.info(
                "Skipping user playbook aggregation for '%s' (agent_version=%s): no aggregator config or min_cluster_size < 2, config: %s",
                playbook_name,
                self.agent_version,
                playbook_aggregator_config,
            )
            return {
                **_empty_stats,
                "skipped": skip_reason,
            }

        # Check if we should run aggregation based on new playbooks count
        # For rerun, use all user playbooks (last_processed_id=0) to determine if aggregation is needed
        if not self._should_run_aggregation(
            playbook_name,
            playbook_aggregator_config,
            rerun=playbook_aggregator_request.rerun,
        ):
            new_count = self._get_new_user_playbooks_count(
                playbook_name,
                rerun=playbook_aggregator_request.rerun,
            )
            trigger_count = (
                playbook_aggregator_config.reaggregation_trigger_count
                if playbook_aggregator_config.reaggregation_trigger_count > 0
                else 2
            )
            logger.info(
                "Skipping user playbook aggregation for '%s' (agent_version=%s) - only %d new user playbooks (need %d)",
                playbook_name,
                self.agent_version,
                new_count,
                trigger_count,
            )
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_gate_evaluated",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="should_skip",
                count_value=new_count,
                metadata={
                    "new_user_playbooks": new_count,
                    "trigger_count": trigger_count,
                },
            )
            return {
                **_empty_stats,
                "skipped": f"not enough new playbooks ({new_count} < {trigger_count})",
            }

        record_usage_event(
            org_id=self.request_context.org_id,
            event_name="aggregation_gate_evaluated",
            event_category="aggregation",
            pipeline="playbook",
            playbook_name=playbook_name,
            agent_version=self.agent_version,
            outcome="should_run",
        )
        logger.info(
            "Running user playbook aggregation for '%s' (agent_version=%s)",
            playbook_name,
            self.agent_version,
        )
        logger.info(
            "Aggregation prompt processor: %s",
            type(self.aggregation_prompt_processor).__name__
            if self.aggregation_prompt_processor is not None
            else "disabled",
        )

        # Get existing APPROVED and PENDING playbooks before archiving (to pass to LLM for deduplication).
        # Singleton aggregation pulls the user's whole set — no name filter.
        existing_playbooks = self.storage.get_agent_playbooks(  # type: ignore[reportOptionalMemberAccess]
            status_filter=[None],  # Current playbooks only
            playbook_status_filter=[PlaybookStatus.APPROVED, PlaybookStatus.PENDING],
        )
        logger.info(
            "Found %s existing playbooks (approved + pending) to preserve",
            len(existing_playbooks),
        )

        # get all user playbooks and generate clusters
        user_playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            agent_version=self.agent_version,
            include_embedding=True,
        )
        full_archive_playbook_names = sorted(
            {
                playbook.playbook_name
                for playbook in [*existing_playbooks, *user_playbooks]
                if playbook.playbook_name
            }
            | {playbook_name}
        )
        clusters = self.get_clusters(user_playbooks, playbook_aggregator_config)

        # Determine which clusters changed (skip for rerun)
        mgr = self._create_state_manager()
        archived_playbook_ids = []
        full_archive = False
        prev_fingerprints: dict = {}  # Populated for incremental mode

        # Deferred-archive flag: full archive is performed AFTER LLM generation,
        # and only when at least one new playbook was produced. Avoids silently
        # dropping existing PENDING/APPROVED playbooks when the LLM returns
        # null (cluster identified as duplicate of existing).
        pending_full_archive = False

        if playbook_aggregator_request.rerun:
            logger.info("Rerun requested: bypassing cluster change detection")
            changed_clusters = clusters
            full_archive = True
            pending_full_archive = True
        else:
            # Load previous fingerprints and detect changes
            prev_fingerprints = mgr.get_cluster_fingerprints(
                name=playbook_name, version=self.agent_version
            )

            if not prev_fingerprints:
                logger.info(
                    "No previous cluster fingerprints found, treating all clusters as changed"
                )
                changed_clusters = clusters
                full_archive = True
                pending_full_archive = True
            else:
                (
                    changed_clusters,
                    archived_playbook_ids,
                ) = self._determine_cluster_changes(clusters, prev_fingerprints)

                if not changed_clusters and not archived_playbook_ids:
                    logger.info(
                        "No cluster changes detected for '%s', skipping LLM calls",
                        playbook_name,
                    )
                    result = {
                        **_empty_stats,
                        "skipped": "no cluster changes detected",
                    }
                    if self.effect_coordinator is None:
                        self._update_operation_state(playbook_name, user_playbooks)
                    else:
                        self.effect_coordinator.prepare([])
                        with self.effect_coordinator.apply_scope():
                            self._update_operation_state(playbook_name, user_playbooks)
                            self.effect_coordinator.complete(result)
                    record_usage_event(
                        org_id=self.request_context.org_id,
                        event_name="aggregation_succeeded",
                        event_category="aggregation",
                        pipeline="playbook",
                        playbook_name=playbook_name,
                        agent_version=self.agent_version,
                        outcome="success",
                        count_value=0,
                        duration_ms=int(
                            (time.perf_counter() - aggregation_start) * 1000
                        ),
                        metadata={"skipped": "no cluster changes detected"},
                    )
                    return result

                logger.info(
                    "Detected %d changed clusters, %d playbooks to archive",
                    len(changed_clusters),
                    len(archived_playbook_ids),
                )

        effect_scope: AbstractContextManager[None] | None = None
        try:
            # Emit the started event inside the protected block so any failure
            # from here on is paired with an aggregation_failed event.
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_started",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="started",
            )
            # Generate new playbooks only for changed clusters while preserving
            # the exact source cluster for each non-duplicate playbook.
            generated_pairs = self._generate_playbooks_with_source_clusters(
                changed_clusters,
                existing_playbooks,
                direction_overlap_threshold=playbook_aggregator_config.direction_overlap_threshold,
            )
            new_playbooks = [playbook for playbook, _, _ in generated_pairs]
            if self.effect_coordinator is not None:
                self.effect_coordinator.prepare(new_playbooks)
                pending_scope = self.effect_coordinator.apply_scope()
                pending_scope.__enter__()
                effect_scope = pending_scope

            previous_fingerprints_for_changed_clusters = {}
            changed_fps_by_previous_fp = {}
            changed_fps_with_replacements = set()
            previous_playbook_id_by_fp = {}
            if not playbook_aggregator_request.rerun and prev_fingerprints:
                for cluster_playbooks in changed_clusters.values():
                    fp = self._compute_cluster_fingerprint(cluster_playbooks)
                    current_user_ids = {
                        fb.user_playbook_id
                        for fb in cluster_playbooks
                        if fb.user_playbook_id is not None
                    }
                    matched_prev_fingerprints = {
                        prev_fp: fp_data
                        for prev_fp, fp_data in prev_fingerprints.items()
                        if fp_data.get("agent_playbook_id") is not None
                        and current_user_ids
                        & set(fp_data.get("user_playbook_ids") or [])
                    }
                    if matched_prev_fingerprints:
                        previous_fingerprints_for_changed_clusters[fp] = (
                            matched_prev_fingerprints
                        )
                        for prev_fp, fp_data in matched_prev_fingerprints.items():
                            changed_fps_by_previous_fp.setdefault(prev_fp, set()).add(
                                fp
                            )
                            playbook_id = fp_data.get("agent_playbook_id")
                            if playbook_id is not None:
                                previous_playbook_id_by_fp[prev_fp] = playbook_id

            # Lazy archive: only full-archive when the LLM produced replacements.
            # Skipping the archive when new_playbooks is empty preserves existing
            # PENDING/APPROVED playbooks that the LLM identified as duplicates.
            if pending_full_archive:
                if new_playbooks:
                    for name in full_archive_playbook_names:
                        self.storage.archive_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                            name, agent_version=self.agent_version
                        )
                else:
                    logger.info(
                        "Skipping full archive of %s (agent_version=%s): LLM produced 0 new playbooks; existing PENDING/APPROVED playbooks preserved",
                        full_archive_playbook_names,
                        self.agent_version,
                    )
                    full_archive = False

            # Build new fingerprint state
            new_fingerprints = {}

            if not playbook_aggregator_request.rerun:
                # Carry forward unchanged fingerprints from previous state
                prev_fps = prev_fingerprints
                current_fp_set = set()
                for cluster_playbooks in clusters.values():
                    fp = self._compute_cluster_fingerprint(cluster_playbooks)
                    current_fp_set.add(fp)

                changed_fp_set = set()
                for cluster_playbooks in changed_clusters.values():
                    changed_fp_set.add(
                        self._compute_cluster_fingerprint(cluster_playbooks)
                    )

                # Carry forward unchanged clusters (still exist and not changed)
                new_fingerprints.update(
                    {
                        fp: fp_data
                        for fp, fp_data in prev_fps.items()
                        if fp in current_fp_set and fp not in changed_fp_set
                    }
                )

            saved_playbook_list: list[AgentPlaybook] = []
            selective_supersede_playbook_ids = set()
            replaced_previous_fingerprints = set()

            # Save each playbook + its aggregate event atomically, then assign
            # fingerprints and source-windows for the saved row.
            for playbook, cluster_playbooks, provenance in generated_pairs:
                run_mode = "full_archive" if full_archive else "incremental"
                member_ids = [
                    str(fb.user_playbook_id)
                    for fb in cluster_playbooks
                    if fb.user_playbook_id
                ]
                if self.effect_coordinator is None:
                    saved_fb = self.storage.save_agent_playbooks(  # type: ignore[reportOptionalMemberAccess]
                        [playbook],
                        lineage_contexts=[
                            LineageContext(
                                op_kind="aggregate",
                                actor="aggregator",
                                request_id=_run_id,
                                source_ids=member_ids,
                                reason=f"{AGGREGATE_REASON_PREFIX}{run_mode}",
                                model_name=(
                                    provenance.model_name if provenance else None
                                ),
                                provider=provenance.provider if provenance else None,
                            )
                        ],
                    )[0]
                else:
                    saved_fb = self.effect_coordinator.save_agent_playbook(
                        playbook,
                        source_ids=member_ids,
                        request_id=_run_id,
                        run_mode=run_mode,
                        provenance=provenance,
                    )
                saved_playbook_list.append(saved_fb)
                if saved_fb and saved_fb.agent_playbook_id:
                    fp_key = self._compute_cluster_fingerprint(cluster_playbooks)
                    changed_fps_with_replacements.add(fp_key)
                    raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
                    new_fingerprints[fp_key] = {
                        "agent_playbook_id": saved_fb.agent_playbook_id,
                        "user_playbook_ids": raw_ids,
                    }
                    for prev_fp in previous_fingerprints_for_changed_clusters.get(
                        fp_key, {}
                    ):
                        all_overlapping_clusters_replaced = (
                            changed_fps_by_previous_fp.get(prev_fp, set()).issubset(
                                changed_fps_with_replacements
                            )
                        )
                        playbook_id = previous_playbook_id_by_fp.get(prev_fp)
                        if all_overlapping_clusters_replaced:
                            replaced_previous_fingerprints.add(prev_fp)
                            if playbook_id is not None:
                                selective_supersede_playbook_ids.add(playbook_id)
                    self.storage.set_source_windows_for_agent_playbook(  # type: ignore[reportOptionalMemberAccess]
                        saved_fb.agent_playbook_id,
                        [
                            AgentPlaybookSourceWindow(
                                user_playbook_id=fb.user_playbook_id,
                                source_interaction_ids=list(fb.source_interaction_ids),
                            )
                            for fb in sorted(
                                cluster_playbooks,
                                key=lambda item: item.user_playbook_id,
                            )
                        ],
                    )

            # Changed clusters that did not get a replacement keep their previous
            # fingerprint/playbook mapping so a later successful replacement can
            # supersede the old playbook. Brand-new duplicate clusters still get
            # a None marker to avoid repeated LLM calls for the same fingerprint.
            for cluster_playbooks in changed_clusters.values():
                fp = self._compute_cluster_fingerprint(cluster_playbooks)
                if fp in new_fingerprints:
                    continue

                previous_for_cluster = previous_fingerprints_for_changed_clusters.get(
                    fp, {}
                )
                preserved_previous = {
                    prev_fp: fp_data
                    for prev_fp, fp_data in previous_for_cluster.items()
                    if prev_fp not in replaced_previous_fingerprints
                }
                if preserved_previous:
                    new_fingerprints.update(preserved_previous)
                    continue

                raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
                new_fingerprints[fp] = {
                    "agent_playbook_id": None,
                    "user_playbook_ids": raw_ids,
                }

            # Store fingerprints in operation state
            mgr.update_cluster_fingerprints(
                name=playbook_name,
                version=self.agent_version,
                fingerprints=new_fingerprints,
            )

            # Update operation state with the highest user_playbook_id processed
            self._update_operation_state(playbook_name, user_playbooks)

            # Remove archived playbooks after successful aggregation. ALWAYS soft-supersede
            # (never hard-delete) so the removal is reconstructable from lineage — mirrors the
            # profile dedup always-soft path (#206).
            archived_ids_without_overlapping_changed_cluster: set[int] = set()
            if new_playbooks and prev_fingerprints:
                archived_id_set = set(archived_playbook_ids)
                for prev_fp, fp_data in prev_fingerprints.items():
                    playbook_id = fp_data.get("agent_playbook_id")
                    if (
                        playbook_id in archived_id_set
                        and prev_fp not in changed_fps_by_previous_fp
                    ):
                        archived_ids_without_overlapping_changed_cluster.add(
                            playbook_id
                        )
            ids_to_supersede = {
                *selective_supersede_playbook_ids,
                *archived_ids_without_overlapping_changed_cluster,
            }
            if not _run_id:
                # Empty request_id makes the removal unreconstructable (lineage events are keyed
                # on it). Fail loud and skip removal — never silently hard-delete.
                capture_anomaly(
                    "lineage.aggregation.missing_request_id",
                    level="error",
                    org_id=self.request_context.org_id,
                )
            else:
                try:
                    if full_archive:
                        for name in full_archive_playbook_names:
                            self.storage.supersede_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                                name,
                                agent_version=self.agent_version,
                                request_id=_run_id,
                            )
                    elif ids_to_supersede:
                        self.storage.supersede_agent_playbooks_by_ids(  # type: ignore[reportOptionalMemberAccess]
                            sorted(ids_to_supersede),
                            request_id=_run_id,
                        )
                    elif archived_playbook_ids:
                        logger.info(
                            "Skipping selective supersede of %s (agent_version=%s): LLM produced 0 new playbooks; existing PENDING/APPROVED playbooks preserved",
                            archived_playbook_ids,
                            self.agent_version,
                        )
                except Exception:
                    with sentry_tags(
                        subsystem="playbook_aggregation",
                        op="supersede_agent_playbooks",
                        org_id=self.request_context.org_id,
                        request_id=_run_id,
                    ):
                        logger.exception(
                            "Failed to soft-supersede archived agent playbooks (run %s)",
                            _run_id,
                        )
                    capture_anomaly(
                        "lineage.aggregation.supersede_failed",
                        level="error",
                        org_id=self.request_context.org_id,
                        request_id=_run_id,
                    )
                    if self.effect_coordinator is not None:
                        raise

            stats = {
                "clusters_found": len(clusters),
                "user_playbooks_processed": len(user_playbooks),
                "playbooks_generated": len(saved_playbook_list),
            }
            if self.effect_coordinator is not None:
                self.effect_coordinator.complete(stats)
                if effect_scope is None:
                    raise RuntimeError("aggregation effect scope was not entered")
                completed_scope = effect_scope
                effect_scope = None
                completed_scope.__exit__(None, None, None)

            self._enqueue_playbook_optimization(saved_playbook_list)
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_succeeded",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="success",
                count_value=len(saved_playbook_list),
                duration_ms=int((time.perf_counter() - aggregation_start) * 1000),
                metadata=stats,
            )
            self._record_learnings_generated(
                learning_ids=[
                    str(saved.agent_playbook_id)
                    for saved in saved_playbook_list
                    if getattr(saved, "agent_playbook_id", None)
                ],
                playbook_name=playbook_name,
                request_id=_run_id,
                metadata=stats,
                total_count=len(saved_playbook_list),
            )
            return stats

        except Exception as e:
            if effect_scope is not None:
                failed_scope = effect_scope
                effect_scope = None
                failed_scope.__exit__(type(e), e, e.__traceback__)
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_failed",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="failed",
                duration_ms=int((time.perf_counter() - aggregation_start) * 1000),
                error_kind=type(e).__name__,
            )
            if self.effect_coordinator is None:
                logger.error(
                    "Error during playbook aggregation for '%s': %s. Restoring archived playbooks.",
                    playbook_name,
                    str(e),
                )
                if full_archive:
                    for name in full_archive_playbook_names:
                        self.storage.restore_archived_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                            name, agent_version=self.agent_version
                        )
                elif archived_playbook_ids:
                    self.storage.restore_archived_agent_playbooks_by_ids(  # type: ignore[reportOptionalMemberAccess]
                        archived_playbook_ids
                    )
            else:
                logger.error(
                    "Error during managed playbook aggregation for '%s': %s. The effect transaction was rolled back.",
                    playbook_name,
                    str(e),
                )
            # Re-raise the exception after restoring
            raise

    def _record_learnings_generated(
        self,
        *,
        learning_ids: list[str],
        playbook_name: str,
        request_id: str,
        metadata: Mapping[str, Any],
        total_count: int | None = None,
    ) -> None:
        """Emit ``learnings_generated`` for a completed aggregation run.

        Prefers one event per learning id (entity-backed) when every saved
        playbook in this run carries a durable ``agent_playbook_id`` — the
        common case, since ``save_agent_playbooks``
        raises rather than returning a partial row. Falls back to the
        count-based aggregate event when ``learning_ids`` is short of
        ``total_count`` (a falsy/unset id slipped through), mirroring
        ``ExtractionResumeWorker._record_finalized_learnings`` — this avoids
        emitting a colliding ``learn:agent_playbook:0`` key. ``total_count``
        defaults to ``len(learning_ids)`` so callers that already guarantee a
        complete id list (e.g. existing tests) are unaffected.
        """
        from reflexio.server.billing_meter import (
            emit_learnings_generated,
            emit_learnings_generated_records,
        )

        total = len(learning_ids) if total_count is None else total_count
        if len(learning_ids) == total:
            emit_learnings_generated_records(
                org_id=self.request_context.org_id,
                configurator=self.configurator,
                learning_ids=learning_ids,
                source="aggregation",
                pipeline="playbook",
                request_id=request_id,
                agent_version=self.agent_version,
                playbook_name=playbook_name,
                entity_type="agent_playbook",
                metadata=metadata,
            )
            return
        emit_learnings_generated(
            org_id=self.request_context.org_id,
            configurator=self.configurator,
            count=total,
            source="aggregation",
            pipeline="playbook",
            request_id=request_id,
            agent_version=self.agent_version,
            playbook_name=playbook_name,
            entity_type="agent_playbook",
            metadata=metadata,
        )

    def get_clusters(
        self,
        user_playbooks: list[UserPlaybook],
        playbook_aggregator_config: PlaybookAggregatorConfig,
    ) -> dict[int, list[UserPlaybook]]:
        """
        Cluster user playbooks based on their embeddings (trigger indexed).

        Args:
            user_playbooks: Contains user playbooks to cluster
            playbook_aggregator_config: AgentPlaybook aggregator config

        Returns:
            dict[int, list[UserPlaybook]]: Dictionary mapping cluster IDs to lists of user playbooks
        """
        if not playbook_aggregator_config:
            logger.info(
                "No playbook aggregator config found, skipping playbook aggregation"
            )
            return {}

        min_cluster_size = playbook_aggregator_config.min_cluster_size
        similarity_threshold = playbook_aggregator_config.clustering_similarity

        if not user_playbooks:
            logger.info("No user playbooks to cluster")
            return {}

        # Mock mode: cluster by trigger
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: clustering by trigger")
            return aggregator_clustering.cluster_by_trigger_mock(
                user_playbooks, min_cluster_size
            )

        # Extract embeddings from user playbooks
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances

        embeddings = np.array([playbook.embedding for playbook in user_playbooks])

        if len(embeddings) < min_cluster_size:
            logger.info(
                "Not enough playbooks to cluster (got %d, need %d)",
                len(embeddings),
                min_cluster_size,
            )
            return {}

        # Compute cosine distance matrix for better text embedding clustering
        distance_matrix = cosine_distances(embeddings)

        # Choose algorithm based on dataset size
        # Convert similarity threshold to distance threshold (distance = 1 - similarity)
        distance_threshold = 1.0 - similarity_threshold
        if len(embeddings) < CLUSTERING_ALGORITHM_THRESHOLD:
            cluster_labels = self._cluster_with_agglomerative(
                distance_matrix, min_cluster_size, distance_threshold
            )
        else:
            cluster_labels = self._cluster_with_hdbscan(
                distance_matrix, min_cluster_size, distance_threshold
            )

        # Group playbooks by cluster
        clusters: dict[int, list[UserPlaybook]] = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Skip noise points from HDBSCAN
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(user_playbooks[idx])

        # Filter out clusters smaller than min_cluster_size
        clusters = {
            label: playbooks
            for label, playbooks in clusters.items()
            if len(playbooks) >= min_cluster_size
        }

        logger.info(
            "Found %d clusters from %d playbooks", len(clusters), len(user_playbooks)
        )
        for cluster_id, cluster_playbooks in clusters.items():
            logger.info("Cluster %d: %d playbooks", cluster_id, len(cluster_playbooks))

        return clusters

    def _cluster_with_agglomerative(
        self,
        distance_matrix: np.ndarray,
        min_cluster_size: int,
        distance_threshold: float,
    ) -> np.ndarray:
        return aggregator_clustering.cluster_with_agglomerative(
            distance_matrix, min_cluster_size, distance_threshold
        )

    def _cluster_with_hdbscan(
        self,
        distance_matrix: np.ndarray,
        min_cluster_size: int,
        distance_threshold: float,
    ) -> np.ndarray:
        return aggregator_clustering.cluster_with_hdbscan(
            distance_matrix, min_cluster_size, distance_threshold
        )

    def _generate_playbooks_with_source_clusters(
        self,
        clusters: dict[int, list[UserPlaybook]],
        existing_approved_playbooks: list[AgentPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> list[tuple[AgentPlaybook, list[UserPlaybook], ModelProvenance | None]]:
        """Generate playbooks with their exact source cluster and provenance."""
        new_playbooks: list[
            tuple[AgentPlaybook, list[UserPlaybook], ModelProvenance | None]
        ] = []
        approved_playbooks_str = (
            "\n".join([f"- {fb.content}" for fb in existing_approved_playbooks])
            if existing_approved_playbooks
            else "None"
        )
        for cluster_playbooks in clusters.values():
            shared_state: dict[str, object] = {}
            processing_context = AggregationPromptProcessingContext(
                data={
                    "agent_version": self.agent_version,
                    "org_id": self.request_context.org_id,
                }
            )
            if self.aggregation_prompt_processor is None:
                prompt_cluster_playbooks = cluster_playbooks
            else:
                prompt_cluster_playbooks = [
                    self._postproc._preprocess_user_playbook_for_prompt(
                        playbook, shared_state, processing_context
                    )
                    for playbook in cluster_playbooks
                ]

            generated = self._generate_playbook_from_cluster(
                prompt_cluster_playbooks,
                approved_playbooks_str,
                direction_overlap_threshold=direction_overlap_threshold,
                processing_context=processing_context,
            )
            if generated is not None:
                playbook, provenance = generated
                new_playbooks.append((playbook, cluster_playbooks, provenance))
        return new_playbooks

    def _enqueue_playbook_optimization(
        self, saved_playbooks: Sequence[AgentPlaybook | None]
    ) -> None:
        config = self.configurator.get_config().playbook_optimizer_config
        if (
            getattr(config, "enabled", False) is not True
            or getattr(config, "optimize_agent_playbooks", False) is not True
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
                playbook is None
                or not playbook.agent_playbook_id
                or playbook.status is not None
                or playbook.playbook_status != PlaybookStatus.PENDING
            ):
                continue
            target = PlaybookOptimizationTarget(
                kind="agent_playbook", target_id=playbook.agent_playbook_id
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

    def _generate_playbook_from_cluster(
        self,
        cluster_playbooks: list[UserPlaybook],
        existing_approved_playbooks_str: str,
        direction_overlap_threshold: float = 0.6,
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> tuple[AgentPlaybook, ModelProvenance | None] | None:
        """
        Generate a playbook from a cluster using structured JSON output.

        Args:
            cluster_playbooks: List of raw playbooks in this cluster
            existing_approved_playbooks_str: Formatted string of existing approved playbooks
            direction_overlap_threshold: Token overlap threshold for grouping by direction

        Returns:
            Generated playbook and its provenance, or None if no new playbook is needed
        """
        if not cluster_playbooks:
            return None

        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            # Extract structured fields directly from cluster
            triggers = [fb.trigger for fb in cluster_playbooks if fb.trigger]

            trigger = triggers[0] if triggers else "in general"

            # Fall back to using content from first playbook if available
            first_content = cluster_playbooks[0].content
            if not first_content:
                logger.info("No valid content in cluster, skipping")
                return None

            # Build content directly as a freeform summary
            content_text = f"When {trigger}, {first_content}."

            response = PlaybookAggregationOutput(
                playbook=StructuredPlaybookContent(
                    content=content_text,
                    trigger=trigger,
                )
            )
            response, artifact_count = self._postproc._postprocess_aggregation_response(
                response,
                processing_context,
            )
            self._postproc._record_postprocessing_artifacts(artifact_count)
            playbook = self._process_aggregation_response(response, cluster_playbooks)
            if playbook is None:
                return None
            return (
                playbook.model_copy(update={"playbook_metadata": "mock_generated"}),
                None,
            )

        # Format raw playbooks for prompt using structured format
        raw_playbooks_str = self._format_structured_cluster_input(
            cluster_playbooks,
            direction_overlap_threshold=direction_overlap_threshold,
        )

        messages = [
            {
                "role": "user",
                "content": self.request_context.prompt_manager.render_prompt(
                    PlaybookServiceConstants.PLAYBOOK_AGGREGATION_PROMPT_ID,
                    {
                        "user_playbooks": raw_playbooks_str,
                        "existing_approved_playbooks": existing_approved_playbooks_str,
                        "aggregation_prompt_extra_instructions": self._postproc._aggregation_prompt_extra_instructions_for_context(
                            processing_context
                        ),
                    },
                ),
            }
        ]

        try:
            completion = self.client.generate_chat_response_with_provenance(
                messages=messages,
                model=self.client.config.model,
                response_format=PlaybookAggregationOutput,
                parse_structured_output=True,
            )
            response = completion.value
            model_provenance = completion.provenance
            if isinstance(response, PlaybookAggregationOutput):
                response, artifact_count = (
                    self._postproc._postprocess_aggregation_response(
                        response,
                        processing_context,
                    )
                )
                self._postproc._record_postprocessing_artifacts(artifact_count)
            else:
                response, artifact_count = (
                    self._postproc._postprocess_aggregation_output(
                        response,
                        processing_context,
                    )
                )
                self._postproc._record_postprocessing_artifacts(artifact_count)
            log_model_response(logger, "Aggregation structured response", response)

            if not isinstance(response, PlaybookAggregationOutput):
                logger.warning(
                    "LLM response was not parsed as PlaybookAggregationOutput (got %s), returning None.",
                    type(response).__name__,
                )
                return None

            playbook = self._process_aggregation_response(response, cluster_playbooks)
            if playbook is None:
                return None
            return playbook, model_provenance
        except Exception as exc:
            processed_error, artifact_count = (
                self._postproc._postprocess_aggregation_output(
                    str(exc),
                    processing_context,
                )
            )
            self._postproc._record_postprocessing_artifacts(artifact_count)
            logger.error(
                "AgentPlaybook aggregation failed due to %s, returning None.",
                processed_error,
            )
            return None

    def _process_aggregation_response(
        self, response: PlaybookAggregationOutput, cluster_playbooks: list[UserPlaybook]
    ) -> AgentPlaybook | None:
        """
        Process structured response from LLM into AgentPlaybook.

        Args:
            response: Parsed PlaybookAggregationOutput from LLM
            cluster_playbooks: Cluster playbooks used only for non-user metadata
                such as playbook name and agent version. Callers may pass
                prompt-preprocessed copies here, so this method must not read
                user-authored fields from them.

        Returns:
            AgentPlaybook or None if no playbook should be generated
        """
        if not response:
            return None

        structured = response.playbook
        if structured is None:
            logger.info("LLM returned null playbook (duplicate of existing)")
            return None

        # content is always the LLM's freeform summary;
        # fall back to formatted structured fields for backward compatibility
        playbook_content = ensure_playbook_content(structured.content, structured)
        if not playbook_content.strip():
            logger.info("Aggregated playbook has no valid content, skipping")
            return None
        logger.info(
            "Aggregated playbook content (freeform): %.200s",
            playbook_content,
        )

        return AgentPlaybook(
            playbook_name=cluster_playbooks[0].playbook_name,
            agent_version=cluster_playbooks[0].agent_version,
            content=playbook_content,
            trigger=structured.trigger,
            rationale=structured.rationale,
            playbook_status=PlaybookStatus.PENDING,
            playbook_metadata="",
        )

    def _get_playbook_aggregator_config(self) -> PlaybookAggregatorConfig | None:
        root_config = self.configurator.get_config()
        playbook_config = getattr(root_config, "user_playbook_extractor_config", None)
        if not playbook_config:
            return None
        return playbook_config.aggregation_config
