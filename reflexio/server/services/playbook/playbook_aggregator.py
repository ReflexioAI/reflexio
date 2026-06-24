from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
    PlaybookAggregatorRequest,
    ensure_playbook_content,
)
from reflexio.server.services.service_utils import log_model_response
from reflexio.server.tracing import capture_anomaly, sentry_tags
from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)

# Threshold for switching between clustering algorithms
# Below this, use Agglomerative (works better with small datasets)
# Above this, use HDBSCAN (scales better, handles noise)
CLUSTERING_ALGORITHM_THRESHOLD = 50


class PlaybookAggregator:
    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        agent_version: str,
    ) -> None:
        self.client = llm_client
        self.storage = request_context.storage
        self.configurator = request_context.configurator
        self.request_context = request_context
        self.agent_version = agent_version

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

    def _format_cluster_input(self, cluster_playbooks: list[UserPlaybook]) -> str:
        """
        Format a cluster of playbooks for the aggregation prompt using per-item format.

        Each playbook is shown as a self-contained unit with content as the
        primary content, followed by optional structured fields as supplementary metadata.

        Args:
            cluster_playbooks: List of raw playbooks in this cluster

        Returns:
            str: Formatted input for the aggregation prompt
        """
        blocks = []
        for idx, fb in enumerate(cluster_playbooks, 1):
            lines = [f"[{idx}]"]
            if fb.content:
                lines.append(f'Content: "{fb.content}"')
            if fb.trigger:
                lines.append(f'Trigger: "{fb.trigger}"')
            if fb.rationale:
                lines.append(f'Rationale: "{fb.rationale}"')
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "(No playbook items)"

    @staticmethod
    def _get_direction_key(fb: UserPlaybook) -> str:
        """
        Extract a similarity key from a user playbook for grouping.

        Returns the raw content used for token-overlap comparison. Grouping is
        purely content-similarity based: under Option B a skill may legitimately
        hold mixed-orientation rules (do-rules and avoid-rules for different
        sub-aspects of one task), so whole-content polarity is NOT derived or
        gated here. Preserving distinct do/avoid rules when similar items are
        merged is the aggregation prompt's responsibility.

        Args:
            fb: A user playbook item

        Returns:
            str: Content used as the similarity key for grouping
        """
        return fb.content or ""

    @staticmethod
    def _token_overlap(str1: str, str2: str, threshold: float = 0.6) -> bool:
        """
        Check if two strings have significant token overlap using asymmetric containment.

        Computes the ratio of shared tokens to the smaller set, so a short string
        contained in a longer one still counts as a match.

        Args:
            str1: First string
            str2: Second string
            threshold: Minimum overlap ratio

        Returns:
            bool: True if overlap ratio >= threshold
        """
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        if not tokens1 or not tokens2:
            return False
        intersection = len(tokens1 & tokens2)
        overlap_ratio = max(intersection / len(tokens1), intersection / len(tokens2))
        return overlap_ratio >= threshold

    @staticmethod
    def _group_playbooks_by_direction(
        cluster_playbooks: list[UserPlaybook],
        threshold: float = 0.6,
    ) -> list[list[UserPlaybook]]:
        """
        Group playbooks by similarity of their content.

        Uses greedy single-linkage: each playbook is assigned to the first
        existing group that has any member with sufficient token overlap.
        Groups are returned sorted by size descending (largest first).

        Grouping is purely content-similarity based and does NOT gate on a
        derived whole-content polarity. Under Option B a skill may hold
        mixed-orientation rules (do-rules and avoid-rules for different
        sub-aspects), and whole-content polarity is undefined for such a skill.
        Keeping a do-rule and an avoid-rule as distinct rules when similar items
        are merged into one skill is the aggregation prompt's responsibility,
        not a mechanical split here.

        Args:
            cluster_playbooks: List of raw playbooks to group
            threshold: Token overlap threshold for grouping

        Returns:
            list[list[UserPlaybook]]: Groups sorted by size descending
        """
        groups: list[list[UserPlaybook]] = []

        for fb in cluster_playbooks:
            key = PlaybookAggregator._get_direction_key(fb)
            matched = False
            for group in groups:
                if any(
                    PlaybookAggregator._token_overlap(
                        key,
                        PlaybookAggregator._get_direction_key(group_fb),
                        threshold,
                    )
                    for group_fb in group
                ):
                    group.append(fb)
                    matched = True
                    break
            if not matched:
                groups.append([fb])

        # Sort by group size descending (largest first)
        groups.sort(key=len, reverse=True)
        return groups

    def _format_structured_cluster_input(
        self,
        cluster_playbooks: list[UserPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> str:
        """
        Format a cluster of playbooks for structured aggregation prompt.

        When the cluster forms a single similarity group, uses the flat-list
        format. When distinct similarity groups are detected (multiple groups),
        uses a grouped format so the LLM can see which items are similar and
        preserve distinct rules (e.g. a do-rule and an avoid-rule) as separate
        rules in the merged skill rather than collapsing them.

        Args:
            cluster_playbooks: List of raw playbooks in this cluster
            direction_overlap_threshold: Token overlap threshold for grouping by direction

        Returns:
            str: Formatted input for the aggregation prompt
        """
        groups = self._group_playbooks_by_direction(
            cluster_playbooks, threshold=direction_overlap_threshold
        )

        if len(groups) <= 1:
            return self._format_flat(cluster_playbooks)
        return self._format_grouped(groups)

    def _format_flat(self, cluster_playbooks: list[UserPlaybook]) -> str:
        """
        Format playbooks as flat bullet lists (original format, used when no conflict).

        Args:
            cluster_playbooks: List of raw playbooks in this cluster

        Returns:
            str: Formatted input with separate field lists
        """
        triggers = []
        rationales = []

        for fb in cluster_playbooks:
            if fb.trigger:
                triggers.append(fb.trigger)
            if fb.rationale:
                rationales.append(fb.rationale)

        lines: list[str] = []

        if triggers:
            lines.append("TRIGGER conditions (to be consolidated):")
            lines.extend(f"- {trigger}" for trigger in triggers)
        else:
            lines.append("TRIGGER conditions: (none specified)")

        if rationales:
            lines.append("RATIONALE summaries:")
            lines.extend(f"- {r}" for r in rationales)

        self._append_freeform_observations(lines, cluster_playbooks)

        return "\n".join(lines)

    def _format_grouped(
        self,
        groups: list[list[UserPlaybook]],
    ) -> str:
        """
        Format playbooks in grouped layout (used when conflicting directions are detected).

        Args:
            groups: AgentPlaybook groups sorted by size descending

        Returns:
            str: Formatted input with group headers and per-playbook fields
        """
        lines: list[str] = [
            "The following playbook items are grouped by similarity. "
            "Groups are ordered by size (largest first).",
            "",
        ]

        for idx, group in enumerate(groups, start=1):
            count_label = "playbook" if len(group) == 1 else "playbooks"
            lines.append(f"Group {idx} ({len(group)} {count_label}):")
            for fb in group:
                parts: list[str] = []
                if fb.trigger:
                    parts.append(f'Trigger: "{fb.trigger}"')
                if fb.rationale:
                    parts.append(f'Rationale: "{fb.rationale}"')
                if not parts and fb.content:
                    parts.append(f'AgentPlaybook: "{fb.content}"')
                if parts:
                    lines.append(f"  - {parts[0]}")
                    lines.extend(f"    {p}" for p in parts[1:])
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _append_freeform_observations(
        lines: list[str], cluster_playbooks: list[UserPlaybook]
    ) -> None:
        """Append freeform observations from cluster playbooks to output lines."""
        freeform_observations = [
            fb.content for fb in cluster_playbooks if not fb.trigger and fb.content
        ]
        if freeform_observations:
            lines.append("Freeform observations (from freeform cluster members):")
            lines.extend(f"- {obs}" for obs in freeform_observations)

    # ===============================
    # private methods - cluster change detection
    # ===============================

    @staticmethod
    def _compute_cluster_fingerprint(cluster_playbooks: list[UserPlaybook]) -> str:
        """
        Compute a fingerprint for a cluster based on its user_playbook_ids.
        The fingerprint is deterministic and order-independent.

        Args:
            cluster_playbooks: List of raw playbooks in this cluster

        Returns:
            str: SHA-256 hash (truncated to 16 hex chars) of sorted user_playbook_ids
        """
        sorted_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
        id_str = ",".join(str(id) for id in sorted_ids)
        return hashlib.sha256(id_str.encode()).hexdigest()[:16]

    def _determine_cluster_changes(
        self,
        clusters: dict[int, list[UserPlaybook]],
        prev_fingerprints: dict,
    ) -> tuple[dict[int, list[UserPlaybook]], list[int]]:
        """
        Compare current cluster fingerprints against stored fingerprints to determine changes.

        Args:
            clusters: Current clusters (cluster_id -> list of UserPlaybook)
            prev_fingerprints: Previous fingerprint state
                (fingerprint_hash -> {"agent_playbook_id": int, "user_playbook_ids": list})

        Returns:
            tuple of:
                - changed_clusters: Only clusters needing new LLM calls
                - playbook_ids_to_archive: Old playbook_ids from changed/disappeared clusters
        """
        # Compute fingerprints for current clusters
        current_fingerprints = {}
        for cluster_id, cluster_playbooks in clusters.items():
            fp = self._compute_cluster_fingerprint(cluster_playbooks)
            current_fingerprints[cluster_id] = fp

        current_fp_set = set(current_fingerprints.values())
        prev_fp_set = set(prev_fingerprints.keys())

        # Changed clusters: fingerprints that are new (not in previous state)
        changed_clusters = {}
        for cluster_id, fp in current_fingerprints.items():
            if fp not in prev_fp_set:
                changed_clusters[cluster_id] = clusters[cluster_id]

        # Playbook IDs to archive: from fingerprints that disappeared or changed
        playbook_ids_to_archive = []
        for fp, fp_data in prev_fingerprints.items():
            if fp not in current_fp_set:
                playbook_id = fp_data.get("agent_playbook_id")
                if playbook_id is not None:
                    playbook_ids_to_archive.append(playbook_id)

        return changed_clusters, playbook_ids_to_archive

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
        _run_id = str(uuid.uuid4())
        _empty_stats = {
            "clusters_found": 0,
            "user_playbooks_processed": 0,
            "playbooks_generated": 0,
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
                    # Still update bookmark
                    self._update_operation_state(playbook_name, user_playbooks)
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
                    return {**_empty_stats, "skipped": "no cluster changes detected"}

                logger.info(
                    "Detected %d changed clusters, %d playbooks to archive",
                    len(changed_clusters),
                    len(archived_playbook_ids),
                )

                # Selectively archive only playbooks from changed/disappeared clusters
                if archived_playbook_ids:
                    self.storage.archive_agent_playbooks_by_ids(archived_playbook_ids)  # type: ignore[reportOptionalMemberAccess]

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
            new_playbooks = [playbook for playbook, _ in generated_pairs]

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
                prev_fps = mgr.get_cluster_fingerprints(
                    name=playbook_name, version=self.agent_version
                )
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

            # Initialize changed cluster fingerprints before assigning saved IDs
            # below. ``generated_pairs`` preserves the exact source cluster for
            # every non-duplicate playbook the LLM produced.
            for cluster_playbooks in changed_clusters.values():
                fp = self._compute_cluster_fingerprint(cluster_playbooks)
                raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)

                new_fingerprints[fp] = {
                    "agent_playbook_id": None,
                    "user_playbook_ids": raw_ids,
                }

            saved_playbook_list: list[AgentPlaybook] = []

            # Save each playbook + its aggregate event atomically, then assign
            # fingerprints and source-windows for the saved row.
            for playbook, cluster_playbooks in generated_pairs:
                run_mode = "full_archive" if full_archive else "incremental"
                member_ids = [
                    str(fb.user_playbook_id)
                    for fb in cluster_playbooks
                    if fb.user_playbook_id
                ]
                saved_fb = self.storage.save_agent_playbook_with_aggregate_event(  # type: ignore[reportOptionalMemberAccess]
                    playbook,
                    source_ids=member_ids,
                    request_id=_run_id,
                    run_mode=run_mode,
                )
                saved_playbook_list.append(saved_fb)
                if saved_fb and saved_fb.agent_playbook_id:
                    fp_key = self._compute_cluster_fingerprint(cluster_playbooks)
                    raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)
                    new_fingerprints[fp_key] = {
                        "agent_playbook_id": saved_fb.agent_playbook_id,
                        "user_playbook_ids": raw_ids,
                    }
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
                    elif archived_playbook_ids:
                        self.storage.supersede_agent_playbooks_by_ids(  # type: ignore[reportOptionalMemberAccess]
                            archived_playbook_ids, request_id=_run_id
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

            self._enqueue_playbook_optimization(saved_playbook_list)

            stats = {
                "clusters_found": len(clusters),
                "user_playbooks_processed": len(user_playbooks),
                "playbooks_generated": len(saved_playbook_list),
            }
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
            return stats

        except Exception as e:
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
            # Restore archived playbooks if any error occurs during aggregation
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
            # Re-raise the exception after restoring
            raise

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
            return self._cluster_by_trigger_mock(user_playbooks, min_cluster_size)

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

    def _cluster_by_trigger_mock(
        self, user_playbooks: list[UserPlaybook], min_cluster_size: int
    ) -> dict[int, list[UserPlaybook]]:
        """
        Simple mock clustering by exact trigger match.

        Args:
            user_playbooks: List of user playbooks with trigger field
            min_cluster_size: Minimum number of playbooks per cluster

        Returns:
            dict[int, list[UserPlaybook]]: Clusters grouped by trigger
        """
        # Group by trigger
        condition_groups: dict[str, list[UserPlaybook]] = {}
        for fb in user_playbooks:
            condition = fb.trigger or ""
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(fb)

        # Convert to cluster format, filtering by min_cluster_size
        clusters: dict[int, list[UserPlaybook]] = {}
        cluster_id = 0
        for playbooks_group in condition_groups.values():
            if len(playbooks_group) >= min_cluster_size:
                clusters[cluster_id] = playbooks_group
                cluster_id += 1

        logger.info(
            "Mock mode: created %d trigger clusters from %d playbooks",
            len(clusters),
            len(user_playbooks),
        )
        return clusters

    def _cluster_with_agglomerative(
        self,
        distance_matrix: np.ndarray,
        min_cluster_size: int,  # noqa: ARG002
        distance_threshold: float,
    ) -> np.ndarray:
        """
        Cluster using Agglomerative Clustering - best for small datasets.

        Args:
            distance_matrix: Precomputed cosine distance matrix
            min_cluster_size: Minimum cluster size (used for logging only,
                              filtering happens in get_clusters)
            distance_threshold: Maximum cosine distance to merge clusters (1 - similarity_threshold)

        Returns:
            np.ndarray: Cluster labels for each point
        """
        from sklearn.cluster import AgglomerativeClustering

        logger.info(
            "Using Agglomerative Clustering for %d playbooks (< %d threshold), distance_threshold=%.2f",
            len(distance_matrix),
            CLUSTERING_ALGORITHM_THRESHOLD,
            distance_threshold,
        )

        clusterer = AgglomerativeClustering(
            n_clusters=None,  # type: ignore[reportArgumentType]
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )

        return clusterer.fit_predict(distance_matrix)

    def _cluster_with_hdbscan(
        self,
        distance_matrix: np.ndarray,
        min_cluster_size: int,
        distance_threshold: float,
    ) -> np.ndarray:
        """
        Cluster using HDBSCAN - best for large datasets with potential noise.

        Args:
            distance_matrix: Precomputed cosine distance matrix
            min_cluster_size: Minimum number of points to form a cluster
            distance_threshold: Maximum cosine distance for cluster merging (1 - similarity_threshold)

        Returns:
            np.ndarray: Cluster labels for each point (-1 indicates noise)
        """
        import hdbscan

        logger.info(
            "Using HDBSCAN for %d playbooks (>= %d threshold), distance_threshold=%.2f",
            len(distance_matrix),
            CLUSTERING_ALGORITHM_THRESHOLD,
            distance_threshold,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="precomputed",
            cluster_selection_epsilon=distance_threshold,
        )

        return clusterer.fit_predict(distance_matrix)

    def _generate_playbooks_from_clusters(
        self,
        clusters: dict[int, list[UserPlaybook]],
        existing_approved_playbooks: list[AgentPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> list[AgentPlaybook]:
        """
        Generate playbooks from clusters, considering existing approved playbooks.

        Args:
            clusters: Dictionary mapping cluster IDs to lists of raw playbooks
            existing_approved_playbooks: List of existing approved playbooks to avoid duplication
            direction_overlap_threshold: Token overlap threshold for grouping by direction

        Returns:
            list[AgentPlaybook]: List of newly generated playbooks (excludes duplicates)
        """
        return [
            playbook
            for playbook, _ in self._generate_playbooks_with_source_clusters(
                clusters,
                existing_approved_playbooks,
                direction_overlap_threshold=direction_overlap_threshold,
            )
        ]

    def _generate_playbooks_with_source_clusters(
        self,
        clusters: dict[int, list[UserPlaybook]],
        existing_approved_playbooks: list[AgentPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> list[tuple[AgentPlaybook, list[UserPlaybook]]]:
        """Generate agent playbooks while preserving their exact source cluster."""
        # Format existing approved playbooks for the prompt
        approved_playbooks_str = (
            "\n".join([f"- {fb.content}" for fb in existing_approved_playbooks])
            if existing_approved_playbooks
            else "None"
        )

        new_playbooks: list[tuple[AgentPlaybook, list[UserPlaybook]]] = []
        for cluster_playbooks in clusters.values():
            playbook = self._generate_playbook_from_cluster(
                cluster_playbooks,
                approved_playbooks_str,
                direction_overlap_threshold=direction_overlap_threshold,
            )
            if playbook is not None:
                new_playbooks.append((playbook, cluster_playbooks))
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
    ) -> AgentPlaybook | None:
        """
        Generate a playbook from a cluster using structured JSON output.

        Args:
            cluster_playbooks: List of raw playbooks in this cluster
            existing_approved_playbooks_str: Formatted string of existing approved playbooks
            direction_overlap_threshold: Token overlap threshold for grouping by direction

        Returns:
            AgentPlaybook | None: Generated playbook, or None if no new playbook needed
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

            return AgentPlaybook(
                playbook_name=cluster_playbooks[0].playbook_name,
                agent_version=cluster_playbooks[0].agent_version,
                content=content_text,
                trigger=trigger,
                playbook_status=PlaybookStatus.PENDING,
                playbook_metadata="mock_generated",
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
                    },
                ),
            }
        ]

        try:
            response = self.client.generate_chat_response(
                messages=messages,
                model=self.client.config.model,
                response_format=PlaybookAggregationOutput,
                parse_structured_output=True,
            )
            log_model_response(logger, "Aggregation structured response", response)

            if not isinstance(response, PlaybookAggregationOutput):
                logger.warning(
                    "LLM response was not parsed as PlaybookAggregationOutput (got %s), returning None.",
                    type(response).__name__,
                )
                return None

            return self._process_aggregation_response(response, cluster_playbooks)
        except Exception as exc:
            logger.error(
                "AgentPlaybook aggregation failed due to %s, returning None.",
                str(exc),
            )
            return None

    def _process_aggregation_response(
        self, response: PlaybookAggregationOutput, cluster_playbooks: list[UserPlaybook]
    ) -> AgentPlaybook | None:
        """
        Process structured response from LLM into AgentPlaybook.

        Args:
            response: Parsed PlaybookAggregationOutput from LLM
            cluster_playbooks: Original cluster playbooks for metadata

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
