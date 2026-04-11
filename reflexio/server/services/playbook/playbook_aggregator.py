import hashlib
import logging
import os

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentPlaybookSnapshot,
    AgentPlaybookUpdateEntry,
    PlaybookAggregationChangeLog,
    PlaybookStatus,
    StructuredData,
    UserPlaybook,
    agent_playbook_to_snapshot,
)
from reflexio.models.config_schema import (
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
        # Only count current user playbooks (status=None), not archived or pending ones
        new_count = self.storage.count_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            playbook_name=playbook_name,
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
            sd = fb.structured_data
            if sd.trigger:
                lines.append(f'Trigger: "{sd.trigger}"')
            if sd.instruction:
                lines.append(f'Instruction: "{sd.instruction}"')
            if sd.pitfall:
                lines.append(f'Pitfall: "{sd.pitfall}"')
            if sd.rationale:
                lines.append(f'Rationale: "{sd.rationale}"')
            if sd.blocking_issue:
                lines.append(
                    f"Blocked by: [{sd.blocking_issue.kind.value}] {sd.blocking_issue.details}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else "(No playbook items)"

    @staticmethod
    def _get_direction_key(fb: UserPlaybook) -> str:
        """
        Extract a direction key from a user playbook for similarity grouping.

        Args:
            fb: A user playbook item

        Returns:
            str: Concatenation of instruction + pitfall, or content as fallback
        """
        key = f"{fb.structured_data.instruction or ''} {fb.structured_data.pitfall or ''}".strip()
        if not key:
            key = fb.content or ""
        return key

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
        Group playbooks by similarity of their actionable content (instruction + pitfall).

        Uses greedy single-linkage: each playbook is assigned to the first existing group
        that has any member with sufficient token overlap. Groups are returned sorted by
        size descending (largest first).

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

        When all playbooks agree (single direction group), uses the flat-list format.
        When conflicting directions are detected (multiple groups), uses a grouped format
        so the LLM can see agreement/disagreement and apply majority-wins resolution.

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
        return self._format_grouped(groups, cluster_playbooks)

    def _format_flat(self, cluster_playbooks: list[UserPlaybook]) -> str:
        """
        Format playbooks as flat bullet lists (original format, used when no conflict).

        Args:
            cluster_playbooks: List of raw playbooks in this cluster

        Returns:
            str: Formatted input with separate field lists
        """
        instructions = []
        pitfalls = []
        triggers = []
        rationales = []

        for fb in cluster_playbooks:
            if fb.structured_data.instruction:
                instructions.append(fb.structured_data.instruction)
            if fb.structured_data.pitfall:
                pitfalls.append(fb.structured_data.pitfall)
            if fb.structured_data.trigger:
                triggers.append(fb.structured_data.trigger)
            if fb.structured_data.rationale:
                rationales.append(fb.structured_data.rationale)

        lines: list[str] = []

        if triggers:
            lines.append("TRIGGER conditions (to be consolidated):")
            lines.extend(f"- {trigger}" for trigger in triggers)
        else:
            lines.append("TRIGGER conditions: (none specified)")

        if instructions:
            lines.append("INSTRUCTION actions:")
            lines.extend(f"- {action}" for action in instructions)

        if pitfalls:
            lines.append("PITFALL actions:")
            lines.extend(f"- {action}" for action in pitfalls)

        if rationales:
            lines.append("RATIONALE summaries:")
            lines.extend(f"- {r}" for r in rationales)

        self._append_blocking_issues(lines, cluster_playbooks)
        self._append_freeform_observations(lines, cluster_playbooks)

        return "\n".join(lines)

    def _format_grouped(
        self,
        groups: list[list[UserPlaybook]],
        cluster_playbooks: list[UserPlaybook],
    ) -> str:
        """
        Format playbooks in grouped layout (used when conflicting directions are detected).

        Args:
            groups: AgentPlaybook groups sorted by size descending
            cluster_playbooks: All playbooks in the cluster (for blocking issues / freeform)

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
                if fb.structured_data.trigger:
                    parts.append(f'Trigger: "{fb.structured_data.trigger}"')
                if fb.structured_data.instruction:
                    parts.append(f'Instruction: "{fb.structured_data.instruction}"')
                if fb.structured_data.pitfall:
                    parts.append(f'Pitfall: "{fb.structured_data.pitfall}"')
                if fb.structured_data.rationale:
                    parts.append(f'Rationale: "{fb.structured_data.rationale}"')
                if not parts and fb.content:
                    parts.append(f'AgentPlaybook: "{fb.content}"')
                if parts:
                    lines.append(f"  - {parts[0]}")
                    lines.extend(f"    {p}" for p in parts[1:])
            lines.append("")

        self._append_blocking_issues(lines, cluster_playbooks)

        return "\n".join(lines)

    @staticmethod
    def _append_blocking_issues(
        lines: list[str], cluster_playbooks: list[UserPlaybook]
    ) -> None:
        """Append blocking issues from cluster playbooks to output lines."""
        blocking_issues = [
            f"[{fb.structured_data.blocking_issue.kind.value}] {fb.structured_data.blocking_issue.details}"
            for fb in cluster_playbooks
            if fb.structured_data.blocking_issue
        ]
        if blocking_issues:
            lines.append("BLOCKED BY issues:")
            lines.extend(f"- {issue}" for issue in blocking_issues)

    @staticmethod
    def _append_freeform_observations(
        lines: list[str], cluster_playbooks: list[UserPlaybook]
    ) -> None:
        """Append freeform observations from cluster playbooks to output lines."""
        freeform_observations = [
            fb.content
            for fb in cluster_playbooks
            if not fb.structured_data.trigger and fb.content
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

    def _build_change_log(
        self,
        playbook_name: str,
        full_archive: bool,
        before_playbooks_by_id: dict[int, AgentPlaybook],
        saved_playbooks: list[AgentPlaybook],
        archived_playbook_ids: list[int],
        prev_fingerprints: dict,
    ) -> PlaybookAggregationChangeLog:
        """Build a PlaybookAggregationChangeLog from the aggregation run results.

        Args:
            playbook_name: The playbook name being aggregated
            full_archive: Whether this was a full archive (rerun/first run)
            before_playbooks_by_id: Snapshot of playbooks before archiving, keyed by playbook_id
            saved_playbooks: Newly saved playbooks from this run
            archived_playbook_ids: AgentPlaybook IDs that were selectively archived (incremental mode)
            prev_fingerprints: Previous cluster fingerprints (empty for full archive)

        Returns:
            PlaybookAggregationChangeLog with added/removed/updated lists populated
        """
        added: list[AgentPlaybookSnapshot] = []
        removed: list[AgentPlaybookSnapshot] = []
        updated: list[AgentPlaybookUpdateEntry] = []

        if full_archive:
            # No 1:1 mapping — all old playbooks are removed, all new are added
            removed = [
                agent_playbook_to_snapshot(fb) for fb in before_playbooks_by_id.values()
            ]
            added = [agent_playbook_to_snapshot(fb) for fb in saved_playbooks if fb]
        else:
            # Incremental mode: map old playbook_ids to new playbooks via fingerprints
            # Build a set of old playbook_ids that were archived
            archived_id_set = set(archived_playbook_ids)

            # Build mapping: old_playbook_id -> new_playbook_id via fingerprint changes
            # prev_fingerprints maps fp_hash -> {playbook_id, user_playbook_ids}
            # new_fingerprints maps fp_hash -> {playbook_id, user_playbook_ids}
            # If an old fingerprint disappeared and a new one appeared, and
            # the old fp had a playbook_id in archived_id_set, we can try to pair them.
            # However, without a direct cluster-level old->new mapping, we use a simpler approach:
            # archived playbooks that have a corresponding new playbook (by position in saved list) are updates.

            # Collect old playbook_ids from disappeared fingerprints
            old_fp_playbook_ids = {}
            for fp, fp_data in prev_fingerprints.items():
                fid = fp_data.get("agent_playbook_id")
                if fid is not None and fid in archived_id_set:
                    old_fp_playbook_ids[fid] = fp

            # For each saved playbook, try to match with an archived old playbook
            matched_old_ids: set[int] = set()
            for saved_fb in saved_playbooks:
                if not saved_fb:
                    continue
                # Try to find an old playbook from the archived set to pair with
                paired_old_id = None
                for old_id in list(old_fp_playbook_ids.keys()):
                    if old_id not in matched_old_ids:
                        paired_old_id = old_id
                        matched_old_ids.add(old_id)
                        break

                if (
                    paired_old_id is not None
                    and paired_old_id in before_playbooks_by_id
                ):
                    updated.append(
                        AgentPlaybookUpdateEntry(
                            before=agent_playbook_to_snapshot(
                                before_playbooks_by_id[paired_old_id]
                            ),
                            after=agent_playbook_to_snapshot(saved_fb),
                        )
                    )
                else:
                    added.append(agent_playbook_to_snapshot(saved_fb))

            # Remaining archived playbooks that weren't paired are removals
            for old_id in archived_id_set:
                if old_id not in matched_old_ids and old_id in before_playbooks_by_id:
                    removed.append(
                        agent_playbook_to_snapshot(before_playbooks_by_id[old_id])
                    )

        return PlaybookAggregationChangeLog(
            playbook_name=playbook_name,
            agent_version=self.agent_version,
            run_mode="full_archive" if full_archive else "incremental",
            added_agent_playbooks=added,
            removed_agent_playbooks=removed,
            updated_agent_playbooks=updated,
        )

    # ===============================
    # public methods
    # ===============================

    def run(self, playbook_aggregator_request: PlaybookAggregatorRequest) -> dict:  # noqa: C901
        """Run playbook aggregation.

        Returns:
            dict: Aggregation stats with keys: clusters_found, user_playbooks_processed, playbooks_generated, skipped (optional)
        """
        _empty_stats = {
            "clusters_found": 0,
            "user_playbooks_processed": 0,
            "playbooks_generated": 0,
        }

        # get playbook aggregator config
        playbook_aggregator_config = self._get_playbook_aggregator_config(
            playbook_aggregator_request.playbook_name
        )
        if (
            not playbook_aggregator_config
            or playbook_aggregator_config.min_cluster_size < 2
        ):
            logger.info(
                "Skipping user playbook aggregation for '%s' (agent_version=%s): no aggregator config or min_cluster_size < 2, config: %s",
                playbook_aggregator_request.playbook_name,
                self.agent_version,
                playbook_aggregator_config,
            )
            return {
                **_empty_stats,
                "skipped": "no aggregator config or min_cluster_size < 2",
            }

        # Check if we should run aggregation based on new playbooks count
        # For rerun, use all user playbooks (last_processed_id=0) to determine if aggregation is needed
        if not self._should_run_aggregation(
            playbook_aggregator_request.playbook_name,
            playbook_aggregator_config,
            rerun=playbook_aggregator_request.rerun,
        ):
            new_count = self._get_new_user_playbooks_count(
                playbook_aggregator_request.playbook_name,
                rerun=playbook_aggregator_request.rerun,
            )
            trigger_count = (
                playbook_aggregator_config.reaggregation_trigger_count
                if playbook_aggregator_config.reaggregation_trigger_count > 0
                else 2
            )
            logger.info(
                "Skipping user playbook aggregation for '%s' (agent_version=%s) - only %d new user playbooks (need %d)",
                playbook_aggregator_request.playbook_name,
                self.agent_version,
                new_count,
                trigger_count,
            )
            return {
                **_empty_stats,
                "skipped": f"not enough new playbooks ({new_count} < {trigger_count})",
            }

        logger.info(
            "Running user playbook aggregation for '%s' (agent_version=%s)",
            playbook_aggregator_request.playbook_name,
            self.agent_version,
        )

        # Get existing APPROVED and PENDING playbooks before archiving (to pass to LLM for deduplication)
        existing_playbooks = self.storage.get_agent_playbooks(  # type: ignore[reportOptionalMemberAccess]
            playbook_name=playbook_aggregator_request.playbook_name,
            status_filter=[None],  # Current playbooks only
            playbook_status_filter=[PlaybookStatus.APPROVED, PlaybookStatus.PENDING],
        )
        logger.info(
            "Found %s existing playbooks (approved + pending) to preserve",
            len(existing_playbooks),
        )

        # get all user playbooks and generate clusters
        user_playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            playbook_name=playbook_aggregator_request.playbook_name,
            agent_version=self.agent_version,
            include_embedding=True,
        )
        clusters = self.get_clusters(user_playbooks, playbook_aggregator_config)

        # Capture all current playbooks before archiving (for change log)
        before_playbooks_by_id: dict[int, AgentPlaybook] = {
            fb.agent_playbook_id: fb for fb in existing_playbooks
        }

        # Determine which clusters changed (skip for rerun)
        mgr = self._create_state_manager()
        playbook_name = playbook_aggregator_request.playbook_name
        archived_playbook_ids = []
        full_archive = (
            False  # True when archive_agent_playbooks_by_playbook_name was used
        )
        prev_fingerprints: dict = {}  # Populated for incremental mode

        if playbook_aggregator_request.rerun:
            # Full rerun: archive all non-APPROVED playbooks, regenerate everything
            logger.info("Rerun requested: bypassing cluster change detection")
            self.storage.archive_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                playbook_name, agent_version=self.agent_version
            )
            changed_clusters = clusters
            full_archive = True
        else:
            # Load previous fingerprints and detect changes
            prev_fingerprints = mgr.get_cluster_fingerprints(
                name=playbook_name, version=self.agent_version
            )

            if not prev_fingerprints:
                # First run: treat all clusters as changed, archive all existing
                logger.info(
                    "No previous cluster fingerprints found, treating all clusters as changed"
                )
                self.storage.archive_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                    playbook_name, agent_version=self.agent_version
                )
                changed_clusters = clusters
                full_archive = True
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
            # Generate new playbooks only for changed clusters
            new_playbooks = self._generate_playbooks_from_clusters(
                changed_clusters,
                existing_playbooks,
                direction_overlap_threshold=playbook_aggregator_config.direction_overlap_threshold,
            )

            # Save playbooks (returns playbooks with playbook_id populated)
            saved_playbooks = self.storage.save_agent_playbooks(new_playbooks)  # type: ignore[reportOptionalMemberAccess]

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

            # Map saved playbooks back to changed clusters by order
            # _generate_playbooks_from_clusters iterates clusters in order and
            # filters out None results, so we need to track which playbooks
            # correspond to which clusters
            for cluster_playbooks in changed_clusters.values():
                fp = self._compute_cluster_fingerprint(cluster_playbooks)
                raw_ids = sorted(fb.user_playbook_id for fb in cluster_playbooks)

                # Try to match saved playbook - the LLM may return None for some
                # clusters (duplicates), so not every cluster has a saved playbook
                playbook_id = None
                # We can't perfectly map without changing _generate_playbooks_from_clusters,
                # so store the fingerprint with whatever playbook_id we have
                new_fingerprints[fp] = {
                    "agent_playbook_id": playbook_id,
                    "user_playbook_ids": raw_ids,
                }

            # Now assign playbook_ids from saved playbooks to fingerprints
            # Since both iterate in cluster order, match by position
            saved_playbook_list = list(saved_playbooks)
            fp_keys_from_changed = [
                self._compute_cluster_fingerprint(cluster_playbooks)
                for cluster_playbooks in changed_clusters.values()
            ]

            # saved_playbooks only contains non-None results, so we just
            # assign playbook_ids to fingerprints that got valid playbooks
            for saved_fb in saved_playbook_list:
                if saved_fb and saved_fb.agent_playbook_id:
                    # Find matching fingerprint by trigger/content matching
                    for fp_key in fp_keys_from_changed:
                        if (
                            fp_key in new_fingerprints
                            and new_fingerprints[fp_key]["agent_playbook_id"] is None
                        ):
                            new_fingerprints[fp_key]["agent_playbook_id"] = (
                                saved_fb.agent_playbook_id
                            )
                            break

            # Store fingerprints in operation state
            mgr.update_cluster_fingerprints(
                name=playbook_name,
                version=self.agent_version,
                fingerprints=new_fingerprints,
            )

            # Update operation state with the highest user_playbook_id processed
            self._update_operation_state(playbook_name, user_playbooks)

            # Build and save change log
            try:
                change_log = self._build_change_log(
                    playbook_name=playbook_name,
                    full_archive=full_archive,
                    before_playbooks_by_id=before_playbooks_by_id,
                    saved_playbooks=saved_playbook_list,
                    archived_playbook_ids=archived_playbook_ids,
                    prev_fingerprints=(prev_fingerprints if not full_archive else {}),
                )
                self.storage.add_playbook_aggregation_change_log(change_log)  # type: ignore[reportOptionalMemberAccess]
                logger.info(
                    "User playbook aggregation change log for '%s' (agent_version=%s): %d agent playbooks added, %d removed, %d updated",
                    playbook_name,
                    self.agent_version,
                    len(change_log.added_agent_playbooks),
                    len(change_log.removed_agent_playbooks),
                    len(change_log.updated_agent_playbooks),
                )
            except Exception:
                logger.exception(
                    "Failed to save playbook aggregation change log for '%s', continuing",
                    playbook_name,
                )

            # Delete archived playbooks after successful aggregation
            if full_archive:
                self.storage.delete_archived_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                    playbook_name, agent_version=self.agent_version
                )
            elif archived_playbook_ids:
                self.storage.delete_agent_playbooks_by_ids(archived_playbook_ids)  # type: ignore[reportOptionalMemberAccess]

            return {
                "clusters_found": len(clusters),
                "user_playbooks_processed": len(user_playbooks),
                "playbooks_generated": len(saved_playbook_list),
            }

        except Exception as e:
            # Restore archived playbooks if any error occurs during aggregation
            logger.error(
                "Error during playbook aggregation for '%s': %s. Restoring archived playbooks.",
                playbook_name,
                str(e),
            )
            if full_archive:
                self.storage.restore_archived_agent_playbooks_by_playbook_name(  # type: ignore[reportOptionalMemberAccess]
                    playbook_name, agent_version=self.agent_version
                )
            elif archived_playbook_ids:
                self.storage.restore_archived_agent_playbooks_by_ids(
                    archived_playbook_ids
                )  # type: ignore[reportOptionalMemberAccess]
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
            user_playbooks: List of user playbooks with structured_data.trigger
            min_cluster_size: Minimum number of playbooks per cluster

        Returns:
            dict[int, list[UserPlaybook]]: Clusters grouped by trigger
        """
        # Group by trigger
        condition_groups: dict[str, list[UserPlaybook]] = {}
        for fb in user_playbooks:
            condition = fb.structured_data.trigger or ""
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
        # Format existing approved playbooks for the prompt
        approved_playbooks_str = (
            "\n".join([f"- {fb.content}" for fb in existing_approved_playbooks])
            if existing_approved_playbooks
            else "None"
        )

        new_playbooks = []
        for cluster_playbooks in clusters.values():
            playbook = self._generate_playbook_from_cluster(
                cluster_playbooks,
                approved_playbooks_str,
                direction_overlap_threshold=direction_overlap_threshold,
            )
            if playbook is not None:
                new_playbooks.append(playbook)
        return new_playbooks

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
            instructions = [
                fb.structured_data.instruction
                for fb in cluster_playbooks
                if fb.structured_data.instruction
            ]
            pitfalls = [
                fb.structured_data.pitfall
                for fb in cluster_playbooks
                if fb.structured_data.pitfall
            ]
            triggers = [
                fb.structured_data.trigger
                for fb in cluster_playbooks
                if fb.structured_data.trigger
            ]

            instruction = instructions[0] if instructions else None
            pitfall = pitfalls[0] if pitfalls else None
            trigger = triggers[0] if triggers else "in general"

            # At least one of instruction or pitfall is required for valid playbook
            if instruction is None and pitfall is None:
                # Fall back to using content from first playbook if available
                first_content = cluster_playbooks[0].content
                if first_content:
                    instruction = first_content
                else:
                    logger.info("No valid structured fields in cluster, skipping")
                    return None

            # Build content directly as a freeform summary
            trigger_text = trigger or "in general"
            action_parts = []
            if instruction:
                action_parts.append(instruction)
            if pitfall:
                action_parts.append(f"avoid: {pitfall}")
            content_text = f"When {trigger_text}, {'; '.join(action_parts)}."

            return AgentPlaybook(
                playbook_name=cluster_playbooks[0].playbook_name,
                agent_version=cluster_playbooks[0].agent_version,
                content=content_text,
                structured_data=StructuredData(
                    trigger=trigger,
                    instruction=instruction,
                    pitfall=pitfall,
                ),
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
        logger.info(
            "Aggregated playbook content (freeform): %.200s",
            playbook_content,
        )
        structured_data = StructuredData(
            rationale=structured.rationale,
            trigger=structured.trigger,
            instruction=structured.instruction,
            pitfall=structured.pitfall,
            blocking_issue=structured.blocking_issue,
            embedding_text=structured.trigger or playbook_content,
        )

        return AgentPlaybook(
            playbook_name=cluster_playbooks[0].playbook_name,
            agent_version=cluster_playbooks[0].agent_version,
            content=playbook_content,
            structured_data=structured_data,
            playbook_status=PlaybookStatus.PENDING,
            playbook_metadata="",
        )

    def _get_playbook_aggregator_config(
        self, playbook_name: str
    ) -> PlaybookAggregatorConfig | None:
        playbook_configs = (
            self.configurator.get_config().user_playbook_extractor_configs
        )
        if not playbook_configs:
            return None
        for agent_playbook_config in playbook_configs:
            if agent_playbook_config.extractor_name == playbook_name:
                return agent_playbook_config.aggregation_config
        return None
