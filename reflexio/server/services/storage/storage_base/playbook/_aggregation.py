"""Durable state contracts for incremental user-playbook aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reflexio.models.api_schema.service_schemas import UserPlaybook

AggregationDisposition = Literal["residual", "cluster_member", "terminal_noop"]

AGGREGATION_RETRY_BASE_SECONDS = 60
AGGREGATION_RETRY_MAX_SECONDS = 3_600
AGGREGATION_INVALIDATION_RETENTION_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class PlaybookAggregationClaim:
    """One database-fenced, organization-wide aggregation claim."""

    agent_version: str
    owner: str
    fence: int
    state_version: int
    expires_at: int


@dataclass(frozen=True)
class PlaybookAggregationBacklog:
    """Bounded-work backlog counts used by scheduling and telemetry."""

    undisposed: int
    residual: int
    invalidations: int
    oldest_residual_age_seconds: int | None = None
    dirty_repairs: int = 0
    residual_retry_after_seconds: int = 0
    repair_retry_after_seconds: int = 0

    @property
    def pending(self) -> bool:
        return bool(
            self.undisposed or self.residual or self.invalidations or self.dirty_repairs
        )

    @property
    def continuation_delay_seconds(self) -> int:
        """Delay retry-only work until its earliest durable cooldown expires."""
        if self.undisposed or self.invalidations:
            return 0
        delays: list[int] = []
        if self.residual:
            delays.append(max(0, self.residual_retry_after_seconds))
        if self.dirty_repairs:
            delays.append(max(0, self.repair_retry_after_seconds))
        return min(delays, default=0)


@dataclass(frozen=True)
class PlaybookAggregationInvalidation:
    invalidation_id: int
    agent_version: str
    operation: str
    entity_id: int
    source_ids: tuple[int, ...]


@dataclass(frozen=True)
class PlaybookAggregationClusterMatch:
    cluster_id: str
    similarity: float
    agent_playbook_id: int


@dataclass(frozen=True)
class PlaybookAggregationRerunSnapshot:
    """Inputs materialized from one pre-compute storage snapshot."""

    user_playbooks: tuple[UserPlaybook, ...]
    invalidation_ids: tuple[int, ...]
    user_high_watermark: int | None
    invalidation_high_watermark: int | None


@dataclass(frozen=True)
class PlaybookAggregationRebuildSample:
    """Bounded prompt inputs for one invalidated cluster rebuild."""

    cluster_id: str
    agent_playbook_id: int
    member_ids: tuple[int, ...]


class PlaybookAggregationStoreMixin:
    """Portable, deliberately narrow aggregation state API.

    Tenant backends scope these records by schema; SQLite scopes them by the
    storage database.  Callers must not use a sequence watermark for discovery.
    """

    def schedule_playbook_aggregation(self, agent_version: str) -> None:
        """Durably mark a version pending without postponing existing work."""
        raise NotImplementedError

    def repair_playbook_aggregation_pending_state(
        self, *, limit: int = 100
    ) -> list[str]:
        """Create pending state for eligible anti-join work after a lost signal."""
        raise NotImplementedError

    def claim_due_playbook_aggregation(
        self,
        *,
        owner: str,
        lease_seconds: int,
        agent_version: str | None = None,
    ) -> PlaybookAggregationClaim | None:
        """Claim the oldest due (or requested admin) version using database time."""
        raise NotImplementedError

    def renew_playbook_aggregation_claim(
        self, claim: PlaybookAggregationClaim, *, lease_seconds: int
    ) -> PlaybookAggregationClaim | None:
        """Renew a live claim or return None after ownership is lost."""
        raise NotImplementedError

    def validate_playbook_aggregation_claim(
        self, claim: PlaybookAggregationClaim
    ) -> bool:
        """Lock and validate the lease fence plus expected state version."""
        raise NotImplementedError

    def finish_playbook_aggregation_claim(
        self,
        claim: PlaybookAggregationClaim,
        *,
        success: bool,
        retry_after_seconds: int,
        backlog_retry_after_seconds: int,
        min_interval_seconds: int,
        backlog: PlaybookAggregationBacklog | None = None,
    ) -> bool:
        """Fenced completion with separate failure, drain, and idle delays."""
        raise NotImplementedError

    def stage_playbook_aggregation_intake(
        self, agent_version: str, *, limit: int, window_limit: int = 20_000
    ) -> list[int]:
        """Stage new rows from the durable newest-N unclustered window."""
        raise NotImplementedError

    def get_playbook_aggregation_bootstrap_status(self, agent_version: str) -> str:
        """Return ``pending`` or ``complete`` for legacy-state adoption."""
        raise NotImplementedError

    def set_playbook_aggregation_bootstrap_status(
        self, agent_version: str, status: Literal["pending", "complete"]
    ) -> None:
        """Persist legacy-state adoption progress for one version."""
        raise NotImplementedError

    def get_playbook_aggregation_cluster_rebuild_cursor(
        self, cluster_id: str
    ) -> tuple[int, str] | None:
        """Return a legacy cluster's member cursor and state, if it exists."""
        raise NotImplementedError

    def adopt_legacy_playbook_aggregation_cluster_page(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        agent_playbook_id: int,
        centroid_embedding: list[float],
        member_embeddings: list[tuple[int, list[float]]],
        embedding_model: str,
        embedding_dimension: int,
        rebuild_cursor: int,
        complete: bool,
    ) -> None:
        """Append one bounded, re-embedded page to a legacy cluster rebuild."""
        raise NotImplementedError

    def reset_playbook_aggregation_version(self, agent_version: str) -> None:
        """Clear typed cluster/item state before a fenced full rerun rebuild."""
        raise NotImplementedError

    def capture_playbook_aggregation_rerun_snapshot(
        self, agent_version: str, *, limit: int
    ) -> PlaybookAggregationRerunSnapshot:
        """Materialize rerun rows and exact invalidations in one read snapshot."""
        raise NotImplementedError

    def stage_playbook_aggregation_snapshot(
        self, agent_version: str, user_playbook_ids: list[int]
    ) -> None:
        """Stage exactly the captured rerun IDs, including later-deleted rows."""
        raise NotImplementedError

    def mark_playbook_aggregation_invalidations_processed(
        self,
        claim: PlaybookAggregationClaim,
        invalidation_ids: list[int],
    ) -> bool:
        """Fenced completion for exact invalidations reflected by a rerun."""
        raise NotImplementedError

    def get_playbook_aggregation_residual_ids(
        self, agent_version: str, *, limit: int
    ) -> list[int]:
        """Return a fair bounded residual page and record attempts."""
        raise NotImplementedError

    def set_playbook_aggregation_disposition(
        self,
        agent_version: str,
        user_playbook_ids: list[int],
        *,
        disposition: AggregationDisposition,
        cluster_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Move staged items to one durable disposition."""
        raise NotImplementedError

    def get_playbook_aggregation_backlog(
        self, agent_version: str
    ) -> PlaybookAggregationBacklog:
        """Return scheduler/telemetry backlog counts."""
        raise NotImplementedError

    def append_playbook_aggregation_invalidation(
        self,
        *,
        agent_version: str,
        operation: str,
        entity_id: int,
        source_ids: list[int] | None = None,
    ) -> None:
        """Append a routed lifecycle invalidation in the caller's transaction."""
        raise NotImplementedError

    def get_playbook_aggregation_invalidations(
        self, agent_version: str, *, limit: int
    ) -> list[PlaybookAggregationInvalidation]:
        """Read an ordered bounded invalidation page."""
        raise NotImplementedError

    def apply_playbook_aggregation_invalidations(
        self,
        claim: PlaybookAggregationClaim,
        invalidation_ids: list[int],
    ) -> bool:
        """Fenced, idempotent removal plus invalidation completion."""
        raise NotImplementedError

    def get_playbook_aggregation_rebuild_cluster_ids(
        self, agent_version: str, user_playbook_ids: list[int]
    ) -> dict[int, str]:
        """Map selected residual members back to their rebuilding cluster."""
        raise NotImplementedError

    def get_playbook_aggregation_rebuild_samples(
        self, agent_version: str, cluster_ids: list[str], *, limit_per_cluster: int
    ) -> list[PlaybookAggregationRebuildSample]:
        """Return recent CURRENT members for each requested rebuilding cluster."""
        raise NotImplementedError

    def defer_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        reason: str,
    ) -> None:
        """Apply one durable retry cooldown to an entire rebuilding cluster.

        Raises:
            RuntimeError: If the expected agent no longer owns the rebuilding cluster.
        """
        raise NotImplementedError

    def complete_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        replacement_agent_playbook_id: int,
        centroid_embedding: list[float],
        embedding_model: str,
    ) -> int:
        """Activate a rebuilt cluster and restore all remaining memberships.

        Raises:
            RuntimeError: If the expected agent no longer owns the rebuilding cluster
                or the cluster has no residual members.
            ValueError: If the replacement embedding dimension changed.
        """
        raise NotImplementedError

    def discard_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        reason: str,
    ) -> int:
        """Terminalize all remaining members and delete a redundant rebuild.

        Raises:
            RuntimeError: If the expected agent no longer owns the rebuilding cluster.
        """
        raise NotImplementedError

    def delete_orphaned_playbook_aggregation_clusters(
        self, agent_version: str
    ) -> list[int]:
        """Delete empty rebuilding clusters and return their former agent IDs."""
        raise NotImplementedError

    def find_nearest_playbook_aggregation_clusters(
        self,
        agent_version: str,
        candidates: list[tuple[int, list[float]]],
        *,
        embedding_model: str,
        limit: int,
    ) -> dict[int, PlaybookAggregationClusterMatch]:
        """Return nearest centroids for a version-isolated candidate batch."""
        raise NotImplementedError

    def create_playbook_aggregation_cluster(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        agent_playbook_id: int | None,
        centroid_embedding: list[float],
        member_count: int,
        embedding_model: str,
    ) -> None:
        """Persist a cluster whose centroid is its current agent embedding."""
        raise NotImplementedError

    def attach_playbook_aggregation_items(
        self,
        *,
        agent_version: str,
        attachments: list[tuple[int, str]],
    ) -> None:
        """Attach residuals without changing the current agent centroid."""
        raise NotImplementedError

    def replace_playbook_aggregation_cluster_agent(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        replacement_agent_playbook_id: int,
        centroid_embedding: list[float],
        embedding_model: str,
    ) -> None:
        """CAS the current agent and its embedding-backed centroid."""
        raise NotImplementedError
