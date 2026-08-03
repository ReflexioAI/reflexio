from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

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
from reflexio.server.error_reporting import capture_anomaly, error_tags
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.embedding_text import (
    embedding_text,
    resolve_clustering_similarity,
)
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
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationClaim,
)
from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)

# Must stay strictly below the smallest server-side row cap of any backend, or a
# capped page would look like a short final page and silently truncate the read.
# PostgREST enforces `max_rows = 1000` (see supabase/*/config.toml).
_AGGREGATION_PLAYBOOK_PAGE_SIZE = 500
_SIGNED_BIGINT_MAX = (1 << 63) - 1
_MAX_EXISTING_PLAYBOOKS_PER_PROMPT = 20


def _select_relevant_existing_playbooks(
    cluster: list[UserPlaybook], existing: list[AgentPlaybook]
) -> list[AgentPlaybook]:
    """Bound repeated context with one linear embedding/lexical-ranked pass."""
    if len(existing) <= _MAX_EXISTING_PLAYBOOKS_PER_PROMPT:
        return existing
    embeddings = [item.embedding for item in cluster if item.embedding]
    dimension = len(embeddings[0]) if embeddings else 0
    embeddings = [value for value in embeddings if len(value) == dimension]
    centroid = (
        [sum(values) / len(embeddings) for values in zip(*embeddings, strict=True)]
        if embeddings
        else []
    )
    centroid_norm = math.sqrt(sum(value * value for value in centroid))
    cluster_tokens = {
        token
        for item in cluster
        for token in f"{item.trigger or ''} {item.content or ''}".lower().split()
    }

    def score(item: AgentPlaybook) -> tuple[float, int]:
        if centroid_norm and len(item.embedding) == dimension:
            item_norm = math.sqrt(sum(value * value for value in item.embedding))
            if item_norm:
                similarity = sum(
                    left * right
                    for left, right in zip(centroid, item.embedding, strict=True)
                ) / (centroid_norm * item_norm)
                return similarity, item.agent_playbook_id
        item_tokens = f"{item.trigger or ''} {item.content}".lower().split()
        token_set = set(item_tokens)
        overlap = len(cluster_tokens & token_set) / max(
            1, min(len(cluster_tokens), len(token_set))
        )
        return overlap, item.agent_playbook_id

    ranked = [(score(item), item) for item in existing]
    ranked.sort(key=lambda pair: (-pair[0][0], pair[0][1]))
    return [item for _rank, item in ranked[:_MAX_EXISTING_PLAYBOOKS_PER_PROMPT]]


def _read_all_pages[T](
    fetch_page: Callable[[int, int], list[T]],
    get_id: Callable[[T], int],
) -> tuple[list[T], int | None]:
    """Read exhaustively by descending ID while excluding later inserts."""
    rows: list[T] = []
    max_id = _SIGNED_BIGINT_MAX
    high_watermark: int | None = None
    while True:
        page = fetch_page(_AGGREGATION_PLAYBOOK_PAGE_SIZE, max_id)
        if page and high_watermark is None:
            high_watermark = get_id(page[0])
        rows.extend(page)
        if len(page) < _AGGREGATION_PLAYBOOK_PAGE_SIZE:
            return rows, high_watermark
        max_id = get_id(page[-1]) - 1


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


AggregationGenerationStatus = Literal["generated", "semantic_null", "retryable_failure"]


@dataclass(frozen=True)
class AggregationGenerationOutcome:
    """Unambiguous result for exactly one selected source cluster."""

    status: AggregationGenerationStatus
    source_cluster: list[UserPlaybook]
    playbook: AgentPlaybook | None = None
    provenance: ModelProvenance | None = None


class PlaybookAggregator:
    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        agent_version: str,
        aggregation_prompt_processor: AggregationPromptProcessor | None = None,
        effect_coordinator: AggregationEffectCoordinator | None = None,
        aggregation_claim: PlaybookAggregationClaim | None = None,
        work_budget: int | None = None,
    ) -> None:
        self.client = llm_client
        storage = request_context.storage
        if storage is None:
            raise ValueError("Playbook aggregation requires configured storage")
        self.storage = storage
        self.configurator = request_context.configurator
        self.request_context = request_context
        self.agent_version = agent_version
        self.aggregation_prompt_processor = aggregation_prompt_processor
        self.effect_coordinator = effect_coordinator
        self.aggregation_claim = aggregation_claim
        self.work_budget = work_budget
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
        self,
        playbook_name: str,
        user_playbooks: list[UserPlaybook],
        processed_high_watermark: int | None = None,
    ) -> None:
        """
        Update operation state with the highest user_playbook_id processed.

        Args:
            playbook_name: Name of the playbook type
            user_playbooks: List of user playbooks that were processed
            processed_high_watermark: Highest ID captured by the aggregation read.
                Falls back to the maximum ID in ``user_playbooks`` for existing
                direct callers.
        """
        if processed_high_watermark is None:
            if not user_playbooks:
                return
            processed_high_watermark = max(
                playbook.user_playbook_id for playbook in user_playbooks
            )

        mgr = self._create_state_manager()
        mgr.update_aggregator_bookmark(
            name=playbook_name,
            version=self.agent_version,
            last_processed_id=processed_high_watermark,
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

    def _run_incremental(
        self,
        *,
        config: PlaybookAggregatorConfig,
        run_id: str,
        aggregation_start: float,
    ) -> dict[str, Any]:
        """Process one durable, bounded residual batch without touching old clusters."""
        budget = (
            aggregator_clustering.max_clustering_playbooks()
            if self.work_budget is None
            else max(0, self.work_budget)
        )
        bootstrap_work, bootstrap_complete = self._adopt_legacy_aggregation_state(
            budget=budget
        )
        budget = max(0, budget - bootstrap_work)
        if not bootstrap_complete:
            return {
                "clusters_found": 0,
                "user_playbooks_processed": bootstrap_work,
                "playbooks_generated": 0,
                "staged": 0,
                "attachments": 0,
                "skipped": "legacy cluster adoption pending",
            }
        # Intake and residual processing share the same row budget. Reserve
        # half for anti-join admission when there is undisposed work; unused
        # capacity is naturally borrowed by residual processing.
        intake_limit = (budget + 1) // 2
        staged_ids = self.storage.stage_playbook_aggregation_intake(  # type: ignore[attr-defined]
            self.agent_version, limit=intake_limit
        )
        residual_limit = max(0, budget - len(staged_ids))
        residual_ids = self.storage.get_playbook_aggregation_residual_ids(  # type: ignore[attr-defined]
            self.agent_version, limit=residual_limit
        )
        user_playbooks = self.storage.get_user_playbooks_by_ids_any_user(  # type: ignore[call-arg]
            residual_ids,
            status_filter=[None],
            include_embedding=True,
        )
        user_playbooks = [
            item for item in user_playbooks if item.content and item.content.strip()
        ]
        trigger_count = max(2, config.reaggregation_trigger_count)

        # Bound the dedup prompt as well. Existing agent playbooks are context,
        # not work discovery, and must never turn this path into an org scan.
        existing_playbooks = self.storage.get_agent_playbooks(  # type: ignore[attr-defined]
            limit=min(500, budget),
            agent_version=self.agent_version,
            status_filter=[None],
            playbook_status_filter=[PlaybookStatus.APPROVED, PlaybookStatus.PENDING],
        )
        similarity_threshold = resolve_clustering_similarity(
            config.clustering_similarity,
            model_name=self.storage.embedding_model_name,
        )
        attachments: list[tuple[UserPlaybook, str]] = []
        stage_b_playbooks: list[UserPlaybook] = []
        missing_embedding_ids: list[int] = []
        candidates: list[tuple[int, list[float]]] = []
        for item in user_playbooks:
            if item.user_playbook_id is None:
                continue
            item_id = int(item.user_playbook_id)
            if not item.embedding:
                missing_embedding_ids.append(item_id)
                continue
            candidates.append((item_id, item.embedding))
        matches = self.storage.find_nearest_playbook_aggregation_clusters(  # type: ignore[attr-defined]
            self.agent_version,
            candidates,
            embedding_model=self.storage.embedding_model_name,
            limit=budget,
        )
        for item in user_playbooks:
            if item.user_playbook_id is None or not item.embedding:
                continue
            match = matches.get(int(item.user_playbook_id))
            if match is not None and match.similarity >= similarity_threshold:
                attachments.append((item, match.cluster_id))
            else:
                stage_b_playbooks.append(item)

        clusters = (
            self.get_clusters(stage_b_playbooks, config)
            if len(stage_b_playbooks) >= trigger_count
            else {}
        )
        outcomes = self._generate_playbook_outcomes_with_source_clusters(
            clusters,
            existing_playbooks,
            direction_overlap_threshold=config.direction_overlap_threshold,
        )
        retryable_outcomes = [
            item for item in outcomes if item.status == "retryable_failure"
        ]
        saved_playbooks: list[AgentPlaybook] = []
        replacement_agent_ids_by_outcome = {
            index: self.storage.get_playbook_aggregation_replacement_agent_ids(  # type: ignore[attr-defined]
                self.agent_version,
                [
                    int(item.user_playbook_id)
                    for item in outcome.source_cluster
                    if item.user_playbook_id is not None
                ],
            )
            for index, outcome in enumerate(outcomes)
            if outcome.status == "generated"
        }
        replaced_agent_ids: set[int] = set()
        generated_playbooks = [
            item.playbook
            for item in outcomes
            if item.status == "generated" and item.playbook is not None
        ]
        prepare = getattr(type(self.storage), "prepare_agent_playbooks_for_save", None)
        if callable(prepare):
            prepare(self.storage, generated_playbooks)
        with self.storage.commit_scope():  # type: ignore[attr-defined]
            self._require_live_aggregation_claim()
            self.storage.set_playbook_aggregation_disposition(  # type: ignore[attr-defined]
                self.agent_version,
                missing_embedding_ids,
                disposition="residual",
                reason="embedding_pending",
            )
            self.storage.attach_playbook_aggregation_items(  # type: ignore[attr-defined]
                agent_version=self.agent_version,
                attachments=[
                    (int(item.user_playbook_id), cluster_id, item.embedding)
                    for item, cluster_id in attachments
                    if item.user_playbook_id is not None and item.embedding
                ],
            )
            for outcome_index, outcome in enumerate(outcomes):
                members = outcome.source_cluster
                member_ids = [
                    int(item.user_playbook_id)
                    for item in members
                    if item.user_playbook_id is not None
                ]
                if outcome.status == "retryable_failure":
                    self.storage.set_playbook_aggregation_disposition(  # type: ignore[attr-defined]
                        self.agent_version,
                        member_ids,
                        disposition="residual",
                        reason="llm_retryable_failure",
                    )
                    continue
                if outcome.status == "semantic_null":
                    self.storage.set_playbook_aggregation_disposition(  # type: ignore[attr-defined]
                        self.agent_version,
                        member_ids,
                        disposition="terminal_noop",
                        reason="semantic_null",
                    )
                    continue
                if outcome.playbook is None:
                    raise RuntimeError("generated outcome is missing its playbook")
                lineage_contexts = [
                    LineageContext(
                        op_kind="aggregate",
                        actor="aggregator",
                        request_id=run_id,
                        source_ids=[str(value) for value in member_ids],
                        reason=f"{AGGREGATE_REASON_PREFIX}incremental",
                        model_name=(
                            outcome.provenance.model_name
                            if outcome.provenance
                            else None
                        ),
                        provider=(
                            outcome.provenance.provider if outcome.provenance else None
                        ),
                    )
                ]
                prepared_save = getattr(
                    type(self.storage), "save_prepared_agent_playbooks", None
                )
                if callable(prepared_save):
                    saved = prepared_save(  # pyright: ignore[reportIndexIssue]
                        self.storage,
                        [outcome.playbook],
                        lineage_contexts=lineage_contexts,
                    )[0]
                else:
                    saved = self.storage.save_agent_playbooks(  # type: ignore[attr-defined]
                        [outcome.playbook], lineage_contexts=lineage_contexts
                    )[0]
                saved_playbooks.append(saved)
                replaced_agent_ids.update(
                    replacement_agent_ids_by_outcome[outcome_index]
                )
                embeddings = [item.embedding for item in members if item.embedding]
                cluster_id = self._stable_aggregation_cluster_id(
                    self._compute_cluster_fingerprint(members)
                )
                self.storage.create_playbook_aggregation_cluster(  # type: ignore[attr-defined]
                    cluster_id=cluster_id,
                    agent_version=self.agent_version,
                    agent_playbook_id=saved.agent_playbook_id,
                    embeddings=embeddings,
                    embedding_model=self.storage.embedding_model_name,
                )
                self.storage.set_playbook_aggregation_disposition(  # type: ignore[attr-defined]
                    self.agent_version,
                    member_ids,
                    disposition="cluster_member",
                    cluster_id=cluster_id,
                    reason="generated",
                )
            supersessions = self.storage.supersede_agent_playbooks_by_ids(  # type: ignore[attr-defined]
                sorted(replaced_agent_ids), request_id=run_id
            )
            self.storage.delete_orphaned_playbook_aggregation_clusters(  # type: ignore[attr-defined]
                self.agent_version
            )
            self._require_live_aggregation_claim()

        stats = {
            "clusters_found": len(clusters),
            "user_playbooks_processed": len(user_playbooks),
            "playbooks_generated": len(saved_playbooks),
            "staged": len(staged_ids),
            "attachments": len(attachments),
            "supersessions": supersessions,
            "retryable_failures": len(retryable_outcomes),
        }
        self._enqueue_playbook_optimization(saved_playbooks)
        record_usage_event(
            org_id=self.request_context.org_id,
            event_name="aggregation_succeeded",
            event_category="aggregation",
            pipeline="playbook",
            playbook_name=SINGLETON_USER_PLAYBOOK_NAME,
            agent_version=self.agent_version,
            outcome="partial_failure" if retryable_outcomes else "success",
            count_value=len(saved_playbooks),
            duration_ms=int((time.perf_counter() - aggregation_start) * 1000),
            metadata=stats,
        )
        return stats

    def _stable_aggregation_cluster_id(self, fingerprint: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.request_context.org_id}:{self.agent_version}:{fingerprint}",
            )
        )

    def _adopt_legacy_aggregation_state(self, *, budget: int) -> tuple[int, bool]:
        """Rebuild legacy fingerprint centroids in bounded, resumable pages."""
        status = self.storage.get_playbook_aggregation_bootstrap_status(  # type: ignore[attr-defined]
            self.agent_version
        )
        if status == "complete":
            return 0, True
        mgr = self._create_state_manager()
        fingerprints = mgr.get_cluster_fingerprints(
            name=SINGLETON_USER_PLAYBOOK_NAME, version=self.agent_version
        )
        if not isinstance(fingerprints, dict):
            fingerprints = {}
        consumed = 0
        for fingerprint in sorted(fingerprints):
            data = fingerprints[fingerprint]
            if not isinstance(data, dict) or data.get("agent_playbook_id") is None:
                # A legacy None is ambiguous (semantic null vs. failed output),
                # so its members remain undisposed and are retried normally.
                continue
            try:
                agent_playbook_id = int(data["agent_playbook_id"])
                member_ids = sorted(
                    {
                        int(value)
                        for value in data.get("user_playbook_ids", [])
                        if int(value) > 0
                    }
                )
            except (TypeError, ValueError):
                continue
            if not member_ids:
                # Empty legacy fingerprints have no centroid to adopt and must not
                # become active clusters that nearest-centroid lookup cannot find.
                continue
            cluster_id = self._stable_aggregation_cluster_id(fingerprint)
            progress = self.storage.get_playbook_aggregation_cluster_rebuild_cursor(  # type: ignore[attr-defined]
                cluster_id
            )
            if progress is not None and progress[1] == "active":
                continue
            cursor = progress[0] if progress is not None else 0
            pending_ids = [value for value in member_ids if value > cursor]
            remaining = budget - consumed
            if pending_ids and remaining <= 0:
                return consumed, False
            page_ids = pending_ids[:remaining]
            consumed += len(page_ids)
            rows = self.storage.get_user_playbooks_by_ids_any_user(  # type: ignore[call-arg]
                page_ids,
                status_filter=[None],
                include_embedding=False,
            )
            rows_by_id = {
                int(item.user_playbook_id): item
                for item in rows
                if item.user_playbook_id is not None
            }
            member_embeddings: list[tuple[int, list[float]]] = []
            rebuild_cursor = cursor
            blocked = False
            embed = getattr(self.storage, "_get_embedding", None)
            if not callable(embed) and page_ids:
                raise RuntimeError("aggregation storage cannot re-embed legacy members")
            embed_fn = cast(Callable[[str], list[float]], embed)
            for user_playbook_id in page_ids:
                item = rows_by_id.get(user_playbook_id)
                if item is None or not item.content or not item.content.strip():
                    rebuild_cursor = user_playbook_id
                    continue
                try:
                    value = embed_fn(embedding_text(item))
                except Exception:  # noqa: BLE001 - retry this exact member next run
                    blocked = True
                    break
                if not value:
                    blocked = True
                    break
                member_embeddings.append((user_playbook_id, cast(list[float], value)))
                rebuild_cursor = user_playbook_id
            complete = not blocked and rebuild_cursor >= max(member_ids, default=0)
            if rebuild_cursor > cursor or not member_ids:
                with self.storage.commit_scope():  # type: ignore[attr-defined]
                    self._require_live_aggregation_claim()
                    self.storage.adopt_legacy_playbook_aggregation_cluster_page(  # type: ignore[attr-defined]
                        cluster_id=cluster_id,
                        agent_version=self.agent_version,
                        agent_playbook_id=agent_playbook_id,
                        member_embeddings=member_embeddings,
                        embedding_model=self.storage.embedding_model_name,
                        embedding_dimension=int(
                            self.storage.embedding_dimensions  # type: ignore[attr-defined]
                        ),
                        rebuild_cursor=rebuild_cursor,
                        complete=complete,
                    )
                    self._require_live_aggregation_claim()
            if blocked or not complete:
                return consumed, False

        with self.storage.commit_scope():  # type: ignore[attr-defined]
            self._require_live_aggregation_claim()
            self.storage.set_playbook_aggregation_bootstrap_status(  # type: ignore[attr-defined]
                self.agent_version, "complete"
            )
            self._require_live_aggregation_claim()
        return consumed, True

    def _require_live_aggregation_claim(self) -> None:
        if self.aggregation_claim is None:
            return
        if not self.storage.validate_playbook_aggregation_claim(  # type: ignore[attr-defined]
            self.aggregation_claim
        ):
            raise RuntimeError("playbook aggregation effect lost its database fence")

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

        if (
            not playbook_aggregator_request.rerun
            and getattr(
                self.storage, "supports_incremental_playbook_aggregation", False
            )
            is True
        ):
            return self._run_incremental(
                config=playbook_aggregator_config,
                run_id=_run_id,
                aggregation_start=aggregation_start,
            )

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
        existing_playbooks, _existing_high_watermark = _read_all_pages(
            lambda limit, max_id: self.storage.get_agent_playbooks(  # type: ignore[reportOptionalMemberAccess]
                limit=limit,
                max_agent_playbook_id=max_id,
                agent_version=self.agent_version,
                status_filter=[None],  # Current playbooks only
                playbook_status_filter=[
                    PlaybookStatus.APPROVED,
                    PlaybookStatus.PENDING,
                ],
            ),
            lambda playbook: playbook.agent_playbook_id,
        )
        logger.info(
            "Found %s existing playbooks (approved + pending) to preserve",
            len(existing_playbooks),
        )

        # Preflight the corpus size with a COUNT before paging it in. The read
        # below pulls every current user playbook WITH its embedding, so an
        # oversized org costs ~1 GB of resident floats before clustering is even
        # reached -- the cap has to be enforced ahead of the load, not just
        # inside get_clusters().
        max_playbooks = aggregator_clustering.max_clustering_playbooks()
        rerun_invalidation_ids: list[int] = []
        if playbook_aggregator_request.rerun and self.aggregation_claim is not None:
            rerun_snapshot = self.storage.capture_playbook_aggregation_rerun_snapshot(  # type: ignore[attr-defined]
                self.agent_version,
                limit=max_playbooks + 1,
            )
            user_playbooks = list(rerun_snapshot.user_playbooks)
            user_high_watermark = rerun_snapshot.user_high_watermark
            rerun_invalidation_ids = list(rerun_snapshot.invalidation_ids)
            total_user_playbooks = len(user_playbooks)
        else:
            total_user_playbooks = self.storage.count_user_playbooks(  # pyright: ignore[reportOptionalMemberAccess]
                min_user_playbook_id=0,
                agent_version=self.agent_version,
                status_filter=[None],
            )
        if total_user_playbooks > max_playbooks:
            logger.error(
                "Skipping user playbook aggregation for '%s' (agent_version=%s): "
                "%d current user playbooks exceeds the %s cap of %d",
                playbook_name,
                self.agent_version,
                total_user_playbooks,
                aggregator_clustering.MAX_CLUSTERING_PLAYBOOKS_ENV,
                max_playbooks,
            )
            record_usage_event(
                org_id=self.request_context.org_id,
                event_name="aggregation_gate_evaluated",
                event_category="aggregation",
                pipeline="playbook",
                playbook_name=playbook_name,
                agent_version=self.agent_version,
                outcome="failed",
                count_value=total_user_playbooks,
                metadata={
                    "failure_reason": "full_rerun_safety_cap_exceeded",
                    "user_playbooks": total_user_playbooks,
                    "max_clustering_playbooks": max_playbooks,
                },
            )
            raise RuntimeError(
                "full aggregation rerun exceeds the safety cap "
                f"({total_user_playbooks} > {max_playbooks})"
            )

        # Non-fenced legacy runs retain ordinary pagination. Fenced reruns use
        # the pre-compute repeatable-read snapshot materialized above.
        if not (
            playbook_aggregator_request.rerun and self.aggregation_claim is not None
        ):
            user_playbooks, user_high_watermark = _read_all_pages(
                lambda limit, max_id: self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
                    limit=limit,
                    max_user_playbook_id=max_id,
                    agent_version=self.agent_version,
                    status_filter=[None],  # Current playbooks only
                    include_embedding=True,
                ),
                lambda playbook: playbook.user_playbook_id,
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
                        self._update_operation_state(
                            playbook_name,
                            user_playbooks,
                            processed_high_watermark=user_high_watermark,
                        )
                    else:
                        self.effect_coordinator.prepare([])
                        with self.effect_coordinator.apply_scope():
                            self._update_operation_state(
                                playbook_name,
                                user_playbooks,
                                processed_high_watermark=user_high_watermark,
                            )
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
            elif self.aggregation_claim is not None:
                prepare = getattr(
                    type(self.storage), "prepare_agent_playbooks_for_save", None
                )
                if callable(prepare):
                    prepare(self.storage, new_playbooks)
                pending_scope = self.storage.commit_scope()  # type: ignore[attr-defined]
                pending_scope.__enter__()
                effect_scope = pending_scope
                self._require_live_aggregation_claim()
            if (
                playbook_aggregator_request.rerun
                and self.aggregation_claim is not None
                and new_playbooks
            ):
                self.storage.reset_playbook_aggregation_version(  # type: ignore[attr-defined]
                    self.agent_version
                )
                self.storage.stage_playbook_aggregation_snapshot(  # type: ignore[attr-defined]
                    self.agent_version,
                    [item.user_playbook_id for item in user_playbooks],
                )

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
                    lineage_contexts = [
                        LineageContext(
                            op_kind="aggregate",
                            actor="aggregator",
                            request_id=_run_id,
                            source_ids=member_ids,
                            reason=f"{AGGREGATE_REASON_PREFIX}{run_mode}",
                            model_name=provenance.model_name if provenance else None,
                            provider=provenance.provider if provenance else None,
                        )
                    ]
                    prepared_save = getattr(
                        type(self.storage), "save_prepared_agent_playbooks", None
                    )
                    if self.aggregation_claim is not None and callable(prepared_save):
                        saved_fb = prepared_save(  # pyright: ignore[reportIndexIssue]
                            self.storage,
                            [playbook],
                            lineage_contexts=lineage_contexts,
                        )[0]
                    else:
                        saved_fb = self.storage.save_agent_playbooks(  # type: ignore[reportOptionalMemberAccess]
                            [playbook], lineage_contexts=lineage_contexts
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
                    if (
                        playbook_aggregator_request.rerun
                        and self.aggregation_claim is not None
                    ):
                        cluster_embeddings = [
                            item.embedding
                            for item in cluster_playbooks
                            if item.embedding
                        ]
                        if cluster_embeddings:
                            cluster_id = self._stable_aggregation_cluster_id(fp_key)
                            self.storage.create_playbook_aggregation_cluster(  # type: ignore[attr-defined]
                                cluster_id=cluster_id,
                                agent_version=self.agent_version,
                                agent_playbook_id=saved_fb.agent_playbook_id,
                                embeddings=cluster_embeddings,
                                embedding_model=self.storage.embedding_model_name,
                            )
                            self.storage.set_playbook_aggregation_disposition(  # type: ignore[attr-defined]
                                self.agent_version,
                                raw_ids,
                                disposition="cluster_member",
                                cluster_id=cluster_id,
                                reason="full_rerun",
                            )
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
            self._update_operation_state(
                playbook_name,
                user_playbooks,
                processed_high_watermark=user_high_watermark,
            )

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
                    with error_tags(
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
                    if (
                        self.effect_coordinator is not None
                        or self.aggregation_claim is not None
                    ):
                        raise

            stats = {
                "clusters_found": len(clusters),
                "user_playbooks_processed": len(user_playbooks),
                "playbooks_generated": len(saved_playbook_list),
            }
            if (
                playbook_aggregator_request.rerun
                and self.aggregation_claim is not None
                and new_playbooks
                and not self.storage.mark_playbook_aggregation_invalidations_processed(  # type: ignore[attr-defined]
                    self.aggregation_claim,
                    rerun_invalidation_ids,
                )
            ):
                raise RuntimeError(
                    "playbook aggregation rerun lost its invalidation fence"
                )
            if self.effect_coordinator is not None:
                self.effect_coordinator.complete(stats)
                if effect_scope is None:
                    raise RuntimeError("aggregation effect scope was not entered")
                completed_scope = effect_scope
                effect_scope = None
                completed_scope.__exit__(None, None, None)
            elif self.aggregation_claim is not None:
                self._require_live_aggregation_claim()
                if effect_scope is None:
                    raise RuntimeError("aggregation claim scope was not entered")
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
            if self.effect_coordinator is None and self.aggregation_claim is None:
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

        if not user_playbooks:
            logger.info("No user playbooks to cluster")
            return {}

        min_cluster_size = playbook_aggregator_config.min_cluster_size
        similarity_threshold = resolve_clustering_similarity(
            playbook_aggregator_config.clustering_similarity,
            model_name=self.storage.embedding_model_name,
        )
        # Mock mode: cluster by trigger
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: clustering by trigger")
            return aggregator_clustering.cluster_by_trigger_mock(
                user_playbooks, min_cluster_size
            )

        # Extract embeddings from user playbooks
        import numpy as np

        embedded_playbooks = [
            playbook for playbook in user_playbooks if playbook.embedding
        ]
        skipped_without_embedding = len(user_playbooks) - len(embedded_playbooks)
        if skipped_without_embedding:
            logger.info(
                "Skipping %d user playbooks without an embedding",
                skipped_without_embedding,
            )
        user_playbooks = embedded_playbooks
        embeddings = np.asarray(
            [playbook.embedding for playbook in user_playbooks], dtype=np.float64
        )

        if len(embeddings) < min_cluster_size:
            logger.info(
                "Not enough playbooks to cluster (got %d, need %d)",
                len(embeddings),
                min_cluster_size,
            )
            return {}

        max_playbooks = aggregator_clustering.max_clustering_playbooks()
        if len(embeddings) > max_playbooks:
            # Refuse rather than sampling down: a sampled subset would silently
            # produce different agent playbooks run-to-run.
            logger.error(
                "Refusing to cluster %d user playbooks (cap %d). Raise %s to "
                "allow it -- runtime grows quadratically, so expect roughly "
                "(n/20000)^2 * 2 minutes of CPU.",
                len(embeddings),
                max_playbooks,
                aggregator_clustering.MAX_CLUSTERING_PLAYBOOKS_ENV,
            )
            return {}

        # Choose algorithm based on dataset size
        # Convert similarity threshold to distance threshold (distance = 1 - similarity)
        distance_threshold = 1.0 - similarity_threshold
        if len(embeddings) < CLUSTERING_ALGORITHM_THRESHOLD:
            # Small input only: the precomputed cosine matrix is O(n^2), which is
            # negligible below the threshold and catastrophic above it.
            from sklearn.metrics.pairwise import cosine_distances

            cluster_labels = self._cluster_with_agglomerative(
                cosine_distances(embeddings), min_cluster_size, distance_threshold
            )
        else:
            # HDBSCAN clusters the raw vectors directly -- no O(n^2) matrix.
            cluster_labels = self._cluster_with_hdbscan(
                embeddings, min_cluster_size, distance_threshold
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
        embeddings: np.ndarray,
        min_cluster_size: int,
        distance_threshold: float,
    ) -> np.ndarray:
        return aggregator_clustering.cluster_with_hdbscan(
            embeddings, min_cluster_size, distance_threshold
        )

    def _generate_playbooks_with_source_clusters(
        self,
        clusters: dict[int, list[UserPlaybook]],
        existing_approved_playbooks: list[AgentPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> list[tuple[AgentPlaybook, list[UserPlaybook], ModelProvenance | None]]:
        """Compatibility view containing only generated outcomes."""
        return [
            (outcome.playbook, outcome.source_cluster, outcome.provenance)
            for outcome in self._generate_playbook_outcomes_with_source_clusters(
                clusters,
                existing_approved_playbooks,
                direction_overlap_threshold,
            )
            if outcome.status == "generated" and outcome.playbook is not None
        ]

    def _generate_playbook_outcomes_with_source_clusters(
        self,
        clusters: dict[int, list[UserPlaybook]],
        existing_approved_playbooks: list[AgentPlaybook],
        direction_overlap_threshold: float = 0.6,
    ) -> list[AggregationGenerationOutcome]:
        """Return one tagged outcome for every selected source cluster."""
        outcomes: list[AggregationGenerationOutcome] = []
        for cluster_playbooks in clusters.values():
            relevant_existing = _select_relevant_existing_playbooks(
                cluster_playbooks, existing_approved_playbooks
            )
            approved_playbooks_str = (
                "\n".join(f"- {item.content}" for item in relevant_existing)
                if relevant_existing
                else "None"
            )
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

            generated = self._generate_playbook_from_cluster_outcome(
                prompt_cluster_playbooks,
                approved_playbooks_str,
                direction_overlap_threshold=direction_overlap_threshold,
                processing_context=processing_context,
            )
            outcomes.append(
                AggregationGenerationOutcome(
                    status=generated.status,
                    source_cluster=cluster_playbooks,
                    playbook=generated.playbook,
                    provenance=generated.provenance,
                )
            )
        return outcomes

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
        """Compatibility view of the tagged generation result."""
        outcome = self._generate_playbook_from_cluster_outcome(
            cluster_playbooks,
            existing_approved_playbooks_str,
            direction_overlap_threshold,
            processing_context,
        )
        if outcome.status != "generated" or outcome.playbook is None:
            return None
        return outcome.playbook, outcome.provenance

    def _generate_playbook_from_cluster_outcome(
        self,
        cluster_playbooks: list[UserPlaybook],
        existing_approved_playbooks_str: str,
        direction_overlap_threshold: float = 0.6,
        processing_context: AggregationPromptProcessingContext | None = None,
    ) -> AggregationGenerationOutcome:
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
            return AggregationGenerationOutcome("retryable_failure", [])

        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            # Extract structured fields directly from cluster
            triggers = [fb.trigger for fb in cluster_playbooks if fb.trigger]

            trigger = triggers[0] if triggers else "in general"

            # Fall back to using content from first playbook if available
            first_content = cluster_playbooks[0].content
            if not first_content:
                logger.info("No valid content in cluster, skipping")
                return AggregationGenerationOutcome(
                    "retryable_failure", cluster_playbooks
                )

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
            processed = self._process_aggregation_response_outcome(
                response, cluster_playbooks
            )
            if processed.status != "generated" or processed.playbook is None:
                return processed
            return AggregationGenerationOutcome(
                "generated",
                cluster_playbooks,
                processed.playbook.model_copy(
                    update={"playbook_metadata": "mock_generated"}
                ),
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
                return AggregationGenerationOutcome(
                    "retryable_failure", cluster_playbooks
                )

            processed = self._process_aggregation_response_outcome(
                response, cluster_playbooks
            )
            return AggregationGenerationOutcome(
                processed.status,
                cluster_playbooks,
                processed.playbook,
                model_provenance if processed.status == "generated" else None,
            )
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
            return AggregationGenerationOutcome("retryable_failure", cluster_playbooks)

    def _process_aggregation_response(
        self, response: PlaybookAggregationOutput, cluster_playbooks: list[UserPlaybook]
    ) -> AgentPlaybook | None:
        """Compatibility view of the tagged response classification."""
        return self._process_aggregation_response_outcome(
            response, cluster_playbooks
        ).playbook

    def _process_aggregation_response_outcome(
        self,
        response: PlaybookAggregationOutput,
        cluster_playbooks: list[UserPlaybook],
    ) -> AggregationGenerationOutcome:
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
            return AggregationGenerationOutcome("retryable_failure", cluster_playbooks)

        structured = response.playbook
        if structured is None:
            logger.info("LLM returned null playbook (duplicate of existing)")
            return AggregationGenerationOutcome("semantic_null", cluster_playbooks)

        # content is always the LLM's freeform summary;
        # fall back to formatted structured fields for backward compatibility
        playbook_content = ensure_playbook_content(structured.content, structured)
        if not playbook_content.strip():
            logger.info("Aggregated playbook has no valid content, skipping")
            return AggregationGenerationOutcome("retryable_failure", cluster_playbooks)
        logger.info(
            "Aggregated playbook content (freeform): %.200s",
            playbook_content,
        )

        return AggregationGenerationOutcome(
            "generated",
            cluster_playbooks,
            AgentPlaybook(
                playbook_name=cluster_playbooks[0].playbook_name,
                agent_version=cluster_playbooks[0].agent_version,
                content=playbook_content,
                trigger=structured.trigger,
                rationale=structured.rationale,
                playbook_status=PlaybookStatus.PENDING,
                playbook_metadata="",
            ),
        )

    def _get_playbook_aggregator_config(self) -> PlaybookAggregatorConfig | None:
        root_config = self.configurator.get_config()
        playbook_config = getattr(root_config, "user_playbook_extractor_config", None)
        if not playbook_config:
            return None
        return playbook_config.aggregation_config
