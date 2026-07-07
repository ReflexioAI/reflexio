"""Parallel subquery execution + fusion for deep unified search.

Runs a :class:`~reflexio.server.services.search.deep_search_schemas.SearchPlan`'s
subqueries concurrently on the shared search fan-out executor, always through
the per-arm storage helpers (never the combined single-RPC path — the per-arm
requests are what carry planner-inferred time windows), and fuses the results
into one deduped, provenance-tracked candidate pool.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.config_schema import SearchMode, SearchOptions
from reflexio.server.services.search.deep_search_schemas import ARM_KEY_PREFIX
from reflexio.server.services.unified_search_service import (
    _DEFAULT_AGENT_PLAYBOOK_STATUSES,
    _SEARCH_FANOUT_EXECUTOR,
    _get_cached_query_embedding,
    _search_agent_playbooks_via_storage,
    _search_profiles_via_storage,
    _search_user_playbooks_via_storage,
    _submit_with_current_context,
)
from reflexio.server.tracing import profile_step

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.server.services.search.deep_search_schemas import PlannedSubquery
    from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

_SUBQUERY_TIMEOUT_SECONDS = 30
_ARM_ID_ATTR = {
    "profiles": "profile_id",
    "user_playbooks": "user_playbook_id",
    "agent_playbooks": "agent_playbook_id",
}
_SECONDS_PER_DAY = 86_400


@dataclass
class Candidate:
    """One fused candidate entity with retrieval provenance.

    Attributes:
        key: Cross-arm candidate key ("P:<id>" / "UP:<id>" / "AP:<id>").
        arm: The arm the entity came from.
        entity: The entity model (UserProfile / UserPlaybook / AgentPlaybook).
        subquery_indices: Plan indices of every subquery that surfaced it.
    """

    key: str
    arm: str
    entity: Any
    subquery_indices: list[int] = field(default_factory=list)

    def age_days(self, now: int) -> float | None:
        """Entity age in days from its timestamp field, or None."""
        ts = getattr(self.entity, "last_modified_timestamp", None) or getattr(
            self.entity, "created_at", None
        )
        if not ts:
            return None
        return (now - int(ts)) / _SECONDS_PER_DAY

    def timestamp(self) -> int:
        """The entity's primary timestamp (0 when missing)."""
        return int(
            getattr(self.entity, "last_modified_timestamp", None)
            or getattr(self.entity, "created_at", None)
            or 0
        )


def candidate_key(arm: str, entity: Any) -> str:
    """Build the cross-arm candidate key for an entity."""
    entity_id = getattr(entity, _ARM_ID_ATTR[arm], "")
    return f"{ARM_KEY_PREFIX[arm]}:{entity_id}"


def _window_bounds(
    subquery: PlannedSubquery, now: datetime
) -> tuple[datetime | None, datetime | None]:
    """Convert relative day offsets into absolute (start, end) datetimes."""
    start = (
        now - timedelta(days=subquery.start_days_ago)
        if subquery.start_days_ago is not None
        else None
    )
    end = (
        now - timedelta(days=subquery.end_days_ago)
        if subquery.end_days_ago is not None
        else None
    )
    if start and end and start > end:
        start, end = end, start
    return start, end


def _subquery_embedding(
    storage: BaseStorage, subquery: PlannedSubquery
) -> list[float] | None:
    if subquery.search_mode == "fts" or not getattr(
        storage, "supports_embedding", False
    ):
        return None
    try:
        return _get_cached_query_embedding(storage, subquery.query)
    except Exception:
        logger.warning("deep search subquery embedding failed", exc_info=True)
        return None


def _submit_subquery(
    *,
    storage: BaseStorage,
    subquery: PlannedSubquery,
    request: UnifiedSearchRequest,
    fetch_k: int,
    threshold: float,
    now: datetime,
) -> Future[Any]:
    """Submit one subquery to the fan-out executor, mapped to its arm helper."""
    mode = SearchMode(subquery.search_mode)
    embedding = _subquery_embedding(storage, subquery)
    start_time, end_time = _window_bounds(subquery, now)
    options = SearchOptions(query_embedding=embedding, search_mode=mode)

    if subquery.arm == "profiles":
        return _submit_with_current_context(
            _SEARCH_FANOUT_EXECUTOR,
            _search_profiles_via_storage,
            storage,
            subquery.query,
            fetch_k,
            threshold,
            request.user_id,
            embedding,
            mode,
            start_time,
            end_time,
        )
    if subquery.arm == "agent_playbooks":
        statuses = (
            list(request.agent_playbook_status_filter)
            if request.agent_playbook_status_filter
            else list(_DEFAULT_AGENT_PLAYBOOK_STATUSES)
        )
        return _submit_with_current_context(
            _SEARCH_FANOUT_EXECUTOR,
            _search_agent_playbooks_via_storage,
            storage,
            subquery.query,
            fetch_k,
            threshold,
            request.agent_version,
            request.playbook_name,
            statuses,
            options,
            start_time,
            end_time,
        )
    user_playbook_request = SearchUserPlaybookRequest(
        query=subquery.query,
        user_id=request.user_id,
        agent_version=request.agent_version,
        playbook_name=request.playbook_name,
        status_filter=None,
        threshold=threshold,
        top_k=fetch_k,
        search_mode=mode,
        start_time=start_time,
        end_time=end_time,
    )
    return _submit_with_current_context(
        _SEARCH_FANOUT_EXECUTOR,
        _search_user_playbooks_via_storage,
        storage,
        user_playbook_request,
        options,
    )


def execute_subqueries(
    *,
    subqueries: list[PlannedSubquery],
    storage: BaseStorage,
    request: UnifiedSearchRequest,
    fetch_k: int,
    threshold: float,
    pool: list[Candidate] | None = None,
    index_offset: int = 0,
) -> list[Candidate]:
    """Run subqueries in parallel and fuse results into a candidate pool.

    Fusion is dedup-by-key in first-seen order; a candidate surfaced by
    multiple subqueries records every surfacing subquery index. Passing an
    existing ``pool`` fuses a corrective round into the prior pool.

    Args:
        subqueries: The subqueries to run (already arm-filtered).
        storage: Storage handle.
        request: The original unified search request (filters/user scope).
        fetch_k: Per-subquery result cap.
        threshold: Match threshold forwarded to the arm helpers.
        pool: Existing pool to fuse into (mutated and returned).
        index_offset: Plan-index offset for provenance in corrective rounds.

    Returns:
        list[Candidate]: The fused pool (same object as ``pool`` when given).

    Raises:
        FuturesTimeoutError: When a subquery exceeds the fan-out timeout —
            propagated so the service-level fallback serves classic instead.
    """
    now = datetime.now(UTC)
    fused = pool if pool is not None else []
    by_key = {candidate.key: candidate for candidate in fused}

    with profile_step("search.deep.execute", subqueries=len(subqueries)) as span:
        futures = [
            _submit_subquery(
                storage=storage,
                subquery=subquery,
                request=request,
                fetch_k=fetch_k,
                threshold=threshold,
                now=now,
            )
            for subquery in subqueries
        ]
        for index, (subquery, future) in enumerate(
            zip(subqueries, futures, strict=True)
        ):
            entities = future.result(timeout=_SUBQUERY_TIMEOUT_SECONDS) or []
            plan_index = index + index_offset
            for entity in entities:
                key = candidate_key(subquery.arm, entity)
                existing = by_key.get(key)
                if existing is not None:
                    existing.subquery_indices.append(plan_index)
                    continue
                candidate = Candidate(
                    key=key,
                    arm=subquery.arm,
                    entity=entity,
                    subquery_indices=[plan_index],
                )
                by_key[key] = candidate
                fused.append(candidate)
        span.set_data("pool_size", len(fused))
    return fused
