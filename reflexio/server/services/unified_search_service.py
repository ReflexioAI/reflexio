"""
Unified search service that searches across all entity types in parallel.

Executes in two phases:
  Phase A: Query reformulation + embedding generation (sequential)
  Phase B: Entity searches across profiles, agent playbooks, user playbooks (parallel)
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from reflexio.models.api_schema.retriever_schema import (
    ConversationTurn,
    ReformulationResult,
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    RetrievalFloorConfig,
    SearchMode,
    SearchOptions,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.embedding_text import resolve_retrieval_threshold
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.retrieval.recency import (
    RecencyConfig,
    ScoredItem,
    additive_penalty,
    decay_for_item,
    multiplicative_factor,
)
from reflexio.server.services.retrieval.relevance_floor import apply_relevance_floors
from reflexio.server.services.retrieval.session_dedup import (
    EntityKey,
    session_seen_cache,
)
from reflexio.server.services.retrieval.temporal import (
    freshness_collapse,
    sort_by_recency,
    window_bounds,
)
from reflexio.server.services.retrieval.user_context_guard import (
    should_suppress_user_context,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.tracing import profile_step, set_span_data

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext

logger = logging.getLogger(__name__)
_DEFAULT_ENTITY_TYPES = frozenset({"profiles", "agent_playbooks", "user_playbooks"})
_SOURCE_USER_PLAYBOOK_IDS_KEY = "_source_user_playbook_ids"
# Statuses returned for agent_playbooks when the caller does not pass an
# explicit ``agent_playbook_status_filter``. Excludes REJECTED so that a
# rejection in the dashboard immediately suppresses the playbook from search
# results — every consumer benefits without opting in. Callers that genuinely
# want REJECTED items (e.g. admin views) must pass the full list explicitly.
_DEFAULT_AGENT_PLAYBOOK_STATUSES: tuple[PlaybookStatus, ...] = (
    PlaybookStatus.APPROVED,
    PlaybookStatus.PENDING,
)
_SEARCH_FANOUT_MAX_WORKERS = max(
    1, int(os.getenv("REFLEXIO_SEARCH_FANOUT_WORKERS", "16") or "16")
)
_SEARCH_FANOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_SEARCH_FANOUT_MAX_WORKERS,
    thread_name_prefix="reflexio-search",
)
_ENV_SINGLE_RPC = "REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC"
# Working-pool floor for temporal reordering (freshness collapse / recency
# sort): matches the RecencyConfig default pool so the fresh version of a
# fact can be promoted even when it ranks below top_k on text relevance.
_TEMPORAL_POOL_SIZE = 20
# Session dedup widens the fetch pool so filtered-out (already served) items
# can be backfilled, but never by more than this many extra candidates.
_SESSION_DEDUP_FETCH_BUMP_CAP = 50
# Singular dedup kinds (session_dedup EntityKey vocabulary) — deliberately
# distinct from the plural request-level ``entity_types`` values.
_ENTITY_ID_ATTRS = {
    "profile": "profile_id",
    "user_playbook": "user_playbook_id",
    "agent_playbook": "agent_playbook_id",
}
_EMBEDDING_CACHE_TTL_SECONDS = max(
    0, int(os.getenv("REFLEXIO_QUERY_EMBEDDING_CACHE_TTL_SECONDS", "300") or "300")
)
_EMBEDDING_CACHE_MAX_SIZE = max(
    1, int(os.getenv("REFLEXIO_QUERY_EMBEDDING_CACHE_MAX_SIZE", "1024") or "1024")
)
_embedding_cache_lock = threading.Lock()
_embedding_cache: OrderedDict[tuple[str, int, str, str], tuple[float, list[float]]] = (
    OrderedDict()
)


def run_unified_search(
    request: UnifiedSearchRequest,
    org_id: str,
    storage: BaseStorage,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    pre_retrieval_model_name: str | None = None,
    retrieval_floor: RetrievalFloorConfig | None = None,
    recency: RecencyConfig | None = None,
) -> UnifiedSearchResponse:
    """
    Search across all entity types (profiles, agent playbooks, user playbooks) in parallel.

    Phase A runs query reformulation and embedding generation sequentially.
    Phase B runs all entity searches in parallel using the results from Phase A.

    Args:
        request (UnifiedSearchRequest): The unified search request
        org_id (str): Organization ID (used for feature flag checks)
        storage: Storage instance (BaseStorage implementation)
        llm_client (LiteLLMClient): Shared LLM client instance
        prompt_manager (PromptManager): Prompt manager for query reformulator
        pre_retrieval_model_name (str, optional): Model name override for query reformulation.
            Caller should resolve this from config and/or site vars.

    Returns:
        UnifiedSearchResponse: Combined results from all entity types
    """
    if not request.query:
        return UnifiedSearchResponse(success=True, msg="No query provided")

    with profile_step("search.user_context_guard", entity_type="all") as span:
        suppress_user_context = should_suppress_user_context(
            request.query,
            include_user_context=request.include_user_context,
        )
        span.set_data("suppressed", suppress_user_context)
    if suppress_user_context:
        requested_entity_types = set(request.entity_types or _DEFAULT_ENTITY_TYPES)
        effective_entity_types = requested_entity_types - {
            "profiles",
            "user_playbooks",
        }
        if not effective_entity_types:
            return UnifiedSearchResponse(success=True)
        request = request.model_copy(
            update={"entity_types": sorted(effective_entity_types)}
        )

    top_k = request.top_k if request.top_k is not None else 5
    threshold = resolve_retrieval_threshold(
        request.threshold,
        model_name=storage.embedding_model_name,
    )

    floor_cfg = retrieval_floor or RetrievalFloorConfig()
    floor_on = floor_cfg.enabled
    recency_on = bool(recency and recency.enabled)

    # --- Phase A: query reformulation (+ temporal signals) + embedding ---
    reformulation, embedding, embedding_failed = _run_phase_a(
        query=request.query,
        storage=storage,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        supports_embedding=storage.supports_embedding,
        conversation_history=request.conversation_history,
        enable_reformulation=bool(request.enable_reformulation),
        pre_retrieval_model_name=pre_retrieval_model_name,
        search_mode=request.search_mode,
    )
    reformulated_query = reformulation.standalone_query

    # Degrade-to-FTS: when embedding GENERATION failed (distinct from the
    # benign None cases), force the effective mode to FTS for the entire
    # Phase B fan-out. Storage then takes the zero-placeholder branch and
    # never re-embeds — closing the historical second-embed leak where a
    # None embedding + a vector/hybrid mode caused the storage layer to retry
    # the embed and raise unguarded. Benign None (FTS-only, unsupported)
    # leaves the requested mode untouched.
    effective_search_mode = request.search_mode
    if embedding_failed:
        effective_search_mode = SearchMode.FTS
        logger.warning(
            "event=search_degraded_to_fts reason=embedding_generation_failed "
            "backend=%s requested_mode=%s",
            _storage_backend_name(storage),
            request.search_mode.value,
        )
    window_start, window_end = window_bounds(
        reformulation.start_days_ago, reformulation.end_days_ago
    )

    # Temporal reordering (freshness collapse / recency sort) happens after
    # relevance ranking, so it needs a wider working pool: the fresh version
    # of a fact may rank below top_k on text relevance alone. When signals
    # are present, fetch and rank a larger pool and cut to top_k only after
    # the temporal pass.
    temporal_reorder = reformulation.wants_current or reformulation.recency_dominant
    fetch_k = max(
        top_k,
        floor_cfg.pool_size if floor_on else 0,
        recency.pool_size if recency_on and recency is not None else 0,
        _TEMPORAL_POOL_SIZE if temporal_reorder else 0,
    )
    result_cap = max(top_k, _TEMPORAL_POOL_SIZE) if temporal_reorder else top_k

    # Session dedup: a request that carries a session_id skips items already
    # served to that (org, session) and backfills from a widened pool.
    session_id = (request.session_id or "").strip()
    seen_keys = (
        session_seen_cache.seen(org_id, session_id) if session_id else frozenset()
    )
    if seen_keys:
        fetch_k += min(len(seen_keys), _SESSION_DEDUP_FETCH_BUMP_CAP)

    # --- Phase B: parallel searches across all entity types ---
    profiles, agent_playbooks, user_playbooks = _run_phase_b(
        request=request,
        org_id=org_id,
        storage=storage,
        embedding=embedding,
        query=reformulated_query,
        top_k=fetch_k,
        threshold=threshold,
        recency_on=recency_on,
        start_time=window_start,
        end_time=window_end,
        search_mode=effective_search_mode,
    )

    if profiles is None:
        return UnifiedSearchResponse(success=False, msg="Search failed")

    if seen_keys:
        # Drop before the floors so seen items spend no cross-encoder budget.
        profiles = _drop_seen_items(profiles or [], "profile", seen_keys)
        agent_playbooks = _drop_seen_items(
            agent_playbooks or [], "agent_playbook", seen_keys
        )
        user_playbooks = _drop_seen_items(
            user_playbooks or [], "user_playbook", seen_keys
        )

    if floor_on:
        profiles, agent_playbooks, user_playbooks = _apply_floors(
            query=reformulated_query,
            profiles=profiles,
            agent_playbooks=agent_playbooks,  # type: ignore[arg-type]
            user_playbooks=user_playbooks,  # type: ignore[arg-type]
            top_k=result_cap,
            cfg=floor_cfg,
            recency=recency if recency_on else None,
        )
    elif recency_on and recency is not None:
        profiles = _apply_combined_score_recency(
            profiles or [], entity_type="profiles", top_k=result_cap, cfg=recency
        )
        agent_playbooks = _apply_combined_score_recency(
            agent_playbooks or [],
            entity_type="agent_playbooks",
            top_k=result_cap,
            cfg=recency,
        )
        user_playbooks = _apply_combined_score_recency(
            user_playbooks or [],
            entity_type="user_playbooks",
            top_k=result_cap,
            cfg=recency,
        )
    else:
        profiles = _unwrap_items(profiles or [])[:result_cap]
        agent_playbooks = _unwrap_items(agent_playbooks or [])[:result_cap]
        user_playbooks = _unwrap_items(user_playbooks or [])[:result_cap]

    # --- Temporal post-processing from query-derived signals ---
    if temporal_reorder:
        arms = [profiles or [], agent_playbooks or [], user_playbooks or []]
        if reformulation.wants_current:
            arms = [freshness_collapse(arm) for arm in arms]
        if reformulation.recency_dominant:
            arms = [sort_by_recency(arm) for arm in arms]
        # Final cut to top_k only after the temporal pass, so items promoted
        # from the wider pool can take the top slots.
        profiles, agent_playbooks, user_playbooks = (arm[:top_k] for arm in arms)

    user_playbooks = _suppress_source_user_playbooks(
        storage=storage,
        agent_playbooks=agent_playbooks or [],
        user_playbooks=user_playbooks or [],
    )

    response = UnifiedSearchResponse(
        success=True,
        profiles=profiles,
        agent_playbooks=agent_playbooks,  # type: ignore[reportArgumentType]
        user_playbooks=user_playbooks,  # type: ignore[reportArgumentType]
        reformulated_query=reformulated_query
        if reformulated_query != request.query
        else None,
        degraded=embedding_failed,
        search_mode_effective=effective_search_mode.value if embedding_failed else None,
    )
    if session_id:
        session_seen_cache.record(org_id, session_id, _served_entity_keys(response))
    return response


def _run_phase_a(
    query: str,
    storage: BaseStorage,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    supports_embedding: bool = True,
    conversation_history: list[ConversationTurn] | None = None,
    enable_reformulation: bool = False,
    pre_retrieval_model_name: str | None = None,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> tuple[ReformulationResult, list[float] | None, bool]:
    """Run query reformulation and embedding generation sequentially.

    Args:
        query (str): The original search query
        storage (BaseStorage): Storage instance
        llm_client (LiteLLMClient): Shared LLM client instance
        prompt_manager (PromptManager): Prompt manager instance
        supports_embedding (bool): Whether the storage backend supports embedding generation.
            When False, skips embedding and returns None (local/self-host storage).
        conversation_history (list, optional): Prior conversation turns for context-aware query reformulation
        enable_reformulation (bool): Whether query reformulation is enabled for this request
        pre_retrieval_model_name (str, optional): Model name override for query reformulation
        search_mode (SearchMode): Search mode; FTS-only mode skips embedding generation entirely

    Returns:
        tuple[ReformulationResult, Optional[list[float]], bool]: The
            reformulation result (standalone query + temporal signals;
            signal-free pass-through of the original query when reformulation
            is disabled), the query embedding (None when unsupported, when the
            mode is FTS-only, or when generation failed), and
            ``embedding_failed`` — True *only* when an embedding attempt was
            made and raised. The benign None cases (FTS-only mode,
            ``supports_embedding`` False) leave ``embedding_failed`` False so
            callers can distinguish a real failure from a mode that never
            needed an embedding.
    """
    reformulator = QueryReformulator(
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        model_name=pre_retrieval_model_name,
    )

    # Query reformulation (rewrite() handles all exceptions internally)
    with profile_step(
        "search.reformulate",
        enabled=enable_reformulation,
        has_conversation_history=bool(conversation_history),
    ):
        if enable_reformulation:
            reformulation = reformulator.rewrite(query, conversation_history)
        else:
            reformulation = ReformulationResult(standalone_query=query)

    # Embedding generation (uses reformulated query for semantic accuracy).
    # FTS-only search has no use for an embedding, so skip the call entirely.
    embedding = None
    embedding_failed = False
    if supports_embedding and search_mode != SearchMode.FTS:
        with profile_step(
            "search.embedding",
            backend=_storage_backend_name(storage),
            purpose="query",
        ) as span:
            try:
                embedding = _get_cached_query_embedding(
                    storage, reformulation.standalone_query
                )
                # A falsy result (None or []) means the embedder produced no
                # usable query vector — e.g. a storage backend that swallows
                # EmbeddingUnavailableError and returns [] on a provider outage.
                # Treat it the same as a raised failure: degrade to FTS rather
                # than run a vector search with an empty embedding.
                if not embedding:
                    embedding = None
                    embedding_failed = True
                span.set_data("embedding_generated", embedding is not None)
            except Exception as e:
                span.set_data("embedding_generated", False)
                embedding_failed = True
                logger.exception("Embedding generation failed: %s", e)

    return reformulation, embedding, embedding_failed


def _run_phase_b(
    request: UnifiedSearchRequest,
    org_id: str,  # noqa: ARG001
    storage: BaseStorage,
    embedding: list[float] | None,
    query: str,
    top_k: int,
    threshold: float,
    recency_on: bool = False,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> tuple[
    list[Any] | None,
    list[Any] | None,
    list[Any] | None,
]:
    """Run parallel searches across all entity types by delegating to storage methods.

    Args:
        request (UnifiedSearchRequest): The search request (for filters)
        org_id (str): Organization ID
        storage (BaseStorage): Storage instance
        embedding (Optional[list[float]]): Pre-computed query embedding, or None for text-only search
        query (str): Query string (possibly rewritten) for FTS
        top_k (int): Maximum results per entity type
        threshold (float): Minimum match threshold
        start_time (Optional[datetime]): Query-derived time-window lower bound
            (from reformulation temporal signals); forces the per-arm fan-out
            path since the combined single RPC has no time parameters.
        end_time (Optional[datetime]): Query-derived time-window upper bound.
        search_mode (SearchMode): The EFFECTIVE search mode for this fan-out —
            equal to ``request.search_mode`` normally, but forced to FTS by the
            caller when embedding generation failed (degrade-to-FTS). Threaded
            into every arm so storage never re-embeds a failed query.

    Returns:
        tuple: (profiles, agent_playbooks, user_playbooks) — all None on timeout/failure
    """
    options = SearchOptions(query_embedding=embedding, search_mode=search_mode)

    entity_types = set(request.entity_types or _DEFAULT_ENTITY_TYPES)
    allowed_agent_statuses = request.agent_playbook_status_filter
    try:
        with profile_step(
            "search.phase_b",
            backend=_storage_backend_name(storage),
            entity_types=sorted(entity_types),
            top_k=top_k,
        ) as span:
            # Recency needs the per-row ``combined_score``, which only the scored
            # single-RPC method threads back. Backends that don't advertise
            # ``supports_unified_hybrid_search`` (e.g. native Postgres, which still
            # inherits ``unified_hybrid_search_scored`` and runs it via the same
            # ``_rpc`` it already uses for ``hybrid_match_*``) opt into the scored
            # path only when recency is on, so non-recency routing is unchanged.
            wants_scored_single_rpc = recency_on and callable(
                getattr(storage, "unified_hybrid_search_scored", None)
            )
            # A query-derived time window forces the per-arm fan-out: the
            # combined single RPC has no time parameters.
            has_time_window = start_time is not None or end_time is not None
            if (
                not has_time_window
                and _unified_single_rpc_enabled()
                and (
                    getattr(storage, "supports_unified_hybrid_search", False)
                    or wants_scored_single_rpc
                )
            ):
                combined = _run_phase_b_single_rpc(
                    request=request,
                    storage=storage,
                    embedding=embedding,
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                    entity_types=entity_types,
                    allowed_agent_statuses=allowed_agent_statuses,
                    recency_on=recency_on,
                    search_mode=search_mode,
                )
                if combined is not None:
                    profiles, agent_playbooks, user_playbooks = combined
                    set_span_data(
                        span,
                        {
                            "single_rpc": True,
                            "profiles_count": len(profiles),
                            "agent_playbooks_count": len(agent_playbooks),
                            "user_playbooks_count": len(user_playbooks),
                        },
                    )
                    return profiles, agent_playbooks, user_playbooks
                span.set_data("single_rpc_fallback", True)
            profiles_future = (
                _submit_with_current_context(
                    _SEARCH_FANOUT_EXECUTOR,
                    _search_profiles_via_storage,
                    storage,
                    query,
                    top_k,
                    threshold,
                    request.user_id,
                    embedding,
                    search_mode,
                    start_time,
                    end_time,
                    tags=request.tags,
                )
                if "profiles" in entity_types
                else None
            )
            agent_playbooks_future = (
                _submit_with_current_context(
                    _SEARCH_FANOUT_EXECUTOR,
                    _search_agent_playbooks_via_storage,
                    storage,
                    query,
                    top_k,
                    threshold,
                    request.agent_version,
                    request.playbook_name,
                    allowed_agent_statuses,
                    options,
                    start_time,
                    end_time,
                    tags=request.tags,
                )
                if "agent_playbooks" in entity_types
                else None
            )
            if "user_playbooks" in entity_types:
                rf_request = SearchUserPlaybookRequest(
                    query=query,
                    user_id=request.user_id,
                    agent_version=request.agent_version,
                    playbook_name=request.playbook_name,
                    tags=request.tags,
                    status_filter=None,
                    threshold=threshold,
                    top_k=top_k,
                    search_mode=search_mode,
                    start_time=start_time,
                    end_time=end_time,
                )
                user_playbooks_future = _submit_with_current_context(
                    _SEARCH_FANOUT_EXECUTOR,
                    _search_user_playbooks_via_storage,
                    storage,
                    rf_request,
                    options,
                )
            else:
                user_playbooks_future = None

            profiles = profiles_future.result(timeout=30) if profiles_future else []
            agent_playbooks = (
                agent_playbooks_future.result(timeout=30)
                if agent_playbooks_future
                else []
            )
            user_playbooks = (
                user_playbooks_future.result(timeout=30)
                if user_playbooks_future
                else []
            )
            set_span_data(
                span,
                {
                    "profiles_count": len(profiles),
                    "agent_playbooks_count": len(agent_playbooks),
                    "user_playbooks_count": len(user_playbooks),
                },
            )
    except FuturesTimeoutError:
        logger.error("Unified search timed out")
        return None, None, None
    except Exception as e:
        logger.error("Unified search failed: %s", e)
        return None, None, None

    return profiles, agent_playbooks, user_playbooks


def _unified_single_rpc_enabled() -> bool:
    """Kill switch for the combined Phase B RPC (default on)."""
    return os.getenv(_ENV_SINGLE_RPC, "1").strip().lower() not in {"0", "false", "off"}


def _run_phase_b_single_rpc(
    *,
    request: UnifiedSearchRequest,
    storage: BaseStorage,
    embedding: list[float] | None,
    query: str,
    top_k: int,
    threshold: float,
    entity_types: set[str],
    allowed_agent_statuses: list[PlaybookStatus] | None,
    recency_on: bool = False,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> tuple[list[Any], list[Any], list[Any]] | None:
    """Run all Phase B arms through one combined storage round trip.

    Trades the per-arm round-trip overhead for serialized execution of the
    three queries inside one database session — a win when round-trip
    overhead dominates per-arm query time (toggle via
    ``REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC`` to compare).

    Returns:
        The three result lists, or None when the combined call fails so the
        caller can fall back to the per-arm fan-out (e.g. the SQL function
        is not yet migrated on this deployment). Timeouts propagate like the
        fan-out path so a hung database is not retried.
    """
    statuses = (
        list(allowed_agent_statuses)
        if allowed_agent_statuses
        else list(_DEFAULT_AGENT_PLAYBOOK_STATUSES)
    )
    # Resolve storage.unified_hybrid_search before submit so missing or stale
    # capability flags can fall back to the fan-out path.
    method_name = (
        "unified_hybrid_search_scored" if recency_on else "unified_hybrid_search"
    )
    unified_hybrid_search = getattr(storage, method_name, None)
    if not callable(unified_hybrid_search):
        if recency_on:
            logger.warning(
                "event=search_recency_missing_scores source=single_rpc method=%s",
                method_name,
            )
        return None

    future = _submit_with_current_context(
        _SEARCH_FANOUT_EXECUTOR,
        unified_hybrid_search,
        query=query,
        query_embedding=embedding,
        top_k=top_k,
        threshold=threshold,
        user_id=request.user_id,
        agent_version=request.agent_version,
        playbook_name=request.playbook_name,
        tags=request.tags,
        agent_playbook_statuses=statuses,
        search_mode=search_mode,
        include_profiles="profiles" in entity_types and bool(request.user_id),
        include_agent_playbooks="agent_playbooks" in entity_types,
        include_user_playbooks="user_playbooks" in entity_types,
    )
    try:
        profiles, agent_playbooks, user_playbooks = future.result(timeout=30)
    except FuturesTimeoutError:
        raise
    except Exception:
        logger.warning(
            "Unified single-RPC search failed; falling back to per-arm fan-out",
            exc_info=True,
        )
        return None

    # Mirror _search_agent_playbooks_via_storage: dedupe by id, cap at top_k.
    deduped: list[Any] = []
    seen_ids: set[str] = set()
    for candidate in agent_playbooks:
        playbook = _unwrap_item(candidate)
        playbook_id = str(getattr(playbook, "agent_playbook_id", ""))
        if playbook_id and playbook_id not in seen_ids:
            seen_ids.add(playbook_id)
            deduped.append(candidate)
            if len(deduped) >= top_k:
                break
    return profiles, deduped, user_playbooks


def _apply_floors(
    query: str,
    profiles: list[UserProfile],
    agent_playbooks: list[AgentPlaybook],
    user_playbooks: list[UserPlaybook],
    top_k: int,
    cfg: RetrievalFloorConfig,
    recency: RecencyConfig | None = None,
) -> tuple[list[UserProfile], list[AgentPlaybook], list[UserPlaybook]]:
    """Apply the per-arm relevance floor with one batched cross-encoder call."""
    floored_profiles, floored_agent, floored_user = apply_relevance_floors(
        query,
        [
            ("profiles", profiles, cfg.profile_floor),
            ("agent_playbooks", agent_playbooks, cfg.agent_playbook_floor),
            ("user_playbooks", user_playbooks, cfg.user_playbook_floor),
        ],
        top_k,
        content_of=lambda item: _unwrap_item(item).content,
    )
    return (
        _finalize_floor_arm(
            floored_profiles, entity_type="profiles", top_k=top_k, recency=recency
        ),
        _finalize_floor_arm(
            floored_agent,
            entity_type="agent_playbooks",
            top_k=top_k,
            recency=recency,
        ),
        _finalize_floor_arm(
            floored_user,
            entity_type="user_playbooks",
            top_k=top_k,
            recency=recency,
        ),
    )


def _finalize_floor_arm(
    result: Any,
    *,
    entity_type: str,
    top_k: int,
    recency: RecencyConfig | None,
) -> list[Any]:
    if not recency or not recency.enabled:
        return _unwrap_items(result.items)[:top_k]
    if result.scores is None:
        return _apply_combined_score_recency(
            result.items, entity_type=entity_type, top_k=top_k, cfg=recency
        )
    now = int(datetime.now(UTC).timestamp())
    rescored = []
    for item, score in zip(result.items, result.scores, strict=True):
        unwrapped = _unwrap_item(item)
        freshness = decay_for_item(unwrapped, entity_type=entity_type, now=now)
        rescored.append(
            (unwrapped, score - additive_penalty(freshness, recency.max_penalty_logit))
        )
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _score in rescored[:top_k]]


def _apply_combined_score_recency(
    items: list[Any],
    *,
    entity_type: str,
    top_k: int,
    cfg: RecencyConfig,
) -> list[Any]:
    if not items:
        return []
    scored_items: list[tuple[Any, float]] = []
    for item in items:
        if not isinstance(item, ScoredItem) or item.score is None:
            logger.warning(
                "event=search_recency_missing_scores entity_type=%s items=%d",
                entity_type,
                len(items),
            )
            return _unwrap_items(items)[:top_k]
        scored_items.append((item.item, item.score))
    now = int(datetime.now(UTC).timestamp())
    rescored = []
    for item, score in scored_items:
        freshness = decay_for_item(item, entity_type=entity_type, now=now)
        rescored.append(
            (
                item,
                score * multiplicative_factor(freshness, cfg.max_penalty_frac),
            )
        )
    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _score in rescored[:top_k]]


def _unwrap_item(item: Any) -> Any:
    return item.item if isinstance(item, ScoredItem) else item


def _unwrap_items(items: list[Any]) -> list[Any]:
    return [_unwrap_item(item) for item in items]


def _entity_key(kind: str, item: Any) -> EntityKey | None:
    """Return the session-dedup key for a (possibly score-wrapped) item."""
    raw_id = getattr(_unwrap_item(item), _ENTITY_ID_ATTRS[kind], None)
    if raw_id in (None, "", 0):
        return None
    return (kind, str(raw_id))


def _drop_seen_items(
    items: list[Any], kind: str, seen: frozenset[EntityKey]
) -> list[Any]:
    return [item for item in items if _entity_key(kind, item) not in seen]


def _served_entity_keys(response: UnifiedSearchResponse) -> list[EntityKey]:
    arms: tuple[tuple[str, list[Any]], ...] = (
        ("profile", response.profiles or []),
        ("user_playbook", response.user_playbooks or []),
        ("agent_playbook", response.agent_playbooks or []),
    )
    return [
        key
        for kind, items in arms
        for item in items
        if (key := _entity_key(kind, item)) is not None
    ]


def _suppress_source_user_playbooks(
    *,
    storage: BaseStorage,
    agent_playbooks: list[AgentPlaybook],
    user_playbooks: list[UserPlaybook],
) -> list[UserPlaybook]:
    """Drop user playbooks already represented by returned agent playbooks."""
    if not agent_playbooks or not user_playbooks:
        return user_playbooks

    source_user_playbook_ids: set[int] = set()
    agent_ids_needing_lookup: list[int] = []
    for playbook in agent_playbooks:
        source_ids = getattr(playbook, _SOURCE_USER_PLAYBOOK_IDS_KEY, None)
        if source_ids is None:
            agent_playbook_id = int(getattr(playbook, "agent_playbook_id", 0) or 0)
            if agent_playbook_id:
                agent_ids_needing_lookup.append(agent_playbook_id)
            continue
        source_user_playbook_ids.update(int(source_id) for source_id in source_ids)

    if agent_ids_needing_lookup:
        lookup = getattr(
            storage, "get_source_user_playbook_ids_for_agent_playbooks", None
        )
        if callable(lookup):
            try:
                source_ids_by_agent = cast(
                    dict[int, list[int]], lookup(agent_ids_needing_lookup)
                )
            except Exception:
                logger.warning(
                    "Failed to resolve source user playbooks for unified search suppression",
                    exc_info=True,
                )
            else:
                for source_ids in source_ids_by_agent.values():
                    source_user_playbook_ids.update(
                        int(source_id) for source_id in source_ids
                    )

    if not source_user_playbook_ids:
        return user_playbooks

    filtered = [
        playbook
        for playbook in user_playbooks
        if int(getattr(playbook, "user_playbook_id", 0) or 0)
        not in source_user_playbook_ids
    ]
    suppressed_count = len(user_playbooks) - len(filtered)
    if suppressed_count:
        with profile_step(
            "search.suppress_source_user_playbooks",
            suppressed_count=suppressed_count,
            source_user_playbook_count=len(source_user_playbook_ids),
        ):
            pass
    return filtered


def _get_cached_query_embedding(
    storage: BaseStorage,
    query: str,
) -> list[float]:
    """Return a cached query embedding when available."""
    model_name = storage.embedding_model_name
    dimensions = int(getattr(storage, "embedding_dimensions", 0) or 0)
    normalized_query = " ".join(query.casefold().split())
    key = (model_name, dimensions, normalized_query, "query")
    now = time.monotonic()
    if _EMBEDDING_CACHE_TTL_SECONDS > 0:
        with _embedding_cache_lock:
            cached = _embedding_cache.get(key)
            if cached is not None:
                created_at, value = cached
                if now - created_at <= _EMBEDDING_CACHE_TTL_SECONDS:
                    _embedding_cache.move_to_end(key)
                    return list(value)
                del _embedding_cache[key]

    embedding = storage._get_embedding(query, purpose="query")  # type: ignore[reportAttributeAccessIssue]
    if _EMBEDDING_CACHE_TTL_SECONDS > 0 and embedding:
        with _embedding_cache_lock:
            _embedding_cache[key] = (now, list(embedding))
            _embedding_cache.move_to_end(key)
            while len(_embedding_cache) > _EMBEDDING_CACHE_MAX_SIZE:
                _embedding_cache.popitem(last=False)
    return embedding


def _search_agent_playbooks_via_storage(
    storage: BaseStorage,
    query: str,
    top_k: int,
    threshold: float,
    agent_version: str | None,
    playbook_name: str | None,
    allowed_statuses: list[PlaybookStatus] | None,
    options: SearchOptions,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    tags: list[str] | None = None,
) -> list[AgentPlaybook]:
    """Search agent playbooks, restricted to one or more approval statuses.

    When ``allowed_statuses`` is None or empty, falls back to
    ``_DEFAULT_AGENT_PLAYBOOK_STATUSES`` (APPROVED + PENDING). Callers that
    genuinely want REJECTED playbooks must opt in by passing the full list.
    ``start_time``/``end_time`` bound ``created_at`` (query-derived time
    windows from reformulation temporal signals; unset when the query names
    no window). ``tags`` matches playbooks having any requested tag (OR
    semantics); None or an empty list disables tag filtering.
    """
    with profile_step(
        "search.branch.agent_playbooks",
        backend=_storage_backend_name(storage),
        top_k=top_k,
    ) as span:
        statuses = (
            list(allowed_statuses)
            if allowed_statuses
            else list(_DEFAULT_AGENT_PLAYBOOK_STATUSES)
        )
        request = SearchAgentPlaybookRequest(
            query=query,
            agent_version=agent_version,
            playbook_name=playbook_name,
            tags=tags,
            status_filter=[None],
            playbook_status_filter=statuses,
            threshold=threshold,
            top_k=top_k,
            search_mode=options.search_mode,
            start_time=start_time,
            end_time=end_time,
        )
        results: list[AgentPlaybook] = []
        seen_ids: set[str] = set()
        for playbook in storage.search_agent_playbooks(request, options):
            playbook_id = str(getattr(playbook, "agent_playbook_id", ""))
            if playbook_id and playbook_id not in seen_ids:
                seen_ids.add(playbook_id)
                results.append(playbook)
                if len(results) >= top_k:
                    break
        span.set_data("result_count", len(results))
        return results


def _search_profiles_via_storage(
    storage: BaseStorage,
    query: str,
    top_k: int,
    threshold: float,
    user_id: str | None,
    embedding: list[float] | None,
    search_mode: SearchMode,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    tags: list[str] | None = None,
) -> list[UserProfile]:
    """Search profiles via storage.search_user_profile, returning [] on error or missing user_id.

    Args:
        storage (BaseStorage): Storage instance
        query (str): Search query text
        top_k (int): Maximum results
        threshold (float): Minimum match threshold
        user_id (Optional[str]): User ID filter (required for profile search)
        tags (Optional[list[str]]): Match profiles having any requested tag
        embedding (Optional[list[float]]): Pre-computed query embedding, or None for text-only search
        search_mode (SearchMode): Search mode (hybrid/vector/fts)
        start_time (Optional[datetime]): Lower bound on last_modified_timestamp
            (query-derived time window from reformulation temporal signals;
            unset when the query names no window)
        end_time (Optional[datetime]): Upper bound on last_modified_timestamp

    Returns:
        list[UserProfile]: Matching profiles, or [] on error/missing user_id
    """
    with profile_step(
        "search.branch.profiles",
        backend=_storage_backend_name(storage),
        top_k=top_k,
    ) as span:
        if not user_id:
            span.set_data("result_count", 0)
            return []
        try:
            profiles = storage.search_user_profile(
                SearchUserProfileRequest(
                    user_id=user_id,
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                    tags=tags,
                    search_mode=search_mode,
                    start_time=start_time,
                    end_time=end_time,
                ),
                status_filter=[None],
                query_embedding=embedding,
            )
            span.set_data("result_count", len(profiles))
            return profiles
        except Exception as e:
            span.set_data("result_count", 0)
            logger.error("Profile search failed: %s", e)
            return []


def _search_user_playbooks_via_storage(
    storage: BaseStorage,
    request: SearchUserPlaybookRequest,
    options: SearchOptions,
) -> list[UserPlaybook]:
    with profile_step(
        "search.branch.user_playbooks",
        backend=_storage_backend_name(storage),
        top_k=request.top_k,
    ) as span:
        user_playbooks = storage.search_user_playbooks(request, options)
        span.set_data("result_count", len(user_playbooks))
        return user_playbooks


def _submit_with_current_context(
    executor: ThreadPoolExecutor,
    fn: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> Future[Any]:
    context = contextvars.copy_context()
    return executor.submit(context.run, fn, *args, **kwargs)


def _storage_backend_name(storage: BaseStorage) -> str:
    class_name = storage.__class__.__name__.lower()
    if "postgres" in class_name:
        return "postgres"
    if "supabase" in class_name:
        return "supabase"
    return class_name


class UnifiedSearchService:
    """Class handle for the classic unified search pipeline.

    Wraps :func:`run_unified_search` so the dispatcher factory can return an
    object whose ``__class__.__name__`` can be inspected uniformly alongside
    the agentic search service (Phase 4).

    Args:
        llm_client (LiteLLMClient): Configured LLM client.
        request_context (RequestContext): Current request context.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        self.llm_client = llm_client
        self.request_context = request_context
