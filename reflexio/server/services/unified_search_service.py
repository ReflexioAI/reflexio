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
from typing import TYPE_CHECKING, Any, cast

from reflexio.models.api_schema.retriever_schema import (
    ConversationTurn,
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
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.retrieval.relevance_floor import apply_relevance_floors
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
RetrievalCaptureHook = Callable[
    [UnifiedSearchRequest, UnifiedSearchResponse, BaseStorage, str], None
]
_retrieval_capture_hook: RetrievalCaptureHook | None = None


def configure_retrieval_capture_hook(hook: RetrievalCaptureHook | None) -> None:
    """Register an optional final-response retrieval capture hook.

    Deployments that capture retrieval logs install the hook; OSS leaves it
    unset so unified search behavior is unchanged by default.
    """
    global _retrieval_capture_hook
    _retrieval_capture_hook = hook


def run_unified_search(
    request: UnifiedSearchRequest,
    org_id: str,
    storage: BaseStorage,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    pre_retrieval_model_name: str | None = None,
    retrieval_floor: RetrievalFloorConfig | None = None,
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

    top_k = request.top_k if request.top_k is not None else 5
    threshold = request.threshold if request.threshold is not None else 0.3

    floor_cfg = retrieval_floor or RetrievalFloorConfig()
    floor_on = floor_cfg.enabled
    fetch_k = max(top_k, floor_cfg.pool_size) if floor_on else top_k

    # --- Phase A: query reformulation + embedding generation ---
    reformulated_query, embedding = _run_phase_a(
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

    # --- Phase B: parallel searches across all entity types ---
    profiles, agent_playbooks, user_playbooks = _run_phase_b(
        request=request,
        org_id=org_id,
        storage=storage,
        embedding=embedding,
        query=reformulated_query,
        top_k=fetch_k,
        threshold=threshold,
    )

    if profiles is None:
        return UnifiedSearchResponse(success=False, msg="Search failed")

    if floor_on:
        profiles, agent_playbooks, user_playbooks = _apply_floors(
            query=reformulated_query,
            profiles=profiles,
            agent_playbooks=agent_playbooks,  # type: ignore[arg-type]
            user_playbooks=user_playbooks,  # type: ignore[arg-type]
            top_k=top_k,
            cfg=floor_cfg,
        )

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
    )
    _maybe_capture_final_response(
        request=request,
        response=response,
        storage=storage,
        org_id=org_id,
    )
    return response


def _maybe_capture_final_response(
    *,
    request: UnifiedSearchRequest,
    response: UnifiedSearchResponse,
    storage: BaseStorage,
    org_id: str,
) -> None:
    hook = _retrieval_capture_hook
    if hook is None:
        return
    try:
        hook(request, response, storage, org_id)
    except Exception:
        logger.warning(
            "Unified search retrieval capture hook failed",
            exc_info=True,
        )


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
) -> tuple[str, list[float] | None]:
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
        tuple[str, Optional[list[float]]]: (standalone_query, embedding_vector) — embedding is None when unsupported or on failure
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
            result = reformulator.rewrite(query, conversation_history)
            standalone_query = result.standalone_query
        else:
            standalone_query = query

    # Embedding generation (uses reformulated query for semantic accuracy).
    # FTS-only search has no use for an embedding, so skip the call entirely.
    embedding = None
    if supports_embedding and search_mode != SearchMode.FTS:
        with profile_step(
            "search.embedding",
            backend=_storage_backend_name(storage),
            purpose="query",
        ) as span:
            try:
                embedding = _get_cached_query_embedding(storage, standalone_query)
                span.set_data("embedding_generated", embedding is not None)
            except Exception as e:
                span.set_data("embedding_generated", False)
                logger.error("Embedding generation failed: %s", e)

    return standalone_query, embedding


def _run_phase_b(
    request: UnifiedSearchRequest,
    org_id: str,  # noqa: ARG001
    storage: BaseStorage,
    embedding: list[float] | None,
    query: str,
    top_k: int,
    threshold: float,
) -> tuple[
    list[UserProfile] | None,
    list[AgentPlaybook] | None,
    list[UserPlaybook] | None,
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

    Returns:
        tuple: (profiles, agent_playbooks, user_playbooks) — all None on timeout/failure
    """
    options = SearchOptions(query_embedding=embedding, search_mode=request.search_mode)

    entity_types = set(request.entity_types or _DEFAULT_ENTITY_TYPES)
    allowed_agent_statuses = request.agent_playbook_status_filter
    try:
        with profile_step(
            "search.phase_b",
            backend=_storage_backend_name(storage),
            entity_types=sorted(entity_types),
            top_k=top_k,
        ) as span:
            if (
                getattr(storage, "supports_unified_hybrid_search", False)
                and _unified_single_rpc_enabled()
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
                    request.search_mode,
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
                    status_filter=None,
                    threshold=threshold,
                    top_k=top_k,
                    search_mode=request.search_mode,
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
) -> tuple[list[UserProfile], list[AgentPlaybook], list[UserPlaybook]] | None:
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
    unified_hybrid_search = getattr(storage, "unified_hybrid_search", None)
    if not callable(unified_hybrid_search):
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
        agent_playbook_statuses=statuses,
        search_mode=request.search_mode,
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
    deduped: list[AgentPlaybook] = []
    seen_ids: set[str] = set()
    for playbook in agent_playbooks:
        playbook_id = str(getattr(playbook, "agent_playbook_id", ""))
        if playbook_id and playbook_id not in seen_ids:
            seen_ids.add(playbook_id)
            deduped.append(playbook)
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
    )
    return floored_profiles, floored_agent, floored_user


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
    model_name = str(getattr(storage, "embedding_model_name", "unknown"))
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
) -> list[AgentPlaybook]:
    """Search agent playbooks, restricted to one or more approval statuses.

    When ``allowed_statuses`` is None or empty, falls back to
    ``_DEFAULT_AGENT_PLAYBOOK_STATUSES`` (APPROVED + PENDING). Callers that
    genuinely want REJECTED playbooks must opt in by passing the full list.
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
            status_filter=[None],
            playbook_status_filter=statuses,
            threshold=threshold,
            top_k=top_k,
            search_mode=options.search_mode,
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
) -> list[UserProfile]:
    """Search profiles via storage.search_user_profile, returning [] on error or missing user_id.

    Args:
        storage (BaseStorage): Storage instance
        query (str): Search query text
        top_k (int): Maximum results
        threshold (float): Minimum match threshold
        user_id (Optional[str]): User ID filter (required for profile search)
        embedding (Optional[list[float]]): Pre-computed query embedding, or None for text-only search
        search_mode (SearchMode): Search mode (hybrid/vector/fts)

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
                    search_mode=search_mode,
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
