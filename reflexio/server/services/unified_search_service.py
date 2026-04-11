"""
Unified search service that searches across all entity types in parallel.

Executes in two phases:
  Phase A: Query reformulation + embedding generation (sequential)
  Phase B: Entity searches across profiles, agent playbooks, user playbooks (parallel)
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

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
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)


def run_unified_search(
    request: UnifiedSearchRequest,
    org_id: str,
    storage: BaseStorage,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    pre_retrieval_model_name: str | None = None,
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

    # --- Phase A: query reformulation + embedding generation ---
    supports_embedding = hasattr(storage, "_get_embedding")
    reformulated_query, embedding = _run_phase_a(
        query=request.query,
        storage=storage,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        supports_embedding=supports_embedding,
        conversation_history=request.conversation_history,
        enable_reformulation=bool(request.enable_reformulation),
        pre_retrieval_model_name=pre_retrieval_model_name,
    )

    # --- Phase B: parallel searches across all entity types ---
    profiles, agent_playbooks, user_playbooks = _run_phase_b(
        request=request,
        org_id=org_id,
        storage=storage,
        embedding=embedding,
        query=reformulated_query,
        top_k=top_k,
        threshold=threshold,
    )

    if profiles is None:
        return UnifiedSearchResponse(success=False, msg="Search failed")

    return UnifiedSearchResponse(
        success=True,
        profiles=profiles,
        agent_playbooks=agent_playbooks,  # type: ignore[reportArgumentType]
        user_playbooks=user_playbooks,  # type: ignore[reportArgumentType]
        reformulated_query=reformulated_query
        if reformulated_query != request.query
        else None,
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

    Returns:
        tuple[str, Optional[list[float]]]: (standalone_query, embedding_vector) — embedding is None when unsupported or on failure
    """
    reformulator = QueryReformulator(
        llm_client=llm_client,
        prompt_manager=prompt_manager,
        model_name=pre_retrieval_model_name,
    )

    # Query reformulation (rewrite() handles all exceptions internally)
    if enable_reformulation:
        result = reformulator.rewrite(query, conversation_history)
        standalone_query = result.standalone_query
    else:
        standalone_query = query

    # Embedding generation (uses reformulated query for semantic accuracy)
    embedding = None
    if supports_embedding:
        try:
            embedding = storage._get_embedding(standalone_query)  # type: ignore[reportAttributeAccessIssue]
        except Exception as e:
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
    options = SearchOptions(query_embedding=embedding)

    executor = ThreadPoolExecutor(max_workers=3)
    try:
        profiles_future = executor.submit(
            _search_profiles_via_storage,
            storage,
            query,
            top_k,
            threshold,
            request.user_id,
            embedding,
        )
        fb_request = SearchAgentPlaybookRequest(
            query=query,
            agent_version=request.agent_version,
            playbook_name=request.playbook_name,
            status_filter=[None],
            threshold=threshold,
            top_k=top_k,
        )
        agent_playbooks_future = executor.submit(
            storage.search_agent_playbooks, fb_request, options
        )
        rf_request = SearchUserPlaybookRequest(
            query=query,
            user_id=request.user_id,
            agent_version=request.agent_version,
            playbook_name=request.playbook_name,
            status_filter=None,
            threshold=threshold,
            top_k=top_k,
        )
        user_playbooks_future = executor.submit(
            storage.search_user_playbooks, rf_request, options
        )

        profiles = profiles_future.result(timeout=30)
        agent_playbooks = agent_playbooks_future.result(timeout=30)
        user_playbooks = user_playbooks_future.result(timeout=30)
    except FuturesTimeoutError:
        logger.error("Unified search timed out")
        return None, None, None
    except Exception as e:
        logger.error("Unified search failed: %s", e)
        return None, None, None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return profiles, agent_playbooks, user_playbooks


def _search_profiles_via_storage(
    storage: BaseStorage,
    query: str,
    top_k: int,
    threshold: float,
    user_id: str | None,
    embedding: list[float] | None,
) -> list[UserProfile]:
    """Search profiles via storage.search_user_profile, returning [] on error or missing user_id.

    Args:
        storage (BaseStorage): Storage instance
        query (str): Search query text
        top_k (int): Maximum results
        threshold (float): Minimum match threshold
        user_id (Optional[str]): User ID filter (required for profile search)
        embedding (Optional[list[float]]): Pre-computed query embedding, or None for text-only search

    Returns:
        list[UserProfile]: Matching profiles, or [] on error/missing user_id
    """
    if not user_id:
        return []
    try:
        return storage.search_user_profile(
            SearchUserProfileRequest(
                user_id=user_id,
                query=query,
                top_k=top_k,
                threshold=threshold,
            ),
            status_filter=[None],
            query_embedding=embedding,
        )
    except Exception as e:
        logger.error("Profile search failed: %s", e)
        return []
