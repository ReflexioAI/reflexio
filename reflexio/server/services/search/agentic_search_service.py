"""AgenticSearchService — single SearchAgent loop replacing the v1 6+2 stack.

Agentic-v2 delegates to a single ``SearchAgent`` that drives a tool loop
(``search_user_profiles``, ``search_user_playbooks``, ``search_agent_playbooks``,
``finish``) and returns a free-text answer.

API contract preserved:
- Constructor: ``AgenticSearchService(llm_client, request_context)``
- Method: ``.search(request: UnifiedSearchRequest) -> UnifiedSearchResponse``
- ``UnifiedSearchResponse.msg`` carries the agent's natural-language answer.

Note: ``profiles``, ``user_playbooks``, and ``agent_playbooks`` are returned empty
in agentic-v2 — the agent returns a synthesised answer rather than ranked entity
lists. Callers that need the Q&A answer should read ``response.msg``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.search.search_agent import SearchAgent

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)


class AgenticSearchService:
    """Agentic search orchestrator wired into the backend dispatcher.

    Construction matches ``UnifiedSearchService`` so ``build_search_service``
    can swap the two transparently: both accept ``llm_client`` and
    ``request_context`` as keyword arguments.

    Args:
        llm_client (LiteLLMClient): Configured LLM client for all agent calls.
        request_context (RequestContext): Request context providing
            ``storage`` and ``prompt_manager``.
    """

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage
        self.prompt_manager = request_context.prompt_manager

    def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """Execute the agentic-v2 search for one request.

        Optionally reformulates the query, then delegates to ``SearchAgent``
        which drives a tool loop and returns a natural-language answer.

        Args:
            request (UnifiedSearchRequest): The unified search request.

        Returns:
            UnifiedSearchResponse: ``success=True``, empty entity lists, and
            the agent's answer in the ``msg`` field. ``reformulated_query``
            carries the (possibly rewritten) query used for the search.
        """
        query = self._reformulate(request)

        agent = SearchAgent(
            client=self.client,
            storage=self.storage,
            prompt_manager=self.prompt_manager,
        )
        result = agent.run(
            user_id=request.user_id or "",
            agent_version=request.agent_version or "",
            query=query,
        )

        answer: str = result.get("answer") or ""
        if result.get("budget_exceeded"):
            logger.warning("search agent hit max_steps budget for query %r", query[:80])

        return UnifiedSearchResponse(
            success=True,
            profiles=[],
            user_playbooks=[],
            agent_playbooks=[],
            reformulated_query=query,
            msg=answer or None,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _reformulate(self, request: UnifiedSearchRequest) -> str:
        """Run QueryReformulator when enabled; otherwise return the raw query.

        Reformulation failures fall back to the raw query (the reformulator
        is responsible for its own exception handling).

        Args:
            request (UnifiedSearchRequest): The search request.

        Returns:
            str: Reformulated query string, or the original query if
            reformulation is disabled or the reformulator returns nothing.
        """
        if not request.enable_reformulation:
            return request.query
        reformulator = QueryReformulator(
            llm_client=self.client, prompt_manager=self.prompt_manager
        )
        result = reformulator.rewrite(request.query, request.conversation_history)
        return result.standalone_query or request.query
