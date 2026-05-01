"""AgenticSearchService — single SearchAgent loop replacing the v1 6+2 stack.

Agentic-v2 delegates to a single ``SearchAgent`` that drives a tool loop
(``search_user_profiles``, ``search_user_playbooks``, ``search_agent_playbooks``,
``finish``) and returns a free-text answer plus populated entity lists harvested
from the tool-loop trace.

API contract preserved:
- Constructor: ``AgenticSearchService(llm_client, request_context)``
- Method: ``.search(request: UnifiedSearchRequest) -> UnifiedSearchResponse``
- ``UnifiedSearchResponse.agent_answer`` carries the agent's natural-language answer.
- ``UnifiedSearchResponse.profiles`` / ``user_playbooks`` / ``agent_playbooks`` are
  populated by filtering per-user storage reads against the IDs seen in the trace.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.search.plan import SearchResult
from reflexio.server.services.search.search_agent import (
    SearchAgent,
    _summarise_tool_calls,
)

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.llm.tools import ToolLoopTrace

logger = logging.getLogger(__name__)

# Tool names that produce profile results in the trace
_PROFILE_TOOLS = {"search_user_profiles", "get_user_profile"}
# Tool names that produce user playbook results in the trace
_USER_PLAYBOOK_TOOLS = {"search_user_playbooks", "get_user_playbook"}
# Tool names that produce agent playbook results in the trace
_AGENT_PLAYBOOK_TOOLS = {"search_agent_playbooks", "get_agent_playbook"}


def _harvest_ids_from_trace(
    trace: ToolLoopTrace,
) -> tuple[list[str], list[str], list[str]]:
    """Walk the trace and harvest entity IDs in first-seen order.

    Args:
        trace (ToolLoopTrace): Full tool-loop trace from a SearchAgent run.

    Returns:
        tuple[list[str], list[str], list[str]]: Three ordered lists of unique IDs:
            profile_ids, user_playbook_ids, agent_playbook_ids.
    """
    profile_ids: list[str] = []
    user_playbook_ids: list[str] = []
    agent_playbook_ids: list[str] = []

    seen_profiles: set[str] = set()
    seen_user_playbooks: set[str] = set()
    seen_agent_playbooks: set[str] = set()

    for turn in trace.turns:
        tool = turn.tool_name
        result = turn.result

        if tool in _PROFILE_TOOLS:
            # search returns {"hits": [...]} each item has "id"
            # get returns {"profile": {...}} with "id"
            items = result.get("hits") or (
                [result["profile"]] if "profile" in result else []
            )
            for item in items:
                pid = item.get("id", "") if isinstance(item, dict) else ""
                if pid and pid not in seen_profiles:
                    seen_profiles.add(pid)
                    profile_ids.append(pid)

        elif tool in _USER_PLAYBOOK_TOOLS:
            items = result.get("hits") or (
                [result["playbook"]] if "playbook" in result else []
            )
            for item in items:
                pid = item.get("id", "") if isinstance(item, dict) else ""
                if pid and pid not in seen_user_playbooks:
                    seen_user_playbooks.add(pid)
                    user_playbook_ids.append(pid)

        elif tool in _AGENT_PLAYBOOK_TOOLS:
            items = result.get("hits") or (
                [result["playbook"]] if "playbook" in result else []
            )
            for item in items:
                pid = item.get("id", "") if isinstance(item, dict) else ""
                if pid and pid not in seen_agent_playbooks:
                    seen_agent_playbooks.add(pid)
                    agent_playbook_ids.append(pid)

    return profile_ids, user_playbook_ids, agent_playbook_ids


def _filter_ordered(
    entities: list,
    id_attr: str,
    ordered_ids: list[str],
    top_k: int,
) -> list:
    """Filter entities by id set and return them in first-seen trace order, capped at top_k.

    Args:
        entities (list): Full list of entities fetched from storage.
        id_attr (str): Attribute name on each entity that holds its string ID.
        ordered_ids (list[str]): IDs in first-seen trace order.
        top_k (int): Maximum number of results to return.

    Returns:
        list: Filtered and ordered entities, at most top_k items.
    """
    id_set = set(ordered_ids)
    by_id = {
        str(getattr(e, id_attr, "")): e
        for e in entities
        if str(getattr(e, id_attr, "")) in id_set
    }
    return [by_id[eid] for eid in ordered_ids if eid in by_id][:top_k]


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
        Entity IDs visited during the loop are harvested from the trace and
        used to populate the response entity lists.

        Args:
            request (UnifiedSearchRequest): The unified search request.

        Returns:
            UnifiedSearchResponse: ``success=True``, entity lists populated from
            the agent's trace, and the agent's answer in ``agent_answer``.
            ``reformulated_query`` carries the (possibly rewritten) query used
            for the search.
        """
        # Reject requests missing the user_id rather than silently coercing
        # to empty strings. An empty user_id flows into storage operations
        # (storage.get_user_profile, storage.add_user_profile) and would
        # either return cross-user data on SqliteStorage or write to an
        # unintended path on DiskStorage. Surface the bug at the boundary.
        # agent_version is NOT required — it scopes AgentPlaybook reads
        # (cross-user rules), and an empty value just means "no AgentPlaybook
        # scope filter," which is safe.
        if not request.user_id:
            raise ValueError(
                "agentic search requires a non-empty user_id; got empty"
            )

        query = self._reformulate(request)

        agent = SearchAgent(
            client=self.client,
            storage=self.storage,
            prompt_manager=self.prompt_manager,
            # Tight budget for benchmark throughput; default is 10.
            # Floor is 2 (one search → finish); 5 accommodates the
            # rehydration-mandated patterns (search → reformulate →
            # rehydrate → finish = 4) plus one optional rerank step,
            # while still bounding wasted work on simple questions.
            max_steps=5,
            enable_agent_answer=bool(request.enable_agent_answer),
        )
        result = agent.run(
            user_id=request.user_id,
            agent_version=request.agent_version or "",
            query=query,
        )

        if result.outcome == "error":
            logger.warning("search agent returned error for query %r", query[:80])
            return UnifiedSearchResponse(
                success=True,
                profiles=[],
                user_playbooks=[],
                agent_playbooks=[],
                reformulated_query=query,
                msg=f"agent error: {result.answer or 'unknown'}",
                agent_answer=None,
                agent_trace=_summarise_tool_calls(result.trace),
            )

        if result.budget_exceeded:
            logger.warning("search agent hit max_steps budget for query %r", query[:80])

        profiles, user_playbooks, agent_playbooks = self._fetch_entities(
            request, result
        )

        rehydrated_text = (
            "\n\n".join(result.rehydrated_excerpts)
            if result.rehydrated_excerpts
            else None
        )
        return UnifiedSearchResponse(
            success=True,
            profiles=profiles,
            user_playbooks=user_playbooks,
            agent_playbooks=agent_playbooks,
            reformulated_query=query,
            msg=None,
            agent_answer=result.answer,
            agent_trace=_summarise_tool_calls(result.trace),
            rehydrated_text=rehydrated_text,
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

    def _fetch_entities(
        self,
        request: UnifiedSearchRequest,
        result: SearchResult,
    ) -> tuple[list, list, list]:
        """Harvest entity IDs from trace, fetch all-user entities once, filter in-memory.

        Args:
            request (UnifiedSearchRequest): The original search request (for user_id,
                agent_version, top_k).
            result (SearchResult): Completed agent run with trace.

        Returns:
            tuple[list, list, list]: (profiles, user_playbooks, agent_playbooks) each
                filtered and ordered by first-seen trace position, capped at top_k.
        """
        top_k = request.top_k or 5
        user_id = request.user_id or ""
        agent_version = request.agent_version or ""

        profile_ids, user_playbook_ids, agent_playbook_ids = _harvest_ids_from_trace(
            result.trace
        )

        storage = self.storage
        if storage is None:
            return [], [], []

        profiles: list = []
        if profile_ids:
            all_profiles = storage.get_user_profile(user_id)
            profiles = _filter_ordered(all_profiles, "profile_id", profile_ids, top_k)

        user_playbooks: list = []
        if user_playbook_ids:
            all_user_playbooks = storage.get_user_playbooks(
                user_id=user_id, agent_version=agent_version
            )
            user_playbooks = _filter_ordered(
                all_user_playbooks, "user_playbook_id", user_playbook_ids, top_k
            )

        agent_playbooks: list = []
        if agent_playbook_ids:
            all_agent_playbooks = storage.get_agent_playbooks(
                agent_version=agent_version
            )
            agent_playbooks = _filter_ordered(
                all_agent_playbooks, "agent_playbook_id", agent_playbook_ids, top_k
            )

        return profiles, user_playbooks, agent_playbooks
