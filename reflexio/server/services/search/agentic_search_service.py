"""AgenticSearchService — 6-agent + 2-synthesizer + optional reconciler orchestrator.

Phase 4 landing: the service runs three profile-intent search agents and
three playbook-intent search agents in parallel, then parallel synthesizers
per lane, and finally the extraction reconciler only when synthesizers raise
cross-entity flags. The service returns a ``UnifiedSearchResponse`` matching
the classic pipeline's contract.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.domain.entities import AgentPlaybook, UserPlaybook
from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.services.extraction.critics import (
    CrossEntityFlag,
    Reconciler,
    summarize,
)
from reflexio.server.services.pre_retrieval import QueryReformulator
from reflexio.server.services.search.search_agents import (
    PlaybookSearchAgent,
    ProfileSearchAgent,
    SearchCtx,
)
from reflexio.server.services.search.synthesizers import (
    PlaybookSynthesizer,
    ProfileSynthesizer,
)

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
        agent_workers (int): ThreadPool workers for the 6 parallel search agents.
        synth_workers (int): ThreadPool workers for the 2 parallel synthesizers.
        agent_timeout (float): Per-future timeout applied while collecting search
            agent results.
    """

    PROFILE_INTENTS: tuple[str, str, str] = ("direct", "context", "temporal")
    PLAYBOOK_INTENTS: tuple[str, str, str] = ("direct", "context", "temporal")

    def __init__(
        self,
        *,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        agent_workers: int = 6,
        synth_workers: int = 2,
        agent_timeout: float = 30.0,
    ) -> None:
        self.client = llm_client
        self.request_context = request_context
        self.storage = request_context.storage
        self.prompt_manager = request_context.prompt_manager
        self._agent_workers = min(agent_workers, 6)
        self._synth_workers = min(synth_workers, 2)
        self._agent_timeout = agent_timeout

    def search(self, request: UnifiedSearchRequest) -> UnifiedSearchResponse:
        """Execute the full 6+2+optional-reconciler pipeline for one request.

        Args:
            request (UnifiedSearchRequest): The unified search request.

        Returns:
            UnifiedSearchResponse: Ranked profile / user_playbook / agent_playbook
            lists, the (possibly reformulated) query, and a ``msg`` field that
            flags partial failures.
        """
        partial = False
        query = self._reformulate(request)

        profile_batches, playbook_batches, partial = self._run_agents(
            query, request, partial
        )

        p_ids, p_flags, b_ids, b_flags = self._run_synthesizers(
            query, profile_batches, playbook_batches
        )

        if p_flags or b_flags:
            self._annotate_flags(p_flags + b_flags)

        ranked_profiles, ranked_playbooks = self._assemble_ranked(
            profile_batches, playbook_batches, p_ids, b_ids
        )

        return UnifiedSearchResponse(
            success=True,
            profiles=ranked_profiles,
            user_playbooks=[p for p in ranked_playbooks if isinstance(p, UserPlaybook)],
            agent_playbooks=[
                p for p in ranked_playbooks if isinstance(p, AgentPlaybook)
            ],
            reformulated_query=query,
            msg="partial: some agents timed out" if partial else None,
        )

    # ---------------- phase helpers ---------------- #

    def _reformulate(self, request: UnifiedSearchRequest) -> str:
        """Run QueryReformulator when enabled; otherwise return the raw query.

        Reformulation failures fall back to the raw query (the reformulator
        is responsible for its own exception handling).
        """
        if not request.enable_reformulation:
            return request.query
        reformulator = QueryReformulator(
            llm_client=self.client, prompt_manager=self.prompt_manager
        )
        result = reformulator.rewrite(request.query, request.conversation_history)
        return result.standalone_query or request.query

    def _run_agents(
        self,
        query: str,
        request: UnifiedSearchRequest,
        partial: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        """Run all 6 intent-specialist agents in parallel.

        Returns:
            Tuple of (profile_batches, playbook_batches, partial_flag). Each
            batch carries ``ids``, ``why``, and the raw ``hits`` list.
        """
        with ThreadPoolExecutor(max_workers=self._agent_workers) as pool:
            profile_futs = [
                pool.submit(
                    ProfileSearchAgent(
                        intent,  # type: ignore[arg-type]
                        client=self.client,
                        prompt_manager=self.prompt_manager,
                        storage=self.storage,  # type: ignore[arg-type]
                    ).run,
                    query=query,
                    req=request,
                )
                for intent in self.PROFILE_INTENTS
            ]
            playbook_futs = [
                pool.submit(
                    PlaybookSearchAgent(
                        intent,  # type: ignore[arg-type]
                        client=self.client,
                        prompt_manager=self.prompt_manager,
                        storage=self.storage,  # type: ignore[arg-type]
                    ).run,
                    query=query,
                    req=request,
                )
                for intent in self.PLAYBOOK_INTENTS
            ]
            profile_batches, profile_partial = self._collect_batches(profile_futs)
            playbook_batches, playbook_partial = self._collect_batches(playbook_futs)
        return (
            profile_batches,
            playbook_batches,
            partial or profile_partial or playbook_partial,
        )

    def _collect_batches(
        self, futures: list[Future]
    ) -> tuple[list[dict[str, Any]], bool]:
        """Collect agent futures into batches; set partial=True on any failure."""
        batches: list[dict[str, Any]] = []
        partial = False
        for fut in futures:
            try:
                ctx: SearchCtx = fut.result(timeout=self._agent_timeout)
                batches.append({"ids": ctx.ids, "why": ctx.why, "hits": ctx.hits})
            except Exception as e:
                logger.warning("search agent failed: %s: %s", type(e).__name__, e)
                partial = True
        return batches, partial

    def _run_synthesizers(
        self,
        query: str,
        profile_batches: list[dict[str, Any]],
        playbook_batches: list[dict[str, Any]],
    ) -> tuple[list[str], list[CrossEntityFlag], list[str], list[CrossEntityFlag]]:
        """Run the 2 synthesizers in parallel and return ranked IDs + flags."""
        playbook_other_lane = summarize(
            [h for b in profile_batches for h in b["hits"]], limit=15
        )
        profile_other_lane = summarize(
            [h for b in playbook_batches for h in b["hits"]], limit=15
        )
        with ThreadPoolExecutor(max_workers=self._synth_workers) as pool:
            profile_fut = pool.submit(
                ProfileSynthesizer(
                    client=self.client, prompt_manager=self.prompt_manager
                ).rank,
                query=query,
                candidates=profile_batches,
                other_lane_summary=profile_other_lane,
            )
            playbook_fut = pool.submit(
                PlaybookSynthesizer(
                    client=self.client, prompt_manager=self.prompt_manager
                ).rank,
                query=query,
                candidates=playbook_batches,
                other_lane_summary=playbook_other_lane,
            )
            p_ids, p_flags = profile_fut.result()
            b_ids, b_flags = playbook_fut.result()
        return p_ids, p_flags, b_ids, b_flags

    def _annotate_flags(self, flags: list[CrossEntityFlag]) -> None:
        """Run the Reconciler on cross-entity flags without dropping candidates.

        Search reconciliation only annotates; the orchestrator leaves the
        ranked lists untouched so downstream consumers can still inspect
        flagged items.
        """
        try:
            Reconciler(client=self.client, prompt_manager=self.prompt_manager).resolve(
                [], [], flags
            )
        except Exception as e:
            logger.warning("search reconciler failed: %s: %s", type(e).__name__, e)

    @staticmethod
    def _assemble_ranked(
        profile_batches: list[dict[str, Any]],
        playbook_batches: list[dict[str, Any]],
        p_ids: list[str],
        b_ids: list[str],
    ) -> tuple[list[Any], list[Any]]:
        """Map ranked IDs back to the raw hits collected by the agents."""
        id_to_profile = {
            h.id: h
            for b in profile_batches
            for h in b["hits"]
            if getattr(h, "id", None) is not None
        }
        id_to_playbook = {
            h.id: h
            for b in playbook_batches
            for h in b["hits"]
            if getattr(h, "id", None) is not None
        }
        ranked_profiles = [id_to_profile[i] for i in p_ids if i in id_to_profile]
        ranked_playbooks = [id_to_playbook[i] for i in b_ids if i in id_to_playbook]
        return ranked_profiles, ranked_playbooks
