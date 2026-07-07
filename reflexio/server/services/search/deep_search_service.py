"""Deep (agentic) unified search: PLAN → parallel fan-out → REFLECT.

Opt-in tier behind ``UnifiedSearchRequest.search_depth == "deep"``. Layered
entirely above the classic engine: subqueries execute through the same
per-arm helpers classic uses, storage methods are untouched, and any
unexpected failure propagates to the dispatch site in ``lib/_search.py``
which serves the classic pipeline instead.

Shape (hard-capped at 2 reflect calls, enforced structurally — no loop):

    plan (1 small-model call) → parallel subqueries → reflect (1 strong call)
        └─ if insufficient + corrective subqueries proposed:
               parallel corrective subqueries → reflect (final, no corrective)

Returns ranked entity lists only — no answer synthesis (``agent_answer``
stays None by design); ``agent_trace`` carries the plan + round outcomes.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.services.search.executor import Candidate, execute_subqueries
from reflexio.server.services.search.planner import plan_subqueries
from reflexio.server.services.search.reflector import reflect_and_rank
from reflexio.server.services.unified_search_service import (
    _suppress_source_user_playbooks,
)
from reflexio.server.tracing import profile_step

if TYPE_CHECKING:
    from reflexio.models.config_schema import DeepSearchConfig
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.search.deep_search_schemas import (
        PlannedSubquery,
        ReflectVerdict,
        SearchPlan,
    )

logger = logging.getLogger(__name__)

_ALL_ARMS = ("profiles", "user_playbooks", "agent_playbooks")
# Per-subquery over-fetch so the reflect stage sees a wider pool than top_k.
_MIN_FETCH_K = 10


class AgenticUnifiedSearchService:
    """Class handle for the deep search pipeline (dispatcher counterpart of
    :class:`~reflexio.server.services.unified_search_service.UnifiedSearchService`).

    Args:
        llm_client (LiteLLMClient): Configured LLM client.
        request_context (RequestContext): Current request context (storage +
            prompt manager).
        config (DeepSearchConfig): Operator knobs (subquery cap, timeouts).
        planner_model_name (str, optional): Explicit planner model override.
        reflector_model_name (str, optional): Explicit reflector model
            override.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        *,
        config: DeepSearchConfig,
        planner_model_name: str | None = None,
        reflector_model_name: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.request_context = request_context
        self.config = config
        self.planner_model_name = planner_model_name
        self.reflector_model_name = reflector_model_name

    def search(
        self, request: UnifiedSearchRequest, org_id: str
    ) -> UnifiedSearchResponse:
        """Run the deep search pipeline for one request.

        Args:
            request (UnifiedSearchRequest): The unified search request.
            org_id (str): Organization id (trace metadata only).

        Returns:
            UnifiedSearchResponse: Ranked entities + agent_trace.

        Raises:
            Exception: Any pipeline failure propagates — the dispatch site
                falls back to classic search.
        """
        storage = self.request_context.storage
        if storage is None:
            raise RuntimeError("deep search requires configured storage")

        allowed_arms = self._allowed_arms(request)
        if not allowed_arms:
            return UnifiedSearchResponse(
                success=True, msg="No searchable entity types for this request"
            )

        top_k = request.top_k if request.top_k is not None else 5
        threshold = request.threshold if request.threshold is not None else 0.3
        fetch_k = max(top_k, _MIN_FETCH_K)

        with profile_step("search.deep", org_id=org_id, arms=allowed_arms) as span:
            plan = plan_subqueries(
                query=request.query,
                conversation_history=request.conversation_history,
                allowed_arms=allowed_arms,
                max_subqueries=self.config.max_subqueries,
                llm_client=self.llm_client,
                prompt_manager=self.request_context.prompt_manager,
                model_name=self.planner_model_name,
                timeout_s=self.config.planner_timeout_s,
            )
            pool = execute_subqueries(
                subqueries=plan.subqueries,
                storage=storage,
                request=request,
                fetch_k=fetch_k,
                threshold=threshold,
            )
            verdict = self._reflect(request, plan, pool, allow_corrective=True)
            rounds = [verdict]

            corrective = self._corrective_subqueries(verdict, allowed_arms)
            if corrective:
                pool = execute_subqueries(
                    subqueries=corrective,
                    storage=storage,
                    request=request,
                    fetch_k=fetch_k,
                    threshold=threshold,
                    pool=pool,
                    index_offset=len(plan.subqueries),
                )
                verdict = self._reflect(request, plan, pool, allow_corrective=False)
                rounds.append(verdict)

            span.set_data("pool_size", len(pool))
            span.set_data("rounds", len(rounds))
            span.set_data("final_sufficiency", verdict.sufficiency)
            return self._assemble_response(
                storage=storage,
                plan=plan,
                corrective=corrective,
                pool=pool,
                verdict=verdict,
                rounds=rounds,
                top_k=top_k,
            )

    def _allowed_arms(self, request: UnifiedSearchRequest) -> list[str]:
        arms = [
            arm
            for arm in _ALL_ARMS
            if request.entity_types is None or arm in request.entity_types
        ]
        if not request.user_id:
            arms = [arm for arm in arms if arm != "profiles"]
        return arms

    def _reflect(
        self,
        request: UnifiedSearchRequest,
        plan: SearchPlan,
        pool: list[Candidate],
        *,
        allow_corrective: bool,
    ) -> ReflectVerdict:
        return reflect_and_rank(
            query=request.query,
            plan_notes=plan.notes,
            candidates=pool,
            allow_corrective=allow_corrective,
            llm_client=self.llm_client,
            prompt_manager=self.request_context.prompt_manager,
            model_name=self.reflector_model_name,
            timeout_s=self.config.reflect_timeout_s,
        )

    def _corrective_subqueries(
        self, verdict: ReflectVerdict, allowed_arms: list[str]
    ) -> list[PlannedSubquery]:
        if verdict.sufficiency == "sufficient":
            return []
        return [
            sq
            for sq in verdict.corrective_subqueries
            if sq.arm in allowed_arms and sq.query
        ][: self.config.max_subqueries]

    def _assemble_response(
        self,
        *,
        storage: object,
        plan: SearchPlan,
        corrective: list[PlannedSubquery],
        pool: list[Candidate],
        verdict: ReflectVerdict,
        rounds: list[ReflectVerdict],
        top_k: int,
    ) -> UnifiedSearchResponse:
        ordered = _order_pool(pool, verdict.ranked_candidate_ids)
        recency_arms = {
            sq.arm for sq in (*plan.subqueries, *corrective) if sq.recency_dominant
        }

        per_arm: dict[str, list[Candidate]] = {arm: [] for arm in _ALL_ARMS}
        for candidate in ordered:
            per_arm[candidate.arm].append(candidate)
        for arm in recency_arms:
            per_arm[arm] = sorted(
                per_arm[arm], key=lambda c: c.timestamp(), reverse=True
            )

        profiles = [c.entity for c in per_arm["profiles"][:top_k]]
        agent_playbooks = [c.entity for c in per_arm["agent_playbooks"][:top_k]]
        user_playbooks = [c.entity for c in per_arm["user_playbooks"][:top_k]]
        user_playbooks = _suppress_source_user_playbooks(
            storage=storage,  # type: ignore[arg-type]
            agent_playbooks=agent_playbooks,
            user_playbooks=user_playbooks,
        )
        return UnifiedSearchResponse(
            success=True,
            profiles=profiles,
            agent_playbooks=agent_playbooks,
            user_playbooks=user_playbooks,
            agent_trace=_build_trace(plan, corrective, rounds, len(pool)),
        )


def _order_pool(pool: list[Candidate], ranked_keys: list[str]) -> list[Candidate]:
    """Ranked candidates first (verdict order), then the rest in fusion order."""
    by_key = {candidate.key: candidate for candidate in pool}
    ordered = [by_key[key] for key in ranked_keys if key in by_key]
    ranked_set = set(ranked_keys)
    ordered.extend(c for c in pool if c.key not in ranked_set)
    return ordered


def _build_trace(
    plan: SearchPlan,
    corrective: list[PlannedSubquery],
    rounds: list[ReflectVerdict],
    pool_size: int,
) -> str:
    """Compact JSON trace of the pipeline for observability."""
    return json.dumps(
        {
            "plan": [sq.model_dump(exclude_none=True) for sq in plan.subqueries],
            "plan_notes": plan.notes,
            "corrective": [sq.model_dump(exclude_none=True) for sq in corrective],
            "rounds": [
                {"sufficiency": r.sufficiency, "rationale": r.rationale} for r in rounds
            ],
            "pool_size": pool_size,
        }
    )
