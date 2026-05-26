"""Thin runner for the agentic-v2 extraction pipeline.

Assembles messages, invokes run_tool_loop with a per-kind tool registry, and
calls commit_plan on termination. Returns a CommitResult.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import ToolLoopTrace, ToolRegistry, run_tool_loop
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.invariants import commit_plan
from reflexio.server.services.extraction.plan import (
    CommitResult,
    ExtractionCtx,
    HandlerBundle,
)
from reflexio.server.services.extraction.tools import EXTRACTION_TOOLS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeferredExtractionRun:
    """Output of ExtractionAgent.run_no_commit — plan is not yet committed.

    The parallel-then-unify pipeline collects one of these per axis,
    feeds them all to ``UnifyAgent`` for cross-axis dedup, then calls
    ``ExtractionAgent.commit_deferred`` on the (possibly pruned) plan.

    Attributes:
        ctx: ExtractionCtx with the populated plan, search_count, known_ids.
        outcome: How the loop terminated.
        kind: Extraction axis label ("UserProfile" | "UserProfileAgentRec" | "UserPlaybook").
    """

    ctx: ExtractionCtx
    outcome: str
    kind: str


_PROMPT_ID_BY_KIND: dict[str, str] = {
    "UserProfile": "extraction_user_profile",  # user-side facts only as of v1.2.0
    "UserProfileAgentRec": "extraction_user_profile_agent_rec",  # agent-named-answer axis (parallel to UserProfile)
    "UserPlaybook": "extraction_user_playbook",
}


def _summarise_tool_calls(trace: ToolLoopTrace) -> str:
    """Return a compact 'tool_a:2, tool_b:1' string from a ToolLoopTrace.

    Args:
        trace (ToolLoopTrace): The completed tool loop trace.

    Returns:
        str: Comma-separated name:count pairs ordered by frequency, or '(none)'.
    """
    counts = Counter(t.tool_name for t in trace.turns)
    return ", ".join(f"{name}:{n}" for name, n in counts.most_common()) or "(none)"


def _summarise_usage(trace: ToolLoopTrace) -> str:
    """Return a per-model 'model_x: N tokens, $0.0078' string aggregated across all turns.

    A single response's usage is attached to every turn it produced, so this
    function deduplicates by (model, prompt_tokens, completion_tokens) to avoid
    double-counting when one LLM call produced multiple tool calls.

    Args:
        trace (ToolLoopTrace): The completed tool loop trace.

    Returns:
        str: Semicolon-separated per-model summaries, or '(none)'.
    """
    seen: set[tuple[str, int, int]] = set()
    per_model: dict[str, dict[str, float]] = {}
    for t in trace.turns:
        if t.model is None or t.prompt_tokens is None or t.completion_tokens is None:
            continue
        key = (t.model, t.prompt_tokens, t.completion_tokens)
        if key in seen:
            continue
        seen.add(key)
        bucket = per_model.setdefault(t.model, {"tokens": 0.0, "cost": 0.0})
        bucket["tokens"] += t.total_tokens or 0
        bucket["cost"] += t.cost_usd or 0.0
    if not per_model:
        return "(none)"
    return "; ".join(
        f"{m}: {int(v['tokens'])} tokens, ${v['cost']:.6f}"
        for m, v in per_model.items()
    )


class ExtractionAgent:
    """Single-loop adaptive extraction agent.

    Assembles the seed message from the extraction prompt, drives
    ``run_tool_loop`` with a per-entity-kind tool registry, and commits the
    accumulated plan via ``commit_plan`` on termination (finish or max_steps).

    Args:
        client (LiteLLMClient): LLM client for the underlying tool loop.
        storage: BaseStorage handle (read + commit targets).
        prompt_manager (PromptManager): Renders the per-kind extraction
            prompt — ``extraction_user_profile`` for ``UserProfile`` runs and
            ``extraction_user_playbook`` for ``UserPlaybook`` runs.
        max_steps (int): Cap on tool-calling turns (default 12; see spec §7.2).
        registry (ToolRegistry | None): Tool registry to use.  Defaults to
            ``EXTRACTION_TOOLS`` (backward-compat union of all tools).  Production
            callers should pass ``PROFILE_EXTRACTION_TOOLS`` or
            ``PLAYBOOK_EXTRACTION_TOOLS`` to restrict the LLM to one entity kind.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        storage: object,
        prompt_manager: PromptManager,
        max_steps: int = 12,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.client = client
        self.storage = storage
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps
        self.registry = registry if registry is not None else EXTRACTION_TOOLS

    def run(
        self,
        *,
        user_id: str,
        agent_version: str,
        extractor_name: str,
        extraction_criteria: str,
        sessions_text: str,
        extraction_kind: Literal[
            "UserProfile", "UserProfileAgentRec", "UserPlaybook"
        ] = "UserProfile",
        request_id: str = "",
    ) -> CommitResult:
        """Run one extraction loop and commit the resulting plan.

        Args:
            user_id (str): Authenticated user scope.
            agent_version (str): Active agent_version for this extractor config.
            extractor_name (str): The ``name`` field of the extractor config
                (used as an implicit storage filter).
            extraction_criteria (str): ``extraction_criteria`` text from the
                extractor config, rendered into the agent's prompt.
            sessions_text (str): Pre-rendered session transcript.
            extraction_kind (Literal["UserProfile", "UserPlaybook"]): Entity
                kind this run targets.  Rendered into the prompt to scope the
                LLM's narrative.  Defaults to ``"UserProfile"`` for backward
                compat with existing test callers that omit this argument.
            request_id (str): Source publish_interaction UUID; embedded into
                every profile/playbook this run creates so callers can trace
                back to the originating publish. Defaults to "" for test
                callers that don't have a publish request in scope.

        Returns:
            CommitResult: Includes applied ops, violations, and outcome.
        """
        deferred = self.run_no_commit(
            user_id=user_id,
            agent_version=agent_version,
            extractor_name=extractor_name,
            extraction_criteria=extraction_criteria,
            sessions_text=sessions_text,
            extraction_kind=extraction_kind,
            request_id=request_id,
        )
        return self.commit_deferred(deferred)

    def run_no_commit(
        self,
        *,
        user_id: str,
        agent_version: str,
        extractor_name: str,
        extraction_criteria: str,
        sessions_text: str,
        extraction_kind: Literal[
            "UserProfile", "UserProfileAgentRec", "UserPlaybook"
        ] = "UserProfile",
        request_id: str = "",
    ) -> DeferredExtractionRun:
        """Run the agent loop but do NOT commit the plan.

        Used by the parallel-then-unify pipeline so the unify pass can prune
        cross-axis duplicates before any storage write.

        Args/returns mirror :meth:`run`, except the return value is a
        ``DeferredExtractionRun`` carrying the populated ctx and outcome.
        """
        ctx = ExtractionCtx(
            user_id=user_id,
            agent_version=agent_version,
            extractor_name=extractor_name,
            request_id=request_id,
        )
        bundle = HandlerBundle(
            storage=self.storage,
            ctx=ctx,
            llm_client=self.client,
            prompt_manager=self.prompt_manager,
        )

        prompt = self.prompt_manager.render_prompt(
            _PROMPT_ID_BY_KIND[extraction_kind],
            variables={
                "sessions": sessions_text,
                "extraction_criteria": extraction_criteria,
                "max_steps": str(self.max_steps),
            },
        )

        t0 = time.monotonic()
        result = run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=self.registry,
            model_role=ModelRole.EXTRACTION_AGENT,
            max_steps=self.max_steps,
            ctx=bundle,
            finish_tool_name="finish",
            log_label=f"extraction_agent[{extractor_name}]",
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "extraction_agent[%s] kind=%s loop_done elapsed_ms=%d turns=%d/%d "
            "tools={%s} outcome=%s plan_size=%d usage={%s}",
            extractor_name,
            extraction_kind,
            elapsed_ms,
            len(result.trace.turns),
            self.max_steps,
            _summarise_tool_calls(result.trace),
            result.finished_reason,
            len(ctx.plan),
            _summarise_usage(result.trace),
        )
        return DeferredExtractionRun(
            ctx=ctx, outcome=result.finished_reason, kind=extraction_kind
        )

    def commit_deferred(self, deferred: DeferredExtractionRun) -> CommitResult:
        """Apply invariants and commit a deferred run's plan.

        The unify pass may have mutated ``deferred.ctx.plan`` in place to drop
        cross-axis duplicates; this still respects per-axis invariants because
        ``ctx.search_count`` and ``ctx.known_ids`` came from the agent's own
        loop and remain valid for the surviving ops.
        """
        commit = commit_plan(deferred.ctx, self.storage, outcome=deferred.outcome)
        logger.info(
            "extraction_agent[%s] kind=%s committed applied=%d violations=%s outcome=%s",
            deferred.ctx.extractor_name,
            deferred.kind,
            len(commit.applied),
            sorted({v.code for v in commit.violations}) or "[]",
            commit.outcome,
        )
        return commit
