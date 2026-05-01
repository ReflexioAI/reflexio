"""Thin runner for the agentic-v2 search pipeline. Read-only — no commit stage."""

from __future__ import annotations

import logging
import time
from collections import Counter

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import ToolLoopTrace, run_tool_loop
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.plan import ExtractionCtx, HandlerBundle
from reflexio.server.services.extraction.tools import (
    SEARCH_TOOLS,
    SearchAgentTurnPlan,
)
from reflexio.server.services.search.plan import SearchResult

logger = logging.getLogger(__name__)


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


class SearchAgent:
    """Single-loop adaptive search agent (read-only).

    Assembles the seed message from the search_agent prompt, drives
    ``run_tool_loop`` with ``SEARCH_TOOLS``, and extracts the answer stashed on
    ctx by ``_handle_search_finish``. No commit stage occurs.

    Args:
        client (LiteLLMClient): LLM client for the underlying tool loop.
        storage: BaseStorage handle (read-only for this agent).
        prompt_manager (PromptManager): Renders the ``search_agent`` prompt.
        max_steps (int): Cap on tool-calling turns (default 10; spec §7.2).
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        storage: object,
        prompt_manager: PromptManager,
        max_steps: int = 10,
        enable_agent_answer: bool = False,
    ) -> None:
        self.client = client
        self.storage = storage
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps
        self.enable_agent_answer = enable_agent_answer

    def run(self, *, user_id: str, agent_version: str, query: str) -> SearchResult:
        """Run one search loop for the given query.

        Args:
            user_id (str): Authenticated user scope.
            agent_version (str): Active agent_version for playbook scoping.
            query (str): The search query to answer.

        Returns:
            SearchResult: Typed outcome with answer, termination reason, budget flag,
                and the full tool-loop trace for entity harvesting by callers.
        """
        ctx = ExtractionCtx(user_id=user_id, agent_version=agent_version)
        bundle = HandlerBundle(
            storage=self.storage,
            ctx=ctx,
            llm_client=self.client,
            prompt_manager=self.prompt_manager,
        )

        prompt = self.prompt_manager.render_prompt(
            "search_agent",
            variables={
                "query": query,
                "max_steps": str(self.max_steps),
                "enable_agent_answer": "true" if self.enable_agent_answer else "false",
            },
        )

        t0 = time.monotonic()
        result = run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=SEARCH_TOOLS,
            model_role=ModelRole.SEARCH_AGENT,
            max_steps=self.max_steps,
            ctx=bundle,
            finish_tool_name="finish",
            multi_stage_schema=SearchAgentTurnPlan,
            log_label="search_agent",
        )

        # In search-only mode the agent is told to call finish() with no answer;
        # we surface None so callers can distinguish "agent declined to answer"
        # from "agent failed". Tests that exercised the answer path keep working
        # because they default-construct SearchAgent with enable_agent_answer=False
        # but populate ctx.search_answer via the mocked finish() call — when off,
        # we deliberately drop whatever the agent wrote so the contract is clear.
        if not self.enable_agent_answer:
            answer: str | None = None
        else:
            answer = ctx.search_answer if ctx.search_answer is not None else "no answer"
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "search_agent elapsed_ms=%d turns=%d/%d tools={%s} outcome=%s "
            "answer_len=%d usage={%s}",
            elapsed_ms,
            len(result.trace.turns),
            self.max_steps,
            _summarise_tool_calls(result.trace),
            result.finished_reason,
            len(answer) if answer is not None else 0,
            _summarise_usage(result.trace),
        )
        return SearchResult(
            answer=answer,
            outcome=result.finished_reason,
            budget_exceeded=result.finished_reason == "max_steps",
            trace=result.trace,
            rehydrated_excerpts=list(ctx.rehydrated_excerpts),
        )
