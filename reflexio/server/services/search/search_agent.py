"""Thin runner for the agentic-v2 search pipeline. Read-only — no commit stage."""

from __future__ import annotations

import logging

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import run_tool_loop
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.extraction_agent import _ExtractionBundle
from reflexio.server.services.extraction.plan import ExtractionCtx
from reflexio.server.services.extraction.tools import SEARCH_TOOLS

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.client = client
        self.storage = storage
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def run(self, *, user_id: str, agent_version: str, query: str) -> dict:
        """Run one search loop for the given query.

        Args:
            user_id (str): Authenticated user scope.
            agent_version (str): Active agent_version for playbook scoping.
            query (str): The search query to answer.

        Returns:
            dict: ``{"answer": str, "outcome": str, "budget_exceeded": bool}``.
        """
        ctx = ExtractionCtx(user_id=user_id, agent_version=agent_version)
        bundle = _ExtractionBundle(storage=self.storage, ctx=ctx)

        prompt = self.prompt_manager.render_prompt(
            "search_agent", variables={"query": query}
        )

        result = run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=SEARCH_TOOLS,
            model_role=ModelRole.SEARCH_AGENT,
            max_steps=self.max_steps,
            ctx=bundle,
            finish_tool_name="finish",
            log_label="search_agent",
        )

        answer = getattr(ctx, "_search_answer", "no answer")
        return {
            "answer": answer,
            "outcome": result.finished_reason,
            "budget_exceeded": result.finished_reason == "max_steps",
        }
