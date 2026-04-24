"""Thin runner for the agentic-v2 extraction pipeline.

Assembles messages, invokes run_tool_loop with a per-kind tool registry, and
calls commit_plan on termination. Returns a CommitResult.
"""

from __future__ import annotations

import logging
from typing import Literal

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import ToolRegistry, run_tool_loop
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.invariants import commit_plan
from reflexio.server.services.extraction.plan import (
    CommitResult,
    ExtractionCtx,
    HandlerBundle,
)
from reflexio.server.services.extraction.tools import EXTRACTION_TOOLS

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """Single-loop adaptive extraction agent.

    Assembles the seed message from the extraction prompt, drives
    ``run_tool_loop`` with a per-entity-kind tool registry, and commits the
    accumulated plan via ``commit_plan`` on termination (finish or max_steps).

    Args:
        client (LiteLLMClient): LLM client for the underlying tool loop.
        storage: BaseStorage handle (read + commit targets).
        prompt_manager (PromptManager): Renders the ``extraction_agent`` prompt.
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
        extraction_kind: Literal["UserProfile", "UserPlaybook"] = "UserProfile",
    ) -> CommitResult:
        """Run one extraction loop over the given session text.

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

        Returns:
            CommitResult: Includes applied ops, violations, and outcome.
        """
        ctx = ExtractionCtx(
            user_id=user_id,
            agent_version=agent_version,
            extractor_name=extractor_name,
        )
        bundle = HandlerBundle(storage=self.storage, ctx=ctx)

        prompt = self.prompt_manager.render_prompt(
            "extraction_agent",
            variables={
                "sessions": sessions_text,
                "extraction_criteria": extraction_criteria,
                "extraction_kind": extraction_kind,
            },
        )

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

        return commit_plan(ctx, self.storage, outcome=result.finished_reason)
