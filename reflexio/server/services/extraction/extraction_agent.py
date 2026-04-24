"""Thin runner for the agentic-v2 extraction pipeline.

Assembles messages, invokes run_tool_loop with EXTRACTION_TOOLS, and calls
commit_plan on termination. Returns a CommitResult.
"""

from __future__ import annotations

import logging

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import run_tool_loop
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.invariants import commit_plan
from reflexio.server.services.extraction.plan import CommitResult, ExtractionCtx
from reflexio.server.services.extraction.tools import EXTRACTION_TOOLS

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """Single-loop adaptive extraction agent.

    Assembles the seed message from the extraction prompt, drives
    ``run_tool_loop`` with ``EXTRACTION_TOOLS``, and commits the accumulated
    plan via ``commit_plan`` on termination (finish or max_steps).

    Args:
        client (LiteLLMClient): LLM client for the underlying tool loop.
        storage: BaseStorage handle (read + commit targets).
        prompt_manager (PromptManager): Renders the ``extraction_agent`` prompt.
        max_steps (int): Cap on tool-calling turns (default 12; see spec §7.2).
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        storage: object,
        prompt_manager: PromptManager,
        max_steps: int = 12,
    ) -> None:
        self.client = client
        self.storage = storage
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def run(
        self,
        *,
        user_id: str,
        agent_version: str,
        extractor_name: str,
        extraction_criteria: str,
        sessions_text: str,
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

        Returns:
            CommitResult: Includes applied ops, violations, and outcome.
        """
        ctx = ExtractionCtx(
            user_id=user_id,
            agent_version=agent_version,
            extractor_name=extractor_name,
        )
        bundle = _ExtractionBundle(storage=self.storage, ctx=ctx)

        prompt = self.prompt_manager.render_prompt(
            "extraction_agent",
            variables={
                "sessions": sessions_text,
                "extraction_criteria": extraction_criteria,
            },
        )

        result = run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=EXTRACTION_TOOLS,
            model_role=ModelRole.EXTRACTION_AGENT,
            max_steps=self.max_steps,
            ctx=bundle,
            finish_tool_name="finish",
            log_label=f"extraction_agent[{extractor_name}]",
        )

        return commit_plan(ctx, self.storage, outcome=result.finished_reason)


class _ExtractionBundle:
    """Glue so tool handlers can access both storage and ctx through one param.

    ``_bundle_handler`` in ``tools.py`` unpacks ``bundle.storage`` and
    ``bundle.ctx`` and forwards them to the underlying 3-arg handler.

    Args:
        storage: BaseStorage instance for read and commit operations.
        ctx (ExtractionCtx): Per-run state accumulator.
    """

    __slots__ = ("storage", "ctx")

    def __init__(self, storage: object, ctx: ExtractionCtx) -> None:
        self.storage = storage
        self.ctx = ctx
