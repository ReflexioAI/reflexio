"""F1 per-turn shadow comparison judge service.

Stateless beyond construction. Routes every LLM call through
``LiteLLMClient`` and every prompt through ``PromptManager`` per project
guardrails — never imports OpenAIClient or ClaudeClient directly.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.services.shadow_comparison.outcome import assign_positions

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import Interaction
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager


logger = logging.getLogger(__name__)

_PROMPT_ID = "shadow_comparison"


class ShadowComparisonJudge:
    """Per-turn LLM-as-judge for Reflexio vs Shadow responses (F1).

    The judge sees Request 1 + Request 2 (position-randomized) and decides
    which is better, or whether they tie. The position assignment is
    recorded on the verdict so the dashboard can derive Reflexio-relative
    win / tie / loss via
    :func:`reflexio.server.services.shadow_comparison.outcome.derive_reflexio_outcome`.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        prompt_manager: PromptManager,
        prompt_version: str,
    ):
        """
        Initialize the judge.

        Args:
            llm_client (LiteLLMClient): Unified LLM client. Used to render
                the verdict via ``generate_chat_response`` with a Pydantic
                ``response_format``.
            prompt_manager (PromptManager): Prompt manager for the
                ``shadow_comparison`` prompt. Prompt-version selection is
                governed by the manager's ``version_override`` dict;
                ``prompt_version`` below is recorded on the verdict for
                later filtering but does not affect rendering.
            prompt_version (str): Semver of the active shadow_comparison
                prompt. Stamped onto every verdict so the dashboard can
                exclude verdicts from a stale rubric.
        """
        self._llm_client = llm_client
        self._prompt_manager = prompt_manager
        self._prompt_version = prompt_version

    # ===============================
    # public methods
    # ===============================

    def judge_turn(
        self,
        *,
        interaction: Interaction,
        session_id: str,
        agent_version: str,
        rng: random.Random,
        user_message: str = "",
    ) -> ShadowComparisonVerdict | None:
        """
        Grade one interaction; return ``None`` when the LLM fails or the
        interaction has no shadow response to compare against.

        Returning ``None`` (rather than raising) lets the regen worker
        skip this turn and continue with the rest of the session — one
        rate-limit blip should not abort an entire batch.

        Args:
            interaction (Interaction): The agent-side interaction. Its
                ``content`` is the Reflexio response and ``shadow_content``
                is the no-Reflexio response. When ``shadow_content`` is
                empty the judge short-circuits with ``None``.
            session_id (str): Denormalized onto the verdict for cheap
                session-scoped queries.
            agent_version (str): Pinned onto the verdict for trend slicing.
            rng (random.Random): Position-randomization source.
                Production callers pass a fresh ``random.Random()`` per
                judge call; tests inject a seeded ``Random`` for
                reproducibility.
            user_message (str): The user turn this interaction responds
                to. Optional — empty string is rendered into the prompt
                if the caller does not provide a value.

        Returns:
            ShadowComparisonVerdict | None: The constructed verdict, or
            ``None`` when shadow content is missing or the LLM call fails.
        """
        if not interaction.shadow_content:
            return None

        request_1, request_2, reflexio_is_request_1 = assign_positions(
            reflexio_response=interaction.content,
            shadow_response=interaction.shadow_content,
            rng=rng,
        )

        prompt = self._prompt_manager.render_prompt(
            prompt_id=_PROMPT_ID,
            variables={
                "user_message": user_message,
                "request_1_response": request_1,
                "request_2_response": request_2,
            },
        )

        output = self._call_judge_llm(prompt, interaction.interaction_id)
        if output is None:
            return None

        return ShadowComparisonVerdict(
            verdict_id=0,  # storage assigns the autoincrement id
            interaction_id=str(interaction.interaction_id),
            session_id=session_id,
            agent_version=agent_version,
            reflexio_is_request_1=reflexio_is_request_1,
            output=output,
            judge_prompt_version=self._prompt_version,
            created_at=datetime.now(UTC),
        )

    # ===============================
    # private helpers
    # ===============================

    def _call_judge_llm(
        self,
        prompt: str,
        interaction_id: int,
    ) -> ShadowComparisonOutput | None:
        """
        Invoke the LLM with structured-output and unwrap the response.

        Args:
            prompt (str): The rendered judge prompt.
            interaction_id (int): Interaction id for failure-logging context.

        Returns:
            ShadowComparisonOutput | None: The parsed structured output,
            or ``None`` if the call failed or returned an unexpected type.
        """
        try:
            response = self._llm_client.generate_chat_response(
                messages=[{"role": "user", "content": prompt}],
                response_format=ShadowComparisonOutput,
            )
        except Exception as exc:  # noqa: BLE001 — judge failure must not abort batch
            logger.warning(
                "shadow_comparison judge failed for interaction %s: %s",
                interaction_id,
                exc,
            )
            return None

        if not isinstance(response, ShadowComparisonOutput):
            logger.warning(
                "Unexpected response type from shadow_comparison judge for "
                "interaction %s: %s",
                interaction_id,
                type(response).__name__,
            )
            return None

        return response
