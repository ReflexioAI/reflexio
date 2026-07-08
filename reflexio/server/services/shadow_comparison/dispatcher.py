"""Publish-time dispatch for per-turn shadow comparisons."""

from __future__ import annotations

import logging
import random

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.shadow_comparison.judge import ShadowComparisonJudge

logger = logging.getLogger(__name__)

_ASSISTANT_ROLES = {"agent", "assistant"}


def dispatch_shadow_comparison_judge(
    *,
    storage,  # noqa: ANN001 - BaseStorage; concrete type would create import cycles
    interactions: list[Interaction],
    session_id: str,
    agent_version: str,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> None:
    """Judge each shadow-bearing assistant turn in a single publish request."""
    if not interactions:
        return

    supports_verdicts = getattr(
        storage, "supports_shadow_comparison_verdicts", lambda: False
    )
    if not supports_verdicts():
        logger.info(
            "Skipping shadow comparison for session=%s because %s does not support verdict storage",
            session_id,
            type(storage).__name__,
        )
        return

    sorted_interactions = sorted(interactions, key=lambda i: i.created_at)
    shadow_interactions = [
        interaction
        for interaction in sorted_interactions
        if interaction.shadow_content
        and interaction.role.strip().lower() in _ASSISTANT_ROLES
    ]
    if not shadow_interactions:
        return

    config = request_context.configurator.get_config()  # type: ignore[reportOptionalMemberAccess]
    judge = ShadowComparisonJudge(
        llm_client=llm_client,
        prompt_manager=request_context.prompt_manager,  # type: ignore[reportOptionalMemberAccess]
        prompt_version=config.shadow_comparison_judge_prompt_version,
    )
    rng = random.Random()  # noqa: S311 - position randomization, not crypto
    saved_count = 0

    for interaction in shadow_interactions:
        conversation_context = _format_request_transcript_before(
            sorted_interactions=sorted_interactions,
            target=interaction,
        )
        try:
            verdict = judge.judge_turn(
                interaction=interaction,
                session_id=session_id,
                agent_version=agent_version,
                rng=rng,
                conversation_context=conversation_context,
            )
        except Exception as exc:  # noqa: BLE001 - one judge failure must not abort the batch
            logger.warning(
                "shadow_comparison dispatch failed for interaction %s: %s",
                interaction.interaction_id,
                exc,
            )
            continue
        if verdict is None:
            continue
        try:
            storage.save_shadow_comparison_verdict(verdict)
            saved_count += 1
        except NotImplementedError:
            logger.info(
                "Stopping shadow comparison for session=%s because %s does not support verdict storage",
                session_id,
                type(storage).__name__,
            )
            return
        except Exception as exc:  # noqa: BLE001 - single-row save failure must not abort batch
            logger.warning(
                "shadow_comparison verdict save failed for interaction %s: %s",
                interaction.interaction_id,
                exc,
            )

    if saved_count:
        logger.info(
            "Saved %d shadow_comparison verdict(s) for session=%s",
            saved_count,
            session_id,
        )


def _format_request_transcript_before(
    *, sorted_interactions: list[Interaction], target: Interaction
) -> str:
    """Format prior turns in the same publish request as judge context.

    ``sorted_interactions`` must already be ordered by ``created_at`` — the
    caller sorts once and passes the shared list so this runs O(n) per turn.
    """
    prior_turns: list[Interaction] = []
    for interaction in sorted_interactions:
        if interaction is target or (
            interaction.interaction_id
            and interaction.interaction_id == target.interaction_id
        ):
            break
        if interaction.content:
            prior_turns.append(interaction)
    if not prior_turns:
        return ""
    return "\n\n".join(
        f"{_display_role(interaction.role)}:\n{interaction.content}"
        for interaction in prior_turns
    )


def _display_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized in {"user", "human"}:
        return "User"
    if normalized in _ASSISTANT_ROLES:
        return "Assistant"
    return role.strip() or "Unknown"
