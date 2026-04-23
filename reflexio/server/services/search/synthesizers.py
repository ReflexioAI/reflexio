"""Synthesizers rank / drop / flag the candidate ID sets from search agents.

Each synthesizer consumes the per-intent batches produced by the three
search agents in its lane ("direct", "context", "temporal"), ranks the
surviving IDs, drops low-confidence items, and raises cross-entity flags
for the orchestrator to reconcile against the other lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop
from reflexio.server.services.extraction.critics import CrossEntityFlag

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager


Lane = Literal["profile", "playbook"]


# ---------------- tool argument schemas ---------------- #


class RankArgs(BaseModel):
    """Emit the final ordered list of candidate IDs.

    Args:
        ordered_ids (list[str]): Candidate IDs in ranked order, best first.
    """

    ordered_ids: list[str]


class DropArgs(BaseModel):
    """Exclude a candidate ID with a one-line reason.

    Args:
        id (str): Candidate ID to drop.
        reason (str): One-line justification.
    """

    id: str
    reason: str


class SynthFlagArgs(BaseModel):
    """Flag a candidate that conflicts with the other lane.

    Args:
        id (str): Candidate ID being flagged.
        reason (str): One-line description of the conflict.
    """

    id: str
    reason: str


class EmptyArgs(BaseModel):
    """No arguments."""


# ---------------- ctx + handlers ---------------- #


@dataclass
class SynthCtx:
    """Mutable accumulator passed to synthesizer tool handlers.

    Attributes:
        lane (Lane): Which lane ("profile" or "playbook") this ctx serves.
        ordered (list[str]): Final ranked IDs emitted by ``rank``.
        dropped (list[str]): IDs excluded via ``drop``.
        flags (list[CrossEntityFlag]): Cross-entity conflicts raised.
        finished (bool): True once ``finish`` has been called.
    """

    lane: Lane
    ordered: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    flags: list[CrossEntityFlag] = field(default_factory=list)
    finished: bool = False


def _rank(args: BaseModel, ctx: SynthCtx) -> dict:
    """Tool handler: record the final ranked ID list."""
    a = cast(RankArgs, args)
    ctx.ordered = list(a.ordered_ids)
    return {"ranked": len(a.ordered_ids)}


def _drop(args: BaseModel, ctx: SynthCtx) -> dict:
    """Tool handler: exclude a candidate ID."""
    a = cast(DropArgs, args)
    ctx.dropped.append(a.id)
    return {"dropped": a.id}


def _flag(args: BaseModel, ctx: SynthCtx) -> dict:
    """Tool handler: raise a cross-entity conflict flag tied to ctx.lane."""
    a = cast(SynthFlagArgs, args)
    ctx.flags.append(
        CrossEntityFlag(candidate_index=-1, reason=f"{a.id}: {a.reason}", lane=ctx.lane)
    )
    return {"flagged": a.id}


def _finish(_args: BaseModel, ctx: SynthCtx) -> dict:
    """Tool handler: terminate the synthesizer loop."""
    ctx.finished = True
    return {"finished": True}


SYNTH_TOOLS = ToolRegistry(
    [
        Tool(name="rank", args_model=RankArgs, handler=_rank),
        Tool(name="drop", args_model=DropArgs, handler=_drop),
        Tool(
            name="flag_cross_entity_conflict",
            args_model=SynthFlagArgs,
            handler=_flag,
        ),
        Tool(name="finish", args_model=EmptyArgs, handler=_finish),
    ]
)


def _candidates_to_block(candidates: list[dict[str, Any]]) -> str:
    """Render per-intent batches into a human-readable block for the prompt.

    Args:
        candidates (list[dict]): Per-intent batches, each with ``ids`` and ``why``.

    Returns:
        str: One line per batch; ``(no candidates)`` when empty.
    """
    if not candidates:
        return "(no candidates)"
    lines = [
        f"[{batch.get('why', '')}] -> {', '.join(batch.get('ids', []))}"
        for batch in candidates
    ]
    return "\n".join(lines)


class ProfileSynthesizer:
    """Synthesizer that ranks candidate profile IDs from the 3 profile search agents.

    Args:
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        max_steps (int): Cap on tool-calling turns for one synthesis run.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 4,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def rank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        other_lane_summary: str,
    ) -> tuple[list[str], list[CrossEntityFlag]]:
        """Run the synthesizer tool loop and return the ranked IDs + flags.

        Args:
            query (str): The (reformulated) user query.
            candidates (list[dict]): Per-intent batches from the 3 search agents.
            other_lane_summary (str): Rendered summary of the playbook-lane hits.

        Returns:
            tuple[list[str], list[CrossEntityFlag]]: Ordered IDs and raised flags.
        """
        ctx = SynthCtx(lane="profile")
        prompt = self.prompt_manager.render_prompt(
            "profile_synthesizer",
            variables={
                "query": query,
                "candidates_block": _candidates_to_block(candidates),
                "other_lane": other_lane_summary,
            },
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=SYNTH_TOOLS,
            model_role=ModelRole.SYNTHESIZER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return ctx.ordered, ctx.flags


class PlaybookSynthesizer:
    """Synthesizer that ranks candidate playbook IDs from the 3 playbook search agents.

    Args:
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        max_steps (int): Cap on tool-calling turns for one synthesis run.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 4,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def rank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        other_lane_summary: str,
    ) -> tuple[list[str], list[CrossEntityFlag]]:
        """Run the synthesizer tool loop and return the ranked IDs + flags.

        Args:
            query (str): The (reformulated) user query.
            candidates (list[dict]): Per-intent batches from the 3 search agents.
            other_lane_summary (str): Rendered summary of the profile-lane hits.

        Returns:
            tuple[list[str], list[CrossEntityFlag]]: Ordered IDs and raised flags.
        """
        ctx = SynthCtx(lane="playbook")
        prompt = self.prompt_manager.render_prompt(
            "playbook_synthesizer",
            variables={
                "query": query,
                "candidates_block": _candidates_to_block(candidates),
                "other_lane": other_lane_summary,
            },
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=SYNTH_TOOLS,
            model_role=ModelRole.SYNTHESIZER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return ctx.ordered, ctx.flags
