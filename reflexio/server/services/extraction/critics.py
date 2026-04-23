"""Critic agents and cross-entity reconciler for agentic extraction.

Each critic reviews a lane's candidates (profile or playbook) and decides per
item: accept, refine, reject, or flag a cross-entity conflict. The reconciler
then resolves the flags produced by both critics, possibly dropping or
merging items across lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, model_validator

from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
)

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager


Lane = Literal["profile", "playbook"]


class VettedProfile(ProfileAddItem):
    """Profile accepted (or refined-then-accepted) by a critic."""


class VettedPlaybook(StructuredPlaybookContent):
    """Playbook accepted (or refined-then-accepted) by a critic."""


class CrossEntityFlag(BaseModel):
    """A cross-entity conflict raised by a critic."""

    candidate_index: int
    reason: str
    lane: Lane


# ---------------- critic tool argument schemas ---------------- #


class AcceptArgs(BaseModel):
    """Accept the candidate at candidate_index unchanged."""

    candidate_index: int


class RejectArgs(BaseModel):
    """Reject the candidate at candidate_index with a one-line reason."""

    candidate_index: int
    reason: str


class RefineProfileArgs(BaseModel):
    """Edit a profile candidate, then accept it."""

    candidate_index: int
    content: str
    time_to_live: str
    notes: str | None = None


class RefinePlaybookArgs(BaseModel):
    """Edit a playbook candidate, then accept it."""

    candidate_index: int
    trigger: str
    content: str
    rationale: str
    notes: str | None = None


class CrossEntityFlagArgs(BaseModel):
    """Flag that this candidate conflicts with the other lane."""

    candidate_index: int
    reason: str


class EmptyArgs(BaseModel):
    """No arguments."""


# ---------------- critic ctx + handlers ---------------- #


@dataclass
class CriticCtx:
    """Mutable accumulator shared by critic tool handlers for one review pass."""

    candidates: list[Any]
    lane: Lane
    vetted: list[Any] = field(default_factory=list)
    flags: list[CrossEntityFlag] = field(default_factory=list)
    finished: bool = False


def _accept(args: BaseModel, ctx: CriticCtx) -> dict:
    a = cast(AcceptArgs, args)
    if not 0 <= a.candidate_index < len(ctx.candidates):
        return {"error": "candidate_index out of range"}
    cand = ctx.candidates[a.candidate_index]
    vetted_cls = VettedProfile if ctx.lane == "profile" else VettedPlaybook
    ctx.vetted.append(vetted_cls(**cand.model_dump()))
    return {"accepted": a.candidate_index}


def _reject(args: BaseModel, _ctx: CriticCtx) -> dict:
    a = cast(RejectArgs, args)
    return {"rejected": a.candidate_index, "reason": a.reason}


def _refine_profile(args: BaseModel, ctx: CriticCtx) -> dict:
    a = cast(RefineProfileArgs, args)
    if not 0 <= a.candidate_index < len(ctx.candidates):
        return {"error": "candidate_index out of range"}
    orig = ctx.candidates[a.candidate_index]
    merged = orig.model_copy(
        update={
            "content": a.content,
            "time_to_live": a.time_to_live,
            "notes": a.notes if a.notes is not None else orig.notes,
        }
    )
    ctx.vetted.append(VettedProfile(**merged.model_dump()))
    return {"refined": a.candidate_index}


def _refine_playbook(args: BaseModel, ctx: CriticCtx) -> dict:
    a = cast(RefinePlaybookArgs, args)
    if not 0 <= a.candidate_index < len(ctx.candidates):
        return {"error": "candidate_index out of range"}
    orig = ctx.candidates[a.candidate_index]
    merged = orig.model_copy(
        update={
            "trigger": a.trigger,
            "content": a.content,
            "rationale": a.rationale,
            "notes": a.notes if a.notes is not None else orig.notes,
        }
    )
    ctx.vetted.append(VettedPlaybook(**merged.model_dump()))
    return {"refined": a.candidate_index}


def _flag(args: BaseModel, ctx: CriticCtx) -> dict:
    a = cast(CrossEntityFlagArgs, args)
    ctx.flags.append(
        CrossEntityFlag(
            candidate_index=a.candidate_index,
            reason=a.reason,
            lane=ctx.lane,
        )
    )
    return {"flagged": a.candidate_index}


def _finish_critic(_args: BaseModel, ctx: CriticCtx) -> dict:
    ctx.finished = True
    return {"finished": True}


PROFILE_CRITIC_TOOLS = ToolRegistry(
    [
        Tool(name="accept", args_model=AcceptArgs, handler=_accept),
        Tool(name="reject", args_model=RejectArgs, handler=_reject),
        Tool(name="refine", args_model=RefineProfileArgs, handler=_refine_profile),
        Tool(
            name="flag_cross_entity_conflict",
            args_model=CrossEntityFlagArgs,
            handler=_flag,
        ),
        Tool(name="finish", args_model=EmptyArgs, handler=_finish_critic),
    ]
)

PLAYBOOK_CRITIC_TOOLS = ToolRegistry(
    [
        Tool(name="accept", args_model=AcceptArgs, handler=_accept),
        Tool(name="reject", args_model=RejectArgs, handler=_reject),
        Tool(name="refine", args_model=RefinePlaybookArgs, handler=_refine_playbook),
        Tool(
            name="flag_cross_entity_conflict",
            args_model=CrossEntityFlagArgs,
            handler=_flag,
        ),
        Tool(name="finish", args_model=EmptyArgs, handler=_finish_critic),
    ]
)


def summarize(items: list[Any], limit: int = 20) -> str:
    """Produce a deterministic bullet summary of candidate items.

    No LLM call — used to feed each critic a compact awareness of the *other*
    lane, and to render vetted lanes and flags for the reconciler prompt.

    Args:
        items (list): Pydantic model instances with ``content`` or
            ``trigger`` attributes and optional ``source_span``.
        limit (int): Max number of items to render before truncation marker.

    Returns:
        str: Multi-line bullet summary; `"(none)"` if items is empty.
    """
    lines: list[str] = []
    for i, it in enumerate(items[:limit]):
        preview = (
            getattr(it, "content", None) or getattr(it, "trigger", None) or str(it)
        )
        src = getattr(it, "source_span", None) or ""
        src_tail = f" / src={src[:40]}" if src else ""
        lines.append(f"- [{i}] {(preview or '')[:80]}{src_tail}")
    if len(items) > limit:
        lines.append(f"  ...({len(items) - limit} more truncated)")
    return "\n".join(lines) if lines else "(none)"


class ProfileCritic:
    """Reviews a batch of profile candidates and emits vetted items + flags.

    Args:
        client (LiteLLMClient): LLM client driving the critic tool loop.
        prompt_manager (PromptManager): Prompt store for the ``profile_critic`` prompt.
        max_steps (int): Cap on critic tool-calling turns.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 6,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def review(
        self,
        candidates: list[ProfileAddItem],
        other_lane_summary: str,
    ) -> tuple[list[VettedProfile], list[CrossEntityFlag]]:
        """Run the critic tool loop over ``candidates``.

        Args:
            candidates (list[ProfileAddItem]): Profile items emitted by the
                3 angle readers (after deduplication upstream, if any).
            other_lane_summary (str): Deterministic summary of the playbook
                lane for cross-entity awareness.

        Returns:
            tuple[list[VettedProfile], list[CrossEntityFlag]]: Vetted
            profiles and any cross-entity flags the critic raised.
        """
        ctx = CriticCtx(candidates=list(candidates), lane="profile")
        prompt = self.prompt_manager.render_prompt(
            "profile_critic",
            variables={
                "candidates_block": summarize(list(candidates)),
                "other_lane": other_lane_summary,
            },
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PROFILE_CRITIC_TOOLS,
            model_role=ModelRole.CRITIC,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return list(ctx.vetted), list(ctx.flags)


class PlaybookCritic:
    """Reviews a batch of playbook candidates and emits vetted items + flags.

    Args:
        client (LiteLLMClient): LLM client driving the critic tool loop.
        prompt_manager (PromptManager): Prompt store for the ``playbook_critic`` prompt.
        max_steps (int): Cap on critic tool-calling turns.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 6,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def review(
        self,
        candidates: list[StructuredPlaybookContent],
        other_lane_summary: str,
    ) -> tuple[list[VettedPlaybook], list[CrossEntityFlag]]:
        """Run the critic tool loop over ``candidates``.

        Args:
            candidates (list[StructuredPlaybookContent]): Playbook items
                emitted by the 3 angle readers.
            other_lane_summary (str): Deterministic summary of the profile
                lane for cross-entity awareness.

        Returns:
            tuple[list[VettedPlaybook], list[CrossEntityFlag]]: Vetted
            playbooks and any cross-entity flags the critic raised.
        """
        ctx = CriticCtx(candidates=list(candidates), lane="playbook")
        prompt = self.prompt_manager.render_prompt(
            "playbook_critic",
            variables={
                "candidates_block": summarize(list(candidates)),
                "other_lane": other_lane_summary,
            },
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PLAYBOOK_CRITIC_TOOLS,
            model_role=ModelRole.CRITIC,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return list(ctx.vetted), list(ctx.flags)


# ---------------- reconciler ---------------- #


class SupersedeArgs(BaseModel):
    """Drop one side because the other supersedes it."""

    keep_lane: Lane
    keep_index: int
    drop_lane: Lane
    drop_index: int


class MergeArgs(BaseModel):
    """Merge two items across lanes into one; keep the item on (keep_lane, keep_index)."""

    keep_lane: Lane
    keep_index: int
    drop_lane: Lane
    drop_index: int
    merged_content: str

    @model_validator(mode="after")
    def lanes_must_differ(self) -> MergeArgs:
        """Prevent same-lane merges which would cause an index-shift hazard."""
        if self.keep_lane == self.drop_lane:
            raise ValueError(
                f"keep_lane and drop_lane must differ; both are '{self.keep_lane}'. "
                "Use supersede instead."
            )
        return self


class KeepBothArgs(BaseModel):
    """Keep both items — the flag was a false alarm."""

    reason: str


@dataclass
class ReconcilerCtx:
    """Mutable accumulator passed to reconciler tool handlers."""

    profiles: list[VettedProfile]
    playbooks: list[VettedPlaybook]
    finished: bool = False


def _lane_list(ctx: ReconcilerCtx, lane: Lane) -> list[Any]:
    return ctx.profiles if lane == "profile" else ctx.playbooks


def _supersede(args: BaseModel, ctx: ReconcilerCtx) -> dict:
    a = cast(SupersedeArgs, args)
    tgt = _lane_list(ctx, a.drop_lane)
    if not 0 <= a.drop_index < len(tgt):
        return {"error": "drop_index out of range"}
    tgt.pop(a.drop_index)
    return {"superseded": [a.drop_lane, a.drop_index]}


def _merge(args: BaseModel, ctx: ReconcilerCtx) -> dict:
    a = cast(MergeArgs, args)
    keep_list = _lane_list(ctx, a.keep_lane)
    drop_list = _lane_list(ctx, a.drop_lane)
    if not (0 <= a.keep_index < len(keep_list) and 0 <= a.drop_index < len(drop_list)):
        return {"error": "index out of range"}
    kept = keep_list[a.keep_index]
    keep_list[a.keep_index] = kept.model_copy(update={"content": a.merged_content})
    # If the two indices refer to the same lane, dropping may shift keep_index;
    # but cross-lane is the usual case here.
    drop_list.pop(a.drop_index)
    return {"merged": True}


def _keep_both(args: BaseModel, _ctx: ReconcilerCtx) -> dict:
    a = cast(KeepBothArgs, args)
    return {"kept_both": True, "reason": a.reason}


def _finish_reconciler(_args: BaseModel, ctx: ReconcilerCtx) -> dict:
    ctx.finished = True
    return {"finished": True}


RECONCILER_TOOLS = ToolRegistry(
    [
        Tool(name="supersede", args_model=SupersedeArgs, handler=_supersede),
        Tool(name="merge", args_model=MergeArgs, handler=_merge),
        Tool(name="keep_both", args_model=KeepBothArgs, handler=_keep_both),
        Tool(name="finish", args_model=EmptyArgs, handler=_finish_reconciler),
    ]
)


class Reconciler:
    """Resolves cross-entity flags by superseding, merging, or keep-both.

    Args:
        client (LiteLLMClient): LLM client driving the reconciler tool loop.
        prompt_manager (PromptManager): Prompt store for the ``reconciler`` prompt.
        max_steps (int): Cap on reconciler tool-calling turns.
    """

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 6,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def resolve(
        self,
        profiles: list[VettedProfile],
        playbooks: list[VettedPlaybook],
        flags: list[CrossEntityFlag],
    ) -> tuple[list[VettedProfile], list[VettedPlaybook]]:
        """Run the reconciler tool loop to resolve cross-entity flags.

        Args:
            profiles (list[VettedProfile]): Vetted profile items from the profile critic.
            playbooks (list[VettedPlaybook]): Vetted playbook items from the playbook critic.
            flags (list[CrossEntityFlag]): Flags emitted by either critic.

        Returns:
            tuple[list[VettedProfile], list[VettedPlaybook]]: Revised lanes
            after supersede/merge resolutions.
        """
        ctx = ReconcilerCtx(profiles=list(profiles), playbooks=list(playbooks))
        if not flags:
            return ctx.profiles, ctx.playbooks
        flags_block = "\n".join(
            f"- ({f.lane}) idx={f.candidate_index}: {f.reason}" for f in flags
        )
        prompt = self.prompt_manager.render_prompt(
            "reconciler",
            variables={
                "profiles_block": summarize(list(profiles)),
                "playbooks_block": summarize(list(playbooks)),
                "flags_block": flags_block,
            },
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=RECONCILER_TOOLS,
            model_role=ModelRole.RECONCILER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return ctx.profiles, ctx.playbooks
