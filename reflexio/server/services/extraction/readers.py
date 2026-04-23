"""Angle-specialist readers that emit profile / playbook candidates.

Each reader drives a tool-calling loop for one extraction angle ("facts",
"context", "temporal" for profiles; "behavior", "trigger", "rationale" for
playbooks). The LLM emits candidates by calling ``emit_profile`` /
``emit_playbook`` and ends the turn by calling ``finish``. Emitted items are
collected into the reader's ``ReaderCtx`` and returned to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

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


ProfileAngle = Literal["facts", "context", "temporal"]
PlaybookAngle = Literal["behavior", "trigger", "rationale"]


class EmptyArgs(BaseModel):
    """No arguments."""


class _EmitProfileArgs(ProfileAddItem):
    """Emit one candidate profile item for the current reader angle."""


class _EmitPlaybookArgs(StructuredPlaybookContent):
    """Emit one candidate playbook item for the current reader angle."""


@dataclass
class ReaderCtx:
    """Mutable accumulator passed to tool handlers during one reader run."""

    candidates: list = field(default_factory=list)
    finished: bool = False


def _append_profile(args: BaseModel, ctx: ReaderCtx) -> dict:
    # Registry validated into _EmitProfileArgs before dispatch.
    ctx.candidates.append(args)
    return {"accepted": True}


def _append_playbook(args: BaseModel, ctx: ReaderCtx) -> dict:
    # Registry validated into _EmitPlaybookArgs before dispatch.
    ctx.candidates.append(args)
    return {"accepted": True}


def _mark_finished(_args: BaseModel, ctx: ReaderCtx) -> dict:
    ctx.finished = True
    return {"finished": True}


PROFILE_READER_TOOLS = ToolRegistry(
    [
        Tool(
            name="emit_profile",
            args_model=_EmitProfileArgs,
            handler=_append_profile,
        ),
        Tool(name="finish", args_model=EmptyArgs, handler=_mark_finished),
    ]
)

PLAYBOOK_READER_TOOLS = ToolRegistry(
    [
        Tool(
            name="emit_playbook",
            args_model=_EmitPlaybookArgs,
            handler=_append_playbook,
        ),
        Tool(name="finish", args_model=EmptyArgs, handler=_mark_finished),
    ]
)


@dataclass
class ReaderInputs:
    """Inputs a reader needs for one run.

    Attributes:
        sessions (str): Rendered session transcripts to feed into the reader prompt.
    """

    sessions: str


class ProfileReader:
    """Angle-specialist reader that emits candidate profile items.

    Args:
        angle (ProfileAngle): Which angle prompt to render ("facts", "context", "temporal").
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        max_steps (int): Cap on tool-calling turns for one reader run.
    """

    def __init__(
        self,
        angle: ProfileAngle,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 8,
    ) -> None:
        self.angle = angle
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def read(self, inputs: ReaderInputs) -> list[ProfileAddItem]:
        """Run the tool loop for one reader angle and return its candidates.

        Args:
            inputs (ReaderInputs): Session transcript input.

        Returns:
            list[ProfileAddItem]: Candidates emitted by the reader, in emission order.
        """
        ctx = ReaderCtx()
        prompt = self.prompt_manager.render_prompt(
            f"profile_reader_{self.angle}",
            variables={"sessions": inputs.sessions},
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PROFILE_READER_TOOLS,
            model_role=ModelRole.ANGLE_READER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return list(ctx.candidates)


class PlaybookReader:
    """Angle-specialist reader that emits candidate playbook items.

    Args:
        angle (PlaybookAngle): Which angle prompt to render ("behavior", "trigger", "rationale").
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        max_steps (int): Cap on tool-calling turns for one reader run.
    """

    def __init__(
        self,
        angle: PlaybookAngle,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        max_steps: int = 8,
    ) -> None:
        self.angle = angle
        self.client = client
        self.prompt_manager = prompt_manager
        self.max_steps = max_steps

    def read(self, inputs: ReaderInputs) -> list[StructuredPlaybookContent]:
        """Run the tool loop for one reader angle and return its candidates.

        Args:
            inputs (ReaderInputs): Session transcript input.

        Returns:
            list[StructuredPlaybookContent]: Candidates emitted by the reader,
            in emission order.
        """
        ctx = ReaderCtx()
        prompt = self.prompt_manager.render_prompt(
            f"playbook_reader_{self.angle}",
            variables={"sessions": inputs.sessions},
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PLAYBOOK_READER_TOOLS,
            model_role=ModelRole.ANGLE_READER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="finish",
        )
        return list(ctx.candidates)
