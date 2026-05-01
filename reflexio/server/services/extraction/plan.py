"""Plan-op types, ExtractionCtx, HandlerBundle, and commit-result types for the agentic-v2 pipeline.

Tool handlers append PlanOp instances to ``ctx.plan`` rather than hitting
storage directly. A deterministic commit stage at ``finish`` (or on
``max_steps``) runs invariants and applies the valid ops atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Mirrors ProfileTimeToLive — kept as Literal to avoid circular import on enum.
ProfileTTL = Literal[
    "one_day", "one_week", "one_month", "one_quarter", "one_year", "infinity"
]

PlaybookStrength = Literal["hard", "soft"]


class _BasePlanOp(BaseModel):
    """Base class for all PlanOp variants. Discriminated union via ``op``."""

    model_config = ConfigDict(frozen=True)


class CreateUserProfileOp(_BasePlanOp):
    op: Literal["create_user_profile"] = "create_user_profile"
    content: Annotated[str, Field(min_length=1)]
    ttl: ProfileTTL
    source_span: Annotated[str, Field(min_length=1)]


class DeleteUserProfileOp(_BasePlanOp):
    op: Literal["delete_user_profile"] = "delete_user_profile"
    id: Annotated[str, Field(min_length=1)]


class CreateUserPlaybookOp(_BasePlanOp):
    op: Literal["create_user_playbook"] = "create_user_playbook"
    trigger: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]
    rationale: str = ""
    strength: PlaybookStrength = "soft"
    source_span: Annotated[str, Field(min_length=1)]


class DeleteUserPlaybookOp(_BasePlanOp):
    op: Literal["delete_user_playbook"] = "delete_user_playbook"
    id: Annotated[str, Field(min_length=1)]


PlanOp = Annotated[
    CreateUserProfileOp
    | DeleteUserProfileOp
    | CreateUserPlaybookOp
    | DeleteUserPlaybookOp,
    Field(discriminator="op"),
]


@dataclass
class ExtractionCtx:
    """Per-run state for the extraction agent.

    Attributes:
        user_id: Authenticated user the run is scoped to.
        agent_version: Agent version from the active config.
        extractor_name: Optional per-extractor scope filter.
        request_id: Source publish_interaction request UUID — embedded into
            every profile/playbook this run creates so retrieval can trace
            back to the originating session. Empty string when called from
            test contexts that don't have a publish request.
        plan: Accumulated PlanOps awaiting commit.
        known_ids: Ids the agent has legitimately seen (from search/get/create
            handlers). Invariant B checks delete ids against this set.
        search_count: Number of search_* tool calls. Invariant A gates on this.
        finished: True once the agent calls the ``finish`` tool.
    """

    user_id: str
    agent_version: str
    extractor_name: str | None = None
    request_id: str = ""
    plan: list = field(
        default_factory=list
    )  # list[PlanOp] — type-erased to avoid forward-ref issues
    known_ids: set[str] = field(default_factory=set)
    search_count: int = 0
    finished: bool = False
    search_answer: str | None = None
    # Compressed rehydration excerpts captured by `read_session_text` calls
    # during the agent loop. Surfaced verbatim on the response so callers can
    # include them in downstream context without going through the search
    # agent's natural-language `finish(answer=…)` synthesis (which paraphrases
    # operands and loses fidelity).
    rehydrated_excerpts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HandlerBundle:
    """Glue so tool handlers can access shared services through one param.

    The run_tool_loop primitive passes a single ``ctx`` param to tool handlers;
    handlers in tools.py need access to BaseStorage and an ExtractionCtx. A few
    handlers (e.g., the rehydration tool) additionally call back into the LLM
    layer for in-tool denoising — those receive ``llm_client`` and
    ``prompt_manager`` here. Both ExtractionAgent and SearchAgent build one of
    these before driving the loop.

    Args:
        storage: BaseStorage handle.
        ctx: ExtractionCtx with per-run state.
        llm_client: Optional LiteLLMClient for in-tool LLM calls (e.g.
            compression). ``None`` in test paths that don't exercise tools
            requiring LLM completions.
        prompt_manager: Optional PromptManager for rendering in-tool prompts.
            Same ``None`` semantics as ``llm_client``.
    """

    storage: object
    ctx: ExtractionCtx
    llm_client: object | None = None
    prompt_manager: object | None = None


class Violation(BaseModel):
    code: Literal["A", "B", "D", "E", "F", "H", "J", "K"]
    severity: Literal["hard", "soft"]
    affected_op_indices: list[int]
    msg: str


class CommitResult(BaseModel):
    applied: list[PlanOp]
    violations: list[Violation]
    outcome: Literal["finish_tool", "max_steps", "error"]
