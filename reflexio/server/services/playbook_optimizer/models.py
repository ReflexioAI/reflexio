"""In-process Pydantic models used by the playbook optimizer.

These types are *not* persisted directly — they describe the inputs and
outputs of the rollout/judge/adapter pipeline. The persisted equivalents
live in ``reflexio.models.api_schema.domain`` (``PlaybookOptimization*``)
and are populated from these models inside the GEPA adapter.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain import Interaction
from reflexio.models.structured_output import StrictStructuredOutput


class ChatMessage(BaseModel):
    """One conversation turn passed to / returned from the assistant backend."""

    role: Literal["user", "assistant", "system"]
    content: str


class ScenarioWindow(BaseModel):
    """One source-of-truth scenario the optimizer replays during rollouts.

    Built by ``ScenarioResolver`` from a user playbook's
    ``source_interaction_ids``. Only the user turns are replayed verbatim;
    the assistant side is regenerated fresh on each rollout against the
    playbook variant under test.
    """

    user_playbook_id: int | None = None
    source_interaction_ids: list[int] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)

    @property
    def user_turns(self) -> list[ChatMessage]:
        turns: list[ChatMessage] = []
        for interaction in self.interactions:
            role = _normalize_role(interaction.role)
            if role == "user" and interaction.content:
                turns.append(ChatMessage(role="user", content=interaction.content))
        return turns


class RolloutTrace(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    playbook_content: str = ""


class JudgeASI(BaseModel):
    failure_modes: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    winning_behaviors: list[str] = Field(default_factory=list)
    missing_instruction: str | list[str] = ""
    recommended_mutation: str | list[str] = ""


class JudgeOutput(StrictStructuredOutput):
    verdict: Literal["candidate", "incumbent", "tie"]
    score: float = Field(ge=0.0, le=1.0)
    likert: int = Field(ge=1, le=5)
    rationale: str = ""
    asi: JudgeASI = Field(default_factory=JudgeASI)


class CandidateEvaluationOutput(BaseModel):
    score: float
    verdict: Literal["candidate", "incumbent", "tie", "aborted"]
    likert: int
    rationale: str
    asi: JudgeASI = Field(default_factory=JudgeASI)
    incumbent_rollout: RolloutTrace = Field(default_factory=RolloutTrace)
    candidate_rollout: RolloutTrace = Field(default_factory=RolloutTrace)


class EvaluationTrajectory(BaseModel):
    scenario: ScenarioWindow
    candidate_content: str
    output: CandidateEvaluationOutput


class GEPAEventPayload(BaseModel):
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)


def _normalize_role(role: str) -> Literal["user", "assistant", "system"]:
    lowered = role.strip().lower()
    if lowered in {"assistant", "agent"}:
        return "assistant"
    if lowered == "system":
        return "system"
    return "user"
