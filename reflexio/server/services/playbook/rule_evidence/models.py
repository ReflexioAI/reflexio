"""Portable source evidence and proposal schemas; these grant no write authority."""

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from reflexio.models.structured_output import StrictStructuredOutput

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]


class EvidenceModel(StrictStructuredOutput):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Reference(EvidenceModel):
    session_id: Text
    turn: int = Field(ge=0)


class Turn(EvidenceModel):
    turn: int = Field(ge=0)
    role: Text
    content: Text


class Session(EvidenceModel):
    session_id: Text
    user_id: Text
    exposed_version: Text
    source_limitations: str
    messages: list[Turn] = Field(min_length=1)


class Incumbent(EvidenceModel):
    id: int = Field(gt=0)
    user_id: Text
    version: Text
    trigger: str | None
    content: Text


class ToolContract(EvidenceModel):
    name: Text
    description: Text


class RuleEvidenceInput(EvidenceModel):
    incumbent: Incumbent
    sessions: list[Session] = Field(min_length=1)
    agent_context: str
    success_definition: str
    available_tools: list[ToolContract]
    tool_context_status: Literal["captured", "partial", "missing"]

    @model_validator(mode="after")
    def check_sources(self) -> Self:
        ids = [session.session_id for session in self.sessions]
        if len(set(ids)) != len(ids):
            raise ValueError("sessions must be unique")
        for session in self.sessions:
            if session.user_id != self.incumbent.user_id:
                raise ValueError("evidence must belong to the incumbent owner")
            if session.exposed_version != self.incumbent.version:
                raise ValueError("evidence must expose the exact incumbent version")
            turns = [message.turn for message in session.messages]
            if len(set(turns)) != len(turns):
                raise ValueError("source turn coordinates must be unique")
        names = [tool.name for tool in self.available_tools]
        if len(set(names)) != len(names):
            raise ValueError("tool contracts must be unique")
        if self.tool_context_status == "missing" and self.available_tools:
            raise ValueError("missing tool context cannot carry tool contracts")
        return self


class Observation(EvidenceModel):
    dimension: Literal[
        "task_completion",
        "constraint_compliance",
        "evidence_grounding",
        "execution_effectiveness",
    ]
    kind: Literal["strength", "problem", "uncertainty"]
    observation: Text
    consequence: Text
    evidence_refs: list[Reference] = Field(min_length=1, max_length=6)


class RuleAssessment(EvidenceModel):
    disposition: Literal[
        "rule_gap", "adherence_failure", "no_issue", "not_applicable", "unknown"
    ]
    incumbent_quote: str = Field(max_length=1200)
    expected_if_followed: Text
    reason: Text
    evidence_refs: list[Reference] = Field(min_length=1, max_length=6)


class Diagnostic(EvidenceModel):
    session_id: Text
    user_id: Text
    observations: list[Observation] = Field(max_length=8)
    rule_assessment: RuleAssessment
    missing_information: list[Text] = Field(max_length=4)


class Proposal(EvidenceModel):
    decision: Literal["hypothesis", "abstain"]
    gap_index: int | None = Field(ge=0)
    hypothesis: str
    replacement: str
    uncertainties: list[Text] = Field(max_length=3)
    abstention_reason: str

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        if self.decision == "abstain":
            if self.gap_index is not None or self.replacement:
                raise ValueError("abstention cannot select a gap or carry an edit")
            if not self.abstention_reason.strip():
                raise ValueError("abstention requires a reason")
        elif (
            self.gap_index is None
            or not self.hypothesis.strip()
            or not self.replacement.strip()
        ):
            raise ValueError("repair requires a gap, hypothesis and replacement")
        return self


class RuleEvidenceResult(EvidenceModel):
    decision: Literal["proposal", "abstain"]
    reason: Text
    diagnostics: list[Diagnostic]
    proposal: Proposal | None
    revised_content: str | None
    evidence_refs: list[Reference]
