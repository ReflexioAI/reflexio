"""Selected procedural extraction contracts; no execution or publication authority."""

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from reflexio.models.structured_output import StrictStructuredOutput

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
RESOURCE_PATTERN = r"^(?:references|scripts|assets)/[a-z0-9][a-z0-9_.-]*$"
PAGE_PATTERN = r"^(SKILL\.md|(?:references|scripts|assets)/[a-z0-9][a-z0-9_.-]*)$"
Check = Literal["pass", "fail", "unknown"]


class ContractError(ValueError):
    """Original source, complete output, or exact binding does not match."""


class Model(StrictStructuredOutput):
    model_config = ConfigDict(extra="forbid")


class Source(Model):
    interaction_id: int = Field(gt=0, strict=True)
    request_id: Text
    owner: Text
    role: Text
    request_source: str
    text: Text
    created_at: int | None = Field(default=None, strict=True)
    request_created_at: int | None = Field(default=None, strict=True)
    # Code-owned display provenance. Omitted from serialized input so ordinary
    # content-only studies retain their exact catalog; catalog attaches labels
    # only to the relevant physical pieces, outside their quotation bytes.
    piece_fields: list[Text] | None = Field(default=None, exclude=True)


class Quotation(Model):
    interaction_id: int = Field(gt=0, strict=True)
    text: Text = Field(max_length=600)


class Input(Model):
    owner: Text
    sources: list[Source] = Field(min_length=1)
    existing: list[dict[str, Any]]
    agent_context: str
    definition: Text
    tool_context: str

    @model_validator(mode="after")
    def source_identity(self) -> "Input":
        ids = [source.interaction_id for source in self.sources]
        if len(ids) != len(set(ids)) or any(
            source.owner != self.owner for source in self.sources
        ):
            raise ContractError("Procedural sources require unique IDs and one owner")
        return self


class Page(Model):
    path: str = Field(min_length=1, pattern=PAGE_PATTERN)
    content: Text


class Document(Model):
    name: str = Field(min_length=1, pattern=NAME_PATTERN, max_length=64)
    pages: list[Page] = Field(min_length=1)


class Finish(Model):
    status: Literal["complete", "incomplete"]
    reason: Text


class Observation(Model):
    topic: Text
    evidence: Text
    observation: Text


class Create(Model):
    group: Text
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=NAME_PATTERN,
        description="Stable lowercase hyphenated slug, not a human-readable title.",
    )
    trigger: Text
    entry: Text
    evidence: Text


class Revise(Create):
    target: Text


class Retire(Model):
    group: Text
    target: Text
    evidence: Text


class Citation(Model):
    group: Text
    evidence: Text


class Resource(Model):
    group: Text
    document_name: str = Field(min_length=1, max_length=64, pattern=NAME_PATTERN)
    path: str = Field(
        min_length=1,
        pattern=RESOURCE_PATTERN,
        description="Optional references/, scripts/ or assets/ file; SKILL.md comes from entry.",
    )
    content: Text
    evidence: Text


class RemoveResource(Model):
    group: Text
    document_name: str = Field(min_length=1, max_length=64, pattern=NAME_PATTERN)
    path: str = Field(
        min_length=1,
        pattern=RESOURCE_PATTERN,
        description="Optional references/, scripts/ or assets/ file; SKILL.md comes from entry.",
    )
    evidence: Text


class Change(Model):
    group: Text
    operation: Literal["upsert", "retire"]
    consume: list[Text]
    document: Document | None
    trigger: str | None
    support: list[Text] = Field(min_length=1)


class Maintenance(Model):
    changes: list[Change]


class Verdict(Model):
    group: Text
    decision: Literal["accept", "reject", "unknown"]
    instruction_support: Check
    scope_conditions: Check
    useful_nonduplicate: Check
    preservation_and_correction: Check
    resource_consistency: Check
    reason: Text


class Review(Model):
    groups: list[Verdict]


class ResultVerdict(Model):
    unit: Text
    decision: Literal["accept", "reject", "unknown"]
    result_support: Check
    scope_conditions: Check
    useful_nonduplicate: Check
    preservation_and_correction: Check
    resource_consistency: Check
    reason: Text
