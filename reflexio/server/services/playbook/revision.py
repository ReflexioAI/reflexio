"""Explicit final fields for one playbook version, without write authority."""

from typing import Annotated, Protocol

from pydantic import ConfigDict, Field

from reflexio.models.structured_output import StrictStructuredOutput


class OriginalPlaybookFields(Protocol):
    @property
    def original_content(self) -> str: ...

    @property
    def original_trigger(self) -> str | None: ...


class FinalPlaybookRevision(StrictStructuredOutput):
    """Both fields are required; explicit null Trigger never means unspecified."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    revised_content: Annotated[str, Field(min_length=1, pattern=r"\S")]
    revised_trigger: str | None

    def changes(self, original: OriginalPlaybookFields) -> bool:
        return (self.revised_content, self.revised_trigger) != (
            original.original_content,
            original.original_trigger,
        )
