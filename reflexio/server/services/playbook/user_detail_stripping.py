from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast


@dataclass(frozen=True)
class DetectedEntity:
    start: int
    end: int
    entity_type: str
    replacement: str
    confidence: float
    source: str


@dataclass(frozen=True)
class StrippingResult:
    text: str
    detections: list[DetectedEntity] = field(default_factory=list)


class UserDetailDetector(Protocol):
    def detect(self, text: str) -> list[DetectedEntity]: ...


class UserDetailStripper(Protocol):
    def strip_user_details(
        self,
        text: str,
        shared_mapping: dict[str, int] | None = None,
    ) -> StrippingResult: ...


class PassthroughStripper:
    def strip_user_details(
        self,
        text: str,
        shared_mapping: dict[str, int] | None = None,  # noqa: ARG002
    ) -> StrippingResult:
        return StrippingResult(text=text, detections=[])


def create_configured_user_detail_stripper(
    configurator: object,
) -> UserDetailStripper | None:
    if not hasattr(type(configurator), "create_user_detail_stripper"):
        return None
    create_stripper = getattr(configurator, "create_user_detail_stripper", None)
    if not callable(create_stripper):
        return None
    typed_create_stripper = cast(
        "Callable[[], UserDetailStripper | None]", create_stripper
    )
    return typed_create_stripper()


def get_configured_playbook_aggregation_prompt_extra_instructions(
    configurator: object,
) -> str | None:
    if not hasattr(
        type(configurator), "get_playbook_aggregation_prompt_extra_instructions"
    ):
        return None
    get_instructions = getattr(
        configurator,
        "get_playbook_aggregation_prompt_extra_instructions",
        None,
    )
    if not callable(get_instructions):
        return None
    typed_get_instructions = cast("Callable[[], str | None]", get_instructions)
    return typed_get_instructions()


_PERSON_PLACEHOLDER_RE = re.compile(r"\[PERSON_\d+\]")


def count_person_placeholders(text: str | None) -> int:
    if text is None:
        return 0
    return len(_PERSON_PLACEHOLDER_RE.findall(text))


def replace_person_placeholders(text: str | None) -> str | None:
    if text is None:
        return None
    return _PERSON_PLACEHOLDER_RE.sub("a user", text)
