from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


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
    return create_stripper()


_PERSON_PLACEHOLDER_RE = re.compile(r"\[PERSON_\d+\]")


def count_person_placeholders(text: str | None) -> int:
    if text is None:
        return 0
    return len(_PERSON_PLACEHOLDER_RE.findall(text))


def replace_person_placeholders(text: str | None) -> str | None:
    if text is None:
        return None
    return _PERSON_PLACEHOLDER_RE.sub("a user", text)
