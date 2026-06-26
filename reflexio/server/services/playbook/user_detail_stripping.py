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


_STRIPPING_PLACEHOLDER_RE = re.compile(r"\[(EMAIL|PHONE|PERSON)_\d+\]")

_PLACEHOLDER_GENERIC_TEXT: dict[str, str] = {
    "EMAIL": "an email address",
    "PHONE": "a phone number",
    "PERSON": "a user",
}


def count_stripping_placeholders(text: str | None) -> int:
    if text is None:
        return 0
    return len(_STRIPPING_PLACEHOLDER_RE.findall(text))


def replace_stripping_placeholders(text: str | None) -> str | None:
    if text is None:
        return None

    def _replace(match: re.Match[str]) -> str:
        return _PLACEHOLDER_GENERIC_TEXT.get(match.group(1), "a user")

    return _STRIPPING_PLACEHOLDER_RE.sub(_replace, text)
