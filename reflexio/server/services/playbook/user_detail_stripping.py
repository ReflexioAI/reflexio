from __future__ import annotations

from collections.abc import Callable
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
    prompt_extra_instructions: str | None

    def strip_user_details(
        self,
        text: str,
        shared_mapping: dict[str, int] | None = None,
    ) -> StrippingResult: ...

    def sanitize_aggregation_output_text(
        self,
        text: str | None,
    ) -> tuple[str | None, int]: ...


class PassthroughStripper:
    prompt_extra_instructions: str | None = None

    def strip_user_details(
        self,
        text: str,
        shared_mapping: dict[str, int] | None = None,  # noqa: ARG002
    ) -> StrippingResult:
        return StrippingResult(text=text, detections=[])

    def sanitize_aggregation_output_text(
        self,
        text: str | None,
    ) -> tuple[str | None, int]:
        return text, 0


UserDetailStripperFactory = Callable[[object], UserDetailStripper | None]


def _default_user_detail_stripper_factory(
    _configurator: object,
) -> UserDetailStripper | None:
    return None


_user_detail_stripper_factory: UserDetailStripperFactory = (
    _default_user_detail_stripper_factory
)


def set_user_detail_stripper_factory(factory: UserDetailStripperFactory) -> None:
    """Register the deployment-specific aggregation stripper factory."""
    global _user_detail_stripper_factory  # noqa: PLW0603
    _user_detail_stripper_factory = factory


def create_aggregation_user_detail_stripper(
    configurator: object,
) -> UserDetailStripper | None:
    """Create the deployment-specific stripper for aggregation, if any."""
    return _user_detail_stripper_factory(configurator)
