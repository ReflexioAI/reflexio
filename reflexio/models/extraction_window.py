"""Current per-kind extraction geometry, independent of storage/server imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from reflexio.models.config_schema import Config

DEFAULT_WINDOW_SIZE = 10
DEFAULT_STRIDE_SIZE = 8


class ExtractionWindow(NamedTuple):
    """Resolved history width and automatic extraction cadence.

    Window-only/manual operations do not consume the automatic stride.
    """

    window_size: int
    stride_size: int

    def require_automatic(self) -> None:
        if not 1 <= self.stride_size <= self.window_size:
            raise ValueError(
                "stride_size must be <= window_size and both must be positive"
            )


def get_extractor_window_params(
    config: Config,
    kind: Literal["profile", "playbook"],
) -> ExtractionWindow:
    """Read the explicit pair chosen by the caller's extraction service."""
    if kind == "profile":
        return ExtractionWindow(config.profile_window_size, config.profile_stride_size)
    if kind == "playbook":
        return ExtractionWindow(
            config.playbook_window_size, config.playbook_stride_size
        )
    raise ValueError("Extraction windows require profile or playbook policy")
