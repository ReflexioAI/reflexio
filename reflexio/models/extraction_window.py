"""Effective extraction geometry, independent of storage and server imports."""

from typing import NamedTuple

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
    extractor_config: object,
    global_window_size: int | None,
    global_stride_size: int | None,
) -> ExtractionWindow:
    """Resolve each extractor override, then global value, then schema default."""
    width = getattr(extractor_config, "window_size_override", None)
    stride = getattr(extractor_config, "stride_size_override", None)
    return ExtractionWindow(
        width
        if width is not None
        else (
            global_window_size
            if global_window_size is not None
            else DEFAULT_WINDOW_SIZE
        ),
        stride
        if stride is not None
        else (
            global_stride_size
            if global_stride_size is not None
            else DEFAULT_STRIDE_SIZE
        ),
    )
