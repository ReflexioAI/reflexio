"""Bucket per-session correction counts into a fixed 6-bin histogram.

The spec §4.5 called for a continuous 0.0-1.0 score distribution; the actual
schema only has `is_success` and `number_of_correction_per_session`, so we
distribute by correction count instead. Six bins keep the visualization
readable while preserving the shape of session quality.
"""

from __future__ import annotations

from collections.abc import Iterable

BUCKET_LABELS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5+")
_BUCKET_COUNT = len(BUCKET_LABELS)


def bucket_corrections(
    corrections: Iterable[int],
) -> tuple[int, int, int, int, int, int]:
    """Return per-bin counts for the corrections histogram.

    Args:
        corrections (Iterable[int]): One non-negative integer per session in
            the window. Values >= 5 fall into the final "5+" bin.

    Returns:
        tuple[int, ...]: Six non-negative counts in BUCKET_LABELS order.
    """
    bins = [0] * _BUCKET_COUNT
    for c in corrections:
        idx = min(_BUCKET_COUNT - 1, max(0, c))
        bins[idx] += 1
    return tuple(bins)  # type: ignore[return-value]
