"""Bucket evaluation results into a 6-bin corrections-per-session histogram."""

from reflexio.server.services.evaluation_overview.distribution import (
    BUCKET_LABELS,
    bucket_corrections,
)


def test_six_bins_in_canonical_order() -> None:
    """Buckets: 0, 1, 2, 3, 4, 5+ — that exact order, six entries."""
    assert BUCKET_LABELS == ("0", "1", "2", "3", "4", "5+")


def test_counts_each_bucket() -> None:
    """Each correction count maps to its bin; 5+ catches everything >=5."""
    corrections = [0, 0, 1, 2, 2, 2, 5, 7, 100]
    bins = bucket_corrections(corrections)
    assert bins == (2, 1, 3, 0, 0, 3)


def test_empty_input_yields_all_zero_bins() -> None:
    """No data → six zero-height bars (preserves chart shape)."""
    assert bucket_corrections([]) == (0, 0, 0, 0, 0, 0)


def test_negative_corrections_clamp_to_zero_bin() -> None:
    """Defensively treat negative values as zero instead of indexing from the end."""
    assert bucket_corrections([-3, 0, 5]) == (2, 0, 0, 0, 0, 1)
