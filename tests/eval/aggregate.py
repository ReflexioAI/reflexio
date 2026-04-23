"""Polars-based aggregator for golden-set eval results.

Reads a parquet file containing per-case judge scores and per-backend cost
metrics and reduces it to a per-backend summary. Used by the weekly eval
report and by the comparison harness.
"""

from __future__ import annotations

import polars as pl


def aggregate_eval_results(results_path: str) -> pl.DataFrame:
    """Group per-case rows by ``backend`` and report means + p95 latency.

    Args:
        results_path (str): Path to a parquet file with columns
            ``backend``, ``signal_f1``, ``answer_correctness``,
            ``grounded_rate``, ``cost_usd``, ``latency_ms``.

    Returns:
        pl.DataFrame: One row per backend with aggregated columns
            ``mean_f1``, ``mean_correctness``, ``grounded_rate``,
            ``mean_cost``, ``p95_latency``.
    """
    return (
        pl.scan_parquet(results_path)
        .group_by("backend")
        .agg(
            [
                pl.col("signal_f1").mean().alias("mean_f1"),
                pl.col("answer_correctness").mean().alias("mean_correctness"),
                pl.col("grounded_rate").mean().alias("grounded_rate"),
                pl.col("cost_usd").mean().alias("mean_cost"),
                pl.col("latency_ms").quantile(0.95).alias("p95_latency"),
            ]
        )
        .collect()
    )
