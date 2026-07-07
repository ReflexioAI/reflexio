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


def aggregate_by_category(results_path: str) -> pl.DataFrame:
    """Group per-case rows by ``(backend, category)`` for the search eval.

    Keeps temporal categories (``temporal_current``, ``temporal_window``,
    ``supersession``) visible instead of averaging them into one number.

    Args:
        results_path (str): Path to a parquet file with columns ``backend``,
            ``category``, ``recall_at_k``, ``mrr``, ``answer_correctness``,
            ``latency_ms``.

    Returns:
        pl.DataFrame: One row per (backend, category) with aggregated
            columns ``mean_recall``, ``mean_mrr``, ``mean_correctness``,
            ``p95_latency``, ``n_cases``.
    """
    return (
        pl.scan_parquet(results_path)
        .group_by(["backend", "category"])
        .agg(
            [
                pl.col("recall_at_k").mean().alias("mean_recall"),
                pl.col("mrr").mean().alias("mean_mrr"),
                pl.col("answer_correctness").mean().alias("mean_correctness"),
                pl.col("latency_ms").quantile(0.95).alias("p95_latency"),
                pl.len().alias("n_cases"),
            ]
        )
        .sort(["backend", "category"])
        .collect()
    )
