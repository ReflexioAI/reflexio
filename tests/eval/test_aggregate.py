"""Unit tests for the eval polars aggregator."""

from __future__ import annotations

import polars as pl

from tests.eval.aggregate import aggregate_eval_results


def _write_fixture(tmp_path) -> str:
    df = pl.DataFrame(
        {
            "backend": ["classic", "classic", "agentic", "agentic"],
            "signal_f1": [0.5, 0.6, 0.8, 0.7],
            "answer_correctness": [0.5, 0.5, 0.7, 0.8],
            "grounded_rate": [0.9, 0.95, 0.98, 1.0],
            "cost_usd": [0.001, 0.001, 0.01, 0.01],
            "latency_ms": [1000, 1100, 2500, 2700],
        }
    )
    path = tmp_path / "r.parquet"
    df.write_parquet(path)
    return str(path)


def test_aggregate_returns_per_backend_stats(tmp_path):
    """Output has one row per backend and the expected aggregated columns."""
    out = aggregate_eval_results(_write_fixture(tmp_path))

    assert set(out["backend"].to_list()) == {"classic", "agentic"}
    assert "mean_f1" in out.columns
    assert "mean_correctness" in out.columns
    assert "grounded_rate" in out.columns
    assert "mean_cost" in out.columns
    assert "p95_latency" in out.columns


def test_aggregate_means_are_correct(tmp_path):
    """Agentic mean_f1 = (0.8 + 0.7) / 2 = 0.75."""
    out = aggregate_eval_results(_write_fixture(tmp_path))

    agentic = out.filter(pl.col("backend") == "agentic").row(0, named=True)
    assert agentic["mean_f1"] == 0.75
    assert agentic["mean_correctness"] == 0.75
    assert agentic["mean_cost"] == 0.01


def test_aggregate_p95_latency_is_tail(tmp_path):
    """p95 latency should be near the tail of each backend's latency distribution."""
    out = aggregate_eval_results(_write_fixture(tmp_path))

    classic = out.filter(pl.col("backend") == "classic").row(0, named=True)
    agentic = out.filter(pl.col("backend") == "agentic").row(0, named=True)
    assert classic["p95_latency"] >= 1000
    assert agentic["p95_latency"] >= 2500
