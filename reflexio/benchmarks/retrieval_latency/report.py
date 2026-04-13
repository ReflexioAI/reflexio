"""
Statistics and markdown rendering for retrieval latency benchmark results.

Input: a flat list of trial rows produced by :mod:`bench`, each with
``{backend, retrieval_type, layer, n, trial, elapsed_ms}``.

Output: markdown tables grouped by retrieval type, optional diff column
against a committed baseline file.

All quantile math uses :mod:`statistics` to avoid pulling in numpy.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reflexio.benchmarks.retrieval_latency.scenarios import QUERIES

# Flag a cell when p95 has grown by more than this factor since the baseline.
REGRESSION_THRESHOLD = 1.20


@dataclass(frozen=True)
class CellKey:
    """
    Immutable key identifying one (backend, retrieval_type, layer, n) cell.

    Attributes:
        backend (str): Storage backend name, e.g. ``"sqlite"``.
        retrieval_type (str): One of ``profile``, ``user_playbook``,
            ``agent_playbook``, ``unified``.
        layer (str): ``service`` or ``http``.
        n (int): Corpus size for this cell.
    """

    backend: str
    retrieval_type: str
    layer: str
    n: int


@dataclass
class CellStats:
    """
    Aggregated latency stats for one benchmark cell.

    Attributes:
        trials (int): Number of timed samples summarized.
        mean_ms (float): Arithmetic mean latency in milliseconds.
        p50_ms (float): Median latency.
        p95_ms (float): 95th percentile.
        p99_ms (float): 99th percentile.
        max_ms (float): Slowest sample observed.
    """

    trials: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass
class RunResults:
    """
    Full results of a benchmark run, aggregated and raw.

    Attributes:
        config (dict): Top-level metadata (CLI args, git sha, timestamp).
        raw (list[dict]): All trial rows, unmodified.
        stats (dict[str, CellStats]): Aggregated stats keyed by
            ``backend|retrieval_type|layer|n``.
    """

    config: dict[str, Any]
    raw: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, CellStats] = field(default_factory=dict)


def _quantile(sorted_samples: list[float], q: float) -> float:
    """
    Return the ``q``-th quantile of an already-sorted sample list.

    Uses the linear interpolation method so small N (e.g. 10 samples for
    the smoke test) still gives a stable number.

    Args:
        sorted_samples (list[float]): Sorted list of samples.
        q (float): Target quantile in ``[0, 1]``.

    Returns:
        float: Interpolated quantile value.
    """
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = q * (len(sorted_samples) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = pos - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


def compute_stats(samples: list[float]) -> CellStats:
    """
    Compute mean / p50 / p95 / p99 / max for one cell's samples.

    Args:
        samples (list[float]): Latency samples in milliseconds.

    Returns:
        CellStats: Aggregated stats. Empty input returns zeros.
    """
    if not samples:
        return CellStats(
            trials=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0
        )
    sorted_s = sorted(samples)
    return CellStats(
        trials=len(samples),
        mean_ms=statistics.fmean(sorted_s),
        p50_ms=_quantile(sorted_s, 0.50),
        p95_ms=_quantile(sorted_s, 0.95),
        p99_ms=_quantile(sorted_s, 0.99),
        max_ms=sorted_s[-1],
    )


def cell_key_str(k: CellKey) -> str:
    """
    Serialize a :class:`CellKey` into a flat string for JSON keys.

    Args:
        k (CellKey): The cell identifier.

    Returns:
        str: ``backend|retrieval_type|layer|n`` form.
    """
    return f"{k.backend}|{k.retrieval_type}|{k.layer}|{k.n}"


def aggregate(raw: list[dict[str, Any]]) -> dict[str, CellStats]:
    """
    Bucket raw trial rows by cell and compute aggregated stats per cell.

    Args:
        raw (list[dict]): Trial rows as emitted by :mod:`bench`.

    Returns:
        dict[str, CellStats]: Mapping from flat cell key to stats.
    """
    buckets: dict[str, list[float]] = {}
    for row in raw:
        k = CellKey(
            backend=row["backend"],
            retrieval_type=row["retrieval_type"],
            layer=row["layer"],
            n=row["n"],
        )
        buckets.setdefault(cell_key_str(k), []).append(row["elapsed_ms"])
    return {key: compute_stats(samples) for key, samples in buckets.items()}


def write_results_json(results: RunResults, path: Path) -> None:
    """
    Persist aggregated + raw results to disk as JSON.

    Args:
        results (RunResults): Fully populated run results.
        path (Path): Output file path. Parent directory is created.
    """
    payload = {
        "config": results.config,
        "stats": {k: asdict(v) for k, v in results.stats.items()},
        "raw": results.raw,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_baseline(path: Path) -> dict[str, CellStats]:
    """
    Load aggregated baseline stats from a previously-written results file.

    Args:
        path (Path): Baseline results.json file.

    Returns:
        dict[str, CellStats]: Stats indexed by flat cell key. Empty dict
        if the file is missing or malformed.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        k: CellStats(**v)
        for k, v in data.get("stats", {}).items()  # type: ignore[arg-type]
    }


def _fmt_cell(stats: CellStats) -> str:
    """
    Render a single cell's stats as ``p50 / p95 (mean)`` ms.

    Args:
        stats (CellStats): The cell's aggregated stats.

    Returns:
        str: Human-readable one-line summary, or ``"—"`` for empty cells.
    """
    if stats.trials == 0:
        return "—"
    return f"{stats.p50_ms:.1f} / {stats.p95_ms:.1f} ({stats.mean_ms:.1f})"


def _diff_cell(current: CellStats, baseline: CellStats | None) -> str:
    """
    Render a baseline-diff cell showing ΔP95 and a regression marker.

    Args:
        current (CellStats): This run's stats for the cell.
        baseline (CellStats | None): Prior run's stats, if any.

    Returns:
        str: ``+42%`` or ``-8%`` with a ``⚠`` marker if current p95 is
        above the regression threshold.
    """
    if baseline is None or baseline.p95_ms == 0 or current.trials == 0:
        return "—"
    ratio = current.p95_ms / baseline.p95_ms
    delta = (ratio - 1.0) * 100
    marker = " ⚠" if ratio >= REGRESSION_THRESHOLD else ""
    return f"{delta:+.0f}%{marker}"


def render_markdown(
    results: RunResults, baseline: dict[str, CellStats] | None = None
) -> str:
    """
    Render the full run results as a markdown report.

    One table per retrieval type, rows grouped by (backend, layer, N),
    with an optional ΔP95 column against the committed baseline.

    Args:
        results (RunResults): Fully populated run results.
        baseline (dict[str, CellStats] | None): Optional baseline stats.

    Returns:
        str: Markdown report suitable for writing to ``report.md``.
    """
    retrieval_types = sorted({row["retrieval_type"] for row in results.raw})
    backends = sorted({row["backend"] for row in results.raw})
    layers = sorted({row["layer"] for row in results.raw})
    sizes = sorted({row["n"] for row in results.raw})

    lines: list[str] = ["# Retrieval latency benchmark", ""]
    lines.append(f"- Trials per cell: {results.config.get('trials', '?')}")
    lines.append(f"- Warmup per cell: {results.config.get('warmup', '?')}")
    lines.append(f"- Query set: {len(QUERIES)} fixed queries (see scenarios.py)")
    lines.append("- Cell format: `p50 / p95 (mean)` in ms")
    if baseline:
        lines.append(f"- Regression threshold: p95 ≥ {REGRESSION_THRESHOLD:.0%}")
    lines.append("")

    for rt in retrieval_types:
        lines.append(f"## {rt}")
        lines.append("")
        header = ["backend", "layer", *[f"N={n}" for n in sizes]]
        if baseline:
            header.append("ΔP95")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        for backend in backends:
            for layer in layers:
                row_cells: list[str] = [backend, layer]
                latest_stats: CellStats | None = None
                for n in sizes:
                    key = cell_key_str(CellKey(backend, rt, layer, n))
                    stats = results.stats.get(
                        key,
                        CellStats(0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    )
                    row_cells.append(_fmt_cell(stats))
                    if stats.trials > 0:
                        latest_stats = stats
                if baseline and latest_stats is not None:
                    # Diff against the largest-N cell we have for this row.
                    largest_n = max(
                        (
                            n
                            for n in sizes
                            if cell_key_str(CellKey(backend, rt, layer, n))
                            in results.stats
                        ),
                        default=None,
                    )
                    base_key = (
                        cell_key_str(CellKey(backend, rt, layer, largest_n))
                        if largest_n is not None
                        else None
                    )
                    base_stats = baseline.get(base_key) if base_key else None
                    row_cells.append(_diff_cell(latest_stats, base_stats))
                elif baseline:
                    row_cells.append("—")
                lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    return "\n".join(lines)
