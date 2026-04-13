"""
Smoke test for the retrieval latency benchmark.

Runs a tiny version of the full benchmark (N=50, 10 trials, sqlite +
service layer only) and asserts that p95 has not regressed by more than
3× against the committed baseline. The 3× tolerance is deliberately
generous: the smoke test is an alarm for catastrophic slowdowns, not a
precise measurement — a p95 that's 3× slower on an unchanged baseline
almost certainly indicates a real regression (e.g. an accidental per-row
LLM call, a missing index, a broken query plan).

Marked ``skip_in_precommit`` so it runs in ``test-all`` but not on every
commit; seeding 150 rows and running 40 timed retrievals still takes a
few seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reflexio.benchmarks.retrieval_latency.bench import main as bench_main
from reflexio.benchmarks.retrieval_latency.report import (
    CellKey,
    cell_key_str,
    load_baseline,
)
from tests.server.test_utils import skip_in_precommit

_SMOKE_SIZES = "50"
_SMOKE_TRIALS = 10
_SMOKE_WARMUP = 2
_REGRESSION_TOLERANCE = 3.0  # Allow up to 3× slowdown before failing.

_BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


@skip_in_precommit
@pytest.mark.integration
def test_retrieval_latency_smoke(tmp_path: Path) -> None:
    """
    End-to-end smoke test of the benchmark harness.

    Runs the tiny benchmark in a tmpdir, then verifies that:

    1. The harness completes successfully (exit code 0, both output files
       present).
    2. Aggregated stats exist for every cell we expect.
    3. If the committed baseline has stats for a cell, this run's p95 is
       within ``_REGRESSION_TOLERANCE × baseline.p95``.

    An empty baseline file (the initial committed state) makes the
    regression check a no-op; the test still verifies the harness runs.

    Args:
        tmp_path (Path): pytest-provided temporary output directory.
    """
    output_dir = tmp_path / "smoke"
    exit_code = bench_main(
        [
            "--sizes",
            _SMOKE_SIZES,
            "--trials",
            str(_SMOKE_TRIALS),
            "--warmup",
            str(_SMOKE_WARMUP),
            "--backend",
            "sqlite",
            "--layer",
            "service",
            "--retrieval",
            "profile,user_playbook,agent_playbook,unified",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0, "benchmark harness exited with non-zero status"
    assert (output_dir / "results.json").exists()
    assert (output_dir / "report.md").exists()

    current = load_baseline(output_dir / "results.json")
    assert current, "smoke run produced no aggregated stats"

    expected_retrievals = {"profile", "user_playbook", "agent_playbook", "unified"}
    missing: list[str] = []
    for rt in expected_retrievals:
        key = cell_key_str(CellKey("sqlite", rt, "service", int(_SMOKE_SIZES)))
        if key not in current:
            missing.append(key)
    assert not missing, f"smoke run missing stats for cells: {missing}"

    baseline = load_baseline(_BASELINE_PATH)
    if not baseline:
        # First-time run before a baseline is committed — harness OK, skip
        # regression check. Regenerate the baseline via the README recipe.
        return

    regressions: list[str] = []
    for key, stats in current.items():
        base = baseline.get(key)
        if base is None or base.p95_ms == 0 or stats.p95_ms == 0:
            continue
        ratio = stats.p95_ms / base.p95_ms
        if ratio > _REGRESSION_TOLERANCE:
            regressions.append(
                f"{key}: p95 {stats.p95_ms:.1f}ms is {ratio:.1f}× "
                f"baseline {base.p95_ms:.1f}ms"
            )
    assert not regressions, (
        "Retrieval latency regressed past tolerance:\n  " + "\n  ".join(regressions)
    )
