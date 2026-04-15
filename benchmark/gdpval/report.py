"""Build the cross-cell comparison report.

Reads every `<host>/<phase>_*/results.jsonl` under a run dir, computes the
headline deltas (host native learning, reflexio marginal contribution, total
savings, quality deltas), and writes `comparison.csv` and `comparison.md`.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)


def _load_phase_records(run_dir: Path, host: str, phase: str) -> list[dict[str, Any]]:
    """Read every result record from `<host>/<phase>_*/results.jsonl`.

    Args:
        run_dir (Path): Root output directory for this benchmark run.
        host (str): Host agent name.
        phase (str): Phase label (`p1`/`p2`/`p3`).

    Returns:
        list[dict[str, Any]]: Loaded records in file order.
    """
    matches = list((run_dir / host).glob(f"{phase}_*/results.jsonl"))
    if not matches:
        return []
    records: list[dict[str, Any]] = []
    for path in matches:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", path)
    return records


def _cell_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-task records into one cell's headline metrics.

    Args:
        records (list[dict[str, Any]]): All task records for one host/phase.

    Returns:
        dict[str, float]: Mean totals for tokens, score, payment, and cost.
    """
    if not records:
        return {
            "tasks": 0,
            "mean_total_tokens": 0.0,
            "mean_score_10": 0.0,
            "mean_actual_payment": 0.0,
            "mean_cost_usd": 0.0,
        }
    totals = [r.get("tokens", {}).get("total_tokens", 0) for r in records]
    costs = [r.get("tokens", {}).get("cost_usd", 0.0) for r in records]
    scores = [r.get("evaluation", {}).get("score_10", 0) for r in records]
    payments = [r.get("evaluation", {}).get("actual_payment", 0.0) for r in records]
    return {
        "tasks": len(records),
        "mean_total_tokens": mean(totals) if totals else 0.0,
        "mean_score_10": mean(scores) if scores else 0.0,
        "mean_actual_payment": mean(payments) if payments else 0.0,
        "mean_cost_usd": mean(costs) if costs else 0.0,
    }


def build_comparison(
    run_dir: Path,
    hosts: list[str],
    phases: list[str],
) -> None:
    """Aggregate all per-phase results into a comparison.csv and comparison.md.

    Args:
        run_dir (Path): Root output directory for this run.
        hosts (list[str]): Hosts whose results should be compared.
        phases (list[str]): Phases that actually ran.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple[str, str], dict[str, float]] = {}
    per_task: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    for host in hosts:
        for phase in phases:
            records = _load_phase_records(run_dir, host, phase)
            cells[(host, phase)] = _cell_metrics(records)
            for rec in records:
                tid = rec.get("task_id", "")
                per_task.setdefault(tid, {})[(host, phase)] = rec

    csv_path = run_dir / "comparison.csv"
    _write_csv(csv_path, hosts, phases, per_task)
    logger.info("Wrote %s", csv_path)

    md_path = run_dir / "comparison.md"
    md_path.write_text(_render_markdown(cells, hosts, phases))
    logger.info("Wrote %s", md_path)


def _write_csv(
    path: Path,
    hosts: list[str],
    phases: list[str],
    per_task: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> None:
    """Emit one row per task with columns for every (host, phase) cell."""
    fieldnames = ["task_id"]
    for host in hosts:
        for phase in phases:
            fieldnames.extend(
                f"{host}_{phase}_{metric}"
                for metric in ("total_tokens", "score_10", "actual_payment", "cost_usd")
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tid, by_cell in sorted(per_task.items()):
            row: dict[str, Any] = {"task_id": tid}
            for host in hosts:
                for phase in phases:
                    rec = by_cell.get((host, phase), {})
                    row[f"{host}_{phase}_total_tokens"] = rec.get("tokens", {}).get("total_tokens", "")
                    row[f"{host}_{phase}_score_10"] = rec.get("evaluation", {}).get("score_10", "")
                    row[f"{host}_{phase}_actual_payment"] = rec.get("evaluation", {}).get("actual_payment", "")
                    row[f"{host}_{phase}_cost_usd"] = rec.get("tokens", {}).get("cost_usd", "")
            writer.writerow(row)


def _render_markdown(
    cells: dict[tuple[str, str], dict[str, float]],
    hosts: list[str],
    phases: list[str],
) -> str:
    """Produce the human-readable comparison summary."""
    lines: list[str] = ["# GDPVal Benchmark — Comparison", ""]
    lines.append("## Per-cell means")
    lines.append("")
    lines.append("| Host | Phase | Tasks | Mean tokens | Mean score/10 | Mean payment | Mean cost USD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for host in hosts:
        for phase in phases:
            m = cells.get((host, phase), {})
            lines.append(
                f"| {host} | {phase} | {int(m.get('tasks', 0))} | "
                f"{m.get('mean_total_tokens', 0):.0f} | "
                f"{m.get('mean_score_10', 0):.2f} | "
                f"${m.get('mean_actual_payment', 0):.2f} | "
                f"${m.get('mean_cost_usd', 0):.4f} |"
            )

    lines.append("")
    lines.append("## Headline deltas")
    lines.append("")
    for host in hosts:
        p1 = cells.get((host, "p1"), {})
        p2 = cells.get((host, "p2"), {})
        p3 = cells.get((host, "p3"), {})
        native = p1.get("mean_total_tokens", 0) - p2.get("mean_total_tokens", 0)
        marginal = p2.get("mean_total_tokens", 0) - p3.get("mean_total_tokens", 0)
        combined = p1.get("mean_total_tokens", 0) - p3.get("mean_total_tokens", 0)
        q_native = p2.get("mean_score_10", 0) - p1.get("mean_score_10", 0)
        q_marginal = p3.get("mean_score_10", 0) - p2.get("mean_score_10", 0)
        lines.append(f"### {host}")
        lines.append("")
        lines.append(f"- Native learning token savings (P1 − P2): **{native:.0f}**")
        lines.append(f"- Reflexio marginal token savings (P2 − P3): **{marginal:.0f}**")
        lines.append(f"- Combined token savings (P1 − P3): **{combined:.0f}**")
        lines.append(f"- Quality Δ from native learning (P2 − P1): **{q_native:+.2f}** score/10")
        lines.append(f"- Quality Δ from reflexio (P3 − P2): **{q_marginal:+.2f}** score/10")
        lines.append("")

    return "\n".join(lines)
