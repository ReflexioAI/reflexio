"""Artifact evaluation via ClawWork's LLMEvaluator.

Ports gdpval_bench's `_evaluate_task` / `_get_evaluator` (run_benchmark.py:227-373)
so scoring is apples-to-apples with OpenSpace's published gdpval_bench numbers:

  1. Walk the workspace for artifact files (same extension set as ClawWork)
  2. Call `LLMEvaluator.evaluate_artifact(task, artifacts, description, max_payment)`
  3. Apply the 0.6 payment cliff (identical to ClawWork's EconomicTracker)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from benchmark.gdpval.config import (
    CLAWWORK_ROOT,
    EVAL_ARTIFACT_EXTENSIONS,
    EVAL_MIN_THRESHOLD,
    ensure_clawwork_importable,
)

logger = logging.getLogger(__name__)

_EVALUATOR_SINGLETON: Any = None
_EVALUATOR_INITIALIZED = False


def _discover_artifacts(workspace_dir: Path, reference_filenames: list[str]) -> list[str]:
    """Scan a workspace for agent-created artifacts.

    Skips reference files (to avoid evaluating inputs) and zero-byte files.

    Args:
        workspace_dir (Path): Directory the agent wrote deliverables into.
        reference_filenames (list[str]): Filenames of reference inputs to ignore.

    Returns:
        list[str]: Sorted list of absolute artifact paths.
    """
    if not workspace_dir.exists():
        return []
    ref_names = set(reference_filenames)
    artifacts: list[str] = []
    for f in workspace_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in EVAL_ARTIFACT_EXTENSIONS:
            continue
        if f.name in ref_names:
            continue
        if f.stat().st_size == 0:
            continue
        artifacts.append(str(f))
    return sorted(artifacts)


def _get_evaluator() -> Any:
    """Lazily create and cache the ClawWork `LLMEvaluator`.

    Uses the same initialization ClawWork does: gpt-4o (or `EVALUATION_MODEL`),
    `EVALUATION_API_KEY`/`OPENAI_API_KEY`, and meta-prompts from
    `CLAWWORK_ROOT/eval/meta_prompts/`.

    Returns:
        Any: A configured LLMEvaluator, or None if initialization failed.
    """
    global _EVALUATOR_SINGLETON, _EVALUATOR_INITIALIZED
    if _EVALUATOR_INITIALIZED:
        return _EVALUATOR_SINGLETON
    _EVALUATOR_INITIALIZED = True

    meta_prompts_dir = CLAWWORK_ROOT / "eval" / "meta_prompts"
    if not meta_prompts_dir.exists():
        logger.warning("ClawWork meta-prompts not found at %s; evaluation disabled", meta_prompts_dir)
        return None

    ensure_clawwork_importable()
    try:
        from livebench.work.llm_evaluator import LLMEvaluator  # noqa: E402

        _EVALUATOR_SINGLETON = LLMEvaluator(
            meta_prompts_dir=str(meta_prompts_dir),
            max_payment=50.0,  # overridden per task
        )
    except Exception as exc:
        logger.warning("LLMEvaluator import/init failed: %s", exc)
        _EVALUATOR_SINGLETON = None
    return _EVALUATOR_SINGLETON


def evaluate_task(task: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    """Score a completed task's deliverables using the ClawWork evaluator.

    Args:
        task (dict[str, Any]): Normalized GDPVal task dict.
        workspace_dir (Path): Where the agent wrote its deliverables.

    Returns:
        dict[str, Any]: Evaluation result matching gdpval_bench's record format —
            has_evaluation, evaluation_score, score_10, payment, actual_payment,
            max_payment, artifact_count, artifact_paths, description, feedback,
            cliff_applied.
    """
    ref_filenames = [Path(rf).name for rf in (task.get("reference_files") or [])]
    artifact_paths = _discover_artifacts(workspace_dir, ref_filenames)

    if not artifact_paths:
        return {
            "has_evaluation": False,
            "evaluation_score": 0.0,
            "score_10": 0,
            "payment": 0.0,
            "actual_payment": 0.0,
            "artifact_count": 0,
            "artifact_paths": [],
            "description": "",
            "feedback": "No artifacts found in workspace.",
        }

    evaluator = _get_evaluator()
    if evaluator is None:
        return {
            "has_evaluation": False,
            "evaluation_score": 0.0,
            "score_10": 0,
            "payment": 0.0,
            "actual_payment": 0.0,
            "artifact_count": len(artifact_paths),
            "artifact_paths": [os.path.basename(p) for p in artifact_paths],
            "description": "",
            "feedback": "Evaluator not available (missing meta-prompts or API key).",
        }

    description = f"Work submission with {len(artifact_paths)} artifact(s)"
    max_payment = task.get("task_value_usd") or 50.0

    try:
        evaluation_score, feedback, payment = evaluator.evaluate_artifact(
            task=task,
            artifact_paths=artifact_paths,
            description=description,
            max_payment=max_payment,
        )
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)
        return {
            "has_evaluation": False,
            "evaluation_score": 0.0,
            "score_10": 0,
            "payment": 0.0,
            "actual_payment": 0.0,
            "artifact_count": len(artifact_paths),
            "artifact_paths": [os.path.basename(p) for p in artifact_paths],
            "description": description,
            "feedback": f"Evaluation error: {exc}",
        }

    cliff_applied = evaluation_score < EVAL_MIN_THRESHOLD
    actual_payment = 0.0 if cliff_applied else payment
    feedback_short = feedback[:500] + "..." if len(feedback) > 500 else feedback

    return {
        "has_evaluation": True,
        "evaluation_score": round(evaluation_score, 4),
        "score_10": round(evaluation_score * 10, 1),
        "payment": round(payment, 2),
        "actual_payment": round(actual_payment, 2),
        "max_payment": round(max_payment, 2),
        "artifact_count": len(artifact_paths),
        "artifact_paths": [os.path.basename(p) for p in artifact_paths],
        "description": description,
        "feedback": feedback_short,
        "cliff_applied": cliff_applied,
    }
