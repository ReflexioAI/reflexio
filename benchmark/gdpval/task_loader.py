"""Thin wrapper re-exporting OpenSpace gdpval_bench.task_loader.

We import directly from OpenSpace instead of vendoring so any upstream fixes
to the HuggingFace download / stratified sampling / prefetch logic flow
through automatically.
"""

from __future__ import annotations

from benchmark.gdpval.config import ensure_openspace_importable

ensure_openspace_importable()

from gdpval_bench.task_loader import (  # noqa: E402
    load_tasks,
    prefetch_reference_files,
    prepare_task_workspace,
)

__all__ = ["load_tasks", "prefetch_reference_files", "prepare_task_workspace"]
