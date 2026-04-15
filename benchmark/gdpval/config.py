"""Defaults and external paths for the GDPVal benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # benchmark/

# External repos — point these at your local clones via env vars. The
# defaults assume the layout documented in the README (`~/repos/<name>`),
# so contributors don't need to bake in any personal absolute paths. The
# `ensure_*_importable` helpers below raise a clear error if the directory
# is missing.
_DEFAULT_REPOS_DIR = Path.home() / "repos"


def _resolve_repo_root(env_var: str, default_dirname: str) -> Path:
    """Resolve an external repo root from an env var, falling back to ``~/repos/<name>``.

    Args:
        env_var (str): Name of the environment variable that points at the repo.
        default_dirname (str): Directory name under ``~/repos`` to use when unset.

    Returns:
        Path: Resolved (but not necessarily existing) absolute path.
    """
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser().resolve()
    return (_DEFAULT_REPOS_DIR / default_dirname).resolve()


OPENSPACE_ROOT = _resolve_repo_root("OPENSPACE_ROOT", "OpenSpace")
HERMES_ROOT = _resolve_repo_root("HERMES_ROOT", "hermes-agent")
CLAWWORK_ROOT = _resolve_repo_root("CLAWWORK_ROOT", "ClawWork")

# Default run configuration.
# Use the bare `minimax/` litellm prefix so OpenSpace (which routes through
# litellm) picks up `MINIMAX_API_KEY` directly. Prepend `openrouter/` to
# force OpenRouter routing when an OpenRouter key is available. Hermes's
# adapter strips either prefix before handing the model ID to AIAgent.
DEFAULT_MODEL = "minimax/MiniMax-M2.7"
DEFAULT_HOSTS = ["openspace", "hermes"]
DEFAULT_PHASES = ["p1", "p2", "p3"]
DEFAULT_TASK_LIST = OPENSPACE_ROOT / "gdpval_bench" / "tasks_50.json"

# Reflexio backend. Honors `REFLEXIO_URL` first, then `BACKEND_PORT` (worktree
# deployments export this to avoid colliding with the main checkout's 8081),
# then falls back to the standard 8081.
DEFAULT_REFLEXIO_URL = os.environ.get(
    "REFLEXIO_URL",
    f"http://localhost:{os.environ.get('BACKEND_PORT', '8081')}",
)
DEFAULT_TOP_K = 10
DEFAULT_SEARCH_THRESHOLD = 0.3

# Evaluation (ClawWork LLMEvaluator).
EVAL_MIN_THRESHOLD = 0.6  # 0.6 payment cliff, same as gdpval_bench
EVAL_ARTIFACT_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".txt", ".csv", ".json", ".md",
    ".py", ".js", ".html", ".css",
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
}

# Output layout.
OUTPUT_DIR = _THIS_DIR / "output"
LOG_DIR = _THIS_DIR / "logs"

# OpenSpace baseline — captured once on first P1 run, then restored as the
# "cold" starting state at the beginning of every subsequent P1. OpenSpace's
# skill store lives in a hardcoded `PROJECT_ROOT/.openspace/`, so we cannot
# isolate per-run state purely through the adapter's `host_state_dir`. This
# baseline dir lets us at least make the cold state reproducible across runs.
OPENSPACE_BASELINE_DIR = _THIS_DIR / ".openspace_baseline"


def ensure_openspace_importable() -> None:
    """Add OpenSpace root to sys.path so `gdpval_bench.*` modules are importable.

    We reuse OpenSpace's `gdpval_bench.task_loader` and `gdpval_bench.token_tracker`
    directly instead of vendoring them, so this benchmark stays in sync with any
    upstream fixes to task loading or token accounting.
    """
    if not OPENSPACE_ROOT.exists():
        raise FileNotFoundError(
            f"OPENSPACE_ROOT does not exist: {OPENSPACE_ROOT}. "
            "Clone OpenSpace or set OPENSPACE_ROOT env var."
        )
    os_path = str(OPENSPACE_ROOT)
    if os_path not in sys.path:
        sys.path.insert(0, os_path)


def ensure_hermes_importable() -> None:
    """Add hermes-agent root to sys.path so `run_agent.AIAgent` is importable."""
    if not HERMES_ROOT.exists():
        raise FileNotFoundError(
            f"HERMES_ROOT does not exist: {HERMES_ROOT}. "
            "Clone hermes-agent or set HERMES_ROOT env var."
        )
    h_path = str(HERMES_ROOT)
    if h_path not in sys.path:
        sys.path.insert(0, h_path)


def ensure_clawwork_importable() -> None:
    """Add ClawWork to sys.path so `livebench.work.llm_evaluator.LLMEvaluator` is importable."""
    if not CLAWWORK_ROOT.exists():
        raise FileNotFoundError(
            f"CLAWWORK_ROOT does not exist: {CLAWWORK_ROOT}. "
            "Clone ClawWork or set CLAWWORK_ROOT env var."
        )
    cw_path = str(CLAWWORK_ROOT)
    if cw_path not in sys.path:
        sys.path.insert(0, cw_path)
