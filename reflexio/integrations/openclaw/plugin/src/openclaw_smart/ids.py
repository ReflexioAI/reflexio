"""Resolve stable identifiers for openClaw sessions.

Two identifiers matter to reflexio:

- ``session_id``: openClaw's per-session id (``sessionKey``), passed via the
  TS shim. We forward it to reflexio's interaction ``session_id`` field so
  individual turns remain attributable to their conversation, but it is no
  longer the scope key for extracted preferences.
- ``project_id``: a stable, cross-session name for the project. We use this
  as reflexio's ``user_id`` for preferences, so user preferences extracted
  in one session are visible to every later session in the same repo.
  ``agent_version`` is hardcoded to ``"openclaw"`` in the adapter so shared
  skills roll up globally across all openclaw projects.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # noqa: S404 — git invocation with a fixed flag set.
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def _resolve_from_git(cwd: Path) -> str | None:
    """Return the git toplevel basename for *cwd*, or ``None`` if not a repo.

    Must never raise — callers depend on this returning ``None`` cleanly so
    hook handlers can fall back to other identifiers. Resolves ``git`` to an
    absolute path first to avoid PATH-dependent process spawning, and catches
    any ``OSError`` (e.g., ``cwd`` is unreadable or has been deleted) along
    with the previously-handled ``FileNotFoundError`` / ``TimeoutExpired``.
    """
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, cwd is a Path.
            [git_bin, "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            toplevel = result.stdout.strip()
            if toplevel:
                return Path(toplevel).name
    except (OSError, subprocess.TimeoutExpired) as exc:
        # OSError covers FileNotFoundError + PermissionError + missing-cwd.
        _LOGGER.debug("git toplevel resolution failed: %s", exc)
    return None


def resolve_project_id(cwd: str | os.PathLike[str] | None = None) -> str:
    """Return a stable project identifier for the given working directory.

    Prefers the basename of the git toplevel (so worktrees, submodules, and
    ``cd src/`` all still map to the same project). Falls back to the cwd
    basename when the directory is not inside a git repo.

    Args:
        cwd: Working directory to resolve. Defaults to ``os.getcwd()``.

    Returns:
        str: A non-empty identifier. Never raises.
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    return _resolve_from_git(base) or base.name or "unknown-project"


def resolve_project_id_with_fallback(
    cwd: str | os.PathLike[str] | None,
    agent_id: str | None,
) -> str:
    """Resolve project ID, falling back to ``agent_id`` when not in a git repo.

    openClaw can run from any directory (or none at all), so when there is no
    git context we prefer the openClaw agent id over a fabricated cwd
    basename, which would be brittle across sessions.

    Args:
        cwd: Working directory of the openClaw session (``ctx.workspaceDir``).
        agent_id: openClaw agent id from ``ctx.agentId``.

    Returns:
        str: git repo basename if cwd is in a repo; else ``agent_id``; else
        the literal ``"openclaw"``.
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    return _resolve_from_git(base) or agent_id or "openclaw"
