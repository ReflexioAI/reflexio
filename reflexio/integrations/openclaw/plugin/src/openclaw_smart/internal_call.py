"""Detect hook invocations that should not be published to reflexio.

The single concern for openclaw-smart is reflexio's own LLM provider. The
``openclaw`` LiteLLM provider (see
``reflexio.server.llm.providers.openclaw_provider``) shells out to the
``openclaw`` CLI to answer extractor prompts. That subprocess is a full
openClaw invocation, so it fires *our* hooks too — and without a guard,
the ``agent_end`` hook would publish the extractor's own system prompt
back into reflexio as a user interaction. Reflexio would then train on
its own internals.

Detection signals, OR'd:
  - Env var ``OPENCLAW_SMART_INTERNAL=1``, set by reflexio's provider
    before spawning ``openclaw``.
  - ``payload.cwd`` resolves inside the reflexio repository. Catches
    direct interactive ``openclaw`` runs from inside the reflexio
    checkout (manual debugging) that would otherwise pollute the corpus.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

INTERNAL_ENV = "OPENCLAW_SMART_INTERNAL"

# Plugin layout (in-repo / editable):
#   <reflexio_repo>/reflexio/integrations/openclaw/plugin/src/openclaw_smart/internal_call.py
# parents[5] = <reflexio_repo>, the directory we want to fence off.
#
# In install-mode layouts (PyPI wheel, ~/.openclaw/plugins/cache/...), this
# fixed depth doesn't hold. Guarding the index lookup keeps module import
# from crashing on unfamiliar layouts — the env marker is the primary
# signal there. ``OPENCLAW_SMART_REFLEXIO_DIR`` lets callers (and tests)
# override the path without touching the module.
_THIS_DIR = Path(__file__).resolve().parent
_override_reflexio_dir = os.environ.get("OPENCLAW_SMART_REFLEXIO_DIR")
if _override_reflexio_dir:
    _REFLEXIO_DIR: Path = Path(_override_reflexio_dir).resolve()
else:
    _parents = _THIS_DIR.parents
    # In an unexpected layout we fall back to _THIS_DIR itself, which makes
    # the relative_to() check below match (correctly) only when cwd is
    # literally inside this module's own directory — effectively a no-op
    # fence, ceding all detection responsibility to the env marker.
    _REFLEXIO_DIR = _parents[5] if len(_parents) > 5 else _THIS_DIR


def is_internal_invocation(payload: dict[str, Any]) -> bool:
    """True if this hook fire originated from reflexio's own LLM provider.

    Args:
        payload (dict[str, Any]): Parsed openClaw hook payload. Only ``cwd``
            (or ``workspaceDir``) is inspected.

    Returns:
        bool: True when the env marker is set or ``cwd`` points inside the
            reflexio repository. False otherwise, including when ``cwd`` is
            missing or unresolvable.
    """
    if os.environ.get(INTERNAL_ENV) == "1":
        return True
    cwd = payload.get("cwd") or payload.get("workspaceDir")
    if not isinstance(cwd, str) or not cwd:
        return False
    try:
        resolved = Path(cwd).resolve()
    except OSError:
        return False
    try:
        resolved.relative_to(_REFLEXIO_DIR)
    except ValueError:
        return False
    return True
