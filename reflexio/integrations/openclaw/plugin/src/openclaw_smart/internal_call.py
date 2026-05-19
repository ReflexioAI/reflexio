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

# Plugin layout:
#   <reflexio_repo>/reflexio/integrations/openclaw/plugin/src/openclaw_smart/internal_call.py
# parents[5] = <reflexio_repo>, the directory we want to fence off.
#
# In install mode the in-repo layout is absent — the env marker is the
# primary signal and this path never matches. ``OPENCLAW_SMART_REFLEXIO_DIR``
# lets callers (and tests) override the path without touching the module.
_THIS_DIR = Path(__file__).resolve().parent
_REFLEXIO_DIR = Path(
    os.environ.get("OPENCLAW_SMART_REFLEXIO_DIR") or _THIS_DIR.parents[5]
)


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
