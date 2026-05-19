"""Shared "search reflexio, render markdown, emit envelope" pipeline.

before_tool_call and before_prompt_build both (a) run a query-aware reflexio
search, (b) render the hits with ``context_format.render_inline_with_registry``,
(c) persist the citation registry for the agent_end hook to resolve, and
(d) emit a ``prependContext`` envelope on stdout. The TS shim wraps stdout
and translates it to the openClaw plugin-SDK return shape.

Per Phase 0 finding B2 we use ``prependContext`` (per-turn) rather than
``prependSystemContext`` (cached) so injected context refreshes every turn.
``before_tool_call`` is observe-only per finding Q1, so handlers there
should NOT call ``emit_context`` — they only append state.

The caller remains responsible for handler-specific framing — see the two
call sites for the small policy differences.
"""

from __future__ import annotations

import json
import sys
import time

from openclaw_smart import context_format, oc_cite, state
from openclaw_smart.reflexio_adapter import Adapter


def emit_context(
    *,
    session_id: str,
    project_id: str,
    query: str,
    top_k: int,
    adapter: Adapter | None = None,
) -> bool:
    """Search reflexio, render hits, emit ``prependContext`` JSON on stdout.

    Args:
        session_id (str): openClaw session id (``sessionKey``); used to scope
            the per-session citation registry.
        project_id (str): reflexio ``user_id`` for this repo.
        query (str): Free-text query routed to reflexio's unified
            ``/api/search`` endpoint, which fans out to user playbooks
            (project-scoped), agent playbooks (global), and preferences
            (project-scoped) server-side.
        top_k (int): Cap on hits per collection.
        adapter (Adapter | None): Injection seam for tests. A fresh
            ``Adapter()`` is used when ``None``.

    Returns:
        bool: ``True`` when markdown was emitted to stdout; ``False`` when
            the search returned nothing to inject.
    """
    user_playbooks, agent_playbooks, profiles = (adapter or Adapter()).search_all(
        project_id=project_id,
        query=query,
        top_k=top_k,
    )
    markdown, registry = context_format.render_inline_with_registry(
        project_id=project_id,
        user_playbooks=user_playbooks,
        agent_playbooks=agent_playbooks,
        profiles=profiles,
    )
    if not markdown:
        return False

    oc_cite.ensure_installed()
    state.append_injected(
        session_id,
        (dict(entry, ts=int(time.time())) for entry in registry),
    )

    sys.stdout.write(json.dumps({"prependContext": markdown}))
    sys.stdout.write("\n")
    return True


__all__ = ["emit_context"]
