"""session_start hook — apply startup defaults without broad memory retrieval.

For openClaw, the ``session_start`` event delivers ``{sessionId, sessionKey,
agentId, workspaceDir}``. The handler:

1. Fetches the current stall state from reflexio and renders a 1-line banner
   to ``prependContext`` if learning is paused.
2. Pushes openclaw-smart's preferred extraction defaults (smaller window /
   stride than reflexio's out-of-box 10/5).
3. Pushes optimizer defaults so reflexio can evaluate candidate playbooks
   via the local ``openclaw infer model run`` CLI.

Output: when there is a banner, emits ``{"prependContext": "<banner>"}`` on
stdout for the TS shim to relay back to openClaw. Otherwise emits nothing.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from openclaw_smart.reflexio_adapter import Adapter
from openclaw_smart.stall_banner import render_banner

# openclaw-smart's preferred extraction cadence — more frequent, smaller
# windows than reflexio's out-of-box 10/5. Applied idempotently to the
# reflexio server on every session_start via
# Adapter.apply_extraction_defaults.
_WINDOW_SIZE = 5
_STRIDE_SIZE = 3
# Optimizer is on by default. Set this env var to "0" to skip pushing the
# openclaw-smart optimizer defaults on session_start (kill switch).
_DISABLE_OPTIMIZER_ENV = "OPENCLAW_SMART_ENABLE_OPTIMIZER"
_OPTIMIZER_TIMEOUT_SECONDS = 300


def _adapter() -> Adapter:
    """Construct the reflexio adapter for this hook invocation.

    Indirected through a factory so tests can monkeypatch the adapter
    construction without touching the ``Adapter`` class itself.

    Returns:
        Adapter: A fresh adapter bound to the current process env.
    """
    return Adapter()


def _stall_banner(adapter: Any) -> str:
    """Return the prepend-able stall banner, or "" if no banner should fire.

    Reads ``adapter.fetch_stall_state()``; if it reports an active,
    not-yet-notified stall, renders a one-line banner via
    ``stall_banner.render_banner``. All exceptions are absorbed: this is
    defense-in-depth — even though the hook dispatcher already wraps
    ``handle`` in try/except, a stall-path bug must never block the
    existing playbook/profile rendering.

    Args:
        adapter (Any): The adapter to query. Duck-typed so tests can stub.

    Returns:
        str: The banner text, or ``""`` when there is nothing to show.
    """
    try:
        state_obj = adapter.fetch_stall_state()
    except Exception:  # noqa: BLE001 — stall path must never crash the hook.
        return ""
    if state_obj is None:
        return ""
    if not getattr(state_obj, "stalled", False):
        return ""
    if getattr(state_obj, "notified_in_cc", False):
        return ""
    try:
        return render_banner(
            reason=getattr(state_obj, "reason", None),
            reset_estimate=getattr(state_obj, "reset_estimate", None),
        )
    except Exception:  # noqa: BLE001 — render_banner bug must not block playbook injection.
        return ""


def handle(payload: dict[str, Any]) -> None:
    """Handle a session_start hook payload.

    Args:
        payload (dict[str, Any]): openClaw event/ctx blob, expected to
            contain ``sessionKey`` (or ``sessionId``).
    """
    session_id = payload.get("sessionKey") or payload.get("sessionId")
    if not session_id:
        return

    adapter = _adapter()

    # Stall banner — emitted via prependContext, fires at most once per stall
    # event (controlled server-side via mark_stall_notified).
    banner = _stall_banner(adapter)

    adapter.apply_extraction_defaults(
        window_size=_WINDOW_SIZE,
        stride_size=_STRIDE_SIZE,
    )
    if os.environ.get(_DISABLE_OPTIMIZER_ENV) != "0":
        adapter.apply_optimizer_defaults(
            script_path=_optimizer_assistant_path(),
            timeout_seconds=_OPTIMIZER_TIMEOUT_SECONDS,
        )

    if not banner:
        return

    sys.stdout.write(json.dumps({"prependContext": banner}))
    sys.stdout.write("\n")

    # Telemetry must not break the session.
    with contextlib.suppress(Exception):
        adapter.mark_stall_notified()


def _optimizer_assistant_path() -> str:
    """Return the absolute path to the openclaw-smart-optimizer-assistant binary."""
    executable = Path(sys.executable)
    suffix = ".exe" if os.name == "nt" else ""
    return str(executable.with_name(f"openclaw-smart-optimizer-assistant{suffix}"))
