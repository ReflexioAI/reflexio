#!/usr/bin/env bash
# Dispatch an openClaw hook event to the openclaw_smart Python package.
# OPENCLAW_PLUGIN_ROOT points at the plugin dir (dev: <repo>/.../openclaw/plugin;
# installed: ~/.openclaw/plugins/cache/.../openclaw-smart/<version>),
# which is also the Python project root with pyproject.toml + uv.lock.
# We invoke via `uv run --project` so the pinned env from uv.lock is used.
#
# If a prior Setup recorded an install failure at
# ~/.openclaw-smart/install-failed, short-circuit with a user-visible
# message instead of trying to run uv and failing silently.
set -eu

HOST="openclaw"
EVENT="${1:-}"
case "$EVENT" in
  openclaw)
    HOST="$EVENT"
    EVENT="${2:-}"
    ;;
esac
if [ -z "$EVENT" ]; then
  echo ''
  exit 0
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_lib.sh
. "$HERE/_lib.sh"
if openclaw_smart_is_internal_invocation_env; then
  echo ''
  exit 0
fi
# Pick up uv from the user's login-shell PATH (covers ~/.local/bin populated
# by the astral.sh installer) so a fresh install works before the user
# restarts their terminal. Matches the pattern used by smart-install.sh.
openclaw_smart_source_login_path
# Explicit fallback for the astral.sh installer's default paths, in case
# the user's login-shell rc hasn't yet been re-sourced to pick them up.
openclaw_smart_prepend_astral_bins

PLUGIN_ROOT="$(cd "$HERE/.." && pwd)"

FAILURE_MARKER="$HOME/.openclaw-smart/install-failed"
STATE_DIR="$HOME/.openclaw-smart"
if [ -f "$FAILURE_MARKER" ]; then
  if [ "$EVENT" = "session-start" ] && command -v python3 >/dev/null 2>&1; then
    python3 - "$FAILURE_MARKER" <<'PY'
import json, pathlib, sys
msg = pathlib.Path(sys.argv[1]).read_text().strip() or "unknown error"
print(json.dumps({
    "prependContext": (
        f"> **openclaw-smart is not installed correctly:** {msg}\n"
        "> Re-run the plugin's Setup (restart openClaw) "
        "or fix the underlying issue and delete "
        "`~/.openclaw-smart/install-failed` to retry."
    )
}))
PY
  else
    echo ''
  fi
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  # Self-heal from skipped Setup/SessionStart bootstrap. SessionStart can
  # afford to wait because it has the install budget; prompt/tool hooks start
  # the same installer detached so normal work is not blocked by first-run
  # dependency setup.
  if [ "${OPENCLAW_SMART_BOOTSTRAPPING:-}" = "1" ]; then
    echo ''
    exit 0
  fi
  if [ -x "$PLUGIN_ROOT/scripts/smart-install.sh" ]; then
    mkdir -p "$STATE_DIR"
    if [ "$EVENT" = "session-start" ]; then
      OPENCLAW_SMART_BOOTSTRAPPING=1 bash "$PLUGIN_ROOT/scripts/smart-install.sh" >&2
      openclaw_smart_prepend_astral_bins
      openclaw_smart_prepend_node_bins
      if command -v uv >/dev/null 2>&1; then
        bash "$HERE/backend-service.sh" start >/dev/null 2>&1 || true
      fi
    else
      openclaw_smart_spawn_detached env OPENCLAW_SMART_BOOTSTRAPPING=1 \
        bash "$PLUGIN_ROOT/scripts/smart-install.sh" \
        >>"$STATE_DIR/install.log" 2>&1 || true
    fi
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo ''
    exit 0
  fi
fi

# Re-check the failure marker after the inline bootstrap. ``smart-install.sh``
# can write ``install-failed`` *after* uv is on PATH (e.g. ``uv sync`` failed),
# in which case we still have a working ``uv`` but a non-functional plugin.
# Without this gate we would proceed to ``uv run`` and crash with a confusing
# downstream error instead of surfacing the recorded failure reason.
if [ -f "$FAILURE_MARKER" ]; then
  echo ''
  exit 0
fi

# Stdin is the hook payload JSON — stream it through to the Python CLI.
exec uv run --project "$PLUGIN_ROOT" --quiet python -m openclaw_smart.hook "$HOST" "$EVENT"
