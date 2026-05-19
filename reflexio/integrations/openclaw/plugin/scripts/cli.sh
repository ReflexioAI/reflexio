#!/usr/bin/env bash
# Wrapper for slash commands that invoke the openclaw_smart CLI via uv.
# openClaw runs `!` bash directives in slash command .md files in a
# non-interactive, non-login shell that does NOT source ~/.zshrc or
# ~/.bash_profile. As a result, binaries installed by smart-install.sh
# at ~/.local/bin (e.g. uv from the astral.sh installer) are invisible
# to those directives until the user manually re-sources their shell rc.
# This wrapper bootstraps PATH the same way hook_entry.sh does so the
# slash commands work on a fresh install.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_lib.sh
. "$HERE/_lib.sh"
openclaw_smart_source_login_path
openclaw_smart_prepend_astral_bins
openclaw_smart_prepend_node_bins

PLUGIN_ROOT="$(cd "$HERE/.." && pwd)"

# If the Setup hook recorded an install failure, surface that reason
# instead of falling through to a generic "uv not found" — mirrors the
# branch at hook_entry.sh so slash commands and hooks behave consistently
# on a broken install.
FAILURE_MARKER="$HOME/.openclaw-smart/install-failed"
if [ -f "$FAILURE_MARKER" ]; then
  msg="$(cat "$FAILURE_MARKER" 2>/dev/null || echo "")"
  [ -n "$msg" ] || msg="unknown error"
  echo "openclaw-smart is not installed correctly: $msg" >&2
  echo "Re-run the plugin's Setup (restart openClaw) or fix the underlying issue and delete $FAILURE_MARKER to retry." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  # Self-heal: the Setup/SessionStart hook may have been skipped (trust
  # prompt declined, plugin enabled mid-session, etc.) leaving the install
  # half-done. Run smart-install.sh inline so the user does not have to
  # restart openClaw just to recover.
  # Guard against recursion: if smart-install.sh ever shells back through
  # this wrapper (e.g. a future migration step) we must not loop forever.
  if [ "${OPENCLAW_SMART_BOOTSTRAPPING:-}" = "1" ]; then
    echo "openclaw-smart: bootstrap recursion detected; aborting." >&2
    exit 1
  fi
  if [ -x "$PLUGIN_ROOT/scripts/smart-install.sh" ]; then
    echo "openclaw-smart: 'uv' not found — bootstrapping dependencies (~1-3 min on first install)..." >&2
    # ``set -e`` would short-circuit on a non-zero installer exit and skip
    # the structured failure-marker checks below. Capture the failure
    # explicitly so the wrapper continues to the unified error reporting.
    OPENCLAW_SMART_BOOTSTRAPPING=1 bash "$PLUGIN_ROOT/scripts/smart-install.sh" >&2 || \
      echo "openclaw-smart: smart-install.sh exited non-zero; continuing to error report" >&2
    openclaw_smart_prepend_astral_bins
    openclaw_smart_prepend_node_bins
  fi
  if ! command -v uv >/dev/null 2>&1; then
    if [ -f "$FAILURE_MARKER" ]; then
      msg="$(cat "$FAILURE_MARKER" 2>/dev/null || echo "unknown error")"
      echo "openclaw-smart: install failed: $msg" >&2
      echo "Fix the underlying issue and delete $FAILURE_MARKER to retry." >&2
    else
      echo "openclaw-smart: 'uv' not found on PATH after bootstrap attempt." >&2
      echo "Install it from https://docs.astral.sh/uv/ or rerun $PLUGIN_ROOT/scripts/smart-install.sh manually." >&2
    fi
    exit 1
  fi
fi

exec uv run --project "$PLUGIN_ROOT" --quiet python -m openclaw_smart.cli "$@"
