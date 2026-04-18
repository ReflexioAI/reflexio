#!/usr/bin/env bash
# openclaw-embedded uninstall.sh — reverses install.sh.
# Leaves workspace/.reflexio/ user data intact unless --purge is passed.
set -euo pipefail

OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
PURGE_DATA="${1:-}"

info() { echo "==> $*"; }

info "Disabling plugin..."
openclaw plugins disable reflexio-embedded 2>/dev/null || echo "(already disabled)"

info "Uninstalling plugin..."
openclaw plugins uninstall --force reflexio-embedded 2>/dev/null || echo "(already uninstalled)"
rm -rf "$OPENCLAW_HOME/extensions/reflexio-embedded"

info "Cleaning up state files..."
rm -f "$OPENCLAW_HOME/reflexio-consolidation-state.json"

info "Removing heartbeat entry..."
if [[ -f "$OPENCLAW_HOME/workspace/HEARTBEAT.md" ]]; then
  sed -i '' '/## Reflexio Consolidation Check/,/^$/d' "$OPENCLAW_HOME/workspace/HEARTBEAT.md" 2>/dev/null || true
fi

if [[ "$PURGE_DATA" == "--purge" ]]; then
  info "Purging .reflexio/ user data per --purge flag..."
  rm -rf "$PWD/.reflexio"
else
  info "User data at .reflexio/ preserved. Use --purge to delete it too."
fi

info "Restarting openclaw gateway..."
openclaw gateway restart

info "Uninstall complete."
