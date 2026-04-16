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

info "Removing cron job..."
openclaw cron rm reflexio-embedded-consolidate 2>/dev/null || echo "(already removed)"

info "Removing skills..."
rm -rf "$OPENCLAW_HOME/workspace/skills/reflexio-embedded"
rm -rf "$OPENCLAW_HOME/workspace/skills/reflexio-consolidate"

info "Removing agent definitions..."
rm -f "$OPENCLAW_HOME/workspace/agents/reflexio-extractor.md"
rm -f "$OPENCLAW_HOME/workspace/agents/reflexio-consolidator.md"

info "Removing plugin resources..."
rm -rf "$OPENCLAW_HOME/workspace/plugins/reflexio-embedded"

if [[ "$PURGE_DATA" == "--purge" ]]; then
  info "Purging .reflexio/ user data per --purge flag..."
  rm -rf "$PWD/.reflexio"
else
  info "User data at .reflexio/ preserved. Use --purge to delete it too."
fi

info "Restarting openclaw gateway..."
openclaw gateway restart

info "Uninstall complete."
