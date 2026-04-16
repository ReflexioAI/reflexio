#!/usr/bin/env bash
# openclaw-embedded install.sh — host-wide plugin installation.
# Per-agent config (active-memory targeting, extraPath) is done at first use via SKILL.md bootstrap.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"

die() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# 1. Prereq checks
info "Checking prerequisites..."
command -v openclaw >/dev/null || die "openclaw CLI required but not found on PATH"
command -v node >/dev/null     || die "node required but not found on PATH"

# 2. Install the plugin (hooks are registered programmatically from index.ts)
# `plugins install --link <path>` rejects `--force`, so we uninstall any prior
# registration first to make the install idempotent.
info "Installing plugin..."
openclaw plugins uninstall --force reflexio-embedded 2>/dev/null || true
openclaw plugins install --link "$PLUGIN_DIR"
# plugins install auto-enables by default. If ever it stops doing so, fall
# back to an explicit enable.
openclaw plugins enable reflexio-embedded 2>/dev/null || true

# 3. Copy main SKILL.md and consolidate command
info "Copying skills to workspace..."
mkdir -p "$OPENCLAW_HOME/workspace/skills/reflexio-embedded"
cp "$PLUGIN_DIR/SKILL.md" "$OPENCLAW_HOME/workspace/skills/reflexio-embedded/"
cp -r "$PLUGIN_DIR/commands/reflexio-consolidate" "$OPENCLAW_HOME/workspace/skills/"

# 4. Copy agent definitions
info "Copying agent definitions..."
mkdir -p "$OPENCLAW_HOME/workspace/agents"
cp "$PLUGIN_DIR/agents/reflexio-extractor.md"     "$OPENCLAW_HOME/workspace/agents/"
cp "$PLUGIN_DIR/agents/reflexio-consolidator.md"  "$OPENCLAW_HOME/workspace/agents/"

# 5. Copy prompts and scripts (referenced by agents at runtime)
info "Copying prompts and scripts..."
mkdir -p "$OPENCLAW_HOME/workspace/plugins/reflexio-embedded"
cp -r "$PLUGIN_DIR/prompts" "$OPENCLAW_HOME/workspace/plugins/reflexio-embedded/"
cp -r "$PLUGIN_DIR/scripts" "$OPENCLAW_HOME/workspace/plugins/reflexio-embedded/"
chmod +x "$OPENCLAW_HOME/workspace/plugins/reflexio-embedded/scripts/"*.sh

# 6. Enable active-memory plugin (host-wide; per-agent targeting is SKILL.md bootstrap's job)
info "Enabling active-memory plugin..."
openclaw plugins enable active-memory || \
  echo "warning: active-memory enable failed — plugin may already be enabled or unavailable; continuing"

# 7. Register daily consolidation cron
info "Registering daily consolidation cron (3am)..."
openclaw cron add \
  --name reflexio-embedded-consolidate \
  --cron "0 3 * * *" \
  --session isolated \
  --agent reflexio-consolidator \
  --message "Run your full-sweep consolidation workflow now. Follow your system prompt in full." \
  || echo "warning: cron registration failed — you can register it manually later with the same flags"

# 8. Restart gateway
info "Restarting openclaw gateway..."
openclaw gateway restart

# 9. Verify
info "Verification:"
if openclaw plugins inspect reflexio-embedded 2>/dev/null | grep -q "Status: loaded"; then
  info "  ✓ plugin registered and loaded"
else
  echo "  ⚠ plugin did not reach 'loaded' status; run 'openclaw plugins inspect reflexio-embedded' to debug"
fi
if openclaw cron list 2>/dev/null | grep -q reflexio-embedded-consolidate; then
  info "  ✓ cron registered"
else
  echo "  ⚠ cron not visible in 'openclaw cron list'"
fi

info "Installation complete."
info "On first use, the SKILL.md bootstrap will guide per-agent configuration (active-memory targeting, extraPath registration, embedding provider)."
