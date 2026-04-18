#!/usr/bin/env bash
# openclaw-embedded install.sh — plugin installation.
# Skills are served from the extension dir via the manifest.
# Agents are injected via extraSystemPrompt at runtime.
# HEARTBEAT.md is appended on first agent session by setup.ts.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "$0")/../plugin" && pwd)"
OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"

die() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# 1. Prereq checks
info "Checking prerequisites..."
command -v openclaw >/dev/null || die "openclaw CLI required but not found on PATH"
command -v node >/dev/null     || die "node required but not found on PATH"

# 2. Install the plugin
info "Installing plugin..."
openclaw plugins uninstall --force reflexio-embedded 2>/dev/null || true
rm -rf "$OPENCLAW_HOME/extensions/reflexio-embedded"
openclaw plugins install "$PLUGIN_DIR"
openclaw plugins enable reflexio-embedded 2>/dev/null || true

# 3. Enable active-memory plugin and configure per-agent targeting
info "Enabling active-memory plugin..."
openclaw plugins enable active-memory || \
  echo "warning: active-memory enable failed — plugin may already be enabled or unavailable; continuing"

info "Configuring active-memory agent targeting..."
openclaw config set plugins.entries.active-memory.config.agents '["*"]' || \
  echo "warning: active-memory agent targeting config failed"

info "Registering .reflexio/ as memory extraPath..."
openclaw config set agents.defaults.memorySearch.extraPaths '[".reflexio/"]' --strict-json || \
  echo "warning: extraPath registration failed"

# 4. Restart gateway
info "Restarting openclaw gateway..."
openclaw gateway restart

# 5. Verify
info "Verification:"
if openclaw plugins inspect reflexio-embedded 2>/dev/null | grep -q "Status: loaded"; then
  info "  ✓ plugin registered and loaded"
else
  echo "  ⚠ plugin did not reach 'loaded' status; run 'openclaw plugins inspect reflexio-embedded' to debug"
fi

info "Installation complete."
info "Skills are served from the extension dir. HEARTBEAT.md is set up on first agent session."
