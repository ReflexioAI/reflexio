#!/usr/bin/env bash
#
# Legacy Claude Code SessionStart hook.
#
# Deprecated: the maintained integration is claude-smart. This shim remains
# for one compatibility release so old settings entries do not fail, but it
# never starts the Reflexio backend.

cat > /dev/null
echo '{}'
echo "[reflexio] legacy SessionStart hook is deprecated and no-op; install claude-smart for active hooks." >&2
exit 0
