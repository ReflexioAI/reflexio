#!/usr/bin/env bash
# Open the reflexio dashboard in the user's browser. The dashboard itself
# (Next.js app) is provided by the claude-smart sibling plugin and listens
# on http://localhost:3001 by default — both plugins target the same URL.
# We only need the "open browser" hop here.
set -eu

URL="${REFLEXIO_DASHBOARD_URL:-http://localhost:3001}"
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1 || true
else
  echo "Open this URL in your browser: $URL"
fi
