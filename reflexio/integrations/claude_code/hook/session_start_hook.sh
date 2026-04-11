#!/usr/bin/env bash
#
# Claude Code SessionStart hook for Reflexio.
#
# Checks if the Reflexio server is running and starts it in the background
# if not. Runs at session start so the server is ready before the first
# user message (and the first search hook).
#
# Designed for minimal latency (~10ms): reads stdin, outputs {}, forks
# all real work to background, and exits immediately.

LOG_DIR="$HOME/.reflexio/logs"
STARTING_FLAG="$LOG_DIR/.server-starting"
STALE_AGE_MIN=2  # flag files older than this (minutes) are stale

# Resolve SERVER_URL: env var > ~/.reflexio/.env > default
if [ -z "${REFLEXIO_URL:-}" ]; then
    _ENV_FILE="$HOME/.reflexio/.env"
    if [ -f "$_ENV_FILE" ]; then
        REFLEXIO_URL=$(grep -o '^REFLEXIO_URL="\{0,1\}[^"]*' "$_ENV_FILE" 2>/dev/null | head -1 | sed 's/^REFLEXIO_URL="\{0,1\}//')
    fi
fi
SERVER_URL="${REFLEXIO_URL:-http://127.0.0.1:8081}"

# 1. Consume stdin (protocol requirement) and output empty result immediately
cat > /dev/null
echo '{}'

# 2. If a recent flag file exists, another start is in progress — exit
if [ -f "$STARTING_FLAG" ]; then
    # Clean up stale flags (older than STALE_AGE_MIN minutes)
    find "$LOG_DIR" -name ".server-starting" -mmin +"$STALE_AGE_MIN" -delete 2>/dev/null
    # Re-check after cleanup — if flag still exists, a recent start is in progress
    if [ -f "$STARTING_FLAG" ]; then
        exit 0
    fi
fi

# 3. Determine if server is local (only local servers can be auto-started)
IS_LOCAL=true
case "$SERVER_URL" in
    *127.0.0.1*|*localhost*) IS_LOCAL=true ;;
    *) IS_LOCAL=false ;;
esac

# 4. Fork health check + conditional server start to background
(
    # Remote server — can't start it locally, skip without even checking
    if [ "$IS_LOCAL" = "false" ]; then
        exit 0
    fi

    # Quick health check — curl returns 0 on 2xx, non-zero otherwise
    if curl -sf --max-time 2 "$SERVER_URL/health" > /dev/null 2>&1; then
        exit 0  # Server is healthy, nothing to do
    fi

    # Server is not running — start it
    mkdir -p "$LOG_DIR"
    echo "$(date +%s)" > "$STARTING_FLAG"
    # Append (>>) so prior server logs are preserved across restarts; the
    # new-lifetime banner emitted by `reflexio services start` provides the
    # visual separator between runs.
    reflexio services start --only backend >> "$LOG_DIR/server.log" 2>&1 &
    sleep 30 && rm -f "$STARTING_FLAG"
) &

exit 0
