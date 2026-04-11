#!/bin/bash

# Configurable ports (must match what was used in run_services.sh)
BACKEND_PORT=${BACKEND_PORT:-8081}
DOCS_PORT=${DOCS_PORT:-8082}

echo "Stopping services..."

# Stop FastAPI server — kill by port AND by process name to catch workers mid-request
PIDS=$(lsof -t -i:${BACKEND_PORT} 2>/dev/null)
UVICORN_PIDS=$(pgrep -f "uvicorn src.server.api:app" 2>/dev/null)
ALL_PIDS=$(echo -e "${PIDS}\n${UVICORN_PIDS}" | sort -u | grep -v '^$')
if [ -n "$ALL_PIDS" ]; then
    echo "$ALL_PIDS" | xargs kill 2>/dev/null
    sleep 1
    # Force kill any survivors (uvicorn reload workers can ignore SIGTERM)
    PIDS=$(lsof -t -i:${BACKEND_PORT} 2>/dev/null)
    UVICORN_PIDS=$(pgrep -f "uvicorn src.server.api:app" 2>/dev/null)
    ALL_PIDS=$(echo -e "${PIDS}\n${UVICORN_PIDS}" | sort -u | grep -v '^$')
    [ -n "$ALL_PIDS" ] && echo "$ALL_PIDS" | xargs kill -9 2>/dev/null
    echo "Stopped FastAPI server (${BACKEND_PORT})"
else
    echo "FastAPI server (${BACKEND_PORT}) not running"
fi

# Stop docs frontend (Next.js) — kill by port
DOCS_PIDS=$(lsof -t -i:${DOCS_PORT} 2>/dev/null)
if [ -n "$DOCS_PIDS" ]; then
    echo "$DOCS_PIDS" | xargs kill 2>/dev/null
    sleep 1
    DOCS_PIDS=$(lsof -t -i:${DOCS_PORT} 2>/dev/null)
    [ -n "$DOCS_PIDS" ] && echo "$DOCS_PIDS" | xargs kill -9 2>/dev/null
    echo "Stopped docs frontend (${DOCS_PORT})"
else
    echo "Docs frontend (${DOCS_PORT}) not running"
fi

echo "All services stopped."
