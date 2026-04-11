#!/bin/bash

# Configurable ports (override via env vars for worktree dev)
BACKEND_PORT=${BACKEND_PORT:-8081}
DOCS_PORT=${DOCS_PORT:-8082}
export API_BACKEND_URL=${API_BACKEND_URL:-"http://localhost:${BACKEND_PORT}"}

# Start FastAPI backend
uvicorn reflexio.server.api:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload --reload-include "reflexio/server/site_var/site_var_sources/*.json" &

# Start docs frontend (Next.js)
(cd docs && npm run dev -- -p ${DOCS_PORT}) &

# Keep container running
wait
