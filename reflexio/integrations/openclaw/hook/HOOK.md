---
name: reflexio-context
description: "Inject user profile at session start, capture conversations at session end for automatic playbook and profile extraction"
metadata:
  openclaw:
    emoji: "brain"
    events: ["agent:bootstrap", "message:received", "message:sent", "command:stop"]
    requires:
      bins: ["reflexio"]
      env: []
---

# Reflexio Context Hook

Automatically connects your OpenClaw agent to [Reflexio](https://github.com/reflexio-ai/reflexio) for continuous self-improvement.

## What It Does

### On `agent:bootstrap` (session start)
Fetches a brief user profile summary via `reflexio user-profiles search --json`. Injects user preferences, expertise, and communication style. Does NOT load playbooks or skills here — those are retrieved per-task via the companion skill's `reflexio search "<task>"` command.

### On `message:received` (before each response)
Runs `reflexio search "<user message>" --top-k 5` synchronously before the agent responds. If results are found, injects a `REFLEXIO_CONTEXT.md` bootstrap file with relevant playbooks and corrections. Skips trivial inputs (< 5 chars, or `yes/no/ok/sure/thanks`). Times out after 5 seconds — never blocks the response.

### On `message:sent` (each turn)
Buffers each (user message, agent response) pair into a local SQLite database (`~/.reflexio/sessions.db`). Lightweight local write — no network calls.

### On `command:stop` (session end)
Flushes the complete buffered conversation to Reflexio via `reflexio interactions publish --file`. The server detects corrections via LLM analysis, extracts playbooks (freeform content summary + optional structured fields: trigger/instruction/pitfall/rationale) and user profiles. Fire-and-forget — does not block session shutdown.

## Prerequisites

- The `reflexio` CLI installed and on PATH (`pip install reflexio`)
- A running Reflexio server (local or remote)
- For cloud/Supabase mode: `REFLEXIO_API_KEY` environment variable set

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REFLEXIO_API_KEY` | — | Required for cloud/Supabase mode. Not needed for local/SQLite. |
| `REFLEXIO_URL` | `http://127.0.0.1:8081` | Reflexio server URL |
| `REFLEXIO_USER_ID` | auto (from agentId) | User ID for profile search |
| `REFLEXIO_AGENT_VERSION` | `openclaw-agent` | A label identifying your agent version. Playbooks are scoped by this tag. |
