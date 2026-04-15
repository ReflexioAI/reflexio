---
name: reflexio-context
description: "Inject relevant past-session playbooks and user profile before each agent response. Search-only — conversation buffering and server-side extraction are handled by the /reflexio-extract slash command instead."
metadata:
  openclaw:
    emoji: "brain"
    events: ["agent:bootstrap", "message:received"]
    requires:
      bins: ["reflexio"]
      env: []
---

# Reflexio Context Hook

Automatically connects your OpenClaw agent to [Reflexio](https://github.com/reflexio-ai/reflexio) for cross-session memory retrieval.

## What It Does

The hook is pure Node.js + native `fetch()`. It does not spawn subprocesses,
invoke the `reflexio` CLI, or write any data to disk. All traffic is HTTP to
the local Reflexio backend at `http://127.0.0.1:8081`.

### On `agent:bootstrap` (session start)
POSTs to `/api/search` with the query `"communication style, expertise, and
preferences"` to fetch a brief user profile summary. Injects user preferences,
expertise, and communication style as a `REFLEXIO_USER_PROFILE.md` bootstrap
file. Does NOT load playbooks here — those are retrieved per-message.

### On `message:received` (before each response)
POSTs to `/api/search` with the user's message and `top_k: 5`. If results are
found, formats them as markdown and injects a `REFLEXIO_CONTEXT.md` bootstrap
file with relevant playbooks and corrections. Skips trivial inputs (< 5
chars, or `yes/no/ok/sure/thanks`). Times out after 5 seconds — never blocks
the response.

### What the hook does NOT do
It does not buffer turns, write to SQLite, or POST to
`/api/publish_interaction`. Extracting playbooks from conversations and
writing them back to Reflexio is the responsibility of the
`/reflexio-extract` slash command, which runs in the agent's own context.
That split is what lets this integration operate without any LLM provider
API key on the Reflexio server side.

## Prerequisites

1. **`reflexio` CLI on PATH** — `pipx install reflexio-ai` (or `pip install --user reflexio-ai`). Needed to start the backend server and run the slash commands.
2. **Local Reflexio server running at `http://127.0.0.1:8081`** — the hook does NOT start it; the skill's First-Use Setup does that once via `reflexio services start --only backend`.

No LLM provider API key is required by this integration. All extraction
happens in the agent's own session when the user runs `/reflexio-extract`;
the Reflexio server only performs CRUD and semantic search against its
local storage.

## Privacy

The hook itself communicates only with `http://127.0.0.1:8081`. It reads no
environment variables, no configuration files, and has no code path that
reaches any other host. It does not persist any conversation data — it only
reads from Reflexio's local store.

The `/reflexio-extract` slash command, when invoked, sends playbook
`add`/`update`/`search` calls to the same local server. Those calls carry
the trigger/instruction/pitfall/content fields that the agent extracted
from the current conversation; the raw conversation transcript is never
sent to the server.

If you want to audit what the server stores, see `~/.reflexio/` on your
machine.

## Security contract — hook side

This hook is **hard-pinned to `http://127.0.0.1:8081`**. The destination is a
hardcoded constant in `handler.js`; changing it requires editing the source.
The hook reads **no environment variables** and **no configuration files**.
All settings are hardcoded at module scope in `handler.js`:

- Server URL: `http://127.0.0.1:8081` (loopback, not configurable)
- Agent label: `openclaw-agent` (not configurable)
- User ID: derived from OpenClaw's session key prefix, with fallback `openclaw`

If you need remote Reflexio from OpenClaw, run a local proxy at
`127.0.0.1:8081` or use the Claude Code integration, which supports a full
set of configuration overrides for remote endpoints.
