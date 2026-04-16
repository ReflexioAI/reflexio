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

The hook is pure Node.js + native `fetch()`. It does not spawn subprocesses
or invoke the `reflexio` CLI — all traffic is HTTP to the local Reflexio
backend at `http://127.0.0.1:8081`.

### On `agent:bootstrap` (session start)
POSTs to `/api/search` with the query `"communication style, expertise, and
preferences"` to fetch a brief user profile summary. Injects user preferences,
expertise, and communication style as a `REFLEXIO_USER_PROFILE.md` bootstrap
file. Does NOT load playbooks here — those are retrieved per-message or
per-task.

### On `message:received` (before each response)
POSTs to `/api/search` with the user's message and `top_k: 5`. If results are
found, formats them as markdown and injects a `REFLEXIO_CONTEXT.md` bootstrap
file with relevant playbooks and corrections. Skips trivial inputs (< 5
chars, or `yes/no/ok/sure/thanks`). Times out after 5 seconds — never blocks
the response.

### On `message:sent` (each turn)
Buffers each (user message, agent response) pair into a local SQLite database
(`~/.reflexio/sessions.db`). Lightweight local write — no network calls. If
the buffer exceeds `BATCH_SIZE * 2` unpublished turns, triggers an incremental
publish (see below).

### On `command:stop` (session end)
POSTs the complete buffered conversation to `/api/publish_interaction` as a
single JSON request. The local Reflexio server detects corrections via LLM
analysis, extracts playbooks (content + top-level fields: trigger, rationale,
blocking_issue) and user profiles. Blocks
briefly on the HTTP round-trip; if it fails, turns stay unpublished and are
retried on the next `agent:bootstrap`.

## Prerequisites

1. **`reflexio` CLI on PATH** — `pipx install reflexio-ai` (or `pip install --user reflexio-ai`). Needed to start the backend server and run the slash commands.
2. **Local Reflexio server running at `http://127.0.0.1:8081`** — the hook does NOT start it; the skill's First-Use Setup does that once via `reflexio services start --only backend`.
3. **An LLM provider API key in `~/.reflexio/.env`** — **required for the system to work end-to-end, even though the hook itself never reads it.** The local Reflexio server uses this key to extract playbooks and profiles from captured conversations via LiteLLM. `reflexio setup openclaw` will prompt you interactively for the provider (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, MiniMax, DashScope, xAI, Moonshot, ZAI, or any local LLM via a custom base URL) and write the key for you. **If you want fully offline operation, provide a local LLM endpoint (Ollama, LM Studio, vLLM) at this step instead of a hosted provider.**

The skill's registry metadata does not declare this LLM key under `requires.env` because that field describes variables the hook's own code path reads, and the hook is deliberately stateless (no env var access, no filesystem config reads — enforced in `handler.js`). The dependency lives one hop away, at the backend server, and is called out here in prose instead.

## Privacy: what the hook guarantees, what it doesn't

The hook itself communicates only with `http://127.0.0.1:8081`. It reads no
environment variables, no configuration files, and has no code path that
reaches any other host.

**The local Reflexio server, however, makes outbound LLM API calls** for
profile/playbook extraction. The destination is whatever you configured in
`~/.reflexio/.env` (OpenAI, Anthropic, Gemini, etc.). If that provider is
external, excerpts of your conversations will be sent to it. If you want a
fully offline setup, configure the server to use a local LLM (Ollama,
LM Studio, vLLM, etc.) before enabling the hook.

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
