---
name: reflexio-hooks
description: "Claude Code hooks for Reflexio: server auto-start, context search, and session capture"
events: ["SessionStart", "UserPromptSubmit", "Stop"]
requires:
  bins: ["reflexio", "node", "bash", "curl"]
---

# Reflexio Hooks

Hooks that integrate Reflexio with Claude Code across the session lifecycle.

## What They Do

### On `SessionStart` (session begins)

1. Checks if the Reflexio server is running via `curl` to `/health`
2. If not running, starts `reflexio services start --only backend` in background
3. Outputs `{}` immediately — adds ~10ms latency, all real work is backgrounded
4. Uses a flag file (`~/.reflexio/logs/.server-starting`) to prevent concurrent starts
5. Cleans up stale flag files older than 2 minutes

This ensures the server is ready before the first `UserPromptSubmit` search hook fires.

### On `UserPromptSubmit` (every user message)

1. Runs `reflexio search "<prompt>"` with the user's message
2. Injects matching profiles and playbooks as context Claude sees before responding
3. Falls back to starting the server if it is down (redundant safety net for mid-session crashes)

### On `Stop` (session end)

1. Reads the session transcript JSONL file from `transcript_path` in the event payload
2. Extracts user queries and assistant text responses (skips thinking blocks, tool calls, system messages)
3. Writes the formatted payload to a temp file
4. Spawns a detached `reflexio interactions publish --file <payload>` process (fire-and-forget)
5. Outputs `{}` on stdout and exits immediately — does not block session shutdown

The Reflexio server then analyzes the conversation for learning signals (corrections, friction, re-steering) and extracts playbooks and user profiles automatically.

## Prerequisites

- The `reflexio` CLI installed and on PATH (`pip install reflexio`)
- Node.js runtime (for search and capture hooks)
- `curl` (for server health checks — pre-installed on macOS and most Linux)

## Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REFLEXIO_URL` | `http://127.0.0.1:8081` (local) or `https://www.reflexio.ai:8081` (remote) | No | Reflexio server URL (configured via `reflexio auth login`) |
| `REFLEXIO_API_KEY` | — | Managed / self-hosted only | API key for authenticated access to remote Reflexio server |
| `REFLEXIO_USER_ID` | `claude-code` | No | User ID for scoping profiles and playbooks |
| `REFLEXIO_AGENT_VERSION` | `claude-code` | No | Agent version tag for playbook filtering |

## Installation

Run `reflexio setup claude-code` to install automatically, or add to your Claude Code `settings.json` manually:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash /path/to/reflexio/integrations/claude_code/hook/session_start_hook.sh"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "node /path/to/reflexio/integrations/claude_code/hook/search_hook.js"
          }
        ]
      }
    ]
  }
}
```

## Safety

- The SessionStart hook adds ~10ms latency — all server work runs in a background process
- The flag file prevents concurrent server starts across hooks and sessions
- The Stop hook checks `stop_hook_active` to prevent infinite loops
- Publishing is fire-and-forget — failures do not affect the Claude Code session
- Transcript data is written to a temp file with restricted permissions (0600)
- The temp file is cleaned up after publishing
