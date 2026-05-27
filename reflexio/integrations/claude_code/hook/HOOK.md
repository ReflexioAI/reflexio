---
name: reflexio-hooks
description: "Claude Code hooks for Reflexio: server auto-start, context search, and session capture"
events: ["SessionStart", "UserPromptSubmit", "SessionEnd"]
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
3. Records the `(kind, real_id)` of every returned profile and user_playbook to a session-scoped JSONL state file at `~/.reflexio/claude-code-sessions/<session_id>.jsonl`. Each line is `{prompt, timestamp, citations: [...]}`. The SessionEnd handler reads this file to attach citations to the assistant interaction that followed each prompt — this is what powers the /evaluations "Rules that moved the needle" attribution panel. The state file is deleted after the SessionEnd publish.
4. Falls back to starting the server if it is down (redundant safety net for mid-session crashes)

### On `SessionEnd` (session end)

1. Reads the session transcript JSONL file from `transcript_path` in the event payload
2. Extracts user queries and assistant responses — preserves text and tool_use blocks (as `tools_used` metadata), skips thinking blocks and system messages
3. Loads the session-scoped citation state file written by the search hook and attaches recorded citations to the assistant interaction following each matching user prompt (matched on the first 200 characters of trimmed prompt text)
4. Writes the formatted payload to a temp file
5. Spawns a detached `reflexio interactions publish --force-extraction --file <payload>` process (fire-and-forget). Both the temp payload and the citation state file are removed after publish.
6. Logs publish output to `~/.reflexio/logs/stop-hook.log` for diagnostics
7. Outputs `{}` on stdout and exits immediately — does not block session shutdown

The `--force-extraction` flag ensures extraction always runs, even if a mid-session publish already happened within the batch interval. The Reflexio server then analyzes the conversation for learning signals (corrections, friction, re-steering) and extracts playbooks and user profiles automatically.

**Installed automatically with expert mode** (`reflexio setup claude-code --expert`). Not installed in normal mode.

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

Run `reflexio setup claude-code` to install automatically (add `--expert` for the SessionEnd hook), or add to your Claude Code `settings.json` manually:

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
    ],
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "node /path/to/reflexio/integrations/claude_code/hook/handler.js"
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
- Hooks skip internal/headless Claude invocations such as Reflexio extractor
  calls (`CLAUDE_SMART_INTERNAL=1`, `REFLEXIO_INTERNAL=1`, or
  `CLAUDE_CODE_ENTRYPOINT` not equal to `cli`) so backend/search/publish flows
  do not recurse into themselves.
- Publishing is fire-and-forget — failures do not affect the Claude Code session
- Publish errors are logged to `~/.reflexio/logs/stop-hook.log` for diagnostics
- Transcript data is written to a temp file with restricted permissions (0600)
- The temp file is cleaned up after publishing
