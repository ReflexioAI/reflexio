# Reflexio CLI

`reflexio` is a first-class CLI for running the Reflexio service, publishing interactions, exploring extracted profiles and playbooks, and wiring results into your own agent. Every command is also runnable as `python -m reflexio ...`.

## Table of Contents

- [Install & invoke](#install--invoke)
- [Global flags](#global-flags)
- [Quick Reference](#quick-reference)
- [Services](#services)
- [Publishing interactions](#publishing-interactions)
- [Search & context](#search--context)
- [Interactions](#interactions)
- [User profiles](#user-profiles)
- [Agent playbooks](#agent-playbooks)
- [User playbooks](#user-playbooks)
- [Config](#config)
- [Auth](#auth)
- [Diagnostics](#diagnostics)
- [Raw API access](#raw-api-access)
- [Common Workflows](#common-workflows)
- [Getting help](#getting-help)

## Quick Reference

Most common commands at a glance:

| Task | Command |
|------|---------|
| Start server | `reflexio services start` |
| Publish conversation | `reflexio publish --user-id USER --data '...'` |
| Search everything | `reflexio search "query"` |
| List profiles | `reflexio user-profiles list --user-id USER` |
| List playbooks | `reflexio agent-playbooks list` |
| Check health | `reflexio status check` |
| Stop server | `reflexio services stop` |

## Install & invoke

The CLI ships with the `reflexio` package. After `uv sync`:

```shell
uv run reflexio --help
uv run reflexio --version
```

## Global flags

Available on every command (set on the root, before the subcommand):

| Flag           | Env var              | Purpose                          |
| -------------- | -------------------- | -------------------------------- |
| `--json`       | —                    | Structured JSON output envelopes |
| `--server-url` | `REFLEXIO_URL`       | Backend API URL                  |
| `--api-key`    | `REFLEXIO_API_KEY`   | API bearer token                 |
| `--version`    | —                    | Show CLI version                 |

Example: `uv run reflexio --json search "refund policy"`.

## Services

Start and stop the backend and docs servers.

```shell
uv run reflexio services start                          # backend :8081, docs :8082
uv run reflexio services start --storage sqlite         # sqlite (default) | supabase | disk
uv run reflexio services start --backend-port 9000 --docs-port 9001
uv run reflexio services start --only backend --no-reload
uv run reflexio services stop
uv run reflexio services stop --force                   # SIGKILL instead of SIGTERM
```

## Publishing interactions

The top-level `publish` shortcut is the fastest way to get a conversation into Reflexio. It forwards to `interactions publish` and supports three input modes: single-turn flags, inline JSON, and file/stdin payloads.

Reflexio learns most from interactions that contain a **signal** — a user correction, a stated preference, or an explicit choice between alternatives. Examples throughout this doc use that kind of content rather than trivial Q&A.

### Mode 1 — Single-turn (shortcut flags)

`--user-message` / `--agent-response` wrap a single user turn and a single assistant turn. Use this for quick preference captures or smoke-tests. It is **hard-coded to exactly 2 turns** — use Mode 2 or 3 below for anything longer.

```shell
uv run reflexio publish \
  --user-id alice \
  --user-message "Stop adding disclaimers and caveats. Give me the answer in one line." \
  --agent-response "Understood — I'll drop the boilerplate from now on." \
  --wait
```

### Mode 2 — Multi-turn via inline JSON (`--data`)

For real dialogues (3+ turns, tool calls, corrections mid-conversation), pass a JSON object whose `interactions` field is a list of `{role, content}` items. Each item becomes one turn, in order. Roles are `user` and `assistant`; `system` and `tool` turns are also accepted.

```shell
uv run reflexio publish --user-id alice --wait --data '{
  "interactions": [
    {"role": "user",      "content": "Deploy the payments service to us-east-1."},
    {"role": "assistant", "content": "Starting deployment to us-east-1..."},
    {"role": "user",      "content": "Wait — production always runs in us-west-2. Never us-east-1."},
    {"role": "assistant", "content": "Understood. Switching to us-west-2."},
    {"role": "user",      "content": "Good. And always confirm the region with me before deploying."},
    {"role": "assistant", "content": "Noted. I'll confirm the region on every deploy going forward."}
  ]
}'
```

This single call can populate **both** a user profile (`production region is us-west-2`) and an agent playbook (`confirm deploy region before executing`), which is exactly the loop Reflexio is designed for.

You can also load inline JSON from a file with `@`: `--data @conversation.json`.

### Mode 3 — JSON / JSONL file or stdin

For bulk publishing or conversations you already have on disk, point at a file. A `.json` file should be a single object (or a list of objects); a `.jsonl` file is one JSON object per line, one conversation per line.

```shell
uv run reflexio publish --file conversations.jsonl --user-id alice --wait
cat conversations.jsonl | uv run reflexio publish --stdin --user-id alice
```

Each payload object accepts the same fields as Mode 2 (`interactions`, and optionally per-payload overrides for `user_id`, `session_id`, `source`, `agent_version`). A payload's own `user_id` wins over the `--user-id` flag, so you can publish a mixed-user JSONL file in a single call.

### Common flags

Apply to all three modes:

| Flag                | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `--wait`            | Block until server-side extraction finishes (returns real counts).      |
| `--session-id`      | Group multiple `publish` calls into one session.                         |
| `--agent-version`   | Tag the interaction with an agent version (used by playbook filtering). |
| `--source`          | Free-form source tag (defaults to `cli`).                                |
| `--skip-aggregation`| Extract profiles/playbooks but skip playbook aggregation.                |
| `--force-extraction`| Bypass `batch_interval` gating and always run extractors.                |

Full options via `uv run reflexio interactions publish --help`.

## Search & context

Unified semantic search across profiles and playbooks:

```shell
uv run reflexio search "deployment region"
uv run reflexio search "response style" --user-id alice --top-k 10 --threshold 0.5
```

Fetch formatted context for injection into your own agent. Both `--user-id` and `--agent-version` are required — the command fails loudly otherwise to avoid producing a misleading bootstrap:

```shell
uv run reflexio context --user-id alice --agent-version v1
uv run reflexio context --user-id alice --agent-version v1 --query "deploy workflow"
```

## Interactions

```shell
uv run reflexio interactions list --user-id alice
uv run reflexio interactions search "deployment"
uv run reflexio interactions delete <interaction-id>
uv run reflexio interactions delete-all
```

## User profiles

```shell
uv run reflexio user-profiles list --user-id alice
uv run reflexio user-profiles search "preferences"
uv run reflexio user-profiles add --user-id alice --data @profile.json
uv run reflexio user-profiles regenerate --user-id alice
uv run reflexio user-profiles delete <profile-id>
uv run reflexio user-profiles delete-all
```

## Agent playbooks

```shell
uv run reflexio agent-playbooks list --agent-version v1
uv run reflexio agent-playbooks search "error handling"
uv run reflexio agent-playbooks aggregate --agent-version v1
uv run reflexio agent-playbooks update-status <playbook-id> --status approved
uv run reflexio agent-playbooks regenerate --agent-version v1
uv run reflexio agent-playbooks delete <playbook-id>
```

## User playbooks

```shell
uv run reflexio user-playbooks list --user-id alice
uv run reflexio user-playbooks search "preferences" --user-id alice
uv run reflexio user-playbooks add --user-id alice --data @playbook.json
uv run reflexio user-playbooks update <playbook-id> --data @patch.json
uv run reflexio user-playbooks delete <playbook-id>
```

## Config

```shell
uv run reflexio config show
uv run reflexio config set --data '{"api_key_config": {"openai": "sk-..."}}'
uv run reflexio config set --data @config.json
uv run reflexio config storage                           # show storage backend info
uv run reflexio config pull                              # pull server config to local file
```

## Auth

```shell
uv run reflexio auth login --api-key $REFLEXIO_API_KEY --server-url http://localhost:8081
uv run reflexio auth status
uv run reflexio auth logout
```

## Diagnostics

```shell
uv run reflexio status check                             # server health
uv run reflexio status whoami                            # resolved identity
uv run reflexio doctor check                             # env + connectivity diagnostics
uv run reflexio setup init                               # interactive setup wizard
```

## Raw API access

Escape hatch for calling any endpoint directly. Supports `GET`, `POST`, `DELETE`:

```shell
uv run reflexio api GET /health
uv run reflexio api POST /api/get_agent_playbooks --data '{"agent_version": "v1"}'
uv run reflexio api POST /api/set_config --data @config.json
```

## Common Workflows

A typical end-to-end workflow: publish a conversation, verify that Reflexio extracted the right data, then browse the results.

### 1. Start services

```shell
uv run reflexio services start
```

### 2. Publish a conversation

```shell
uv run reflexio publish --user-id alice --wait --data '{
  "interactions": [
    {"role": "user",      "content": "Always use dark mode in the dashboard."},
    {"role": "assistant", "content": "Noted — I will default to dark mode for you."}
  ]
}'
```

### 3. Search to verify extraction

```shell
uv run reflexio search "dark mode"
```

You should see a profile entry reflecting the user's preference.

### 4. Browse profiles and playbooks

```shell
uv run reflexio user-profiles list --user-id alice
uv run reflexio user-playbooks list --user-id alice
uv run reflexio agent-playbooks list
```

## Getting help

Every command and subcommand supports `--help`:

```shell
uv run reflexio --help
uv run reflexio user-profiles --help
uv run reflexio user-playbooks --help
uv run reflexio agent-playbooks --help
```
