# openclaw-smart

openclaw-smart is the openClaw plugin counterpart of
[claude-smart](https://github.com/reflexio-ai/claude-smart): a thin TS shim
plus a Python (`openclaw_smart`) package that wires openClaw's plugin hooks
into a local [Reflexio](https://github.com/reflexio-ai/reflexio) backend
running at `http://localhost:8071/`. The port matches claude-smart so both
plugins share one bundled reflexio (one SQLite store, one extractor), and
leaves reflexio's own 8081 default free for developer use. Conversations
are buffered to a JSONL session log, published for extraction (skills +
preferences) via your own LLM, and the top-matching items are injected
back into every subsequent turn as `prependContext`. The plugin degrades
silently when the backend is unreachable.

## Quick install

Guided Reflexio setup:

```bash
reflexio setup openclaw
```

Or install the OpenClaw plugin from npm:

```bash
npx openclaw-smart install
```

The unscoped `openclaw-smart` npm package is a thin `npx` alias around the
scoped plugin package, `@reflexioai/openclaw-smart`.

If your OpenClaw version supports npm package specs in its plugin installer,
you can also use the native installer directly:

```bash
openclaw plugins install @reflexioai/openclaw-smart
```

The guided `reflexio setup openclaw` command walks you through:

1. Picking an LLM provider and storage backend (SQLite by default).
2. Writing `OPENCLAW_BIN` + `OPENCLAW_SMART_USE_LOCAL_CLI=1` to
   `~/.reflexio/.env` so the backend's `openclaw` LiteLLM provider knows
   which CLI to spawn for extraction.
3. Installing the bundled plugin into openClaw
   (`openclaw plugins install <plugin_dir>`).
4. Running the plugin's first-run installer (`scripts/smart-install.sh`)
   so the next session doesn't pay a cold-install cost.
5. Calling `openclaw plugins inspect reflexio-openclaw-smart` to verify
   the plugin loaded.

`reflexio setup openclaw --uninstall [--purge]` reverses everything;
`--repair` re-runs only the first-run installer. The npm wrapper exposes
the same maintenance operations as `openclaw-smart uninstall [--purge]`
and `openclaw-smart repair`.

## How it works

* **session_start** — pushes openclaw-smart's preferred extraction window /
  stride and the shared-skill optimizer defaults to the reflexio backend.
  Emits a stall banner (`prependContext`) if learning has been idle.
* **before_prompt_build** — appends the user prompt to the session JSONL
  buffer, then runs an `/api/search` query against reflexio scoped to this
  project. Top hits are rendered as markdown and emitted under
  `prependContext` (per-turn, not cached).
* **before_tool_call / after_tool_call** — `before_*` is an observe-only
  stub (openClaw does not honor injection at that point). `after_*`
  redacts secrets, truncates oversized strings, normalizes camelCase
  payloads to snake_case, and appends an `Assistant_tool` record.
* **agent_end** — reads `payload["messages"]`, appends the latest
  assistant turn, then publishes everything since the last
  `published_up_to` watermark (no force-extraction).
* **session_end** — drains the buffer one last time with
  `force_extraction=True` so the session's learnings are available before
  the next openClaw run.

The full design lives in
[`docs/superpowers/specs/2026-05-19-openclaw-smart-design.md`](../../../../docs/superpowers/specs/2026-05-19-openclaw-smart-design.md).

## Skills

Six skill folders ship under `plugin/skills/` (`reflexio` is the always-on contract, the other five are user-invocable):

| Skill       | Purpose                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| `reflexio`  | Always-on contract: trust injected context, cite with `[oc:…]`, stay quiet |
| `learn`     | Force-publish the current session for immediate extraction                 |
| `show`      | Dump currently-known skills + preferences for this project as markdown     |
| `dashboard` | Open the local reflexio web UI (`http://localhost:3001`, shared with claude-smart) |
| `restart`   | Restart the local reflexio backend cleanly                                 |
| `clear-all` | Delete all locally-stored skills + preferences (destructive, prompts)      |

## Configuration

Plugin-side config lives in `plugin/openclaw.plugin.json` (no env vars
required for normal use). The shell scripts honour these env knobs:

* `OPENCLAW_SMART_INTERNAL=1` — recursion guard. Set automatically by
  `openclaw_provider._call_cli` when the reflexio backend's own
  LiteLLM-routed extraction subprocess fires hooks back into openclaw-smart.
  Any handler that sees this env in the payload short-circuits.
* `OPENCLAW_SMART_USE_LOCAL_CLI=1` — opt-in for the
  `openclaw_provider` LiteLLM plugin on the reflexio side. Set by
  `reflexio setup openclaw`.
* `OPENCLAW_BIN` — absolute path to the openclaw CLI used by the local
  LLM provider. Set by `reflexio setup openclaw`.
* `OPENCLAW_DEFAULT_MODEL` — default model passed to
  `openclaw infer model run` when no model is requested explicitly.
* `OPENCLAW_SMART_STATE_DIR` — overrides the JSONL session buffer
  location (default `~/.openclaw-smart/sessions/`). Used by tests.
* `OPENCLAW_SMART_BACKEND_AUTOSTART=0` — opt out of the
  `session_start` backend auto-start.
* `OPENCLAW_PLUGIN_ROOT` — pointer used by slash commands;
  `scripts/ensure-plugin-root.sh` symlinks `~/.reflexio/openclaw-plugin-root`
  to the active install.

## Recursion guard

The reflexio backend uses `openclaw_provider` (a LiteLLM CustomLLM) to
invoke the openclaw CLI for extraction. That CLI can in turn fire openClaw
hooks back into openclaw-smart, which would re-publish the extractor's
own prompt into reflexio — a tight feedback loop that explodes the
buffer. The guard at `openclaw_smart.internal_call.is_internal_invocation`
detects this by:

1. checking `OPENCLAW_SMART_INTERNAL=1` (set by the provider on spawn),
2. checking whether `workspaceDir` lives inside the reflexio install dir.

When either is true the hook dispatcher exits with empty stdout
immediately — no buffer writes, no search, no publish.

## Multi-session limitation

The TS shim tracks one `activeSessionKey` per plugin instance to route
the `reflexio_publish` tool to the right session. In concurrent
multi-session use the last session-key seen wins. This matches
claude-smart's behaviour and is intentional — see spec §10.

## Troubleshooting

* **Hook timing out / no inject** — check
  `~/.openclaw-smart/backend.log` for reflexio startup errors. The
  session-start backend autostart is best-effort; you can also start it
  manually via `bash plugin/scripts/backend-service.sh start`.
* **`uv` not found** — re-run `bash plugin/scripts/smart-install.sh`; on
  failure it writes the reason to `~/.openclaw-smart/install-failed`.
* **Stale plugin after a code change** — `reflexio setup openclaw --repair`
  re-runs the first-run installer and clears the failure marker.
* **Recursion loop / sessions multiplying** — the extractor lost the
  `OPENCLAW_SMART_INTERNAL=1` env. Confirm `openclaw_provider._call_cli`
  is still passing it (see
  `reflexio/server/llm/providers/openclaw_provider.py`).
