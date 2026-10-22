# OpenClaw integration

Cross-session memory for [OpenClaw](https://openclaw.ai) agents. The plugin
(`@reflexioai/openclaw-smart`, id `reflexio-openclaw-smart`) runs inside the
OpenClaw Gateway and talks to a Reflexio backend over HTTP:

| OpenClaw hook | What the plugin does |
| --- | --- |
| `before_prompt_build` | Searches Reflexio with the user's prompt and injects matching preferences and skills as `prependContext`, each tagged `[oc:…]` |
| `agent_end` | Publishes the finished run (user message, assistant reply, tool calls) with the learnings that were injected and any the reply cited |
| `session_end` | Flushes queued publishes and calls `finalize_session` |
| `/reflexio status\|show\|learn` | Backend reachability and failure counts; current learnings; finalize now |

Search and publish share one `request_id` per run, derived from OpenClaw's
`runId`, so Reflexio can join what was retrieved to the turn it shaped.
Learnings are scoped per agent workspace (`openclaw:<agent>:<dir>-<hash>`), not
per person: everyone talking to one agent in a shared channel shares them.
Incognito sessions, heartbeats and cron runs are skipped entirely.

Every hook fails open: if Reflexio is down the turn is unchanged, and the
failure is logged at `warn` and counted in `/reflexio status`. A publish that
fails is kept in memory and retried (same `request_id`) on the next turn; a
retry the server reports as already admitted counts as delivered, and a request
it rejects (4xx) is dropped. The queue does not survive a Gateway restart.
`session_end` (but not compaction) finalizes the session once nothing is queued.
Tool arguments and output are redacted for common credential shapes and
size-capped before they are published.

Requires OpenClaw `>=2026.9.33` and Node `>=24.22.1`.

## Install

With Reflexio installed (`pip install reflexio-ai`):

```bash
reflexio setup openclaw          # provider + storage, then installs and configures the plugin
reflexio services start          # local backend (skip for managed / self-hosted Reflexio)
```

Manually:

```bash
openclaw plugins install @reflexioai/openclaw-smart --accept-capabilities
openclaw config set plugins.entries.reflexio-openclaw-smart.hooks.allowConversationAccess true
openclaw config set plugins.entries.reflexio-openclaw-smart.config.url https://your-reflexio/
openclaw config set plugins.entries.reflexio-openclaw-smart.config.apiKey '${REFLEXIO_API_KEY}'
openclaw gateway restart
```

`allowConversationAccess` is required: OpenClaw blocks conversation hooks for
plugins it did not bundle. Without `url`, the plugin uses `REFLEXIO_URL`, then
`http://127.0.0.1:8061/`.

`reflexio setup openclaw --uninstall` removes it.

## Extraction through OpenClaw's model (optional)

Reflexio normally extracts with its own configured LLM provider. To reuse the
model OpenClaw is signed in to, set `OPENCLAW_SMART_USE_LOCAL_CLI=1` (and
`OPENCLAW_BIN` if `openclaw` is not on `PATH`) for the Reflexio server; the
OpenClaw provider is then detected automatically (`openclaw/default` uses
OpenClaw's configured model; `openclaw/<provider>/<model>` picks one). The
server calls `openclaw infer model run`. Extraction is a tool loop and the CLI
returns text, so tool calls travel in the prompt, the same protocol the Claude
Code provider uses (`server/llm/providers/cli_tool_protocol.py`). Prompts above
~120 KB are refused because the CLI only accepts the prompt as an argument.
Run a Gateway alongside: two embedded OpenClaw processes on one state directory
contend for it.

## Develop and test

```text
plugin/   the TypeScript plugin (npm test, npm run typecheck — against the real SDK types)
lab/      an isolated OpenClaw + Reflexio environment and end-to-end scenarios
```

See [lab/README.md](lab/README.md). The plugin's request bodies are pinned in
`plugin/tests/fixtures/wire-requests.json` and checked against the server's
request models by `tests/integrations/test_openclaw_wire_contract.py`.
