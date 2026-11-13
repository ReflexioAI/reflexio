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

The plugin ships in the Python SDK. Setup compiles its TypeScript with the
Node runtime OpenClaw already requires, then installs the resulting directory.
There is one source and build path for SDK setup and npm development packing;
a separate registry publication is not a release prerequisite.

Setup preserves supported search and publication policy across reinstalls and
removes obsolete fields. It builds before changing host configuration, rejects
invalid supported policy, and attempts to restore the previous settings if
installation fails. For development, `--plugin-path` selects an explicit local
directory or archive instead of the bundled runtime.

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
~120 KB are refused by this adapter’s existing argv transport; the native CLI
transport/version contract is unchanged.
Selected `openclaw/...` request/client models now remain authoritative when a
completion role is supplied, including the resumable extraction agent. Generic
role detection cannot switch that selected route to a cloud provider just because
another API key is present. An explicit per-call cloud model still uses standard
provider resolution; omit `model_role` when that exact cloud model must win.
An explicit `custom_llm_provider` transport override takes precedence over the
model prefix. Embedding APIs keep their separate selection path.

`OPENCLAW_DEFAULT_MODEL` no longer selects a model. Preserve a Reflexio-specific
choice in the existing role model fields, such as
`llm_config.generation_model_name="openclaw/anthropic/claude-sonnet-4-6"`.
Alternatively, remove the old assignment to deliberately use OpenClaw's native
default. Unused stale assignments do not block explicit models or unrelated
startup. A default native request with a nonblank old override refuses before
Reflexio retries or dispatches any fallback; no old value is copied or logged.
The native CLI permission, executable, timeout and fallback controls remain.

This corrects selection precedence; it does not rewrite frozen work or claim
transparent continuation for an old recipe whose role selected another provider.
Existing frozen-binding checks keep refusing changed recipes before dispatch.
The request-derived response model label is not proof of the model actually
served. Observed model/provider attribution stays unknown until independently
established by the native bridge; it is never filled from that request echo.
Offline shared compaction requires an observed model in its dispatch journal.
It therefore refuses native OpenClaw calls before spending a CLI call or
debiting a journal attempt; explicit HTTP transport overrides remain eligible.

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
