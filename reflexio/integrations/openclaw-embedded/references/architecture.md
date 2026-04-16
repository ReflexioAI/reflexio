# Architecture

Deep-dive for maintainers. For a design-level overview, see the spec at
`docs/superpowers/specs/2026-04-16-reflexio-openclaw-embedded-plugin-design.md`.

## Three flows

| Flow | Trigger | Actor | Purpose |
|------|---------|-------|---------|
| A | User message signals a profile | Main agent (skill-guided) | Capture durable user facts mid-turn |
| B | Correction → adjust → confirmation | Main agent (skill-guided) | Capture procedural rules after confirmation |
| C | `session:compact:before`, `command:stop`, `command:reset` | `reflexio-extractor` sub-agent spawned via hook | Batch extract + shallow dedup over full transcript |
| Cron | Daily 3am | `reflexio-consolidator` sub-agent | Full n-way consolidation + TTL sweep |

## File system invariants

- `.reflexio/profiles/*.md` and `.reflexio/playbooks/*.md` — the only user-owned data.
- All files are **immutable in place**. Dedup = delete old + create new (atomic via `.tmp` + rename).
- Every file has frontmatter with `type`, `id`, `created`. Profiles also have `ttl` and `expires`.
- `supersedes: [id1, id2]` — optional, records merge lineage.

## Concurrency model

- Flow A + Flow B are create-only; no coordination.
- Flow C runs as a sub-agent spawned via `api.runtime.subagent.run()` — non-blocking, tracked by Openclaw's Background Tasks ledger.
- Parallel Flow C runs can occur; rare race resolves next cycle (later write wins, orphans swept by full consolidation).

## Openclaw primitives leveraged

- **Memory engine** (`concepts/memory-builtin`): indexes `.md` files under `extraPaths`.
- **Active Memory** (`concepts/active-memory`): optional, injects retrieved context into turns.
- **Hooks** (`automation/hooks`): lifecycle events (bootstrap, compact, stop, reset).
- **Sub-agents** (`tools/subagents`): fire-and-forget work via `sessions_spawn` / `api.runtime.subagent.run()`.
- **LLM-task** (`tools/llm-task`): structured LLM calls with schema validation.
- **Cron** (`automation/cron-jobs`): daily consolidation.
- **exec** (`tools/exec`): allows the agent and sub-agents to invoke `./scripts/reflexio-write.sh`.

## Prompt loading

Prompts live in `prompts/` and are loaded at runtime by sub-agents. Frontmatter follows Reflexio's prompt_bank convention (`active`, `description`, `changelog`, `variables`). When the sub-agent builds a `llm-task` call, it reads the relevant prompt file, substitutes variables from the event / memory_search results / candidate data, and sends the result to `llm-task` with an output schema.

## Graceful degradation

| Missing prereq | Behavior |
|---|---|
| `active-memory` not enabled | SKILL.md instructs agent to run `memory_search` fallback at turn start |
| No embedding provider | Falls back to FTS/BM25 only; vector search unavailable but plugin functional |
| `exec` denied | SKILL.md falls back to printed manual commands; install.sh exits with instructions |
| No `openclaw cron add` | install.sh prints warning; consolidation runs only on `/skill reflexio-consolidate` |
