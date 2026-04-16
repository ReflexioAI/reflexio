---
name: reflexio-consolidate
description: "Run a full-sweep consolidation over all .reflexio/ files — TTL sweep + n-way cluster merge. Use when the user asks to 'clean up reflexio', 'consolidate memory', 'deduplicate playbooks', or suspects drift across sessions."
---

# Reflexio Consolidate

User-invocable via `/skill reflexio-consolidate`. Same workflow that runs daily at 3am via the plugin's cron job, but on-demand.

## What it does

1. TTL sweep: delete expired profile files.
2. For each of profiles and playbooks:
   - Cluster similar files via `memory_search`.
   - For clusters of 2+ members, call `llm-task` with `prompts/full_consolidation.md` to decide merge / subset / keep-all.
   - Apply decisions: write merged files with `supersedes:` frontmatter, unlink merged originals.

## How to run

Delegate to the `reflexio-consolidator` sub-agent:

```
sessions_spawn(
  task: "Run your full-sweep consolidation workflow now. Follow your system prompt in full.",
  agentId: "reflexio-consolidator",
  runTimeoutSeconds: 300,
  mode: "run",
)
```

Report the returned `runId` to the user. They can inspect progress via `openclaw tasks list`.

## When to use

- User asks to "consolidate", "clean up reflexio", "dedupe memory"
- User reports seeing duplicate or contradictory entries in retrieval
- After a long period without daily cron runs (e.g. host was offline)

## When NOT to use

- Routine maintenance — the daily cron at 3am handles this.
- Immediately after Flow A/B writes — shallow dedup at write time + Flow C at session end cover the fresh-extraction cases.

## Failure modes

If `sessions_spawn` is unavailable or `reflexio-consolidator` agent is not registered, the plugin's install.sh did not complete. Tell the user to re-run `./scripts/install.sh` in the plugin directory.
