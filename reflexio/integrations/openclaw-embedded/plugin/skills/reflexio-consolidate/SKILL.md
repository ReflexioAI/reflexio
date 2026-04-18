---
name: reflexio-consolidate
description: "Run a full-sweep consolidation over all .reflexio/ files — TTL sweep + n-way cluster merge. Use when the user asks to 'clean up reflexio', 'consolidate memory', 'deduplicate playbooks', or suspects drift across sessions."
---

# Reflexio Consolidate

User-invocable via `/skill reflexio-consolidate`. Same workflow that runs automatically via heartbeat (every 24h of active use), but on-demand.

## What it does

1. TTL sweep: delete expired profile files.
2. For each of profiles and playbooks:
   - Cluster similar files via `memory_search`.
   - For clusters of 2+ members, call `llm-task` with `prompts/full_consolidation.md` to decide merge / subset / keep-all.
   - Apply decisions: write merged files with `supersedes:` frontmatter, unlink merged originals.

## How to run

Call the `reflexio_run_consolidation` tool. It spawns the consolidator sub-agent and returns a `runId`.

After successful consolidation, call the `reflexio_consolidation_mark_done` tool to update the heartbeat timer.

## When to use

- User asks to "consolidate", "clean up reflexio", "dedupe memory"
- User reports seeing duplicate or contradictory entries in retrieval
- Heartbeat check returns ALERT (automatic trigger)

## When NOT to use

- Routine maintenance — heartbeat handles this automatically.
- Immediately after Flow A/B writes — shallow dedup at write time + Flow C at session end cover the fresh-extraction cases.
