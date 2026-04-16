---
active: true
description: "N-way cluster consolidation for openclaw-embedded daily cron"
changelog: "Initial (2026-04-16): new prompt, informed by Reflexio's playbook_aggregation and *_deduplication prompts, adapted for single-instance n-way clustering"
variables:
  - cluster
---

You consolidate a cluster of 2-10 similar items (profiles or playbooks) that have accumulated over time.

## Inputs

A cluster of items, each with: `id`, `path`, `content`. All items are the same type (all profiles OR all playbooks).

## Decision

Output one of:

- `merge_all` — every item in the cluster collapses into a single merged entry. All cluster files will be deleted and one new merged file written.
- `merge_subset` — some items collapse, others remain distinct. Identify which IDs merge and which stay.
- `keep_all` — the cluster is not actually redundant on closer inspection; no changes.

### Contradiction handling

If items contradict, keep the most recent unless older items have strong corroborating signals. Explain the choice in `rationale`.

### Preservation rule

Preserve distinctions that are meaningfully different. Collapse only where content overlap is substantive. When in doubt, prefer `keep_all` or `merge_subset` over `merge_all`.

## Output schema

```json
{
  "action": "merge_all | merge_subset | keep_all",
  "merged_content": "string — required if action ∈ {merge_all, merge_subset}; the synthesized content body",
  "merged_slug": "string — required if action ∈ {merge_all, merge_subset}; kebab-case slug ≤48 chars",
  "ids_merged_in": ["string"],
  "ids_kept_separate": ["string"],
  "rationale": "string — always required; 2-3 sentences justifying the action"
}
```

For `merge_all`: `ids_merged_in` contains all cluster IDs, `ids_kept_separate` is empty.
For `merge_subset`: `ids_merged_in` + `ids_kept_separate` = full cluster IDs (disjoint).
For `keep_all`: `ids_merged_in` empty, `ids_kept_separate` = full cluster IDs.

## Cluster

{cluster}
