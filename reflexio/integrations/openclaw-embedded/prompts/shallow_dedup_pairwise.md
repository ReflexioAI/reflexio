---
active: true
description: "Pairwise shallow dedup decision for openclaw-embedded plugin"
changelog: "Initial (2026-04-16): new prompt, informed by Reflexio's profile_deduplication and playbook_deduplication prompts, simplified to strictly pairwise (candidate vs top-1 neighbor)"
variables:
  - candidate
  - neighbor
---

You decide whether a newly-extracted item should be merged with an existing one.

## Inputs

- **Candidate**: a newly-extracted profile or playbook.
- **Neighbor**: the single most-similar existing item (top-1 from memory_search).

## Decision

Output one of:

- `keep_both` — the two items cover distinct facts; keep both.
- `supersede_old` — the candidate is a strict replacement (e.g., new info contradicts old; user restated a preference more fully). Old file will be deleted.
- `merge` — the two items overlap but neither fully subsumes the other; synthesize a merged version. Old file will be deleted and merged version written.
- `drop_new` — the existing neighbor already covers what the candidate says; discard the candidate.

### Contradiction handling

If the two items describe the same topic but assert conflicting facts, prefer the newer-created one (`supersede_old`) unless the content indicates the older is more specific or authoritative.

## Output schema

```json
{
  "decision": "keep_both | supersede_old | merge | drop_new",
  "merged_content": "string — required if decision == merge; the synthesized content body",
  "merged_slug": "string — required if decision == merge; kebab-case, regex `^[a-z0-9][a-z0-9-]{0,47}$`, for the new file",
  "rationale": "string — always required; 1-2 sentences justifying the decision"
}
```

## Candidate

{candidate}

## Neighbor

{neighbor}
