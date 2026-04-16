# Openclaw-Embedded Prompts

LLM prompt templates used by Flow C sub-agents and the consolidation cron job.

## Files

- `profile_extraction.md` — extract durable user facts from a transcript
- `playbook_extraction.md` — extract procedural rules from correction+confirmation patterns
- `shallow_dedup_pairwise.md` — decide how to handle a new candidate vs its top-1 neighbor
- `full_consolidation.md` — consolidate a cluster of 2-10 similar items

## Format

Each file is a `.prompt.md` with YAML frontmatter (matches Reflexio's
`server/prompt/prompt_bank/` convention):

```yaml
---
active: true
description: "one-line description"
changelog: "what changed in this version"
variables:
  - var1
  - var2
---

prompt body, with {var1} and {var2} substitution points
```

## Upstream sync

`profile_extraction.md` and `playbook_extraction.md` are ports of Reflexio's
prompt_bank entries. When the upstream bumps a new major version, re-apply
the adaptations documented in `../references/porting-notes.md`.
