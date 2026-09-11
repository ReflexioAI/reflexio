# /reflexio/integrations/openclaw-embedded/plugin/prompts
Description: LLM prompt templates used by Flow C sub-agents and consolidation.

## Main Entry Points


- `profile_extraction.md` — extract durable user facts from a transcript
- `playbook_extraction.md` — extract procedural rules from correction+confirmation patterns
- `full_consolidation.md` — consolidate a cluster of similar items into individual facts

## Purpose


Supply extraction and consolidation instructions to the embedded plugin without a Reflexio server dependency.

## Architecture Pattern


The plugin ships one active version of each prompt as a flat Markdown asset, with upstream-style YAML metadata and variable substitution.

## Key Endpoints / Commands / Contracts


### Format


Each file is a `.md` asset with YAML frontmatter (matches Reflexio's
`server/prompt/prompt_bank/` convention). Unlike upstream's versioned layout
(`<name>/v<ver>.prompt.md`), we store prompts flat (`<name>.md`) since the
plugin ships atomically with one active version at a time.

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

## Requirements / Problems to Avoid


### Upstream sync


`profile_extraction.md` and `playbook_extraction.md` are ports of Reflexio's
prompt_bank entries. On upstream bumps, review the prompt diff against our
adapted versions. The corresponding upstream templates are under
`../../../../server/prompt/prompt_bank/`; preserve plugin-specific variables when syncing.
