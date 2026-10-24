# /reflexio/server/services/shadow_comparison
Description: LLM judging of regular and shadow responses with Reflexio-relative outcomes.

## Main Entry Points


- `dispatcher.py` schedules assistant turns carrying shadow content from publish.
- `worker.py` owns bounded background execution and verdict persistence.
- `judge.py` owns prompt rendering and the LLM call for one interaction.
- `rubric.py` binds rendering and verdict cohorts to one frozen template identity.
- `outcome.py` owns pure position randomization and Reflexio-relative win/loss/tie derivation.

## Purpose


Produce independent comparison verdicts for evaluation reporting.

## Architecture Pattern


`judge.py` renders the prompt and invokes the model; `outcome.py` randomizes answer positions and converts the result back to a Reflexio-relative win/loss/tie.

`ShadowComparisonRubric` resolves the actual PromptManager version and the digest
of its unrendered template. Both verdict writes and current-series reads use its
`v<version>@sha256:<digest>` key. Rendering uses that frozen identity and refuses
template drift; user/session inputs do not enter the key. Existing explicit
PromptManager version overrides still choose the actual template.

The former `shadow_comparison_judge_prompt_version` setting never selected the
rendered prompt and is retired. Current schemas reject it; explicit saved-file
conversion validates and removes it without changing other settings. Hosts must
run their versioned upgrade before serving the strict reader. Historical verdict
rows remain unchanged: semver-only labels cannot establish which template
actually ran and do not mix into the new verified series. An unconfigured or
unrequested series does not invent a rubric identity.

## Requirements / Problems to Avoid


This package intentionally does not use `components/`: dispatch, worker execution, judging, and pure outcome logic already have focused files.
