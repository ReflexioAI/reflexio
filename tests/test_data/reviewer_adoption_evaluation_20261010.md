# Reviewer adoption qualification — 2026-10-10

Status: in progress. The active reviewer remains v1.3.0. No activation or
production change is justified by the completed measurements below.

## Frozen coverage

The public synthetic corpus contains 66 cases / 67 candidates: 38 development
cases and 28 held-out cases / 29 candidates, spanning 25 domain categories.
It includes the earlier 17 controls, 12 cross-domain manifest positives,
artifact-versus-absence boundaries with and without diagnostic wording, supported
workflows, subtraction, agent-controlled tool responses, evidence isolation,
duplication, and injected reviewer instructions. Semantic oracles were inspected
independently before model calls; the holdout has not been measured.

Private production reads captured 36 retained-source windows across three
domains. Of those, 34 resolve every cited interaction and request; two incomplete
windows are excluded. Independent inspection of original content, trigger,
rationale and source metadata identified 22 scoreable stored candidates.
Ambiguous examples are excluded rather than forced into a label. Six useful-core
controls span retail assistance, software QA and creative/educational assistance;
they require revisions, rather than acceptance of unsupported original clauses.
These are reconstructed retained-evidence windows, not recovered original
extraction pools or full conversations. Production text/configuration remain in
private local artifacts and are not committed.

## Completed development diagnosis

Both arms used MiniMax-M3, temperature 0.7, no fallback, identical inputs and
five repeats. The actual reviewer disables transport retries.

| Focused cued controls | Decisions | Reason codes | Call errors |
| --- | --- | --- | --- |
| v1.3.0 | 55/55 | 50/55 | 0 |
| v1.6.0 | 55/55 | 55/55 | 0 |

This focused set contains five positive controls and six artifact/absence
controls. All 25 v1.6 accepted positives preserved the supplied guidance; all
15 revisions removed unavailable-artifact claims and preserved the supported
preference. All 15 absence controls rejected. An independent review checked all
55 v1.6 outputs. A non-persisted retry explanation overclaimed success; it did
not change accepted playbook text.

The subsequent broader v1.6 development diagnosis was stopped after 107 outputs
when its preregistered label gate failed. For the existing unavailable-artifact
control, the first three v1.6 repeats chose unsupported_evidence,
absence_inference and speculative instead of unseen_artifact. All rejected,
but differing labels matter when a grounded core could otherwise be revised.
Checkpoints and failures are retained; this incomplete run is not an aggregate
qualification result. No holdout answers were used.

## Current candidate and remaining gates

Inactive v1.7 clarifies missing support for artifact contents versus using record
omissions as proof of absence in fact. Events, properties and conditions remain
covered by the fatal absence gate; internal-status proof of value remains
distinct. Subject ownership, decision order, revision boundaries and the output
contract remain unchanged. Independent policy review cleared development
evaluation, not activation. Fresh repeated measurements are running.

Before activation: finish development and inspect every saved survivor; pass the
unchanged candidate on independently classified private production windows and
the untouched holdout; validate the complete pipeline in staging. The isolated
replay bypasses scheduling, durable resume, consolidation and existing-playbook
search. Expert windows are excluded. Production runtime model/source identity
must be verified before relying on this MiniMax qualification.

The replay tool freezes effective org/project context and configured tools,
source/action/tool metadata and batch composition. It rejects incomplete source
windows, malformed extractor responses and failed pools; errors cannot count as
pruning or declines. Reports retain expected/processed/error counts, complete
revisions, citations and input/template hashes. Private artifacts use mode 0600.
