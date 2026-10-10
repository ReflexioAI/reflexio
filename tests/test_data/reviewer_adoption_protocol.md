# Reviewer adoption protocol (#428)

The current default is v1.3.0. Measure v1.6.0 first; preserve its wording and
results. If development cases require another prompt version, measure that
version afresh. Do not activate a version by averaging away a positive-control
loss or an unsafe saved rule.

## Frozen coverage and scoring

`reviewer_generalization_cases.json` separates development and held-out cases.
It adds task-scoped unseen-artifact versus absence controls, supported multi-step
workflows, safe subtraction, agent-controlled responses to tool failures,
candidate-specific evidence isolation and injected instruction controls.
Existing cross-domain manifest positives permit accept or revise; their reason
labels are deliberately unscored. Some older manifest rules were narrowed to
remove unsupported scope. This corpus is synthetic, not production tenant data.

Oracle decisions, reason labels and semantic preservation/removal requirements
are independently inspected before paid calls. `preserve_any` lists alternative
complete useful cores, not fragments to mix into one compound learning.
These semantic requirements require manual inspection; string matching alone
does not establish grounding or safe generalization.

Both arms use MiniMax-M3, temperature 0.7, no fallback or transport retries, identical frozen inputs,
and five repetitions per case. Alternate arm order and retain hashes, raw
decisions, revision text, explanations and evidence IDs. Print decision and
label denominators separately: survival-only controls have no label oracle.
Transport errors invalidate the affected matched comparison. If recovery is
needed, rerun every repetition in both arms for the affected cases and retain
the failed report; never select reruns by model answer.

## Acceptance gates declared before measurement

- Known failure and fatal-versus-revisable boundary cases make the expected
  decisions in every repetition, including preservation of supported cores.
- The specific reason-code precedence under investigation appears correctly
  on its targeted boundary controls; a better aggregate score is insufficient.
- Every healthy control preserves its evidenced reusable guidance. No unexplained
  false rejection or lost supported recipient, timing, limit or workflow step.
- Every surviving output is supported by its own retained evidence. No invented
  action, benefit, authorization, negative constraint or borrowed candidate fact.
- Run the held-out split only after the final candidate is frozen. If its results
  cause a prompt edit, those examples become development data and a new untouched
  holdout is required. Report each failed control; do not hide losses in totals.
- Confirm the same gates on private production-derived windows, including
  historical problem cases, healthy same-tenant cores and an unrelated domain.
  Stored learnings are not automatically ground truth. Incomplete evidence
  windows cannot qualify a grounded survivor.

Production inputs, customer configuration and raw outputs stay in private local
artifacts, outside the public repository. Record non-sensitive coverage and
results plus the full hashes of the private evidence/config snapshots. Distinguish
reconstructed historical windows from the unavailable original candidate pool;
do not claim faithful original-defect reproduction without the original inputs.

## Adoption

Only a qualified, unchanged candidate may become the default in an activation
PR. Require affected tests, local Codex P0/P1 clearance and CI. Validate the
complete extraction/review path in staging, then use the managed prepared-release
workflow for production. Keep the prior prompt for a reviewed rollback. No schema
migration is expected for selecting a prompt version. Record production inference
model identity; qualification on MiniMax cannot justify a different runtime model.

## Approved evidence-first experiment

Inactive v1.14.0 deliberately changes the reviewer message and response contract;
it is not an isolated prompt-wording comparison. Both arms retain frozen inputs,
model, temperature, five repeats and alternating order. The candidate receives
trusted policy separately from untrusted source data and returns internal excerpts
before decisions. Every survivor needs bounded exact excerpts from its final
retained evidence; rejects retain none. Role and request-source metadata remain
visible and cannot be upgraded by quoting. Excerpts are retained only in owner-only
qualification reports, never saved as rationale or exposed publicly.

The unchanged semantic gates apply to content, trigger and rationale. Quote
validation does not establish safe generalization or prove the model's reasoning
order. Stop if unsafe guidance persists or a supported healthy core is lost.
Measure development and private cases before freezing the untouched holdout;
complete extraction/review/consolidation staging validation precedes activation.
