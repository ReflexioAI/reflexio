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

## Approved task-scope and inference-profile experiment

Inactive v1.15 retains the v1.14 response/message contract while making selected
parameters conditional on their availability for the current task. A historical
selection is not a standing preference or proof of cross-session carry-over.
Acknowledgment occurrence alone establishes neither its necessity nor a benefit.
The initial comparison uses GLM-5.2 with explicit `high` reasoning and an 8192-token
limit for both v1.3 and v1.15, temperature 0.7, five serial repeats per frozen
priority case, alternating arms, no fallback or outer transport retries. A live
probe verified the explicit settings on the outbound request and a structured
response. Both prompt policy and reasoning profile differ from the earlier
default-profile v1.14 run; do not attribute cross-run differences to wording alone.

Per-call private metrics verify every initial and repair dispatch and retain
recovered failures, usage/cost and latency alongside final outcomes. Missing
observations invalidate measurement rather than counting as zero repairs.
Code checks and independent review precede paid comparison. Only after the
focused gates pass should the unchanged candidate/profile reach the broader
development and private sets, then the untouched holdout and staging pipeline.
The 68-case corpus, frozen oracles and held-out inputs remain unchanged. This
experiment configures evaluation calls only; production activation and any
reviewer-specific provider routing still require a separate reviewed change.

## Approved concise-rationale experiment

Inactive v1.16 preserves the v1.15 decision/evidence contract and task-local rules.
Revised rationales state only a concise, own-evidence basis for the retained rule.
Conversation narration is omitted from both rationale and review reason. Concise
grounded diagnostics belong in reason, which is persisted as survivor notes and
is included in semantic audit. Required workflow sequence remains in content/trigger. No decision oracle or corpus input is edited.
Repeat the same five private priority controls five times in both arms, serial and
alternating, GLM-5.2 high/enabled with 8192 tokens, temperature 0.7, no fallback or
outer transport retries. Stop on an unsupported or unverified saved clause, lost
healthy core, frozen-oracle mismatch, or invalid observation. Exact excerpts and
correct labels alone do not qualify the result. Broader development/private,
untouched holdout and staging gates follow only a successful focused comparison.

## Approved corrected-oracle and benefit-precedence experiment

Inactive v1.17 clarifies unsupported causal benefits (including prospective purpose
claims) versus speculative actions under the existing ordered taxonomy. It changes
no fatal gates, evidence contract, task-local guidance or inference defaults.
Every original candidate field contributes to defect classification. Inspect
content, trigger, rationale and persisted reason/notes for semantic grounding.

Private oracle v2 is preregistered for prospective measurement only: 21 original
source pools retained; one decision-ambiguous control quarantined. Five previously
scored labels are corrected and two previously unscored labels become explicit.
One remaining reason remains deliberately unscored. Both independent audits and
the original oracle are retained; no historical score or output is replaced.
The 68-case synthetic corpus and untouched holdout are byte-identical.

- Original oracle SHA-256: `0b74a51fef5abd5015b12191fd6c23b74cbd7e8ceec8b2b654ae80daf581ca16`.
- Independent audit SHA-256: `fa95312231a2d8349237e1d9244827480a94f220a2cdca6683ac326c95629c15`.
- Prospective oracle v2 SHA-256: `e97e913d4df0492090d7f3bf14351ab301ec3155855d99667fef2a952eff3fe4`.

Begin with the same five priority cases, five serial alternating repeats per arm,
GLM-5.2 high/enabled/8192, temperature 0.7, no seed override/fallback/outer retry.
The retail control now expects unsupported_causality before speculative; the other
four priority oracles are unchanged. Stop on oracle mismatch, invalid observation,
unsupported/ambiguous saved guidance or lost useful core. The unchanged candidate
can proceed to broader development/private, holdout and staging only after all
focused gates pass. This experiment does not qualify MiniMax or activate GLM.


### Preregistered v1.18 artifact-boundary experiment (2026-10-11)

Inactive v1.18 changes only the applicability boundary of `unseen_artifact`: name
the particular artifact/output and unavailable evidence for its claimed contents,
properties, reception or result; contents may be available while reception is not.
Missing evidence for a conversational action's benefit is insufficient to invoke
that code; unsupported causal effects remain rule 5. Genuine unavailable-artifact
claims retain priority. This is a hypothesis, not a proven cause of v1.17 failure.

First compare v1.3/v1.18 on unchanged development controls
`unseen-over-causality` and `causality-over-speculation`, plus separately frozen
`reviewer_artifact_boundary_cases.json` (visible report/unobserved reception),
three serial alternating repeats per arm per case (18 expected outputs). Require every candidate decision
and reason correct, complete measurements and independent semantic inspection.
The existing evaluator finishes a six-output case block before checking its
classification gate; fail stops before the next case/private phase. No holdout
calls or corpus edits. Same GLM-5.2 high/enabled/8192/.7 settings in both arms,
zero outer retries/fallbacks/seed override; internal repairs recorded.

Only after all three boundary controls clear, run the same frozen oracle-v2 five
private priority cases with five repeats per arm (50 expected outputs), alternating
serial arms and the existing fail-closed per-output gates. Preserve original
oracle and v1.17 stopped reports; no retrospective rescoring or label relaxation.
Oracle v2 SHA-256: `e97e913d4df0492090d7f3bf14351ab301ec3155855d99667fef2a952eff3fe4`.
Each driver pins inputs before calls and records source/driver/report hashes.
Broader coverage, unchanged holdout, production-provider compatibility and staging
pipeline remain subsequent gates. This experiment cannot activate a replacement.

Supplemental boundary file SHA-256: `a9cbd9ed7afd6899f8c11691e30147e4926e49aa76765401c9b6df54ff5e11aa`. Its oracle must be independently
reviewed before calls; the original corpus/holdout remain byte-identical.


### Preregistered v1.19 saved-support-basis experiment (2026-10-11)

Inactive v1.19 changes only the instructions for saved rationale and reason/notes.
Rationale states the source-backed instruction, correction, limitation, current
parameter or task prerequisite. Notes diagnose the removed claim and retained
rule. Neither narrates historical agent application/execution. Verified actions
still support content/trigger; these instructions must not remove supported
workflow actions or reject merely announced-but-explicitly-authorized procedures.

Before calls, independently review the three supplemental development oracles in
`reviewer_saved_basis_cases.json`: two announced-only setup cases in separate
domains, and a setup with an explicit successful action result. All require
revision preserving explicit recurring input collection/use while removing an
unsupported no-clarification prohibition. Their reason labels are deliberately
unscored: this phase isolates decision/core preservation and saved-field grounding,
not an exact reason-code boundary. Private oracle v2 still scores exact reasons.
The original 68-case corpus, holdout, oracle and stopped reports remain unchanged.
Supplemental fixture SHA-256:
`06a48ca1c081f0028f1d0a52cd5d1ae64cf55cfcf4122d32f87c94e9a6bf5a8d`.

Run three serial alternating repeats per arm per supplemental case against v1.3
(18 expected outputs), with equal GLM-5.2 high/enabled/8192/.7, no fallback, seed
override or outer retries. Check complete both-arm counts and measurements after
each six-output case block; stop before the next block/private phase on failure.
Require independently grounded content/trigger/rationale/reason-notes and exact
own-source excerpts, preservation of authorized future actions, and no execution
claim from an announcement or mere parameter supply.

Only after all three controls clear, compare the same five private oracle-v2
priority cases with five repeats per arm (50 expected outputs), now placing the
previously failing setup case FIRST, then retail, absence, current-task parameter,
and unseen artifact. Serial alternating arm order and per-output classification/
measurement gates remain; stop on unsafe/unverified saved guidance. Oracle v2 SHA:
`e97e913d4df0492090d7f3bf14351ab301ec3155855d99667fef2a952eff3fe4`.
Drivers pin source/inputs and boundary receipts before private calls. Broader
qualification, unchanged holdout, production-provider compatibility and staging
remain subsequent gates; no activation follows from this focused experiment.
