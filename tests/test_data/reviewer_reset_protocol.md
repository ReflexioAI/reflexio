# Reviewer iteration reset (2026-10-14)

The user approved this reset after the v1.3/v1.54 diagnostic. The default stays
**v1.3.0**. No new prompt, response schema, provider configuration, or production
change is selected here. This protocol governs new work; the older adoption
protocol and evaluation reports remain historical evidence, not instructions to
resume or replace a failed trial.

## Common policy

Optimize the existing contract: a lesson must improve an agent-controlled
decision in a bounded similar **future** task. Preserve supported reusable
instructions, corrections and complete procedures. A value or choice explicitly
limited to one task does not become a standing preference. Current-task-only
memory is a separate policy experiment, not a baseline reasoning failure.

Source scope, complete workflow, earliest trigger and truthful explanation are
part of correctness. Equivalent paraphrases are allowed. Literal quotation
membership is a mechanical property, not proof that an action is supported.

## What the previous diagnostic established

The actual v1.3/v1.54 comparison planned 18 returns. It stopped automatically
after 13 measured returns, one candidate length-limited return and four unbegun
returns. Six pairs completed. The candidate improved agreement with the selected
reason policy and preserved job scope in those pairs, but cost more. One baseline
rejection explicitly followed its one-task durability policy. Eligibility,
precedence, schema and sampling differed, so this was not an isolated test of
architecture. There is no qualified winner. Keep the original sources, grades,
failed trial and unknown failed-entry cost; do not resume or pool replacements.

## Implementation

Use the existing `scripts/evaluate_review_reason_codes.py`, not another evaluation
framework. Record the actual `apply_decisions` result, including saved notes and
provenance. Deep-copy the prepared input for each arm. A fresh checkpoint must
not overwrite an earlier trial. The CLI stops after any call/application error
or incomplete provider/cost observation, retaining that call's output and metrics.
Semantic failures continue the fixed schedule. Legacy in-memory callers retain
the original continue-on-error default for compatibility; reset callers must use
`stop_on_incomplete=True`. Legacy results do not qualify under this protocol.
Test first/second-arm stops, raw applied values, input
isolation and checkpoint preservation before paid calls.

## Development screen

`reviewer_reset_development.json` contains eight synthetic development cases:
six selected unchanged from `reviewer_reason_codes.json`, plus a matched
future-scoped instruction and overgeneralization pair. They cover positive
preferences/procedures, safe subtraction, scope and missing-capture rejection.
They do not constitute comprehensive domain coverage or an untouched holdout.

Independently inspect the source expectations before paid calls. The primary
judgment is whether the saved guidance, scope, trigger, rationale, evidence and
notes are correct. Classify harmful saved guidance, lost useful guidance and
unsupported explanations separately. Judge reason-code agreement separately;
a label-only disagreement is not loss of a useful lesson. Unscored reason codes
remain unscored. Optional diagnostic reason/notes must be truthful, but do not
extend a prose-field label prohibition to them after observing an output.

First characterize the baseline on this common-policy set. Existing v1.4 is a
small precedence-only change, but it already has recorded disagreements and
orders causality before internal status. Do not repeat it without inspecting its
existing evidence, or score it against a later precedence policy as though its
instructions were unchanged. No next candidate has been selected.

Use the existing CLI's baseline-only mode, a fresh private output path, MiniMax-M3,
two serial repeats per case, temperature 0.7, no fallback/transport retries and
the same 8192-token cap. This is 16 planned calls, not a candidate comparison.
The CLI enables strict stopping automatically. Freeze source/settings and
independently verify measurement before judging the actual saved results:

```bash
uv run python scripts/evaluate_review_reason_codes.py \
  --cases tests/test_data/reviewer_reset_development.json --baseline-only \
  --expected-model minimax/MiniMax-M3 --repeats 2 --max-tokens 8192 \
  --out /path/to/new-private-trial.json
```

Run only after exporting the intended provider key through an isolated loader;
never print secrets or raw customer data. The suite is synthetic. Preserve any
incomplete call, stop and record unbegun calls; do not resume or rerun for a more
favorable answer. A baseline screen establishes no candidate win or generality.

## One hypothesis per experiment

Before each fresh comparison record the demonstrated baseline failure, exact
change, expected benefit, likely regression, frozen source/candidate identities,
shared schema/model/sampling/token limit, repeat schedule and stop/promotion
criteria. A reason-policy change must be named explicitly. Use the same prepared
sources and counterbalance arm order globally, not just within each case.

Stop this line of optimization when it adds no public benefit or introduces
harmful guidance, core loss or unreliable completion. Do not add proof fields or
another stage without a measured benefit. A failed diagnostic does not justify
an automatic next prompt version. Semantic judgments require source inspection;
automated decision/code counts alone are not qualification.

## Promotion

A development improvement earns a broader comparison, not activation. Freeze
the candidate and numerical promotion criteria before opening untouched data.
Require preserved healthy controls, all mandatory source/safety assertions,
measured completion/cost/latency, broader coverage, actual configured model and
fallback validation, and full pipeline/staging validation. Repeated calls assess
consistency, not independent domain coverage. Never edit a holdout or threshold
to repair a failed qualification. Keep runtime tests, current-head Codex P0/P1
review, CI and a separately authorized release as adoption gates.
