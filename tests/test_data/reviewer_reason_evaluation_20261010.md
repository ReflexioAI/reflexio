# Reviewer qualification, 2026-10-10 (#428)

The default remains **v1.3.0**. v1.5.0 and v1.6.0 are inactive experiments.
These results qualify synthetic examples, not the original production defect.

## Method

- Real `PlaybookCandidateReviewer.decide` calls, MiniMax `minimax/MiniMax-M3`,
  temperature 0.7, no fallback, Python 3.12.3, LiteLLM 1.103.1.
- Five repetitions per case and arm, alternating baseline/treatment call order.
  Two network workers processed disjoint case groups. No database writes.
- Independently inspected expected answers before measurement. Clarified two
  ambiguous inputs and added three boundary controls to the original 14 cases.
  Do not aggregate these measurements with the earlier 14-case experiment.
- Focused phase: ten cases, including all five accepted positive controls.
  Full phase: all 17 cases. The phases are separate measurements.
- v1.5 source: `b8fdd034f171c72a6a29bf5a97bec1b64c5f75ca`.
  v1.6 source: `8948c13485309974d9ac4aa351d745c57adb4ea7`.
  No measured templates or cases changed during either run.
- Full input SHA-256:
  `9494b8d4efb54647fcda198ac8a79d3d32551efb2eeb178e20eac2f3ff9cfbb0`.
  Focused input SHA-256:
  `4fc646d1280d79e494d810797779f15842f774c296f80587b32ef8ad63ae0513`.
  Both input hashes match between v1.5 and v1.6 measurements.
- Baseline template digest:
  `c6f03e36e2772775d17a147a4e8c697a49962199c4424bcf60cf881d6c8a04fc`.
  v1.6 template digest:
  `bc3009f57d1c5ec12d08c7a8d90e75dd70393214011d6f609575c97255467851`.

## Matched results

Decision and label columns count matches to the independently inspected oracle.
They do not measure the quality of rewritten content.

| Experiment | Arm | Decision matches | Label matches | Positive accepts |
| --- | --- | --- | --- | --- |
| v1.5 focused | v1.3 | 39/50 | 29/50 | 25/25 |
| v1.5 focused | v1.5 | 47/50 | 47/50 | 22/25 |
| v1.5 full, recovered | v1.3 | 73/85 | 59/85 | 25/25 |
| v1.5 full, recovered | v1.5 | 85/85 | 84/85 | 25/25 |
| v1.6 focused | v1.3 | 39/50 | 29/50 | 25/25 |
| v1.6 focused | v1.6 | 50/50 | 50/50 | 25/25 |
| v1.6 full | v1.3 | 75/85 | 63/85 | 25/25 |
| v1.6 full | v1.6 | 85/85 | 83/85 | 25/25 |

v1.5 dropped supported report delivery steps in three focused repetitions.
Its later full-run acceptance does not erase that regression. Manual inspection
also found invented benefits and reviewer-policy language in revision rationales.
v1.6 clarifies complete workflows and requires rationale grounded in the recorded
instruction. The focused v1.6 audit found no semantic concerns in its 50 outputs.
Independent inspection of all 85 full v1.6 outputs (30 revisions, 30 rejects,
25 accepts) found no unsupported guidance, lost supported workflow details or
evidence errors in surviving content.

Two full v1.6 outputs still mislabeled `unseen-over-causality`: zero-based repeats
0 and 2 used `absence_inference` and `unsupported_evidence`, respectively,
instead of `unseen_artifact`. Both correctly rejected the candidate. The prompt
therefore improved decisions in this sample without eliminating label variability.
The incorrect absence label matters beyond reporting: it forces rejection in
the runtime policy. Its possible effect on salvageable candidates remains
unmeasured here and needs a boundary control before activation.

## Transport failures and retained evidence

The initial v1.5 full run had nine TLS/connection failures (four baseline, five
treatment). They were recorded as errors, not reviewer decisions. All five
repetitions in both arms were rerun for each of the three affected cases;
successful original rows for those cases were replaced too. Recovery selected
cases by transport failure, not by answer quality. The initial report and logs
remain available alongside the recovered report. Both v1.6 phases had zero
transport errors and needed no replacement calls.

Raw reports, frozen inputs and logs are retained locally under
`~/.reflexio/evaluations/issue428-v15-20261010/` and
`~/.reflexio/evaluations/issue428-v16-20261010/`. Each full comparison contains
170 matched rows. The committed harness can reproduce the protocol with a
configured real provider; stochastic output need not reproduce these counts.

## Checks and activation boundary

At the measured v1.6 source: 39 focused tests passed; the complete affected
reviewer/prompt/script suite passed **694 tests, with no skips**, including the
optional tier. Ruff lint/format and Pyright passed. Local Codex review cleared
P0/P1 findings. Four installed-artifact CI jobs passed. CodeRabbit's draft skip
is not substantive review clearance.

Activation remains pending: qualify the unchanged candidate against the original
frozen production pools, healthy same-tenant windows and another domain using
the enterprise paired-review harness and normal production-read permissions.
Inspect actual revision content and all mismatches. Five repetitions over a
small synthetic corpus cannot establish production reliability or statistical
significance. No default prompt, production configuration or data was changed.
