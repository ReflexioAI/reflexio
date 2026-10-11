# Reviewer adoption qualification — 2026-10-10

Status: in progress. The active reviewer remains v1.3.0. No activation or
production change is justified by the completed measurements below.

## Receipt-bound replay and model identity diagnostic — 2026-10-11

### Fresh v1.21 MiniMax comparison: stopped, unqualified

A separately preregistered five-case, five-repeat comparison stopped after its
first pair. Both final calls reported MiniMax-M3, passed identity/timing/usage
measurement, and needed no repair. The candidate chose the expected `revise`
decision but reported `speculative` instead of frozen `unsupported_causality`.
Independent source-first review confirmed causal precedence for the original
unsupported blocking/delay claim; the oracle remains unchanged.

The candidate's saved fields contain no evidence-absence assertion, and its two
supporting excerpts are exact own-source passages. However, its condition for
reusing a recorded selection does not explicitly bind that selection to the
current task. One semantic auditor retained this as unverified scope; another
found an unsupported no-reasking restriction as well as the scope defect. Neither
granted clearance. The driver stopped on the classification mismatch
before requesting a semantic verdict. No later case, broader benchmark or
holdout was run; the failed trial cannot be resumed or selectively retried.

Baseline: 25.280 seconds, 3675 input/2764 output tokens, estimated $0.004384.
Candidate: 31.822 seconds, 5795 input/6557 output tokens, estimated $0.009576.
There are two complete reports, two provider requests, zero repairs, zero final
transport errors, zero measurement errors, and no pending call. The manifest
remains incomplete with its original classification gate error preserved.

- Enterprise source: `43f834b820113de79d586e742c768a5506363699`.
- OSS source: `c03b2951cbe953d07d23ff509879e69d5dcff50e`.
- Candidate prompt byte SHA-256: `669f36619fb0ca5190cb0cb309a75141c6d9924603e38a6f64c2de140cf4e711`.
- Protocol byte SHA-256: `0492b15d177ce11f7162de4cd5c12cc2c449c455581570d3314aeef7fe7f97e8`.
- Manifest byte SHA-256: `0a56262c329a200e1baf4f79dc5b5c993fb06e4200a8c745b8a075ea22035498`.
- Independent semantic audit byte SHA-256: `f8709884bb0b21a5c785a155635b0dfca4f6391c8a3f307f0072a61ae70210f7`.
- Independent diagnostic byte SHA-256: `78fe83a446fd7c44e112176bf7fa8029f0ebf77a60980d75cad23a064544288c`.

The synthetic MiniMax identity probe passed before this comparison. It verifies
the tested route's reported identity, not historical production reviewer serving
identity. Nine additional receipt-backed stored-learning controls were captured
read-only and independently audited before any control model calls. Neither
audit found a clean whole-candidate accept control; decision/reason boundaries
remain under prospective reconciliation. These samples do not establish healthy
production coverage. The candidate remains inactive and unqualified; default is
v1.3.0. The next correction must clarify causal-label precedence and task-bound
parameter reuse before a fresh comparison with unchanged labels.

The receipt-bound v1.20 comparison is unqualified. It stopped after 30 completed
outputs (15 candidate, 15 baseline) and one failed candidate call. That call
failed structured parsing after both initial and repair attempts reached 8192
output tokens. This suggests an output-budget problem but does not establish
its cause. Original response finish metadata was not captured.

All 15 candidate revisions preserved their supported cores. Twelve persisted
explanations cleared semantic review; three could imply detail absence across
the full retained conversation when that detail existed in supplemental
chronology. Independent adjudication retained all three as unverified. No
historical report or oracle was changed to turn the trial into a pass.

A separate prospective diagnostic completed two calls with stop finish reasons,
one per arm, without reproducing the structured-output failure. Both requested
`zai/glm-5.2` and reported response model `zai/glm-5.3`. Further diagnostic
calls were stopped because the routing equivalence is unverified; one in-flight
call was interrupted with incomplete observations. Earlier requested-model
labels do not establish the actual serving identity.

Inactive v1.21 removes presence/absence explanations in favor of claim-level
support judgments. It has no fresh qualification result yet. Evaluation now
requires returned identity observations for initial and repair calls, and
future trials must stop on an unverified semantic verdict before further calls.
Production evidence currently verifies the requested MiniMax-M3 reviewer route,
not the returned model identity. Broader coverage, untouched model holdout,
production-provider and staging qualification remain pending.

Private semantic audit byte SHA-256:
`364462ee595318d508b1b8a743874d16dfc5ffd0e88c39eaf4eb683da4242807`.
Independent note-scope adjudication byte SHA-256:
`8dbbe51a504e1a2784e17b50507d4090547c7dc8caabd9206f9bdf33d239e79c`.
Inputs and raw outputs remain in owner-only private storage.

## Frozen coverage

The original frozen public corpus contains 66 cases / 67 candidates: 38
development cases and 28 held-out cases / 29 candidates across 25 domain categories.
After private development diagnosis, two new synthetic development controls were
added: the current corpus has 68 cases / 69 candidates, 40 development cases and
the unchanged 28-case / 29-candidate holdout across 26 domain categories.
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
55 v1.6 outputs. A review-note explanation overclaimed success; accepted content, trigger and
rationale were unchanged.

The subsequent broader v1.6 development diagnosis was stopped after 107 outputs
when its preregistered label gate failed. For the existing unavailable-artifact
control, the first three v1.6 repeats chose unsupported_evidence,
absence_inference and speculative instead of unseen_artifact. All rejected,
but differing labels matter when a grounded core could otherwise be revised.
Checkpoints and failures are retained; this incomplete run is not an aggregate
qualification result. No holdout answers were used.

## Current candidate and remaining gates

Inactive v1.7 clarified missing support for artifact contents versus using record
omissions as proof of absence in fact. Events, properties and conditions remain
covered by the fatal absence gate; internal-status proof of value remains
distinct. Subject ownership, decision order, revision boundaries and the output
contract remain unchanged. Independent policy review cleared development
evaluation, not activation. V1.7 subsequently failed the workflow gate below.
Inactive v1.8 clarified requested-workflow atomicity but failed the artifact
label gate below. Inactive v1.9 consolidates the procedure into a shorter
prompt while preserving fatal gates, revisions and workflows. It failed the
persisted-rationale gate recorded below. Inactive v1.11 additionally enforces
subtraction-only revision and checks each negative clause for its own evidence. It failed the private gates below. Inactive v1.12 explicitly removes unsupported original branches and preserves
bounded task-specific prerequisites. It failed the private grounding gate below.
The current inactive candidate is v1.13, adding worked subtraction examples.
V1.10 failed the private semantic gate recorded below. V1.13 requires fresh repeated qualification; no previous version
qualifies it.

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

## Failed v1.7 development diagnosis

The run stopped after 91 saved outputs, with no call errors, when the supported
weekly reporting workflow failed its preservation gate in repeats 3 and 4.
One compound revision dropped totals-first formatting; the other dropped Friday
delivery to the legal recipient. V1.7 is not qualified for activation. Its saved
first five unavailable-artifact rejection controls chose unseen_artifact, but
that does not compensate for losing a healthy workflow. No holdout calls ran.

## Reproducible identities

The full source SHA records the launch checkout. Template digests identify selected
unrendered prompt content; corpus/case hashes identify inputs. Private raw
artifacts and snapshot manifests are retained outside source control.

- Frozen corpus byte SHA-256: `10514be06e673a20a09380977f795b497d534224da9f602606468719960a40de`.
- Completed cued v1.6 launch source: `180cd6e77500d9b713268082250aa6f4d5c22d65`.
- Completed cued case SHA-256: `00b27deba8eaadc41d38b1228033a3b912bff6837f1f8f27ef9d7d24d3e9574f`.
- Completed cued report byte SHA-256: `02295bd476f5968444b8c85ae88c63fe2387913113c830bfc44bd23bdb347dd8`.
- Reviewer v1.3.0 template content digest: `c6f03e36e2772775d17a147a4e8c697a49962199c4424bcf60cf881d6c8a04fc`.
- Reviewer v1.6.0 template content digest: `bc3009f57d1c5ec12d08c7a8d90e75dd70393214011d6f609575c97255467851`.
- Stopped v1.6 development launch source: `529e08f03f771a9a2efa7de477ab5374f417abe6`.
- Stopped v1.6 development group 0 case SHA-256: `d7f26ee33bc20060d55b86fc8e635f3f05e0fee20a09055ae21772ecb3904850`.
- Stopped v1.6 development group 1 case SHA-256: `22532f354f5af888c27eee117f89d6e2f921d17d25399dec313bc89209918c31`.
- Stopped v1.7 boundary launch source: `c0c85a1f829205293885f5ead317b977d75ff003`.
- Stopped v1.7 boundary group 0 case SHA-256: `a9b667e0bf956221561fca29a4392d0d32dcf26b848a9435e259d1930e6e14de`.
- Stopped v1.7 boundary group 1 case SHA-256: `f18f170de5fe27eb2ffc7940eb188288c2f04649f41f0e5929b068da6eed33ce`.
- Reviewer v1.7.0 template content digest: `2533b9c0143987213f1f565f12e056e267bdf740e5ef6e8960f65b99abaf0bcd`.

Private production snapshots are frozen but not yet qualified. The following
input hashes include serialized evidence, effective context/tools and original
content/trigger/rationale; no customer text is included here.

| Private case | Input SHA-256 |
| --- | --- |
| 01 | `5d50f9ea69bd676857f000cc454fff549c6005b20e185a3dad63e2673e994580` |
| 02 | `847d59d894f0c9b6ca747913fe1defa8097347f017bb4d7aef96aca878902579` |
| 03 | `fc8abc9c850b04bbd3a2e6cc735a501e776f744b2673db82ec06ca9f84fffb6a` |
| 04 | `75445f12adafe0ba3581b4317c81b216542ef7fd45001f433060d53bb5080d8a` |
| 05 | `2579663179406ddd31e0679a5e5d95bb6e885e81cb7f91bd44e41d0c95b4babe` |
| 06 | `a47de81a79c979c79dbc4e90e720d8afeca1c1447c41b491afd7b683f7e461c4` |
| 07 | `e569c1b219e135db08366e952f4ac23e57c599ea973895496b2a6b9c6dbb2f74` |
| 08 | `ffc66e550107f6a37f156628a3d8f9f20e152ed4e17f2da1bd3c0bb427bea396` |
| 09 | `560e80bc53e72f9c3f1ed914b125a5a51f562e0c8d9a6135cbbc662ab416387b` |
| 10 | `215135ccc0efd272faecd127a619f7af3c0c50f93d4b89ad9d22a5546f9a644b` |
| 11 | `d779688c89f256ab44fdc6057629f508422ef2214ff0cbac468c87b5835d7c82` |
| 12 | `a7077d12ba57c647a24a64a5956c9fd732d9074b880c561657894f36a663f004` |
| 13 | `0ac8fa97e9f8d25ff10bfbaaaaba188fb53237dc517c33551d67256d8d703905` |
| 14 | `aee4f280d71276a949be751cb9ca0e4181db3aa88d1f838aac62ff7235bbb250` |
| 15 | `06d621d6547d0435eebca71deed29d9372517f6f863561a0ece7d77a43d5376f` |
| 16 | `6095dd260ffb0ad13dc916a0d9de378825ab68188e6514c22de81b9867167541` |
| 17 | `c36de3ec11ebcb58279282c21c077deff067d4268522f5fd5f6453ca170318af` |
| 18 | `7aa428d8703c94d1ed28d05d3086d8618baf43f331355ad01905ff8699aa4591` |
| 19 | `a95635200193444840b322380775aa587de0fa79fa131bc686e9e52b05c127d2` |
| 20 | `da6859883aa3564f3abcd45256abe98ebeed70f896bc907c61463dfd0a47be4b` |
| 21 | `787e614648d6de919df9d5762512bc16b95a0acbfb4568f7cacf9ec1646afca2` |
| 22 | `33146a58a63bf0ede7bed5d5e77350f73959c77bbb8780a519dbf3fd88d1bf83` |

## Failed v1.8 development diagnosis

The targeted run stopped after 34 saved outputs with no call errors. The
weekly workflow accepted unchanged in all five repeats. The unavailable-artifact
control rejected in every repeat, but chose unsupported_evidence in repeats
0, 2 and 4 instead of unseen_artifact. This fails the declared specificity gate.
These are complete per-case repeats inside an incomplete broader run, not an
aggregate qualification claim. V1.8 remains inactive.

- V1.8 launch source: `478f49e70a579086da3a1fb08ce94d25abfd3c89`.
- V1.8 group 0 case SHA-256: `31b02e1db47ff1850122aa7ab0e7c59aba66b62229cf399bfa76f5efe80bfde4`.
- V1.8 group 1 case SHA-256: `3e3ca69b893a6e09cbaf1b9dc197dffc039ad5d58c2690d8e8a3aa8e91044835`.
- Reviewer v1.8 template content digest: `bf10fbceb3ee8c4d3e27a913d34351c3c7ed95cdd1cc4a20d8cb22f747ec6830`.

## Production inference scope

Read-only ECS inspection found production task revision 231 and release
`1f1c59f38e1e2f8f04d60255636c2e8a0102cb42`, with OSS gitlink
`512dbc8ec4b5d64c88973aee542828411c951773`. That source defaults generation
to MiniMax-M3; effective configuration for all three sampled orgs also resolves
to MiniMax-M3 with no explicit generation override. This verifies source/default
and configuration resolution, not a live inference trace. Production configures
a `zai/glm-5.2` fallback; the MiniMax-only runs do not qualify fallback behavior.
That gap must be addressed before a production activation claim.

## Completed v1.9 priority development checks

Before expanding development evaluation, v1.9 was paired with v1.3 on two
existing controls, five repetitions each. Both made all 10 expected decisions.
V1.9 selected all 10 expected reason codes; v1.3 selected 8/10. Neither arm had
call errors. Independent semantic inspection confirmed that all five v1.9
workflow accepts retained totals-first formatting, the legal recipient and
Friday cadence, with valid citations; all five artifact rejects retained no
unsupported rule. The artifact control explicitly says “unseen chart”: these
results do not establish performance on uncued claims.

The subsequent remaining-development diagnosis failed as recorded below.
Private-window, fallback, holdout and full-pipeline gates remain pending.

- Priority launch source: `b9c95ce5e429470a00164efe8c6c8778e0796ccb`.
- Priority cases SHA-256: `ccbfc01bb30bee5c4286e6c2eda67f7f0d07950178729ae1daa98853e93ebac1`.
- Priority report byte SHA-256: `478faa464014e712cfde7d8a6fb92803a1a4cc3955bb9a95ff44290fe6d07039`.
- Priority prompt identities (unrendered content digests):
  - 1.3.0: `{"active_version": "1.3.0", "prompt_id": "playbook_candidate_review", "template_content_digest": "c6f03e36e2772775d17a147a4e8c697a49962199c4424bcf60cf881d6c8a04fc"}`.
  - 1.9.0: `{"active_version": "1.9.0", "prompt_id": "playbook_candidate_review", "template_content_digest": "2d0a7fba4cd69be75652cdb61f0ff605eb3affecf2bc5cdcb674c2d2abcc5d94"}`.

## Failed v1.9 broader development diagnosis

The remaining-development run was stopped after 22 outputs, with no call errors.
Independent semantic inspection caught a revision whose content and trigger
passed but whose rationale retained the temporary `[C1-E1]` label and review
commentary about the removed artifact claim. Decision/reason scores did not
catch this defect. The persisted-prose guard previously rejected only turn labels;
it now also rejects bracketed candidate and reviewer evidence labels. Regression
tests demonstrated the missing guard before the fix. Inactive v1.10 separates
retained-core rationale from review diagnostics and requires fresh qualification.
The holdout remains untouched.

- Stopped v1.9 report byte SHA-256: `8003891358a52433baacd977a33dec81274e888eee973b9a6a471e63f5bfe26b`.

## Completed v1.10 priority development checks

Three existing development cases were paired for five repeats each: artifact
rejection, supported weekly delivery, and revision of a language preference with
an unsupported artifact claim. Both arms made all 15 expected decisions. V1.10
selected all 15 expected reason codes; v1.3 selected 8/15. Neither had call errors.
All five v1.10 localization revisions retained formal Spanish, removed artifact
claims, and used plain-language rationales supported by retained evidence. The
artifact cases contain diagnostic availability wording; uncued development,
private production, fallback, holdout and full-pipeline gates remain pending.

Both arms now use the strengthened persisted-prose guard. This compares prompt
versions under identical new validation; it is not a replay of production's
unchanged v1.3 prompt plus its previous validation implementation.

- Priority launch source: `04d9322ef939bb144ea9d9417a5bfc8829ff3c97`.
- Priority cases SHA-256: `844edb769ec6d595d7954912e4c17c1ed826cb602e4a00170481b17fa89d09bf`.
- Priority report byte SHA-256: `87d98401be520702470409f38488133105ff8bbcb001a8d8bc2f0e6f47db8326`.
- Reviewer v1.10 template content digest: `19cc47b7a23c35b82d31f526e3d586c490cacf7b2e8f0a3155636bc14226cd4c`.

## Failed v1.10 private development diagnosis

The private priority run was stopped after 24 saved outputs, with no call errors
before interruption (two additional checkpoints represent interrupted calls).
Independent inspection of eight treatment outputs found that a retail revision
added an unsupported fallback branch and QA revisions retained no-reconfirmation
prohibitions unsupported by the retained evidence. Grounded cores survived, but
full revised guidance was unsafe. Three retail repetitions also differed from
the preregistered speculative reason label. Oracles remain unchanged. This
incomplete run cannot qualify v1.10, despite its perfect focused synthetic score.
All private text stays outside the repository.

- Stopped private report byte SHA-256: `637ef2b0ac6910ef33de9450bf664cf52d8ba4f317ee9ed338240032ead5c961`.
- Expanded development corpus byte SHA-256: `998255ffba588c558d0e56f49f09e989f12a47fd444e0a24a6ad6de0a832a08d`.
- Unchanged held-out split canonical SHA-256: `87b0e112987d3e0d6b320b6d85fce52409b195b01c15ecc78ec0cfbaa2da6b34`.

The two new development oracles were independently inspected before v1.11
model calls. Full affected tests for the guard/v1.10 source:
712 passed with low-priority controls enabled; no skips. Activation remains gated.


## Failed v1.11 private development diagnosis

The run stopped after 24 saved outputs, with zero call errors before interruption
and two interrupted checkpoints. Independent inspection found an unsupported
original fallback retained in a revision, temporary shorthand evidence labels in
a saved rationale, and a false generic rejection of a useful task-specific
prerequisite procedure. Other revisions removed unsupported reasking prohibitions,
but this improvement does not compensate for the failures. Oracles and the
untouched holdout are unchanged. V1.11 remains inactive.

- Launch enterprise source: `4e6c637e8c8a8f97fa5cefd1e5ad00182a19c492`.
- Launch OSS source: `8b08982d64aa09d7f4e6df6bef64deb6df8318eb`.
- Reviewer v1.11 template content digest: `b4080db4abe31628a5e5831bcc5b41723776fcb96a5f874521da02475b426130`.
- Stopped report byte SHA-256: `bd3c125c1727f55a9cd282619c3f530c056f2ffc15b940d575b1cae9991cbaa5`.

The shared prose guard is extended to bracketed/parenthesized shorthand evidence
citations. Bare E-number product identifiers remain allowed; the guard recognizes
citation syntax and cannot determine the meaning of every bare identifier.
New inactive v1.12 keeps the same gates and requires fresh qualification.


## Failed v1.12 private development diagnosis

The run stopped after 14 saved outputs, with no call errors before interruption
and two interrupted checkpoints. Independent inspection of seven candidate
outputs found one revision retaining an unsupported generic escalation path in
both content and rationale after deleting its unsupported specific example.
All seven audited decisions preserved a useful core, but the invented surviving
action fails the grounding gate. Only three of those seven matched their frozen
reason labels; the mismatches are retained separately from semantic inspection.
This incomplete run cannot qualify v1.12. No holdout calls ran.

- Launch enterprise source: `8e0edb00af466f6e21b7d8faa352d3fb569f3141`.
- Launch OSS source: `10bfe3b622395970d7c5bf8cbc4b98ae7453bddb`.
- Stopped report byte SHA-256: `8f524b345677dcf0eefbfc94407aaa200602fa848c6ee63864a874c8ff1fa370`.

Inactive v1.13 adds cross-domain worked examples of removing unsupported
continuations and preserving task-specific prerequisites. Neither example is
customer text. It requires independent policy review and fresh measurement.


## Latest implementation validation

The full affected extraction/reviewer/evaluator/prompt suite passed 726 tests
with low-priority controls enabled and no skips after the shorthand citation
guard. Subsequent inactive v1.13 selection checks passed 25 tests; Ruff and
Pyright passed. Independent local review cleared the validation changes and
v1.13 policy for development evaluation. This is code/policy clearance, not
model qualification or activation approval.

## Failed v1.13 private development diagnosis

The run stopped after 17 saved outputs, with one baseline structured-output
repair error and two interrupted checkpoints. Independent inspection of all nine
completed candidate outputs found two decisive grounding failures: an invented
example answer format and unsupported benefit/incorrect acknowledgment claims.
All nine candidate decisions and reason labels matched their frozen oracles,
showing why those scores cannot substitute for inspecting saved guidance. Other
outputs preserved their useful cores; announced-versus-completed wording remains
a separate concern on some procedure rationales. No holdout calls ran.

- Launch enterprise source: `7241ed4f551af62c14d980ee3a3009481bf9bb35`.
- Launch OSS source: `a42e0c666ecd5b95e23c532bc475a70f1e7a69d5`.
- Reviewer v1.13 raw prompt byte SHA-256: `2193b326097573bbb4c5d9ec89f54023311010bf62953f607a1c8450c1773852`.
- Stopped report byte SHA-256: `468de15c42156e95af9caa22a66d1ab7777a2747d6c39d3336a3769c10ea9b79`.

## Independent grounding-check diagnostic

A private diagnostic tests four unsafe proposed outputs and two grounded controls,
five times each. Its checker receives proposed content/trigger/rationale, own
retained evidence with request/source/session metadata, and task/tool context;
the original candidate and reviewer explanation are omitted. Frozen expected
verdicts are used only for scoring, never supplied to the model. Returned clauses
must be verbatim substrings of the proposed field and are independently audited.
This is a development capability test, not an integrated reviewer or production
acceptance gate. Inputs and raw reports remain private.

MiniMax-M3 completed all 30 calls with zero errors. Both healthy controls passed
all five repetitions. Unsupported example format was detected 5/5; benefit or
incorrect acknowledgment 4/5; escalation 1/5; added negative restriction 0/5.
Thus 10/20 unsafe outputs passed the checker incorrectly. Only 6/20 unsafe outputs
had every preregistered defect detected across all affected fields; a single
correct unsafe verdict can still miss another defect. One escalation explanation
also overstated absence of alternatives as contradiction rather than missing
support. These results disqualify this diagnostic checker as a reliable gate.

- Frozen private case canonical SHA-256: `24b3cc4e59125773ed48488c8febe00fce411be4da1c85ddae95ac3da4f46811`.
- MiniMax driver byte SHA-256: `91fbd9ed5057f0690d887b2efc633c829045f4cb01db86097ea313e7059289f6`.
- MiniMax report byte SHA-256: `5e67782bdce0bf6ab4dae35e6d317d432aec18f63813b983b380fdd62055e256`.

No reviewer replacement or additional production checking stage is qualified.
The active prompt remains v1.3.0. The untouched holdout, broader private coverage,
fallback qualification and complete staging-pipeline gates remain pending.

The same diagnostic prompt, inputs, oracles, temperature and five-repeat schedule
were then tested with GLM-5.2, using isolated provider credentials. This run stopped
after 18 saved outputs and four client errors following provider rate-limit
failures. All 18 completed verdicts matched their expected answers; all 13
successful unsafe outputs identified their preregistered defects in every affected
field. Five grounded procedure outputs passed; the grounded prohibition control
was not reached. Additional acknowledgment/usefulness objections were separately
marked ambiguous, not treated as valid just because the unsafe verdict was right.
Incomplete coverage and provider failures prevent qualification; no selective
reruns were used.

- GLM diagnostic driver byte SHA-256: `2c568330f3634de7edeefdd8d31e51108080e147ec62172e8b67577eff9494b0`.
- Stopped GLM report byte SHA-256: `9e7fb8bd1222f6964c8b4c14298bfac123af681625e64b72a9ca8672dbe6fdee`.

A separate MiniMax diagnostic prompt makes grounding of future recommendations
and negative constraints explicit and requires detection of every defect in all
three fields. It uses the same six frozen development cases and was measured below. Because these cases informed the new wording, success would be an
in-sample capability result, not generalization or activation clearance.

The stricter MiniMax diagnostic completed all 30 calls without errors. It detected
all 20 unsafe outputs, but incorrectly rejected two of ten healthy controls
(both on a valid illustrative trigger category). It found every preregistered
defect in 19/20 unsafe outputs, missing the invented rationale in one example
format repetition. An additional unsupported session-adjacency claim was missed
in every negative-restriction repetition, and one explanation still overstated
internal status as proof that alternatives were unavailable. This diagnostic
trades false negatives for healthy-control suppression and is not qualified.
The preregistered secondary synthetic diagnostic remains unmeasured.

- Revised diagnostic prompt SHA-256: `aa6e9f890fed1b583cc3c1ba12537e6f79ab13ebb78cc4c3b266a5991610f2af`.
- Revised diagnostic driver byte SHA-256: `91739936cdcebdc2341e1fe485ea1153112bb262849d271baa607b4897a7300e`.
- Revised diagnostic report byte SHA-256: `d893d59a83a716be4a75dff63bcd7f8e0baf955e5a8b63f273234aa1a34f8c76`.

## Failed lower-temperature replacement diagnostic

A new diagnostic compares the current v1.3/temperature-0.7 configuration with
v1.13/temperature-0.0 on the same five private priority cases, with five planned
repetitions and alternating arm order. This compares a declared replacement
configuration, not an isolated causal temperature effect. Real client construction
changes temperature only; generation, parsing and validation remain unpatched.
The seed override is asserted absent before calls so it cannot silently force
both arms to temperature zero.

The run stopped after seven saved outputs (four baseline, three candidate), zero
call errors and two interrupted checkpoints. The first candidate retail revision
still retained an unsupported emotional-benefit claim; the next retail revision
and the inspected QA revision were grounded. These results fail the semantic gate.
Two retail outputs differed despite temperature zero; determinism is not assumed.
Neither a lower temperature nor this diagnostic qualifies a replacement.

- Diagnostic driver byte SHA-256: `da2a0c1311ed1bc2fca8e8390e73e3811210b64360af8cc2c39f883b6891b84a`.
- Stopped report byte SHA-256: `d18cab9fab7370f7154213c8a323aa6fcdc8c4e785a810cab160b37064e64eb3`.

## Incomplete serial GLM reviewer diagnostic

A separate diagnostic calls the actual reviewer with GLM-5.2 for both v1.3 and
v1.13, temperature 0.7, alternating arm order and five planned repetitions over
the same five priority cases. Calls run serially with isolated credentials,
fallback disabled and no outer transport retries. Generation, parsing and
validation remain unpatched; the seed override is asserted absent before calls.
This is a provider-specific diagnostic, not qualification of MiniMax primary
behavior or proof of production capacity.

The run stopped after 15 saved outputs: eight candidate and seven baseline,
zero call errors and one interrupted checkpoint, out of 50 planned outputs.
All eight candidate decisions and reason codes matched the frozen oracles and
used valid own evidence references. Six candidate revisions were clearly
grounded. Two persisted rationales remain ambiguous under the all-clauses gate:
one retail rationale can imply a further user report after the final assistant
recommendation; one procedure rationale says the agent proceeded with choices
where the evidence establishes an announcement of work, not verified execution.
The procedure's conditional source choice is supported by retained evidence;
independent adjudication did not substantiate an invented choice or durable
user preference. Neither ambiguity is recorded as an unequivocally invented
event, and neither output is counted as verified safe.

The other priority cases were not reached. Zero errors in this serial run do not
establish that serial execution resolves the earlier rate limits. Broader
development, private evidence, untouched holdout and full-pipeline gates remain
pending. The active reviewer remains v1.3.0; no model or checker is adopted.

- Launch enterprise source: `102ebba1c02918ac51bd4b0911002954ead6c123`.
- Launch OSS source: `959123d9c4b87df0b52302e7c853cddc02c55c42`.
- Diagnostic driver byte SHA-256: `a04391266c28a43fdecdfa46228b1fbb003b5189dfa88da7c27b62e22f93ad27`.
- Stopped report byte SHA-256: `353eeb26800f0677194735bf4b8fecc250807e42a97b390515e9f5e09988123a`.

## Evidence-first contract: MiniMax failure

Inactive v1.14.0 separates static trusted policy from JSON conversation data and
requires bounded internal supporting excerpts before decisions. The server checks
exact own-source substrings and final retained evidence membership; role and source
metadata remain visible. Supporting-excerpt objects are not copied into saved playbook fields. This
compares a response/message contract change against v1.3, not isolated wording.

The implementation passed 203 affected extraction/reviewer/prompt tests; 75 focused
checks passed after strengthening version-selection assertions, and 17 enterprise
replay tests passed. The own-excerpt regression failed under deliberate removal of
its guard; source was restored with a verified byte hash. Ruff, type checks and
independent correctness/privacy review cleared the implementation. Two P2 findings
(evaluator selection and report permissions) were fixed and re-reviewed. These are
implementation gates, not semantic qualification.

The real MiniMax comparison used the same five private priority cases, temperature
0.7, five planned repeats and alternating arms, serial calls, no fallback and no
outer transport retries. It stopped after seven outputs (four candidate, three
baseline), zero errors and one interrupted checkpoint, out of 50 planned outputs.
All four candidate decisions/codes matched their frozen oracles; quotes and final
own references were valid. Two revisions were clearly grounded. One retained an
observed acknowledgment clause whose reusable necessity remains unverified, not a
fabricated event. Another decisively retained an unsupported emotional-benefit
claim in saved content despite accurate quotes. Its rationale also overstated
support for acknowledgment as reusable guidance. Quote validation cannot certify
safe generalization. The remaining cases were not reached; v1.14 is not qualified
on MiniMax. No holdout calls or default/model activation occurred.

- Launch enterprise source: `75d3f57281572fc6226a96ee56c6aaa53f397b63`.
- Launch OSS source: `6f38d6c421270c767e49bb03ab245ab68fce460b`.
- Diagnostic driver byte SHA-256: `7745f0d3edc83664173cbe22e2542875f67796d273e5e373e0f2184cabe5331e`.
- Stopped report byte SHA-256: `216f757a41a98cd7d6af8b49d230c47d03cf8b09fe1afb843234c773e94cdd3c`.

## Evidence-first contract: incomplete GLM failure

The same frozen v1.14 contract was compared with v1.3 using GLM-5.2,
temperature 0.7, serial alternating arms and five planned repeats on each of
the five priority cases. No fallback, seed override or outer transport retry
was used. Launch code, prompts and input hashes remained unchanged through
documentation-only commits. This is a provider-specific contract diagnostic,
not qualification of the MiniMax primary or an exact historical replay.

The run stopped after 34 final outputs: 17 candidate and 17 baseline, zero
final call errors and one interrupted empty checkpoint, out of 50 planned
outputs. Candidate decisions and reason codes each matched 16/17 frozen
oracles; baseline decisions matched 12/17 and codes 1/17. These partial scores
do not override the semantic gates. Independent inspection cleared all five
retail revisions, five procedure revisions and five fatal absence rejections.
The first parameter revision remained ambiguous/unverified: a dynamic
configured-value reading is possible, but saved content and rationale can
promote a single task selection into a standing value. The next repeat rejected
the entire parameter candidate and lost the preregistered useful selected-value
core. It failed the healthy-control gate; the oracle was not edited to fit the
rejection. This is a survival failure, not an invented saved fact. The unseen
artifact case was not reached.

All 17 candidate input hashes, retained references and 40 exact own-source
excerpts were independently verified; rejected candidates retained no excerpts,
references or revisions. Two absence-rejection explanations contained imprecise
chronology/source descriptions. They generated no surviving rule, but remain
diagnostic quality findings rather than being described as wholly grounded.

Both arms use the existing client's one same-model structured-response repair
inside a logical reviewer call. Thus logical output counts are not provider
completion counts. The candidate needed five successful repairs (one validation
failure and four parse failures following 8192-token responses); the baseline
needed one successful parse repair following an 8192-token response. Logs record
22 completed candidate provider requests and 18 baseline requests, plus one
interrupted candidate request. Across logged completed requests, estimated costs
were $0.532 and $0.360, and summed provider durations were 1397 and 948 seconds,
respectively. These are small-sample diagnostic totals, exclude the interrupted
request and other overhead, and do not establish production capacity. Replay
JSON retains final outcomes but does not separately expose repair attempts;
the private provider log is preserved alongside the stopped bundle.

Neither provider qualifies v1.14. Broader development, private coverage,
untouched holdout and full staging-pipeline gates remain pending. The default
reviewer and production provider remain unchanged; no activation occurred.

- Launch enterprise source: `75d3f57281572fc6226a96ee56c6aaa53f397b63`.
- Launch OSS source: `6f38d6c421270c767e49bb03ab245ab68fce460b`.
- Diagnostic driver byte SHA-256: `7745f0d3edc83664173cbe22e2542875f67796d273e5e373e0f2184cabe5331e`.
- Stopped report byte SHA-256: `e89959c4ce5bb25f5795cb488e83eac7cb8b489b157bfd3638384e0420a41151`.
- Provider log byte SHA-256: `81290dd1a8f8fe9352ef1c2c7b5b3cde6445d0211b7b3d15410ecc4a94cc2278`.

## Task-local policy and explicit reasoning: incomplete v1.15 diagnostic

Inactive v1.15 narrows configured-value guidance to the current task, forbids
carrying a historical numeric choice into later sessions, and separates observed
acknowledgment from its necessity or claimed benefit. The comparison additionally
sets GLM-5.2 `high` reasoning and 8192 output tokens for both arms. A real outbound
HTTP-body/structured-response probe verified this inference path. This compound
policy/profile experiment cannot isolate a wording effect from the prior
v1.14 default-profile run. Production's active v1.3 and provider remain unchanged.

The same five frozen priority cases were scheduled for five serial alternating
repeats (50 outputs), with temperature 0.7, no seed override, no fallback and no
outer transport retry. The run stopped on the first case after three final
outputs: two candidate and one baseline. Both candidate decisions/reason codes
matched the original oracle; baseline decision matched but its reason code did
not. Final reports contain zero reviewer errors and zero measurement errors.
One additional baseline request was interrupted, produced no decision and has an
incomplete-observation checkpoint. It is not a successful or failed model answer.

Two independent semantic audits found the first candidate's content and trigger
preserved the useful correction and removed the unsupported continuation,
acknowledgment requirement and emotional-benefit claim. Its saved rationale,
however, remained ambiguous about chronology: it can imply a further user report
after the final recommendation, where none is retained. A narrower reading is
possible, so this is an unverified survivor rather than a definitive fabricated
event. The second candidate cleared semantic audit. Thus one candidate is grounded,
one is unverified and none is classified as definitively unsafe. All eight exact
supporting excerpts and retained references match their own frozen source units;
provenance did not resolve the first rationale's ambiguity. It cannot count as a
safe qualification result; no frozen oracle was changed to fit the output.
The experiment remains unqualified, and the other four priority cases, broader
development/private gates, untouched holdout and staging pipeline were not run.

Every completed call verified its initial and repair settings before dispatch
and retained complete lifecycle, timing, usage and cost observations. The two
candidate outputs consumed three provider requests, including one successful
parse repair; one initial response used 8192 output tokens. Token-limit equality
alone does not prove truncation. Recorded candidate logical wall time totaled
193.080 seconds and estimated cost $0.080027; the single baseline output used one
request, no repair, 32.436 seconds and $0.016721. These unequal small samples are
not a general latency/cost comparison, and exclude the interrupted request and
any unobserved charges. Repair overhead is retained rather than hidden behind
zero final-call errors.

Implementation gates: 501 affected OSS tests and 19 enterprise replay tests
passed, plus Ruff and Pyright. Independent correctness, privacy, verification
and policy/docs review cleared P0/P1 and reverified two metrics P2 fixes: default
repair token-limit drift and falsely complete observations without timing/usage.
The four OSS installed-artifact checks passed at the implementation head.
These checks establish implementation readiness, not semantic qualification.
The 68-case corpus and 28-case holdout hashes remain unchanged and the holdout
has not been called. Customer evidence and raw results remain owner-only.

- Launch enterprise source: `0008e83830a451fa786dead9307d65a4be95b149`.
- Launch OSS source: `ffe5470c8de8af707d73a240ca9587215496538e`.
- Driver byte SHA-256: `8f65c1bc1ec3a3a8ac7f696c4645cb4670b7ab67632bcb882a80a5baefe1fdda`.
- Probe receipt byte SHA-256: `e7dad71a3f417775ab0c415c2d0b575d0cc009dd0015fe9b0dede9d10d9a2247`.
- Stopped bundle byte SHA-256: `2eda7d0bd8e3a62dd501ab81ecafead40997449c20c47e3253c72771b75c37d4`.
- Private log byte SHA-256: `569ab271ec6235c87c384264b78268793b582de8f9d1dfe62d7e24fe6d4d00f1`.

## Concise rationale/notes: v1.16 stopped on a frozen-label disagreement

Inactive v1.16 keeps the v1.15 evidence contract, fatal gates, useful-core rules
and explicit GLM high/enabled/8192 profile. Revised rationales should give only a
concise own-evidence basis. Conversation retelling is omitted from rationale and
review reason: reason is persisted in survivor notes, so qualification audits
notes alongside content, trigger and rationale. Required workflow sequence remains
in content/trigger. No original oracle, source pool or held-out input was edited.

The five priority controls again scheduled five serial alternating repeats per
arm, temperature 0.7, no fallback, seed override or outer transport retry. The
runner stopped automatically on the fourth retail candidate's reason label,
after seven final outputs (four candidate, three baseline; three complete pairs).
It returned `unsupported_causality`, while the frozen oracle expected `speculative`.
All four candidate decisions matched; frozen reason labels matched 3/4. There
were zero final reviewer errors, measurement errors, repairs or interruptions.

Independent semantic inspection cleared all four candidate outputs, including
persisted notes. They retained the supported failed-channel correction and
removed the unsupported continuation, acknowledgment requirement and benefit.
No unsupported or ambiguous additional follow-up event was found. All 15 exact
own-source excerpts and final retained references were verified. Some reasons
and rationales still counted or narrated supported events despite the concise
wording. This is an instruction-following limitation, not an invented event.

Two independent original-first taxonomy reviews found that the frozen expected
label misses a higher-priority defect already present in the original candidate:
a claimed emotional benefit of acknowledgment that the source does not establish.
Under unchanged precedence, this is `unsupported_causality`, which comes before
the invented escalation extension's `speculative` defect. The fourth output
followed this priority; the first three did not. Preserve the original stopped
run and its 3/4 legacy score rather than change its oracle or retrospectively
call it a pass. Any corrected label needs independent, versioned preregistration
before a new comparison. This exposes both an oracle defect and inconsistent
model prioritization; it does not establish unsafe surviving guidance.

The four candidate calls used four provider requests, 263.236 seconds of recorded
logical wall time and estimated cost $0.118691. The three baseline calls used three
requests, 94.951 seconds and $0.042922. Every completed call retained full timing,
usage and cost observations and verified its frozen dispatch settings. Unequal
small samples do not qualify general latency/cost performance. The other four
priority cases, broader development/private coverage, untouched holdout and
staging pipeline were not reached. v1.16 remains inactive and unqualified.

Implementation gates passed 503 affected OSS tests, 19 enterprise replay tests,
16 overlapping version/policy checks after final wording, Ruff and Pyright (zero
errors/warnings). Local correctness, architecture, privacy, verification and
policy/docs review cleared P0/P1. Both wording findings were fixed and reviewed
before calls. All four OSS installed-artifact checks passed at the implementation
head. The 68-case corpus and 28-case holdout hashes remain unchanged.

- Launch enterprise source: `c8e467bbba080036b38a432b4ca4c679ee954012`.
- Launch OSS source: `be8e3ae983afb7dc313a5d627507c62469231542`.
- Prompt byte SHA-256: `d6576f0a277d683df1dc5e7837bfadf12627847ac11433d04ab8ffdf617ac532`.
- Driver byte SHA-256: `9bd0f9f97fad4bcd20b20ffce3361251eca73ac398178b32f02e07e64bee3e3d`.
- Stopped bundle byte SHA-256: `2ef0aff9fb3afd95885fc3af74fafd2df520aeaf824b127818e716aac470a602`.
- Private log byte SHA-256: `5c935fb697c3959e60e7603f072c8b9d8b0c0f290ae5359a7ffc2fea1b0c0d6b`.

### Original-only private oracle audit before another measurement

A subsequent read-only review inspected all 22 original candidates and their own
retained sources, without changing pools or rerunning models. A second reviewer
independently confirmed all proposed changes and the decision-ambiguous case from
original inputs, rather than choosing labels to match measured answers.

Five previously scored labels need correction under unchanged precedence:
retail 42/118 and UI 48/580 include unsupported causal benefits/effects;
43/190 is the fatal system-output subject case; 48/712 and 48/5140 make unavailable
artifact value/compliance claims that precede the generic internal-status defect.
Two previously reason-unscored cases have proposed more-specific labels: 43/193
uses omitted delivery as evidence of non-delivery; 48/5142 can preserve a task
prerequisite while removing unsupported realized-artifact properties. These two
were not previously wrong scored labels. Case 42/116 has an unresolved useful-core
threshold and should not be a confident strict decision control before resolution.
Case 42/121 retains a supported rejection with its reason code deliberately
unscored; no new exact label was silently assigned.

The original oracle remains byte-identical. The audit is a proposal, not a new
qualification oracle: separately version/review/preregister any corrected labels
and ambiguity treatment before another run. The stopped v1.16 comparison remains
incomplete and unqualified. Previous semantic failures are not erased by finding
these label defects; historical raw outputs and original scores remain preserved.

- Original oracle byte SHA-256: `0b74a51fef5abd5015b12191fd6c23b74cbd7e8ceec8b2b654ae80daf581ca16`.
- Private audit proposal byte SHA-256: `fa95312231a2d8349237e1d9244827480a94f220a2cdca6683ac326c95629c15`.


### v1.17 prospective oracle v2: precedence clarification stopped on classification

Before calls, independently reviewed oracle v2 retained 21 unchanged-input private
controls, quarantined one unresolved decision and scored 20 reasons. Five previously
scored labels were corrected and two previously unscored labels were added; one
reason remains deliberately unscored. Original oracle, prior scores and outputs
remain unchanged. The inactive prompt clarifies that unsupported claimed future
benefits are causal defects before generic speculative actions, while more-specific
artifact/status defects retain their existing priority.

The preregistered serial GLM-5.2 high/enabled/8192 comparison stopped automatically
on its second candidate output: `unseen_artifact` disagreed with expected
`unsupported_causality`. Three final outputs completed (two candidate, one baseline),
with one complete pair. Candidate decisions matched 2/2 and reasons 1/2. No repair,
final reviewer error, measurement error or manual interruption occurred. This is
a classification-gate failure, not a semantic failure in the surviving guidance:
independent audits cleared both saved content/trigger/rationale/reason-notes and
verified eight exact own-source excerpts and all retained references. A second
original-first audit confirmed that this candidate contains an unsupported causal
benefit and no unavailable-artifact claim; the preregistered reason remains correct. The second
rationale still narrates supported events despite the concise-basis instruction.

Candidate calls used two provider requests, 138.562 seconds of logical wall time
and estimated cost $0.056544; the baseline used one request, 32.768 seconds and
$0.014547. These small unequal samples do not establish general performance.
The four other priority controls, broader development/private qualification,
untouched holdout and staging pipeline were not reached. v1.17 is inactive and
unqualified; do not change the frozen oracle to accommodate the measured answer.

Implementation gates passed 505 affected OSS tests and 19 enterprise replay tests,
Ruff and Pyright (zero errors/warnings). Scoped local reviews cleared P0/P1 and
verified the pre-call frozen-input hash guard fix before launch. The 68-case
synthetic corpus and 28-case holdout hashes remain unchanged.

- Launch enterprise source: `3e505451b8100e8fe069a78f726cab6db9665b6a`.
- Launch OSS source: `19b3921f05608e75e75dfe321d4907b0f2d9f773`.
- Prompt byte SHA-256: `ae9ac3f50b31fb5d8f0ddc50c2c7435808be488acb2d15b185c21d51a05ad61d`.
- Prospective oracle v2 byte SHA-256: `e97e913d4df0492090d7f3bf14351ab301ec3155855d99667fef2a952eff3fe4`.
- Driver byte SHA-256: `3b4409f3007b5a086e187df4a439146360ca5c4b1ce00161fa741b00f9ae56d5`.
- Manifest byte SHA-256: `64520998020f4762219821d0ca5d9aa30038d6c6888b20633a2d442493d991f8`.
- Stopped bundle byte SHA-256: `e845f19a608e0a856f5129ff56a11374eed4d4023692614ff3cf184ec379452e`.
- Private log byte SHA-256: `21ecfab265cbe2b6450b2bdb399ad6ef2950c551f57aeb314823de2c1522d603`.


### v1.18 artifact boundary: development passes, private semantic gate stops

The preregistered inactive change requires an identified artifact/output and
unavailable evidence for its claimed contents, properties, reception or result.
Available contents do not establish reception. Missing general conversational
benefits do not by themselves create an artifact claim. No fatal/useful-core or
inference-profile changes were made. Two independent original-first reviews
confirmed the new supplemental oracle before calls; the original corpus and
holdout stayed byte-identical.

All three boundary controls completed three paired serial repeats (18 outputs).
Candidate decisions and reason codes matched 9/9; baseline decisions matched 9/9
and reason codes 6/9. Independent semantic audit cleared all nine candidates:
six correct rejects and three grounded revisions preserving the explicitly
requested recurring format while removing invented reception/effect claims.
Three retained own-source excerpts were exact. All 18 calls/dispatches had complete
observations, zero repairs, final errors or measurement errors. Candidate calls
used 194.692 logical seconds and estimated $0.076731; baseline 124.724 seconds
and $0.041953. These small diagnostic samples do not qualify general performance.

After this gate passed, the same pinned source ran the frozen private five-control
comparison against oracle v2. It was deliberately interrupted at the semantic
gate after 15 final outputs (eight candidate, seven baseline), seven complete
pairs. Candidate decisions and labels matched 8/8. Retail cleared all five repeats;
setup prerequisites cleared two. The third setup revision's rationale and notes
can describe supplied choices as actually used, while the own record establishes
an announcement, not verified execution. Two independent source-first reviewers
classified this as UNVERIFIED under the strict gate: a narrow interpretation is
possible, but cannot be adopted as a grounded pass. Content/trigger preserve the
supported prerequisite core. This is not a reason-code or definitive fabricated-
completion finding. Preserve the stopped run; do not resume on a charitable
interpretation or change the frozen oracle.

Private candidate grounding: seven clear, one unverified, zero definitive unsafe
findings; 30 exact excerpts and all final references verified. The 15 completed
logical calls recorded 16 dispatches, one recovered parse repair, zero final
reviewer errors and zero measurement errors. The recovered call's first attempt
reported 8192 output tokens; token-limit equality alone does not prove truncation.
The interrupted next baseline call retained its own incomplete checkpoint: one
started dispatch, 67.505 logical seconds, no final output/usage/cost and an
`IncompleteProviderObservation`. This is additional to the completed-call totals;
its unknown cost is not zero. The stopped manifest is incomplete/interrupted.

Eight completed candidate calls used nine requests, 660.353 logical seconds and
estimated $0.240126; seven completed baseline calls used seven requests, 317.436
seconds and $0.107144. Independent baseline retail audit found one grounded
survivor, three unsupported requirement survivors and one unverified outcome
survivor across five repeats; decisions 5/5, labels 0/5. Candidate retail was
5/5 grounded, decisions and labels. This supports a narrow comparative improvement,
not generalization. Remaining absence, task-parameter and unseen-artifact private
controls, broader coverage, untouched holdout, production-provider compatibility
and staging were not reached. v1.18 remains inactive and unqualified.

Implementation passed 507 affected OSS tests, 19 enterprise replay tests and 20
overlapping final version/policy checks; Ruff/Pyright passed (zero type errors or
warnings). Scoped local P0/P1 reviews cleared and reverified the artifact-reception
boundary and complete-both-arm runner gate before source freeze/calls.

- Launch enterprise source: `85ac41f35a187d1c7a303911c4f95d41f4adcace`.
- Launch OSS source: `fa57ceb429d6af6a2fe7afd320130c55ae7e06e0`.
- Prompt byte SHA-256: `32e927ce19453826006de1c9d2707d93836b15532e36d395aab799fe8b8e0107`.
- Supplemental fixture byte SHA-256: `a9cbd9ed7afd6899f8c11691e30147e4926e49aa76765401c9b6df54ff5e11aa`.
- Development driver byte SHA-256: `b4cef60f7e79aef5f3ddd94517c4250e0f5406ca15d46faba11f97144a03af52`.
- Private driver byte SHA-256: `3cf812e6991aecbc8f1f622f12d0a91e76e0ace0c43770b168b997b81efec40d`.
- Development manifest byte SHA-256: `7dddc432cf5dab91b2e2035ac3ead3d311fee0119ac67fbc426a5b67408721c3`.
- Development bundle byte SHA-256: `5688e4445937b0d1593dd04da374e049b315b6dd780f4a11a908cd0adae7f156`.
- Development log byte SHA-256: `eba85f3df08824b76500534fe3c4d788626f0508e0ff0ffdaedfdb09432ddd3c`.
- Private manifest byte SHA-256: `485bcf7433b176f0748649dc25fa5c73db0831cd5bebba5870b7fb6c62adf2da`.
- Private stopped bundle byte SHA-256: `04ff87d7877bda34963043a739968136dfa21825b71a112ac3c69ece86dae42e`.
- Private log byte SHA-256: `07fe1794e26e4b9c22744922d4d9ba001c7bb6661b04d61cf8c448fc509ff20e`.


### v1.19 saved support basis: execution ambiguity removed, notes scope stops

Inactive v1.19 changes rationale/notes instructions to explain requirements and
review decisions rather than historical agent execution. Supported actions remain
in content/trigger. Three separately frozen, independently reviewed development
controls (two announced-only tasks, one tool-confirmed action) completed three
paired repeats each: 18 outputs; both arms' decisions matched 9/9. Exact reason
labels were explicitly unscored. All nine candidate saved-field/core-preservation
audits cleared; 27 exact own-source excerpts and final references were verified.
The 18 logical calls recorded 19 dispatches, one recovered candidate parse repair
and zero final or measurement errors. Candidate calls used 349.127 seconds and
estimated $0.138230; baseline used 246.420 seconds and $0.092021, with complete
observations.

The private replay placed the previously failing setup case first. Five paired
repeats completed (10 outputs): candidate decisions/labels matched 5/5; baseline
decisions matched 5/5 and labels 0/5. All candidate content/triggers/rationales are
grounded and no historical setup execution/application claim survived. Four
outputs cleared all saved fields, but the fifth notes' unqualified denial of
recorded re-asking is UNVERIFIED: the record includes a repeated prerequisite
question before answers, while the supported distinction concerns re-asking after
answers. Two independent source-first audits confirmed this scope ambiguity; do
not qualify on a narrower charitable reading or change the oracle. One earlier
notes output narrates a source-correct question sequence despite the concise
instruction (style miss).

Stop was deliberate at the semantic gate (exit 130), leaving an interrupted,
incomplete manifest and a zero-output retail candidate checkpoint. The 10 completed
logical calls recorded 10 dispatches, zero repairs, zero final/measurement errors,
and 18 exact excerpts. The interrupted additional dispatch retained 59.790 seconds
of logical time and an IncompleteProviderObservation; usage/cost remain unknown,
not zero. Completed candidate calls used 357.353 seconds and estimated $0.128662;
baseline used 227.943 seconds and $0.079803. Retail and other priority controls
were not measured. This narrows the known execution defect in this sample, but
v1.19 remains inactive and unqualified; broader coverage, unchanged holdout,
production-provider compatibility and staging remain pending.

Implementation passed 509 affected OSS and 19 enterprise tests, plus 22 overlapping
final policy/version checks. Ruff/Pyright passed with zero type errors/warnings.
Local scoped P0/P1 reviews cleared and reverified the policy-exception fix before
calls. Original corpus, holdout, oracle v2 and prior reports remain unchanged.

- Launch enterprise source: `6d7587926cc2eae26706a3a2335d73370924ddfe`.
- Launch OSS source: `471b72bff53226a235fd6d022386f1dd0e19efda`.
- Prompt byte SHA-256: `a252291b64cd1bb29c26a433a8638d715e0e58baf39adc44dead9fe36eb7cace`.
- Supplemental fixture byte SHA-256: `06a48ca1c081f0028f1d0a52cd5d1ae64cf55cfcf4122d32f87c94e9a6bf5a8d`.
- Development driver byte SHA-256: `250e495712d02bfcd9404e4f0a4c32f061e62fe5687ee29b2f9387f4c9369c9b`.
- Private driver byte SHA-256: `f5af1d434a9a879292b8382782d97c596d418f782c54c44e566f30013ec87a06`.
- Development manifest byte SHA-256: `ff732fba4bc3b632bf1c8e1c529c74bc26a8b6ccb41d5857a09157383af8b889`.
- Development bundle byte SHA-256: `2784f25994a23ccf5a2a0868ab340bc81ba326d766ee96f0ac7e02a083b9ed87`.
- Development log byte SHA-256: `68ee87e57f74443bfdc43e9617c4317cfdf1a35bbabde89e2b77ce168a0dda55`.
- Private manifest byte SHA-256: `8a252ad84a813ced4f690fd4e6b15d6645ffa052947e7c2cf0ee23b1b99691ee`.
- Private stopped bundle byte SHA-256: `24fd38e181c88118983edf8478f8e56bf03b328668bba697d2bedce72cb0ab25`.
- Private log byte SHA-256: `47dbd0d71c715d76d11e198180a9a1c0609cc54bf3ae9307b37dce1e1bfcdf70`.


### v1.20 claim-level diagnostics: grounded outputs, frozen-label disagreement

Inactive v1.20 restricts reason/notes to the unsupported claim removed (if any)
and supported rule retained. A lack of justification for a prohibition does not
establish that an event or question was absent. Content/trigger actions, rationale
support requirements, evidence contract and taxonomy precedence are unchanged.
One separately frozen, independently adjudicated repeated-question control joins
two unchanged announced/verified-action controls. The original corpus, untouched
holdout, prospective oracle v2 and prior results remain unchanged.

The development gate completed three paired repeats per control: 18 outputs,
both arms' decisions correct 9/9, candidate scored labels 3/3 versus baseline
0/3. Six outputs per arm have deliberately unscored labels. All nine candidate
saved-field/core-preservation audits cleared, with 19 exact own-source excerpts
and valid final references. One diagnostic distinguishes an unsupported FAILED
re-ask from the visible pre-answer repeated question: grounded, but still a
presentation-policy miss. Eighteen logical calls recorded 18 dispatches, zero
repairs, final errors or measurement errors, and complete observations. Candidate
calls used 353.433 seconds and estimated $0.135462; baseline used 249.781 seconds
and $0.089896.

The private priority run stopped automatically at exit 2 after 31 final outputs:
16 candidate, 15 baseline, 15 complete pairs and one unpaired task-parameter
candidate. The setup, causal-benefit and absence controls each completed five
paired repeats. Two independent source-first audits cleared all 16 candidate
saved outputs and their useful cores; 39 exact own-source excerpts and final
references were verified. Candidate decisions match the frozen oracle 16/16,
reason labels 15/16. Some supported narration misses the concise diagnostic
instruction; semantic clearance does not imply perfect instruction compliance.

| Completed private control | Candidate decision / label | Candidate semantic audit | Baseline decision / label | Baseline semantic audit |
|---|---|---|---|---|
| Setup, five pairs | 5/5; 5/5 | 5 grounded | 5/5; 0/5 | 5 unsafe/unverified survivors |
| Causal benefit, five pairs | 5/5; 5/5 | 5 grounded | 5/5; 0/5 | 3 grounded; 2 unsafe survivors |
| Absence inference, five pairs | 5/5; 5/5 | 5 grounded rejects | 0/5; 0/5 | 5 unsafe survivors |
| Task parameter, one unpaired candidate | 1/1; 0/1 | 1 grounded | Not measured | Not measured |

The task-parameter output revised correctly but selected `unsupported_causality`
instead of frozen `speculative`. Both independent original-first audits find the
original candidate also attributes delay/blocking to clarification without
supporting evidence. Under the declared all-original-fields precedence, a causal
defect precedes speculation; the unique expected label omitted that claim. The
saved revision preserves conditional current-task parameter use and removes
standing numeric defaults, cross-session transfer and unsupported prohibitions.
This is a frozen-oracle disagreement, not a demonstrated unsafe survivor. Keep
this run stopped and scored against its original oracle. Any label correction
must be separately audited and preregistered for future calls; no retrospective
passing result, selective retry or continuation is applied.

All 31 logical calls have complete observations: 33 dispatches, two recovered
candidate parse repairs, zero final call errors and zero measurement errors.
Candidate calls used 1304.950 seconds and estimated $0.463525; baseline used
843.504 seconds and $0.280567. There is no interrupted or pending call. The
manifest remains incomplete, not interrupted, with its sole frozen-label gate
error preserved. Task-parameter repeats beyond the first and the unseen-artifact
control were not reached. Broader development/private coverage, unchanged
holdout, production-provider compatibility and staging remain pending. Equal
GLM-5.2 high/enabled/8192/.7 settings do not qualify MiniMax compatibility.
Both arms use the shared strengthened label validator, so the baseline is not
an exact replay of historical production code. No candidate is qualified or
activated; active/default remains v1.3.

Implementation passed 510 affected OSS tests, 19 enterprise replay tests and 24
overlapping final policy/version checks. Ruff/Pyright passed with zero type
errors/warnings. Scoped local P0/P1 reviews cleared before calls. These readiness
checks do not replace semantic or adoption gates. Raw production inputs and
outputs remain in owner-only private artifacts.

- Launch enterprise source: `d4a389b087729b6300bd70df7f1ddd8d5f5d2271`.
- Launch OSS source: `c995fd12e9170f610c9847b00a8a3d8d54a2e056`.
- Prompt byte SHA-256: `38aef8e50c7f9d13a9796fc7f7df4f2cc5652a7e03a018a65c3483333dfed794`.
- New supplemental fixture byte SHA-256: `30d1f63cbbf6cb99000075e6cb368ae27154bc8dfaa269a39db54fe3a39d5348`.
- Development driver byte SHA-256: `7c554e6cbc45d649a5a89792b5e802715e5640c12f8841a38172939c1d0579f9`.
- Private driver byte SHA-256: `454455bac34b53d2f86de057a41b72a255400b9b08692b8e38ede3be305510c7`.
- Development manifest byte SHA-256: `806b1e4d805deaa4c417f61025f96a493c06dd5b0b8466b4424ec5cbc7d89772`.
- Development bundle byte SHA-256: `ee5dfc3c5478359c9b7e9430cef0290b9ae4deb1b4cff06b82f907102ac1727b`.
- Development log byte SHA-256: `bb1e1083d989297191476061a30fe1c82d85cacc713e93b9f42633c6240a8df4`.
- Private manifest byte SHA-256: `1ae987f1b7cf6ed92846ee8fb8f873643fff7ead5e201f23f4090c5789d4ca32`.
- Private stopped bundle byte SHA-256: `4b430d78a4f0272cd127e773517c0ecb21c3ce28e106ed5b038c06a6b5acc2af`.
- Private log byte SHA-256: `66d7b3ca019544b6eedd8d93a752d089312525a5094f38777948668f8f88a79f`.
