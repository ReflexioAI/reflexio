# Reviewer adoption qualification — 2026-10-10

Status: in progress. The active reviewer remains v1.3.0. No activation or
production change is justified by the completed measurements below.

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
