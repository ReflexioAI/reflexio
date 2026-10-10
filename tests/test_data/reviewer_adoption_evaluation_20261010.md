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

Inactive v1.7 clarified missing support for artifact contents versus using record
omissions as proof of absence in fact. Events, properties and conditions remain
covered by the fatal absence gate; internal-status proof of value remains
distinct. Subject ownership, decision order, revision boundaries and the output
contract remain unchanged. Independent policy review cleared development
evaluation, not activation. V1.7 subsequently failed the workflow gate below.
Inactive v1.8 clarified requested-workflow atomicity but failed the artifact
label gate below. Inactive v1.9 consolidates the procedure into a shorter
prompt while preserving fatal gates, revisions and workflows. It requires
fresh repeated qualification; no previous version qualifies it.

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

The unchanged candidate is now undergoing the remaining 36 development cases.
Private-window, fallback, holdout and full-pipeline gates remain pending.

- Priority launch source: `b9c95ce5e429470a00164efe8c6c8778e0796ccb`.
- Priority cases SHA-256: `ccbfc01bb30bee5c4286e6c2eda67f7f0d07950178729ae1daa98853e93ebac1`.
- Priority report byte SHA-256: `478faa464014e712cfde7d8a6fb92803a1a4cc3955bb9a95ff44290fe6d07039`.
- Priority prompt identities (unrendered content digests):
  - 1.3.0: `{"active_version": "1.3.0", "prompt_id": "playbook_candidate_review", "template_content_digest": "c6f03e36e2772775d17a147a4e8c697a49962199c4424bcf60cf881d6c8a04fc"}`.
  - 1.9.0: `{"active_version": "1.9.0", "prompt_id": "playbook_candidate_review", "template_content_digest": "2d0a7fba4cd69be75652cdb61f0ff605eb3affecf2bc5cdcb674c2d2abcc5d94"}`.
