# Rule evidence judge+tuner

`run_rule_evidence(input, invoke)` is a standalone, storage-free proposal method.
`RuleEvidenceInput` contains the actual incumbent, owned sessions exposing that
exact version, numbered source turns, agent context, success definition, and
historically captured tool contracts. Mark tool context captured, partial or
missing; a tool seen in a trace does not establish a complete tool catalog.
Source limitations (for example, native compaction) are passed to both stages.

The caller's `invoke(system_prompt, payload, output_schema)` uses its existing
LLM client and returns an instance of the requested structured-output model.
Credentials, transport retries, usage and latency accounting belong to that
caller. The method raises on provider/invalid-output failure; that must be
reported as a failure, not a successful abstention.

The judge returns useful observations in task completion, constraint compliance,
evidence grounding and execution effectiveness. It distinguishes a rule gap
(literal compliance causes the problem) from an adherence failure (the existing
rule is adequate but ignored). It does not return an overall completion label.
The existing `AgentSuccessEvaluator` and its public `is_success` output are
unchanged. Neither this method nor importing it schedules any background work.

For example, a trace says the tool returned `queued` but the agent reported
`delivered`. If the incumbent says `Report queued as delivered.`, that is a rule
gap. If it says `Report queued as queued; do not claim delivery.`, that is an
adherence failure and no edit is proposed.

The judge selects lossless, code-owned clause IDs. Code resolves them into source
quotations and validates source coordinates before tuning. The whole-rule tuner
receives the complete original rule, original session evidence and diagnostics;
there is no previous draft edit. It audits all guidance against task authority,
can correct at most three cited scalar evidence fields, and generates a complete
replacement or abstains. The source audit also runs when the judge admits no gap.
Corrections retain missing facts, observation kinds/dimensions and source identity.

`wire.py` owns `IndexedDiagnostic`, input-specific `WholeRuleTune` schemas,
correction reconstruction and exact edit derivation. These are raw model outputs;
`models.Diagnostic` and `models.Proposal` are the resolved application projection.
Callers must return the requested schema and preserve raw outputs for replay.
A tuner selects an admitted source session, not an edit span or quote. For trimmed
rules up to 1,200 characters, code replaces the whole rule. For longer rules it
derives one unique bounded edit and requires that applying it reproduce the
**entire** generated rule. Repeated tuning therefore accepts expanded rules.
An unchanged/blank rule or rewrite that cannot fit a unique 1,200-character source
span fails explicitly. No partial repair or fallback is applied.

Citations remain bound to the selected corrected gap. These checks establish
source/application integrity, not semantic safety or permission to publish.
Callers still verify authority, ownership, exposure/version and publication policy.
The measured MiniMax profile is judge: 12,288 tokens/thinking enabled; tuner:
4,096 tokens/thinking disabled, temperature 0, top_p 1, seed 42, timeout 240s.
The method itself does not own transport; `generation_profile` lets callers bind
stage settings without passing MiniMax thinking options to other providers.

Inputs are rejected above 60,000 `cl100k_base` proxy tokens, including prompt and
schema. The proxy is not the provider tokenizer. Full supplied turns are retained;
this implementation does not silently clip or select turns. Referenced coordinates
must belong to the supplied session and turn set. Callers with compacted sources
must describe those limitations and preserve their original-coordinate mapping.

This replaces the earlier exact-clause-generation method. The whole-rule Lab DEV
comparison passed both exposed Git cases versus one for the actual baseline,
with two independent output assessments; this is not population or online proof.
The transfer checks replay the original effective requests/responses/results and
exercise native durable publication. Deployment and real-traffic measurement are
separate steps. There is one maintained engine; release history supports rollback.
