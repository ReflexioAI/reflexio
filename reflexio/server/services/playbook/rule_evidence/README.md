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

With admitted gaps, the tuner selects a gap index and generates its replacement.
Code supplies the uniquely occurring original clause and inherited source
references. Every byte outside that clause stays identical. This structural
guarantee does not prove the replacement is semantically safe: downstream
validation/adoption must check unsupported permissions, scope and usefulness.
With no admitted gap, the method abstains without calling the tuner. The tuner
may also abstain after a diagnosis. A proposal never authorizes publication.

Inputs are rejected above 60,000 `cl100k_base` proxy tokens, including prompt and
schema. The proxy is not the provider tokenizer. Full supplied turns are retained;
this implementation does not silently clip or select turns. Referenced coordinates
must belong to the supplied session and turn set. Callers with compacted sources
must describe those limitations and preserve their original-coordinate mapping.

This is the first native transfer of the lab DEV method. It has not replaced the
online completion evaluator or the enterprise tuner. Native qualification must
measure differences in source rendering, schema, transport and repair behavior
before the lab's performance result can be attributed to this implementation.
