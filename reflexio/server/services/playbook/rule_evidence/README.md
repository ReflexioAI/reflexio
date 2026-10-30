# Source-linked offline playbook tuning

One active method: four-dimensional session-local evidence followed by a global
audit of the complete original rule. This is the RSI global-rule-audit V7 transfer;
there is no competing baseline router, fallback tuner or scalar-outcome prerequisite.

`run_rule_evidence(data, invoke)` receives one owned incumbent and original sessions
exposing its exact version. `invoke` uses the caller's existing provider transport,
idempotency and bounded structural correction. The method owns no storage, retries,
credentials or publication permission.

- `models.py`: frozen input, resolved diagnostics, source-linked audit and result.
- `wire.py`: code-owned clause/reference catalogs, input-bound judge/tuner schemas,
  raw-output resolution and atomic application.
- `judge.md`: compact session evidence through completion, constraints, grounding
  and effectiveness; strengths/problems/uncertainties and deciding missing facts.
- `tune.md`: source-supported global defect hypothesis and up to four exact edits,
  or explicit abstention. Diagnostics remain immutable.

Raw `BoundedTargetDiagnostic` selects fixed original turn slots and fallible local
attribution. Raw `GlobalRuleAudit` selects original `CLAUSE_n` spans and up to six
fixed `REF_n` slots per edit. Code checks exact source ownership, exposed version,
coordinates, nonblank ordered spans of at most1200characters, no overlap/no-op,
boundary whitespace and complete nonblank revised guidance. All edits apply
atomically to the original; unselected content is copied exactly. References are
resolved without truncation. A healthy session needs no fabricated problem for an
independently source-supported rule repair.

The result has `decision`, `reason`, unchanged `diagnostics`, explicit `rule_audit`,
`revised_content` and resolved `evidence_refs`. Old gap-index and diagnostic-correction
outputs are retired. Observed raw wire output remains separate from this projection.
The caller still owns durable invocation, publication and lifecycle authority.
Invalid execution raises; it is not a successful abstention.

Original-turn slots accept exact integer IDs from decoded provider JSON as well as
JSON-mode validation. Strings, booleans, floats and IDs absent from the supplied
session are rejected. This conversion is local to the source-bound enum and keeps
the provider JSON schema and native SQL authorization domain unchanged.
A reply the schema rejects raises after the client's one corrective retry; both the
retry line (`event=llm_parse_retry_corrective`) and the final failure line
(`event=llm_request_end success=False`) carry `errors=` with the failing field
paths and pydantic error types. Only field names the schema declares, list indices
and error tags are logged; any other key is shown as `<redacted>`, so provider
content never reaches the log.

Both measured MiniMax stages reserve4096output tokens with thinking disabled.
Other providers retain the same token bound without MiniMax-specific settings.
Do not silently trim input; an oversized source batch is rejected explicitly.

The bounded exposed DEV result was8/9versus7/9development, with one gain and no
regressions/new critical harm. Watering remains UNKNOWN and diagnostic errors remain.
Exact saved-output replay covers all nine ported cases with no new inference.
This is proposal-quality evidence, not measured edited-agent or production benefit.

Consumer admission may reuse `wire.diagnostic_request`, `proposal_request`, and
`request_input_tokens` to measure the same prompt, source payload and dynamic
schema that dispatch uses. These pure builders own no model calls or resource
capacity. A proposal preview with empty diagnostics/audits measures only its fixed
source envelope; the normal 60,000-token proxy check still validates the complete
request after diagnosis, and the caller must enforce native provider headroom.
