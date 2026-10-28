# /reflexio/server/services/playbook_optimizer
Description: GEPA-driven optimization of playbook content using assistant rollouts and judged candidates.

## Main Entry Points


- `optimizer.py` orchestrates one optimization run and persists candidates, evaluations, events, and optional successor playbooks.
- `scheduler.py` owns optimizer admission and captured failure policy.
- `models.py` owns optimizer-local data shapes.
- `judge.py`, `rollout.py`, `gepa_adapter.py`, `assistant_webhook.py`, and `scenario_resolver.py` are mature implementation units and intentionally remain at the package root.

## Purpose


Generate and compare content candidates while retaining the user/agent playbook adoption rules.

`PlaybookOptimizerConfig.allows_target()` owns admission for execution and both
scheduling paths. The nonnegative `metric_call_budget` owns the overall work
limit: zero admits no new runs, and a positive value selects GEPA’s metric-call stopping threshold. GEPA may
finish a validation batch before checking that threshold.
`user_playbooks_enabled` and `agent_playbooks_enabled` keep target consent
independent. Historical admission/budget inputs are translated only at explicit
upgrade boundaries; normal runtime readers reject them. Frozen GEPA publication
authority retains its historical budget and adoption output keys and is never
rewritten from live configuration.


## Architecture Pattern


The scheduler invokes `optimizer.py`, which resolves source scenarios, runs assistant rollouts through the configured backend, judges candidates, and persists run artifacts and eligible successors.

## Requirements / Problems to Avoid


Do not introduce a `components/` package without a separate design that proves the dependency direction is clearer after the move.

Merge admission has one configuration input: `max_merge_invocations`. Zero
means disabled; positive values bound GEPA merges. The schema normalizes legacy
`use_merge=false` to zero before raw upgrade pruning and canonical saves. GEPA's
boolean and recorded merge identity are derived from the budget; saved job
authority remains immutable; durable publication uses its recorded snapshot.

## Scheduling policy

Both production enqueue paths resolve `PlaybookOptimizerConfig.scheduling_policy`
and pass one validated immutable `OptimizerSchedulingPolicy` to admission.
`abort_cooldown_threshold` and `cooldown_after_aborts_seconds` remain supported;
defaults are two aborts and 3600 seconds. Zero cooldown disables that delay.
A queued callback keeps its enqueue-time policy even when configuration changes.
Completed/skipped outcomes reset the abort counter. Failure state remains
process-local and scoped by org, project and target. Cooldown suppresses new
admissions; previously accepted successors can still run.

The shared bounded callback executor owns dispatch. Each target has at most one
active callback and one queued successor. A new queued callback replaces its
predecessor without moving its queue position. Different targets remain runnable
while a target is active. Queue overflow retains observable drop-oldest behavior.
This is best-effort process-local work, not a durable queue or rate limiter.

The timer heap, daemon and `scheduler_jitter_seconds` setting are retired. Work
becomes runnable immediately; coalescing follows queued/in-flight work rather than
an elapsed delay. Normal prelaunch preparation upgrades saved files to generation6:
it validates historical jitter before removing only that field. Strict runtime
reads reject obsolete jitter, including a former default value. Custom failure
policy and unrelated data survive conversion. Malformed/nonfinite/unsafe settings
raise `OptimizerSchedulingMigrationError` before pruning or any stored-file write;
storage adapters and CLI bootstrap must propagate that error.
