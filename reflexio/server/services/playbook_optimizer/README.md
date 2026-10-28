# /reflexio/server/services/playbook_optimizer
Description: GEPA-driven optimization of playbook content using assistant rollouts and judged candidates.

## Main Entry Points


- `optimizer.py` orchestrates one optimization run and persists candidates, evaluations, events, and optional successor playbooks.
- `scheduler.py` owns deferred optimization scheduling.
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
and pass one validated immutable `OptimizerSchedulingPolicy` to the scheduler.
The original org fields `scheduler_jitter_seconds`, `abort_cooldown_threshold`
and `cooldown_after_aborts_seconds` remain supported and saved. Defaults are
1.0 seconds, two aborts and 3600 seconds; zero jitter/cooldown removes that delay.
A queued callback keeps its enqueue-time policy even when configuration changes.
Completed/skipped outcomes reset the abort counter. Debounce and failure state
remain process-local and scoped by org, project and target.

Malformed/nonfinite/unsafe settings raise `OptimizerSchedulingMigrationError`
before stored-config pruning. Storage adapters and CLI bootstrap must propagate
that error rather than loading defaults or overwriting saved configuration.
Valid custom scheduling values are never stripped or replaced with constants.
