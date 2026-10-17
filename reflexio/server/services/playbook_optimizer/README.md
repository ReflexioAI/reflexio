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
scheduling paths. User optimization and adoption share `user_playbooks_enabled`;
legacy user switches normalize to their conjunction before stored-config pruning.
Both agent switches remain independent because integration defaults preserve an
agent-adoption veto. Frozen GEPA publication authority keeps its historical
`auto_update_user_playbooks` key; it is not rewritten from live configuration.

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

Production enqueue paths use the scheduler's code-owned defaults: up to one
second of jitter and a one-hour target-scoped cooldown after two consecutive
aborts. Completed/skipped outcomes reset the counter. The scheduler's internal
arguments remain available for focused tests; org configuration does not set
them. Default-valued legacy scheduler fields are stripped before typed validation
or raw upgrade pruning. Custom/malformed fields raise
`OptimizerSchedulingMigrationError`; storage adapters and CLI bootstrap must
propagate it rather than loading defaults or overwriting saved configuration.
