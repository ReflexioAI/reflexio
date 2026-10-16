# /reflexio/server/services/playbook_optimizer
Description: GEPA-driven optimization of playbook content using assistant rollouts and judged candidates.

## Main Entry Points


- `optimizer.py` orchestrates one optimization run and persists candidates, evaluations, events, and optional successor playbooks.
- `scheduler.py` owns deferred optimization scheduling.
- `models.py` owns optimizer-local data shapes.
- `judge.py`, `rollout.py`, `gepa_adapter.py`, `assistant_webhook.py`, and `scenario_resolver.py` are mature implementation units and intentionally remain at the package root.

## Purpose


Generate and compare content candidates while retaining the user/agent playbook adoption rules.

## Architecture Pattern


The scheduler invokes `optimizer.py`, which resolves source scenarios, runs assistant rollouts through the configured backend, judges candidates, and persists run artifacts and eligible successors.

## Requirements / Problems to Avoid


Do not introduce a `components/` package without a separate design that proves the dependency direction is clearer after the move.

Merge admission has one configuration input: `max_merge_invocations`. Zero
means disabled; positive values bound GEPA merges. The schema normalizes legacy
`use_merge=false` to zero before raw upgrade pruning and canonical saves. GEPA's
boolean and recorded merge identity are derived from the budget; saved job
authority remains immutable; durable publication uses its recorded snapshot.
