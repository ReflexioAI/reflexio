# playbook_optimizer

GEPA-driven playbook content optimizer.

- `optimizer.py` orchestrates one optimization run and persists candidates, evaluations, events, and optional successor playbooks.
- `scheduler.py` owns deferred optimization scheduling.
- `models.py` owns optimizer-local data shapes.
- `judge.py`, `rollout.py`, `gepa_adapter.py`, `assistant_webhook.py`, and `scenario_resolver.py` are mature implementation units and intentionally remain at the package root.

Do not introduce a `components/` package without a separate design that proves the dependency direction is clearer after the move.
