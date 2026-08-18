# evaluation_overview

Read-side aggregation module for `POST /api/get_evaluation_overview`.

- `service.py` is the request-path entry point. It bulk-loads evaluation results (without embeddings), first-request sources, citations, Braintrust scores, and optional shadow verdicts, then composes `GetEvaluationOverviewResponse` for the dashboard.
- `components/` contains pure read-side aggregation helpers used by the service and focused tests.
- `eval_sampler.py` stays at the package root because regenerate jobs also use it to sample evaluation sessions.
- `reflexio/models/api_schema/eval_overview_schema.py` defines the public request/response contract, including optional `source_sets` cohorts and `source_set_comparison` output.

The overview reports task success across every evaluated session and a separate
behavior-success metric that excludes `failure_type=system_error` rows from
both numerator and denominator. A window with no behavior-evaluable rows
returns a null behavior rate plus eligible/excluded counts for honest UI
rendering. Source-set comparison groups sessions by the first request source,
returns collision-safe `(user_id, session_id)` identities, and rejects duplicate
labels or overlapping source values so the frontend can compare cohorts without
client-side re-aggregation.

This module mutates no core state. Keep response-shape changes in API schema tests and service integration tests.
