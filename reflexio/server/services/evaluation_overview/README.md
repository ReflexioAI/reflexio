# evaluation_overview

Read-side aggregation module for `POST /api/get_evaluation_overview`.

- `service.py` is the request-path entry point. It loads evaluation, citation, Braintrust, and optional shadow verdict data, then composes `GetEvaluationOverviewResponse`.
- `components/` contains pure read-side aggregation helpers used by the service and focused tests.
- `eval_sampler.py` stays at the package root because regenerate jobs also use it to sample evaluation sessions.

The overview reports task success across every evaluated session and a separate
behavior-success metric that excludes `failure_type=system_error` rows from
both numerator and denominator. A window with no behavior-evaluable rows
returns a null behavior rate plus eligible/excluded counts for honest UI
rendering.

This module mutates no core state. Keep response-shape changes in API schema tests and service integration tests.
