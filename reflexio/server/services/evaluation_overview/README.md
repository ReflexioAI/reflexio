# evaluation_overview

Read-side aggregation module for `POST /api/get_evaluation_overview`.

- `service.py` is the request-path entry point. It loads evaluation, citation, Braintrust, and optional shadow verdict data, then composes `GetEvaluationOverviewResponse`.
- `components/` contains pure read-side aggregation helpers used by the service and focused tests.
- `eval_sampler.py` stays at the package root because regenerate jobs also use it to sample evaluation sessions.

This module mutates no core state. Keep response-shape changes in API schema tests and service integration tests.
