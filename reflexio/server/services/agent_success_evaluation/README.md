# agent_success_evaluation

Session-level agent success evaluation module.

## Module Shape

- `service.py`: `AgentSuccessEvaluationService`, the request-path service that runs configured evaluators and saves result rows.
- `runner.py`: `run_group_evaluation(...)`, the background/manual workflow entry point that loads a session, runs the service, and marks operation state. After agent-success work it also runs the retrieved-learning evaluation (independent completion; generation + session-fingerprint fenced) and returns a `GroupEvaluationOutcome` carrying both statuses.
- `scheduler.py`: `GroupEvaluationScheduler`, the deferred inactivity scheduler.
- `sampling.py`: deterministic per-session sampling gates for the session-success and retrieved-learning judge families. `generation_service.py` is the only scheduled-publish caller; it samples once, admits the session when either family is selected, and passes booleans to the runner.
- `components/evaluator.py`: `AgentSuccessEvaluator`, the LLM evaluator component.
- `components/retrieved_learning_evaluator.py`: `RetrievedLearningEvaluator`, per-learning relevance/impact judges over `Interaction.retrieved_learnings`; results go to the `retrieved_learning_evaluation` table.
- `regen_jobs.py`: regeneration job planning and execution; remains root-level because API/admin regenerate flows import it directly.
- `_eval_health.py`: producer/scheduler health counters.
- `agent_success_evaluation_constants.py`: prompt/model output constants.
- `agent_success_evaluation_utils.py`: request DTO and prompt-message construction helpers.

## Sampling Contract

- `AgentSuccessConfig.sampling_rate` controls the session-success judge; `evaluation_only_sampling_rate` can override it for evaluation-only publishes.
- `AgentSuccessConfig.retrieved_learning_sampling_rate` controls the retrieved-learning relevance/impact judges and inherits `sampling_rate` when unset.
- Direct runner callers (regen jobs and on-demand grade routes) leave the sampling booleans at their defaults and run both families; sampling remains a scheduler/publish decision.

## Prompt IDs

- Owns `agent_success_evaluation`, `retrieved_learning_relevance`, and `retrieved_learning_impact`.
- Keeps historical/configured prompt ID `agent_success_evaluation_with_comparison` stable where prompt mapping tests require it.

Do not reintroduce the deleted service/evaluator/runner/scheduler legacy
module files.
