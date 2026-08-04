# agent_success_evaluation

Session-level agent success evaluation module.

## Module Shape

- `service.py`: `AgentSuccessEvaluationService`, the request-path service that runs configured evaluators and saves result rows.
- `runner.py`: `run_group_evaluation(...)`, the background/manual workflow entry point that loads a session, runs the service, and marks operation state. After agent-success work it also runs the retrieved-learning evaluation (independent completion; generation + session-fingerprint fenced) and returns a `GroupEvaluationOutcome` carrying both statuses.
- `scheduler.py`: `GroupEvaluationScheduler`, the deferred inactivity scheduler.
- `components/evaluator.py`: `AgentSuccessEvaluator`, the LLM evaluator component.
- `components/retrieved_learning_evaluator.py`: `RetrievedLearningEvaluator`, per-learning relevance/impact judges over `Interaction.retrieved_learnings`; results go to the `retrieved_learning_evaluation` table.
- `regen_jobs.py`: regeneration job planning and execution; remains root-level because API/admin regenerate flows import it directly.
- `_eval_health.py`: producer/scheduler health counters.
- `agent_success_evaluation_constants.py`: prompt/model output constants.
- `agent_success_evaluation_utils.py`: request DTO and prompt-message construction helpers.

## Failure Classification

Unsuccessful sessions use `system_error`, `missing_tool`, `wrong_tool`,
`insufficient_info_from_tool`, or `wrong_answer`. `system_error` is reserved for
reliability and availability failures outside agent behavior, including hangs,
missing responses, persistent timeouts or 5xx responses, unavailable services,
and quota or credit failures. The evaluation overview keeps these rows in task
success while excluding them from the separate behavior-success denominator.

## Prompt IDs

- Owns `agent_success_evaluation`, `retrieved_learning_relevance`, and `retrieved_learning_impact`.
- Keeps historical/configured prompt ID `agent_success_evaluation_with_comparison` stable where prompt mapping tests require it.

Do not reintroduce the deleted service/evaluator/runner/scheduler legacy
module files.
