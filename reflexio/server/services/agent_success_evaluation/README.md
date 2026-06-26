# agent_success_evaluation

Session-level agent success evaluation module.

## Module Shape

- `service.py`: `AgentSuccessEvaluationService`, the request-path service that runs configured evaluators and saves result rows.
- `runner.py`: `run_group_evaluation(...)`, the background/manual workflow entry point that loads a session, runs the service, dispatches shadow comparison, and marks operation state.
- `scheduler.py`: `GroupEvaluationScheduler`, the deferred inactivity scheduler.
- `components/evaluator.py`: `AgentSuccessEvaluator`, the LLM evaluator component.
- `regen_jobs.py`: regeneration job planning and execution; remains root-level because API/admin regenerate flows import it directly.
- `_eval_health.py`: producer/scheduler health counters.
- `agent_success_evaluation_constants.py`: prompt/model output constants.
- `agent_success_evaluation_utils.py`: request DTO and prompt-message construction helpers.

## Prompt IDs

- Owns `agent_success_evaluation`.
- Keeps historical/configured prompt ID `agent_success_evaluation_with_comparison` stable where prompt mapping tests require it.

Do not reintroduce the deleted service/evaluator/runner/scheduler legacy
module files.
