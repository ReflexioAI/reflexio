# GDPVal Comparison Benchmark

Run the GDPVal dataset through two host agents (OpenSpace, Hermes) in a
three-phase cold → warm → warm+reflexio protocol to measure:

- **P1 → P2**: the host's own native learning lift
- **P2 → P3**: reflexio's *marginal* contribution on top of a warm host

Same tasks, same model, same `LLMEvaluator`, same 0.6 payment cliff. Six cells
total (2 hosts × 3 phases). The headline we care about is `mean(P2 − P3)`.

## Prerequisites

1. Clone the dependency repos. By default `config.py` looks under `~/repos/`:
   ```bash
   mkdir -p ~/repos
   git clone https://github.com/openspace-ai/OpenSpace      ~/repos/OpenSpace
   git clone https://github.com/HKUDS/hermes-agent           ~/repos/hermes-agent
   git clone https://github.com/HKUDS/ClawWork               ~/repos/ClawWork
   ```
   If you cloned them elsewhere, point the benchmark at them via env vars
   before running:
   ```bash
   export OPENSPACE_ROOT=/path/to/OpenSpace
   export HERMES_ROOT=/path/to/hermes-agent
   export CLAWWORK_ROOT=/path/to/ClawWork
   ```

2. Start a reflexio backend in a separate terminal:
   ```bash
   ./run_services.sh
   ```
   (or point `REFLEXIO_URL` at an already-running backend)

3. Export API credentials:
   ```bash
   export OPENROUTER_API_KEY=...          # host agent LLM calls
   export OPENAI_API_KEY=...              # ClawWork LLMEvaluator (gpt-4o)
   export REFLEXIO_API_KEY=...            # reflexio client
   ```

4. For HuggingFace GDPVal auto-download:
   ```bash
   uv pip install datasets
   ```

## Running

```bash
uv run python -m benchmark.gdpval.run_benchmark \
    --hosts openspace,hermes \
    --phases p1,p2,p3 \
    --max-tasks 5 \
    --run-name smoke
```

Verification ladder:

| Command | Checks |
|---|---|
| `--dry-run --hosts hermes --max-tasks 2` | Imports, env, CLI wiring. No LLM calls. |
| `--hosts hermes --phases p1 --max-tasks 1 --no-eval` | Hermes adapter + token path + workspace. |
| `--hosts openspace --phases p1,p2 --max-tasks 1` | Snapshot/restore + litellm token path. |
| `--hosts openspace,hermes --phases p1,p2,p3 --max-tasks 1` | All 6 cells, end-to-end. |
| `--task-list <tasks_50.json> --hosts openspace,hermes` | Headline run. |

## Output

```
output/<run_name>/
  config.json
  <host>/
    snapshots/post_p1/          # state frozen after P1
    host_state_p1/
    host_state_p2/              # copy of post_p1 snapshot
    host_state_p3/              # copy of post_p1 snapshot
    p1_cold/
      results.jsonl
      workspace/<task_id>/      # deliverables the evaluator scores
    p2_warm/
      results.jsonl
      workspace/<task_id>/
    p3_warm_reflexio/
      results.jsonl
      workspace/<task_id>/
  comparison.csv                # per-task × cell
  comparison.md                 # headline deltas
```

## Design notes

- **Host isolation.** OpenSpace writes SkillStore state to `$OPENSPACE_ROOT/.openspace/`;
  Hermes writes MEMORY.md/skills to `$HERMES_HOME` (default `~/.hermes`). The
  OpenSpace adapter copies the relevant dirs in and out per phase. The Hermes
  adapter sets `HERMES_HOME=<host_state_dir>/hermes_home/` before constructing
  `AIAgent`.

- **LLM layer.** OpenSpace routes through litellm (we attach its
  `TokenTracker` callback). Hermes uses the native OpenAI SDK pointed at
  OpenRouter (`base_url=https://openrouter.ai/api/v1`); we read tokens from
  `run_conversation()`'s result dict. Both paths produce the same `TokenStats`.

- **Reflexio seeding.** In P1, every finished trajectory is published via
  `ReflexioClient.publish_interaction(wait_for_response=True)` so the memory
  store is seeded from the same raw experience that warmed the host's own
  native memory. P3 retrieves via `ReflexioClient.search()` (unified search
  across profiles + agent_playbooks + user_playbooks).

- **Evaluator.** `benchmark/gdpval/evaluation.py` is a port of
  `OpenSpace/gdpval_bench/run_benchmark.py:_evaluate_task` — same artifact
  extensions, same `LLMEvaluator` call, same 0.6 payment cliff. Scoring is
  apples-to-apples with OpenSpace's published gdpval_bench numbers.
