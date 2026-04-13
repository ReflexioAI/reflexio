# Retrieval Latency Benchmark

Measures end-to-end latency for profile search, user/agent playbook search,
and unified cross-entity search across storage backends and corpus sizes.

## What it measures

Four retrieval types × two layers × N storage backends × K corpus sizes.

**Layers:**

| Layer      | What it times                                               |
|------------|-------------------------------------------------------------|
| `service`  | `reflexio.search_*()` called directly in-process            |
| `http`     | FastAPI `TestClient` POST to `/api/search_*` (no TCP — ASGI transport only) |

The `http` layer measures FastAPI routing, Pydantic serialization, and
middleware cost on top of the service layer. It does *not* include kernel
socket or network RTT — the goal is to isolate framework overhead from
storage work. Add a uvicorn-backed variant later if you need wall-clock HTTP.

**Retrieval types:** `profile`, `user_playbook`, `agent_playbook`, `unified`.

**Backends:** `sqlite` (always available) and `supabase` (requires a local
`supabase start` + `SUPABASE_ANON_KEY` / `SUPABASE_DB_URL` env vars; skipped
gracefully otherwise).

## Controlling embedder cost

Query embeddings are pre-cached on disk at
`~/.cache/reflexio-benchmarks/embeddings-<model>.json`. First run populates
the cache via the real embedding API (requires `OPENAI_API_KEY` or the
configured provider key); subsequent runs are offline.

Document embeddings during corpus seeding use deterministic hash-derived
fake vectors. Document vector *content* doesn't affect query-time latency
(only row count does), so this is a sound optimization. Query-time
retrieval always uses the real cached query vector.

Query reformulation (a separate LLM call) is disabled on every request so
it doesn't contaminate the timing — it's a known LLM cost, orthogonal to
retrieval.

## Usage

```bash
# Default sweep: sizes 100/1000/10000, sqlite + supabase if available,
# both layers, all four retrieval types, 50 trials + 5 warmup per cell.
uv run python -m reflexio.benchmarks.retrieval_latency.bench

# Small smoke run (for local iteration):
uv run python -m reflexio.benchmarks.retrieval_latency.bench \
    --sizes 100 --trials 10 --warmup 2 \
    --backend sqlite --layer service --retrieval profile

# Diff against a committed baseline:
uv run python -m reflexio.benchmarks.retrieval_latency.bench \
    --baseline tests/benchmarks/baseline.json
```

## Output

Each run writes `results.json` and `report.md` to
`reflexio/benchmarks/retrieval_latency/results/<timestamp>/` — next to the
script itself, so reports travel with the code that produced them (override
with `--output-dir`). The `results/` directory is gitignored; commit a report
manually only when you want to save it as a baseline or reference.

`results.json` contains the full config block, per-cell aggregated stats,
and every raw trial row. `report.md` renders the stats as markdown tables,
one per retrieval type, rows grouped by `(backend, layer)`. Cell format:
`p50 / p95 (mean)` in milliseconds. When `--baseline` is passed, a ΔP95
column flags cells where p95 has grown by 20% or more (`⚠`).

## Interpreting the numbers

Sanity checks to run on any report:

- p95 should scale sub-linearly with N (both FTS and vector indexes are built).
- HTTP layer > service layer on the same cell (framework tax).
- Unified search ≈ `max(profile, user_playbook, agent_playbook)` plus a small
  Phase A fixed cost — unified runs the three entity searches in parallel.
- If a cell has wildly higher variance than its neighbors, suspect GC or a
  cold cache — re-run with more warmup.

## Pytest smoke test

`tests/benchmarks/test_retrieval_latency_smoke.py` runs a tiny version of
this benchmark at `N=50, trials=10, sqlite + service only` and asserts that
p95 has not regressed past 3× the committed baseline. It's marked
`skip_in_precommit`, so it runs in `test-all` but not on every commit. To
regenerate the baseline after an intentional perf change:

```bash
uv run python -m reflexio.benchmarks.retrieval_latency.bench \
    --sizes 50 --trials 10 --warmup 2 \
    --backend sqlite --layer service \
    --output-dir tests/benchmarks/tmp/
cp tests/benchmarks/tmp/results.json tests/benchmarks/baseline.json
```

Then commit the new `baseline.json`.
