# Developer Guide

## Project Structure

```
reflexio/
├── reflexio/              # Main Python package
│   ├── client/            # ReflexioClient implementation
│   ├── cli/               # Command-line interface
│   ├── data/              # Data storage / fixtures
│   ├── integrations/      # LLM and external integrations
│   ├── lib/               # Core library functions
│   ├── models/            # Data models and API schemas
│   │   └── api_schema/    # API request/response schemas
│   ├── server/            # FastAPI backend
│   │   ├── api_endpoints/ # Route handlers
│   │   ├── services/      # Business logic and storage
│   │   ├── llm/           # LLM provider integration
│   │   ├── prompt/        # Prompt templates
│   │   └── site_var/      # Site configuration
│   └── test_support/      # Testing utilities
├── docs/                  # Next.js 16 docs frontend (ShadCN UI)
├── tests/                 # Test suite (pytest)
├── scripts/               # Utility scripts (e.g. reset_db.py)
├── client_dist/           # Lightweight client distribution package
└── notebooks/             # Jupyter notebooks (examples, quickstart)
```

Agent configuration lives in one place: `.agents/skills/` and `.agents/rules/`.
`.claude/skills` and `.claude/rules` are symlinks into it, so there is one stored
copy rather than two that drift. Codex reads the skills from `.agents/skills`
directly; it takes durable guidance from the root `AGENTS.md`, not from
`.agents/rules`, so those rules apply to Claude Code.

Those two are **Git symlinks**. A checkout with `core.symlinks=false` — the
default on Windows without Developer Mode or an elevated shell — materializes
them as plain text files — `.claude/skills` holding the literal text
`../.agents/skills` and `.claude/rules` holding `../.agents/rules` — and no
skill or rule below them is reachable. Clone with `git clone -c core.symlinks=true`, or run
`git config core.symlinks true` followed by `git checkout -- .claude` in an
existing checkout.

## Services

Two services, started together via `./run_services.sh`:

| Service | Framework | Default Port | Env Var |
|---------|-----------|-------------|---------|
| Backend | FastAPI (uvicorn) | 8061 | `BACKEND_PORT` |
| Docs | Next.js 16 | 8062 | `DOCS_PORT` |

`API_BACKEND_URL` is derived automatically as `http://localhost:${BACKEND_PORT}`.

**Storage backend** — pass `--storage sqlite` (default) or `--storage supabase` to select the data storage backend:
```bash
uv run reflexio services start --storage sqlite    # local SQLite (default)
uv run reflexio services start --storage supabase  # Supabase PostgreSQL
```

Stop services with `./stop_services.sh`.

## Dev mode vs. daemon mode

The `reflexio services start` command has two distinct modes:

### Dev mode (default for interactive use)

- `reflexio services start` (no flag) → uvicorn `--reload` ON, single process.
- The backend reloads on every source-file change. Convenient for fast iteration.
- Single-process means concurrency is asyncio (one CPU core). Sufficient for local testing.

### Daemon mode (set-and-forget deployments)

- `reflexio services start --no-reload` → uvicorn multi-worker manager, no reload.
- Workers exit after ~`--max-requests` served requests; the manager respawns them.
- Memory accumulation from any source resets periodically. No external supervisor needed beyond uvicorn itself.

| Flag | Default | Purpose |
|---|---|---|
| `--no-reload` | (off; opt in for daemon mode) | Switches to daemon mode |
| `--workers N` | 2 | Worker count. Higher = more parallelism; must be ≥1 |
| `--max-requests N` | 10000 | Worker recycles after this many requests; 0 disables |
| `--max-requests-jitter J` | 1000 | Per-worker random 0..J added to threshold (avoid synchronized recycles) |
| `--graceful-shutdown-sec S` | 30 | Drain window for in-flight requests on shutdown |

The `--reload + --workers > 1` combination is rejected at CLI parse time (autoreload is incompatible with multi-worker mode).

When the storage backend is SQLite and `--workers > 1`, a warning is logged at startup — SQLite supports concurrent reads but serializes writes (even in WAL mode), so high-QPS writes hit `SQLITE_BUSY`. Switch to Postgres/Supabase for higher write throughput.

### Why this matters

Long-uptime processes accumulate memory regardless of source (request handlers, ORM caches, third-party libraries, fragmented allocators). Without recycling, RSS grows monotonically over days/weeks. With request-count recycling at workers≥2, one worker exits cleanly after ~max_requests served, the manager respawns it under a fresh PID, and the peer worker absorbs traffic during the ~1-2s respawn window — zero downtime.

## API Usage

```bash
curl http://localhost:$BACKEND_PORT/...
```

Or with the Python client:
```python
from reflexio import ReflexioClient
client = ReflexioClient(url_endpoint=f"http://localhost:{BACKEND_PORT}")
```

## Package Management

- **Python**: Use `uv` (`uv sync`, `uv add`, `uv run <cmd>`, or activate `.venv`)
- **Docs frontend**: Use `npm` (run from `docs/` directory)

## Environment Variables

Copy `.env.example` to `.env` and fill in values. Key variables:

- **LLM API keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc.
- **Storage**: `LOCAL_STORAGE_PATH` (defaults to `~/.reflexio/data`) — houses the SQLite DB file. (Not used by `supabase`/`postgres`, which use external connections.)
- **Storage backend**: `REFLEXIO_STORAGE` — `sqlite` (default), `supabase`, or `postgres`. Selects the data storage backend independently from auth configuration.
- **Testing**: `IS_TEST_ENV`, `DEBUG_LOG_TO_CONSOLE`, `MOCK_LLM_RESPONSE`

Never change env variable values in `.env` directly for port overrides — use shell exports instead.

## Supported LLM Providers

| Provider | Env Variable | Model Prefix | Example Usage |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | (default) | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/` | `anthropic/<model>` |
| Google Gemini | `GEMINI_API_KEY` | `gemini/` | `gemini/<model>` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/` | `openrouter/<provider>/<model>` |
| MiniMax | `MINIMAX_API_KEY` | `minimax/` | `minimax/<model>` |
| Azure OpenAI | via config | `azure/` | `azure/<deployment>` |
| Custom endpoint | via config | — | — |

To change which models Reflexio uses, edit [`reflexio/server/site_var/site_var_sources/llm_model_setting.json`](reflexio/server/site_var/site_var_sources/llm_model_setting.json).
Use the provider prefix shown above (e.g., `anthropic/` for Anthropic models). Set the corresponding API key in your `.env` file.

## Modifying API Schemas

Edit files in `reflexio/models/api_schema/`:
- `service_schemas.py` — main API request/response schemas
- `internal_schema.py` — internal data models
- `retriever_schema.py` — retriever-related schemas
- `validators.py` — validation logic

## Code Quality Tools

**Python:**
- **Ruff** — linting + formatting (config in `pyproject.toml`)
- **Pyright** — type checking (config in `pyrightconfig.json`, basic mode, Python 3.14)

```bash
uv run ruff check .             # Lint
uv run ruff format .            # Format
uv run pyright                  # Type check
```

**TypeScript/JavaScript (docs frontend):**
- **ESLint** — linting (config in `docs/eslint.config.mjs`)
- **tsc** — type checking

```bash
cd docs
npx eslint .                    # Lint
npx tsc --noEmit                # Type check
```

## Testing

- Framework: **pytest** with `pytest-xdist` (parallel via `-n auto`)
- Timeout: 120 seconds per test
- Coverage minimum: 65% (branch coverage enabled)
- Markers: `unit`, `integration`, `e2e`, `requires_credentials`

Run tests:
```bash
uv run pytest                          # all tests
uv run pytest tests/server/            # specific directory
uv run pytest -m unit                  # by marker
uv run pytest -k "test_name"           # by name
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure (e.g., `tests/server/` for `reflexio/server/`)
- Name test files `test_<module>.py`
- Use markers: `@pytest.mark.unit` (no network), `@pytest.mark.integration` (needs services), `@pytest.mark.e2e` (full stack), `@pytest.mark.requires_credentials` (needs API keys)
- Keep tests independent — no shared mutable state between tests

### Self-bootstrapping test harnesses

The test bootstrap redirects `REFLEXIO_LOG_DIR` and `LOCAL_STORAGE_PATH` to a
temporary directory before test collection uses storage. In-process E2E
fixtures also configure `StorageConfigSQLite` against a `tmp_path` fixture
(`tests/e2e_tests/conftest.py`). Together these guards keep tests from binding
production ports or writing to `~/.reflexio`.

If you add a future harness that boots services from a clean checkout, it must:

1. Sandbox storage: point `LOCAL_STORAGE_PATH` (or the SQLite `db_path`) at a temp dir, never the default `~/.reflexio/data/`.
2. Use non-default ports: pick a `1XXXX` form that mirrors the production `8061`/`8062` while staying clear of common dev ranges (e.g. `BACKEND_PORT=19061`, `DOCS_PORT=19062`, or higher), and refuse production ports (`8061`, `8062`) unless the user explicitly opts in.
3. Never use the real `$HOME` as the integration home; create a temp `INTEG_HOME` and export `HOME=$INTEG_HOME` before launching services so every subprocess inherits the isolated home.

Without these guards the harness binds `8061`/`8062` (already taken by a developer's running `reflexio services`) or writes to `~/.reflexio/data/`, clobbering the developer's installed state.

## Commit & PR Conventions

**Commit messages** — use conventional prefixes:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code change that neither fixes a bug nor adds a feature
- `docs:` documentation only
- `test:` adding or updating tests
- `chore:` maintenance (deps, CI, scripts)

**Pull requests:**
1. Create a feature branch from `main` (`feat/short-description` or `fix/short-description`)
2. Keep PRs focused — one concern per PR
3. Ensure lint, type checks, and tests pass before submitting
4. Write a clear PR description explaining **what** and **why**

## Client Distribution

The `client_dist/` directory contains a separate lightweight package (`reflexio-client`) for distribution. It symlinks back to `reflexio/` and builds only the client, models, and integrations submodules.

## Git Worktree Development

When working in a git worktree, services must run on different ports to avoid conflicts.

### Setup Checklist

1. `git worktree add ../reflexio-feature feature-branch`
2. `cd ../reflexio-feature`
3. Copy `.env` from main worktree
4. `uv sync && (cd docs && npm install)`
5. `export BACKEND_PORT=8091 DOCS_PORT=3001`
6. `./run_services.sh`

### Notes

- Do NOT modify `.env` for port variables — export in shell instead

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | `./stop_services.sh` or `lsof -i :8061` to find the process |
| Services won't start | Check `.env` has at least one LLM API key set |
| `uv sync` fails | Ensure Python >= 3.14, try `uv self update` |
| Docs frontend won't start | Run `npm --prefix docs install` first |
| Import errors after pull | Run `uv sync` to update dependencies |

### Storage requirements for publishing

Custom storage backends must implement `add_request_if_absent(request) -> bool`
as an atomic insert that participates in `commit_scope`: return false for an
existing request ID without modifying it. The base implementation raises
`NotImplementedError`; a check followed by an upsert cannot safely substitute
for this operation. `add_request` retains its existing upsert contract for other
callers. Duplicate publishes may prepare embeddings before rejection; external
admission receipt replays still skip that preparation.

`ExtractionStreamStore.extraction_report(user_id, request_id)` returns the status
and output counts together using one SQL scope and admission lookup. Backends
with custom coverage implementations should override this operation too.
Standalone `extraction_status` and `extraction_counts` remain available. Publish
omits reporting fields on a post-commit reporting failure while preserving
`success=True` for the durable write. HTTP and client schemas are unchanged.

### Publish and retention ownership

Publish diagnostics have two independently thresholded/throttled records under
the existing `REFLEXIO_PUBLISH_TIMING_*` settings (at most two lines per org per
interval). `publish_timing.total_ms` retains its admission-to-worker-completion
boundary. `publish_http_timing.http_total_ms` measures ASGI entry through sending
the final response body, including authentication/dependencies and response work.
Both share a generated `timing_id`; no request bodies or credentials are logged.
The HTTP record reports `before_handler_ms`, `handler_ms`, `after_handler_ms`,
HTTP status and `response_complete`/`handler_completed`. Missing handler fields
mean the handler did not start or had not finished when HTTP ended; a later
shielded worker can still emit its own record with the same ID. Status 0 means
no response status was observed. Post-response background work is excluded.
Explicit extraction waiting counts in HTTP duration and is marked
`wait_for_response=1`; it remains outside the handler timer. HTTP measurements
exclude network time before ASGI entry and after ASGI send, so are not client RTT.

New top-level handler phases are `publish_config` (the three pre-admission
configuration reads), `worker_dispatch`, and `evaluation_schedule`. The latter
contains `evaluation_config` and `sampling_decision`; these child timers must not
be added again when calculating unattributed time. Configuration freshness and
the sampling write are unchanged. Enterprise HTTP `auth_binding` and
`billing_gate` timers are subsets of `before_handler_ms`: billing dependency
construction and other middleware/framework work remain in the remainder.

Each app composer installs the pure-ASGI timing middleware outermost. Enterprise's
outer instance owns collection, including limited-key rejection, while the
inherited OSS instance passes through the already-open scope.

HTTP publishing runs inside `scheduler_managed_retention()` in the server
adapter. The library's inline retention helper skips that context without
advancing its per-org throttle. Direct embedded calls still sweep at most once
every 300 seconds, including calls with `defer_learning=True`. Context ownership
is reset on exit and does not suppress independent embedded calls in other threads.
`retention_ms` records time in that helper (normally zero for HTTP).

Server row caps are enforced by `LineageGCScheduler` under each project's scope
and application credential. Positive caps start the scheduler independently of
other GC feature flags. It attempts a tick at startup when elected leader, then
waits `lineage_gc.poll_interval_seconds` after a successful tick (default 86400,
24 hours). Failed ticks receive bounded retries at up to 300-second intervals.
The sweep still protects unfinished extraction inputs, overlap context and
retention holds; caps may be exceeded between sweeps. Publishing acknowledges
the durable admission transaction and does not wait for this housekeeping.
