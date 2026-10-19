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

Local inference starts before the backend. Its readiness wait allows up to five
minutes for the first embedding/reranker model download and warmup; the ordinary
service readiness wait remains one minute. A failed inference startup stops the
started processes and reports whether the child exited (with its exit code) or
the model download/warmup timed out. Check the preceding `[embedding]` logs for
the failing stage. This deadline accommodation is platform-independent; a
Windows-only model initialization failure still needs its own diagnosis.

### Standalone OSS access

The standalone backend binds `127.0.0.1` by default. With no
`REFLEXIO_API_KEY`, data endpoints accept only loopback clients using a loopback
hostname; browser origins must also be loopback. Remote callers and untrusted
browser origins receive 401. Health, version, and API documentation remain public. In Swagger, use Authorize
to supply the key. In the bundled docs, enter it in the top-bar API key field;
it stays in memory and clears on endpoint changes or page reloads.

For network access, configure a strong `REFLEXIO_API_KEY` before startup and use
`reflexio services start --backend-host 0.0.0.0`. Every data request, including
local requests, then needs `Authorization: Bearer <key>`; the Python client reads
the same environment variable. Use TLS at the network boundary. Binding all
interfaces without a key does not grant remote data access. The equivalent
standalone entrypoint is `python -m reflexio.server --host 0.0.0.0 --port 8061`.

This policy belongs to `reflexio.server.api:app`. Enterprise and other hosts
composing `create_app` supply their own authentication and are responsible for
their listener configuration. Their authentication dependency is unchanged.
The standalone launcher disables proxy-header rewriting, and no-key access
rejects forwarding headers. Reverse-proxy deployments must configure the key.

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
| `--backend-host ADDRESS` | `127.0.0.1` | Backend bind address; network access requires `REFLEXIO_API_KEY` |
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

The shared Slowapi limiter defaults to client-IP buckets. Enterprise calls
`configure_rate_limiter` to use validated token buckets; the override applies to
routes decorated before or after configuration. Default decorators keep a stable
dispatcher instead of capturing a stale key function. Passing
`get_rate_limit_key` restores IP buckets. Explicit per-route key functions retain
their own behavior, and configuring a key does not change any numerical limits.

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

Nomic uses the maintained native Transformers implementation with remote model
code disabled. The full package declares the tested minimum `transformers>=5.6.2`
without an upper cap. Release qualification must run real installed-wheel encode
checks via `scripts/verify_nomic_artifact.py`; see the README's Nomic dependency
compatibility section. Keep its checkpoint, 512-dimensional normalization and
serialized encoding contract intact when upgrading dependencies.

### Dependency compatibility

Prefer updating our caller to the maintained dependency API and declaring a
tested minimum version. Do not add an upper bound, exact pin, downgrade or
resolver override merely to hide a failure with a newer package. Reproduce the
failure in a clean installed artifact, inspect current upstream support, and
exercise the operation that failed. Importing a package or loading model weights
alone does not verify inference.

A necessary compatibility restriction must have a reproduced failure or upstream
issue, the narrowest affected range, an owning component, qualification evidence,
and a documented widening/removal condition. Review it at the next owning release
or upstream fix, and immediately for relevant security updates. Do not blindly
remove bounds that protect security, API contracts or model/runtime compatibility.

Keep release lockfiles and frozen builds reproducible. Dedicated deployment pins
belong to a qualified environment and require a tested upgrade path; they should
not automatically become restrictions on the public package. Check manifests,
extras, overrides, Docker install steps and runtime version guards across every
repository, including submodules. A source lockfile does not constrain what a
fresh PyPI installation resolves. Keep server/model dependencies out of the
lightweight client distribution.

For example, SDK 0.2.32's fresh-install Nomic warm-up failed on Transformers
5.19.0 when the loader selected legacy remote model code. Native Nomic loading
with `trust_remote_code=False` passed diagnostic inference using the same weights
on 5.19.0, with matching normalized 512-dimensional vectors for the tested inputs.
The proposed `transformers<5.17` restriction was never applied. Qualify that
forward-compatible loader rather than treating the legacy failure as proof that
newer Transformers must be prohibited. Full release qualification remains
separate from these diagnostic probes.

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

### Current model compatibility

LiteLLM `>=1.103.1,<2` is required. Explicit model overrides support OpenAI
`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, and Anthropic `claude-fable-5-1`,
`claude-opus-5-5`, `claude-sonnet-5-5`, plus the existing Haiku 4.5 model.
Direct `openai/` and `anthropic/` prefixes are supported; provider defaults
remain unchanged.

The latest-model adapter tests use the bundled LiteLLM catalog and local HTTP
fixtures: `uv run pytest tests/server/llm/test_latest_provider_models.py -o addopts=`.
They verify provider payloads, Responses routing, native schemas, and reasoning
continuity through tool loops and subprocess snapshots without paid calls.

`ToolCallingChatResponse` carries optional `reasoning_content`, `thinking_blocks`,
`reasoning_items`, and `provider_specific_fields`. Tool history must replay these
unchanged along with tool calls and any accompanying text. For the qualified
GPT-6 and new Claude models, sampling fields are omitted after all overrides
except Luna temperature/top_p with explicit `reasoning_effort="none"`;
new Claude models reject forced tool selection. Missing catalog capabilities
are supplied only for the explicitly qualified IDs, without fabricated pricing
or persistent overrides of subsequently downloaded metadata.

MiniMax `minimax/MiniMax-M3.1-Flash-Preview` and Z.ai `zai/glm-5.3`,
`zai/glm-5.3-flash`, and `zai/glm-5.3-flashx` are also qualified. Their tests
cover prompt-schema output, explicit reasoning-effort forwarding, rejection of
disabled thinking, and real subprocess tool exchanges against a local HTTP
server. These providers retain sampling settings and existing defaults.
FlashX routes to the general API and requires a corresponding key; the other
GLM models retain the existing coding-plan endpoint.

Live subscription smoke tests are opt-in, capped at 4096 output tokens per call
and two tool-loop steps, with no fallback ladder. Load `MINIMAX_API_KEY` and
`ZAI_API_KEY` into the process before running:

```bash
RUN_LOW_PRIORITY=1 uv run pytest tests/e2e_tests/test_latest_secondary_providers_real_llm.py -q -o addopts= -rs
```

This file checks MiniMax M3/M3.1 Preview and GLM 5.2/5.3/5.3 Flash text, JSON,
and tool calls with `assert_litellm_unpatched()` and actual process isolation.
The tests intentionally consume provider quota; normal offline suites skip them.

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

### CLI aggregation operations

`agent-playbooks aggregate` uses the existing durable submit/status APIs rather
than competing for immediate admission to the legacy aggregation endpoint. Only
one request is submitted per selected agent version. `--wait` follows its exact
operation/request/agent identity and reports success only from a committed
successful receipt with a result. Failure, mismatched identity, missing result
or polling deadline is a nonzero CLI outcome. Admission without `--wait` is
reported separately from completion. See [CLI usage](reflexio/cli/README.md#agent-playbooks)
for operation status, timeout and JSON output behavior.

### Storage requirements for publishing

SQLite dataset ownership checks read an established owner without taking a writer
lock, so another connection's write transaction does not block that lookup in
WAL mode. First claims still use an atomic write transaction across processes;
foreign owners and legacy erasure barriers retain their existing rejection rules.
The read path requires the unchanged application-created ownership schema;
custom definitions, indexes or triggers retain the original claim validation.

SQLite initialization reads the current evaluation-index and aggregation schema
before replaying those migrations. Exact current aggregation definitions and the
current evaluation index skip their write work. Missing or changed definitions
retain the original upgrade and rollback paths, including atomic erasure-trigger
replacement. Other migrations still run; this is not a database initialization
cache. Equivalent legacy aggregation definitions may retain the slower path.
Current optimizer indexes are also retained when no legacy jobs need repair;
their comparison accepts whitespace differences emitted by the constraint rebuild.
Current session-outcome and playbook-diagnosis migrations inspect before taking
a writer lock, then recheck under the lock if an upgrade or subject repair is
needed. These checks preserve legacy classification, duplicate retirement,
governance subject backfills and serialized column upgrades. The stall-state
singleton is seeded only when absent, without resetting an existing stall.

SQLite deferred index failures raise `PostCommitStorageError` after the primary
commit; publish preserves the admitted request and reports deferred follow-up.
Extraction-runner and evaluation scheduling failures are contained independently
so one cannot prevent the other follow-up from running. If the extraction runner
cannot start, synchronous callers return deferred work without entering the
coverage waiter.
SQLite standalone write failures roll back their owned implicit transactions;
failures inside `commit_scope` leave rollback to the outer owner. Interrupted
scopes also roll back. Self-committing FTS/vector index operations roll back a
failed replacement so they cannot leave a writer lock behind. Aggregation prepares
agent-playbook embeddings and document expansion before its fenced write scope,
then uses prepared persistence for atomic row, lineage and cluster changes.

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

Publish diagnostics capture one policy at HTTP entry and share it with the
handler and shielded worker. `REFLEXIO_PUBLISH_TIMING_MIN_MS=off` disables timing
(the default); a nonnegative integer enables it at that millisecond threshold,
with `0` reporting all requests. `REFLEXIO_PUBLISH_TIMING_INTERVAL_SECONDS`
remains the independent per-org throttle window (default 60 seconds; `0` disables
throttling). The two record types have separate throttle populations, allowing
at most two lines per org per interval. An in-flight request retains its captured
policy; environment edits affect the next request.

Absent or blank `REFLEXIO_PUBLISH_TIMING_MIN_MS` disables diagnostics. Removed
legacy enable/threshold assignments fail with a migration message, even when
canonical policy is supplied. Invalid canonical thresholds raise without echoing
the value; invalid throttle intervals retain their 60 second fallback. Preserve
managed presets' effective policy when upgrading. See the
[configuration upgrade guide](docs/configuration-upgrade.md).
`publish_timing.total_ms` retains its admission-to-worker-completion
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
waits the code-owned daily housekeeping cadence after a successful tick (86400
seconds, 24 hours). Failed ticks receive bounded retries at up to 300-second intervals.
The sweep still protects unfinished extraction inputs, overlap context and
retention holds; caps may be exceeded between sweeps. Publishing acknowledges
the durable admission transaction and does not wait for this housekeeping.

Cleanup's simple lock is acquired atomically and released only with its holder's
token. If storage raises during release, the manager retains that token for an
explicit retry. A definitive release result forgets the token, including when a
stale takeover means the original holder no longer owns the lock. Retrying after
a lost commit response therefore cannot clear a successor's lock. The retention
sweep itself still attempts release once; this does not add automatic retries.

### Sparse aggregation discovery

`PlaybookAggregationScheduler` accepts an optional `scope_inventory_provider`.
A provider that yields only contexts with due work must also supply the complete
live set of `(org_id, project_id)` tuples for pruning retry and repair state.
An unavailable inventory returns `None` or raises; an empty inventory means
there are no live scopes. Failed or interrupted scans never prune. Existing
full-sweep providers need no changes. Claims and execution are unchanged.

Organization failure streaks belong to the scheduler. Explicit aggregation
reports claim completion back to it, including contention, before executing work;
operation and finalization failures keep their separate scopes. A failed claim
does not reset the organization streak. Durable-operation finalization uses
`playbook-aggregation-operation-finalize:`; incremental finalization uses
`playbook-aggregation-finalize:` so one path cannot clear the other's failures.

## Operational health measurements

`reflexio.server.operational_metrics` is an optional, vendor-neutral health sink.
Deployments register it with `HookRegistry.set_operational_metrics`; without a
sink, emission is a no-op. Sink errors never change application behavior.
This is separate from billable usage events and trace sampling.

The durable worker records claimed non-idle attempt outcomes and duration,
explicit retries, and successful durable commits after the transaction exits.
Window attempts and side-effect delivery attempts are separate. LLM measurements
cover the whole fallback ladder: recovered fallback is success; exhausted calls
are failure; guard cancellation is separate. Call sites emit no user, tenant,
prompt, model endpoint, or request identifiers.

Sparse aggregation consumers can use `on_work_claimed(context)` to requeue a scope
for the next normal tick. The callback runs only after a successful authoritative
claim, within the existing failure/finalization handling; an empty claim does not
requeue. This preserves draining cadence when many items share one scope.

Sparse providers can retain failed scopes with `on_scope_deferred(context, delay)`;
the scheduler emits it on pre-claim failures and when skipping existing backoff.
The provider should retain the scope until that remaining delay expires.


### Search request deadlines

The HTTP `/api/search` path has a five-second server-arrival deadline and returns
504 with `reason=search_deadline` on expiry (no partial results). Set
`REFLEXIO_SEARCH_DEADLINE_ENABLED=false` to disable enforcement while retaining
request timing. Embedded calls have no newly imposed deadline. Timeout responses
do not imply a worker thread has stopped: request admission remains held until
its application and queued/running retrieval work settles. Timing logs for every search
and application outcome metrics are independent of sampled tracing. See the
enterprise managed-cloud health guide for rollout and phase interpretation.

### Mutation testing

The enterprise weekly workflow runs the locked mutmut version on self-hosted
background compute. The configuration uses mutmut 3.8+ list-based `source_paths` and
`pytest_add_cli_args_test_selection`
and `pytest_add_cli_args`; pytest runs serially inside each of the two mutation
children and excludes integration, e2e and paid-provider cases.
Use the console entry point with the locked version: older mutmut 3.5 crashes
when baseline tests start multiprocessing spawn workers.

Run locally with `nice -n 10 uv run mutmut run --max-children 2`, then
`uv run mutmut results --all true` and `uv run mutmut export-cicd-stats`.
Results are in `mutants/mutmut-cicd-stats.json`. The old `--paths-to-mutate`
option and `html` command are unsupported. The former feedback utility target
was removed because that source module no longer exists. `also_copy` supplies
the remaining package so selected mutated modules can import their dependencies,
plus the skill bundles, method documentation and committed `.env.example`
template used during test collection and CLI contract checks. Actual `.env`
files and developer credentials are not copied into mutation workspaces.

Mutation workers exit via `os._exit`. The mutation-only pytest hooks therefore
remove each session's factory-owned temporary directory after fixture teardown,
and record fixture-drain calls against the originating test. Uncaught ENOSPC or
SQLITE_FULL (including wrapped causes and nested exception groups) exits with
infrastructure status 35, so scratch exhaustion cannot count as a killed mutant.
Ordinary pytest temporary file retention is unchanged. Real-engine lifecycle
controls live in
`tests/scripts/test_mutation_scratch.py`.


The test bootstrap adds the directory immediately above `tests/` to the import
path. During mutation testing this is `mutants/`, not the original checkout;
importing original sources hides generated mutants from coverage. The bootstrap
regression exercises both layouts in isolated processes with distinct packages.

### Qualifying reviewer reason-code precedence (#428)

The follow-up [adoption protocol](tests/test_data/reviewer_adoption_protocol.md)
defines development/held-out coverage, semantic oracles and activation gates.
Use `--cases tests/test_data/reviewer_generalization_cases.json --split development`
or `--split holdout` to select one split. Survival-only controls allow acceptance
or a grounded revision and have no exact reason-label oracle; `code_expected`
is their separate label denominator. Inspect saved content as well as scores.
Synthetic fixture turns may specify `source` alongside `role` and `content`.
It becomes `Request.source` and reaches the reviewer's own-evidence metadata,
so a generated/internal `User` turn can retain its producer provenance. Omitted
`source` defaults to an empty string for existing fixtures; role alone does not
establish direct user origin.
Private qualification artifacts require POSIX with `fchmod` support. The CLI
and evaluations with a checkpoint reject unsupported platforms before provider
calls or opening/truncating reports. In-memory evaluation without a checkpoint
does not require that filesystem capability. Unsupported-platform behavior is
covered by simulated tests; native Windows execution has not been qualified.
Report updates write an owner-only temporary file in the destination directory
and atomically replace the destination. They never truncate an existing inode
or follow a destination symlink, and failed replacements remove the temporary
file while preserving the previous report.
The inactive v1.7.0 candidate clarifies unavailable artifact claims versus using
record omissions as proof of absence. It also keeps internal-status proof of
value distinct. It failed a workflow preservation control. Inactive v1.8
clarified preparation-and-delivery atomicity but failed artifact-label stability.
Inactive v1.9 shortened the procedure but failed a persisted-rationale contract.
See the [adoption evaluation report](tests/test_data/reviewer_adoption_evaluation_20261010.md)
for retained failures and remaining gates. Earlier-version results cannot qualify
a new version.
Inactive v1.10 keeps revised rationales restricted to the retained supported core;
review diagnostics belong in the separate reason (retained in review notes),
not in the persisted guidance fields. The shared persisted
prose guard rejects turn labels, bracketed candidate labels and reviewer evidence
labels, while retaining ordinary bare product identifiers such as C1.
V1.10 failed private semantic checks by retaining unsupported prohibitions and
adding an escalation fallback. Inactive v1.11 audits each negative clause and
restricts revision to subtraction and faithful paraphrase of the supported core.
Two additional development controls cover those failures; the holdout is unchanged.
V1.11 failed private checks by retaining an unsupported alternative path, leaking
shorthand evidence citations and rejecting a useful task-specific procedure.
Inactive v1.12 explicitly audits unsupported original branches and preserves
bounded task-specific prerequisites. The shared guard also rejects shorthand
citation lists such as `(E1, E3)`, while allowing bare E-number product identifiers
and formulas such as `SUM(E1, E3)`. It cannot infer every bare identifier's meaning.
V1.12 failed by retaining an unsupported generic escalation path. Inactive v1.13
adds worked subtraction examples in different synthetic domains, but failed private
checks: saved revisions retained unsupported guidance despite matching decision and
label oracles. Independent grounding checks and a lower-temperature replacement
also remain unqualified. See the adoption evaluation report for counts and limits.

Inactive reviewer v1.14.0 tests a different single-call response contract. Trusted
review policy is a system message; context and source text are user-message data.
The model supplies bounded supporting excerpts before decisions. Validation checks
exact substrings against each candidate's final retained evidence, keeping role
and request-source metadata visible; this establishes provenance, not semantic
grounding. Excerpts remain transient, outside saved playbooks, public responses
and ordinary logs. Owner-only evaluation reports retain them for auditing. The
current v1.3 default uses its existing messages and response schema. Qualification
still audits complete instructions, triggers and rationales, including healthy
multi-turn workflows, before holdout, staging and a separate activation change.

Inactive v1.15 keeps the evidence-first contract and explicitly scopes selected
parameters to their current task. Historical values cannot become standing
preferences; acknowledgment occurrence proves neither a requirement nor a benefit.
The evaluation-only `--max-tokens` and `--reasoning-effort` settings apply equally
to both arms. The current GLM-5.2 experiment uses 8192 output tokens and `high`
reasoning, sent through `extra_body` because the installed adapter drops the
standard reasoning argument. A live wire/body and structured-response probe
verified this path. These options do not configure production services.

`reflexio/test_support/reviewer_metrics.py` supplies both evaluation tools with
per-call request, repair, token/cost and timing observations. It pins each initial
and repair request before dispatch and keeps only explicit non-content fields.
Private reports record `calls` even on errors, and `measurement_errors` when
request observations are missing/incomplete; both commands fail such measurements.
Attempt timing includes client/provider slot wait; wall timing covers the complete
logical review. Token-limit equality alone is not proof of truncation. Reports
do not retain provider reasoning, raw logs, headers, credentials or conversation
text in these metric objects. Complete outcome/revision data remains private in
the existing report fields. Qualification still needs independent semantic audit.


Evaluation observation coverage is explicit: `request_lifecycle_complete`,
`timing_complete`, `usage_complete`, and `estimated_cost_complete`.
`observation_complete` requires matched request/end counts, per-request timing,
and token usage; missing observations fail the measurement gate. Missing cost
remains unknown (never zero) and is reported separately from token coverage.
Without an explicit token override, the first observed effective limit is frozen
for every subsequent repair in that logical call.

See the [2026-10-10 qualification report](tests/test_data/reviewer_reason_evaluation_20261010.md)
for pinned sources, repeated real-model results, observed regressions and activation limits.

Inactive v1.17 additionally clarifies that unsupported claimed benefits precede
speculative actions. Its private oracle v2 is separately preregistered; the old
oracle and stopped results are preserved. No default/provider change is made.

Inactive v1.16 additionally limits revised rationales to the retained rule’s
evidence basis, omitting event narration from both rationale and review reason.
Reason is saved as survivor notes and is included in semantic qualification. The frozen controls
and explicit GLM evaluation profile remain unchanged.

The default candidate reviewer remains v1.3.0. The experimental, inactive
v1.4.0 prompt chooses the decision first, then uses the first applicable
reason code: ownership, absence, unseen artifact, causality, internal status,
speculation, other unsupported facts, duplicate, generic, late trigger, compound.
Accepted candidates use `grounded_useful`; revisions label the defect removed.
This preserves the measured subject gate and the fatal-code/revision distinction.
The separate inactive v1.5.0 candidate clarifies the measured disagreements:
independent future tasks are compound even when stated together, necessary
steps of one supported procedure stay together, internal-status proof has
precedence over unsupported causality, and invented future instructions are
speculative rather than unsupported factual attributes. Grounded preferences
still require a duplication and scope check before acceptance. v1.3.0 and
v1.4.0 remain byte-for-byte unchanged. The further inactive v1.6.0 candidate
clarifies that evidence explicitly joining preparation and delivery of one
requested result defines a complete workflow; sharing a topic alone does not.
It also requires preference rationales to cite the stated instruction without
adding an unstated benefit or timing constraint. v1.5.0 is preserved unchanged
because it lost supported delivery steps in the focused positive control.

Run the paired evaluation with a real provider key exported in the shell:

```bash
uv run python scripts/evaluate_review_reason_codes.py \
  --expected-model minimax/MiniMax-M3 --candidate-version 1.6.0 \
  --repeats 5 --out /tmp/reason-codes.json
```

Both arms see the same frozen cases, chronology, model, temperature and empty
fallback list. The command refuses a patched LiteLLM and returns nonzero on
call errors; its report keeps expected, processed and error counts separate.
`--expected-model` checks the generation model resolved from the normal provider
configuration; it does not select or override that model. A mismatch fails
before paid calls. Export only the intended provider's key for the documented
MiniMax run. The candidate-version flag selects the inactive treatment; the
baseline remains v1.3.0. Reports hash the input cases and both prompt templates,
and retain revision content and evidence IDs for manual
inspection of useful-core preservation; correct decision/code labels alone
cannot establish revision quality.
The corpus covers every label plus accepted and revisable controls in reporting,
code-review, translation, troubleshooting and upload contexts. The v1.5/v1.6 evaluation
is a new matched comparison: two formerly ambiguous inputs were clarified and
three boundary controls added. Do not aggregate these scores with the original
14-case measurements. Run repeated paired disagreement cases and positive
controls first, then repeat the full frozen corpus without editing the prompt. It is synthetic except for a publicly reported
candidate shape from #428; it is not the original production candidate pool.
Do not use lexical similarity to group candidates or equate label consistency
with correct accept/revise/reject decisions.

Before activating either experimental candidate, require matched, error-free processing and improved
label accuracy without loss of decision accuracy or any positive-control
survivors. Inspect each mismatch and the model's explanation, rather than
hiding losses in aggregate totals. Then qualify against the original frozen
production candidate pools, healthy same-tenant windows, and a different domain
using the enterprise paired-review harness. That requires the normal
production-read permissions. The synthetic corpus alone cannot justify a
production-quality or original-defect-resolution claim. Re-measure after any
prompt edit, preserving both fatal gates and useful revision cores.

Mutation pytest bootstraps select an empty, owned temporary environment file before loading configuration or importing the server, preventing collection from creating a persistent user `.env`. Ordinary pytest retains its provider credential loading.

Inactive v1.18 requires an identifiable artifact claim with missing supporting evidence before choosing
that reason; unsupported conversational benefits remain causal defects. Compare
unchanged development boundary controls plus the separately frozen visible-report/
unobserved-reception control before the frozen private priority run.

Inactive v1.19 writes rationale as the source-backed requirement and reason/notes
as the review decision, without historical agent-execution narration. Supported
workflow actions remain in content/trigger. Separately frozen announced/verified
setup controls precede private replay; no default activation.

Inactive v1.20 keeps review notes about removed claims and retained rules, avoiding
claims that questions/events were absent. Separately frozen repeated-question
coverage precedes private replay; original benchmarks remain unchanged.

### Local docs network binding

The docs server started by `reflexio services start` binds to `127.0.0.1`, just
like the standalone backend. For remote browsing, forward the docs and backend
ports through SSH; an API key is still required when configured. This does not
change the bind address of production containers or custom embedded backend apps.

## Generation completion and CLI errors

`publish` exits nonzero when the server reports a failure (`success=False`),
in both human and `--json` modes. JSON failures use the standard `ok:false`
error envelope. File/JSONL batches stop at the first rejection: earlier accepted
payloads stay accepted and later payloads are not sent. Inspect those earlier
results before retrying; rerunning the entire input can duplicate accepted data.
Post-commit follow-up failures return accepted work with a warning rather than
a false admission failure. An accepted publish with deferred extraction remains successful; acceptance and
learning completion are separate states.

`user-profiles regenerate --wait` and `user-playbooks regenerate --wait` report
success after generation completes and any nonempty output is promoted. A
successful zero-output attempt skips automatic promotion and reports zero
promoted records; it does not change existing rows through promotion. Explicit
upgrade commands retain their existing no-pending behavior. An omitted generated
count stays unknown (`None`) in the SDK and does not trigger the zero-output skip. Failed or canceled
batches exit nonzero in text and `--json` modes, leave valid pending output stored,
and skip automatic promotion. A promotion failure also exits nonzero. Inspect
operation status and pending records before retrying; regeneration can delete
prior pending records, so a blind whole-batch retry is not a safe recovery.

Batch progress is `failed` if any selected user fails. Cancellation remains
`cancelled`; both terminal outcomes retain processed/failed-user statistics and
generated counts. A successful batch may legitimately produce zero records.
A synchronous SDK polling error reports `success=False`: acceptance does not
prove completion. Generated counts represent records, rather than processed users.

Synchronous regeneration requires a server that assigns an attempt ID at admission.
Deploy the matching server before rolling out SDK 0.2.32 clients using `--wait`.
Older servers return a service label instead; the SDK reports non-success rather
than treating an unrelated completion as its own. Admission is atomic; a busy
service rejects another rerun without starting it. Selection/no-work/preparation
failures are recorded against the admitted attempt. Cancellation requested during
selection survives batch initialization; stale workers cannot replace newer progress.
Exact attempt IDs tolerate client/server clock skew and remain in final SDK responses.
Asynchronous submission remains fire-and-forget.

### Prepared playbook evidence references

Prepared playbook extraction uses exact visible source-span references such as
`i7:content:0:32`; a turn label such as `T1` can refer to several fragments and
cannot substitute for a span reference. Shared ordinary and expert message
construction includes the existing prepared-source citation instruction whenever
an overflow presentation is active, including direct and resumed construction.
The overflow call factory reuses these messages rather than appending a second
instruction. Unprepared extraction keeps its turn-reference contract. Strict
source visibility validation, bounded corrective repair, and evidence-carrier
limits are unchanged.

### Supabase storage schema compatibility

`StorageConfigSupabase` accepts legacy `read_url`/`read_key` fields only to discard
them during normalization; they are absent from canonical output and the schema.
Enterprise Supabase reads use native SQL, so these HTTP reader fields did not
select a replica. `StorageConfigPostgres.read_db_url` and reader pool settings
remain supported for actual native replica routing.

### Stored configuration validation

Persisted reads use `validate_stored_config`: removed unknown fields are ignored
and missing fields use schema defaults, but invalid recognized values raise
`StoredConfigurationError` from `reflexio.config_migration`. Existing
`ConfigurationMigrationError` subclasses remain within this family. Local-file
loading and CLI storage discovery propagate these errors rather than creating or
writing a default policy. Correct the saved configuration before retrying.
Errors and adapter logs name only top-level schema settings; supplied values and
dynamic dictionary keys are omitted. API writes still use ordinary Pydantic
validation. Missing-file, unreadable JSON/decryption and explicit legacy-disk
handling are separate existing contracts, not changed by this validation rule.

### Durable extraction timing

`reflexio/durable_learning_policy.py` resolves background discovery timing once
before worker construction. `REFLEXIO_DURABLE_LEARNING_POLL_SECONDS` controls
pickup latency and discovery read load (default 2 seconds);
`REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS` controls renewable background claim
lifetime and worker-loss recovery (default 300 seconds). These are independent
supported deployment inputs again: the previous retirement fixed their values
without removing their operational purpose. Valid custom values now take effect;
review complete resolved deployment settings when upgrading from the retirement.

Blank values select defaults. Poll seconds must be finite and positive within
`threading.TIMEOUT_MAX`; lease seconds must be positive integral seconds within
signed-64-bit epoch and heartbeat wait headroom. Integral decimal spellings remain
accepted. Invalid values fail startup without disclosing supplied values.
`resolve_discovery_timing(mapping)` and `validate_discovery_timing_env(mapping)`
use the supplied mapping exclusively, even when empty. Scheduler and enterprise
failure reporting capture the same policy at construction; restart to change it.

Worker capacity remains independently configurable through
`REFLEXIO_PUBLISH_LEARNING_WORKERS`. Claim fencing and heartbeat renewal are
unchanged. Explicit/manual user leases and pending-run claims retain separate
lifetimes and are not controlled by these background discovery settings.


Housekeeping cadence and lineage timing migration checks live in
`reflexio/housekeeping_policy.py`. The former `lineage_gc.poll_interval_seconds`
accepts only its legacy default (86400) and is absent from canonical configuration;
custom lineage cadence still requires explicit migration. In contrast,
`governance_retention.audit_events_delete_batch_limit` is supported again: a
positive integer up to PostgreSQL int32, default 500. Enterprise audit cleanup
applies the configured limit separately to audit events and erasure tombstones
in the existing transaction. The earlier retirement did not eliminate this
throughput/transaction-size choice. Stored custom values are preserved; invalid
values fail loading without resetting other saved policy. Retention enablement,
ages, grace windows and holds remain independent. OSS accepts this enterprise
policy without executing audit-event cleanup.

### Session outcome cutovers

SQLite cutover membership matches both user and session. Transactional stores
may mark `SessionJudgmentInputs.requests_ordered` to preserve database ordering
before public timestamps are converted to whole seconds. The internal
`OutcomePrefixPrecondition.verdict_settled_at` accepts fractional epoch seconds;
public outcome timestamps keep their existing integer representation.


### Inference endpoint and process ownership

`reflexio.inference_launch` is the import-light shared resolver for provider URLs,
CLI ordinary-child ownership, and enterprise Docker startup/health gating. An
explicit `REFLEXIO_EMBEDDING_SERVICE_URL`, including localhost, attaches to a
caller-owned process. With no URL, the launcher owns the separate inference
child; API workers never load the local model. The enterprise API-local query
child retains its independent reserved route.

The legacy daemon hostname and disable-local switch are retired. Custom legacy
hostname-only routes or disabled ordinary startup without an endpoint raise a
value-free `ConfigurationMigrationError`; specify the full URL to migrate.
Hostname-based local-service routes used a 30-second request default, while an
explicit URL selects the existing 2-second internal-service default. Set
`REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS` explicitly when preserving that older
budget. Shadowed legacy settings do not override a canonical URL.


### Pending information request admission

`max_pending_followups_per_scope`, including its `tool_overrides.*` form, is
retired. It selected a warning after a separate backlog query; it never limited
admission. Old values are ignored and omitted from canonical configuration,
without replacing them with a fixed threshold. `ask_human` now goes directly to
the durable create-or-attach operation, so failure of an observational count
cannot prevent request creation. TTLs, per-tool answer relevance/cache/validity,
retry budgets and signing-secret rotation retain their existing authority.

The `pending_followup_soft_cap_exceeded` warning is deliberately removed. Migrate
any log rules using that event explicitly. Existing created/attached events and
`extraction_agent_async_info_requested` usage outcomes remain; those describe
requests, not an aggregate unresolved-dependency backlog gauge.

Admitted profile packages retain their original pending-policy snapshot and
contract digest when reconstructed or consumed after upgrade. Only the retired
warning fields may differ from live configuration; unknown fields, signing-secret
digests and real execution policy remain strict. Never rewrite a stored snapshot
or reset paid-attempt journals to make an old package resume.

### Shared maintenance scheduling

`REFLEXIO_MAINTENANCE_INTERVAL_SECONDS` sets the process-level cadence for
lineage, row-cap and registered retention maintenance (positive integer seconds,
default 86400; blank uses the default, invalid/zero/negative values fail).
`MaintenanceSchedule` captures that interval, the leader gate and healthy-wait
provider before bootstrap configuration reads. The composer installs one complete
schedule with `set_maintenance_schedule`; later installs do not change an
existing worker. Standalone OSS waits the captured interval after healthy work;
enterprise may align first and subsequent healthy waits to its UTC anchor.

Three bounded fast retries use min(300,cadence), then back off to the healthy
schedule. The separate no-organizations bootstrap retry stays five seconds.
Always-global sweeps still run before bootstrap discovery, and registered sweeps
and row caps still admit the worker when bootstrap policy cannot be read. This
does not make an unreadable organization's own storage/config usable.

Per-org retention holds, ages, batches and scope remain independent. Retired
`lineage_gc.poll_interval_seconds` custom values still require explicit migration
to a fleet policy; a bootstrap org is not a schedule authority for every org.
