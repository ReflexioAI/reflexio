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

## Services

Two services, started together via `./run_services.sh`:

| Service | Framework | Default Port | Env Var |
|---------|-----------|-------------|---------|
| Backend | FastAPI (uvicorn) | 8081 | `BACKEND_PORT` |
| Docs | Next.js 16 | 8082 | `DOCS_PORT` |

`API_BACKEND_URL` is derived automatically as `http://localhost:${BACKEND_PORT}`.

**Storage backend** — pass `--storage sqlite` (default) or `--storage supabase` to select the data storage backend:
```bash
uv run reflexio services start --storage sqlite    # local SQLite (default)
uv run reflexio services start --storage supabase  # Supabase PostgreSQL
```

Stop services with `./stop_services.sh`.

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
- **Storage**: `LOCAL_STORAGE_PATH`, `SQLITE_FILE_DIRECTORY` (both default to `reflexio/data`)
- **Storage backend**: `REFLEXIO_STORAGE` — `sqlite` (default) or `supabase`. Selects the data storage backend independently from auth configuration.
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
6. `./run_services.sh` (or `/run-services` skill for automatic port handling)

### Notes

- Do NOT modify `.env` for port variables — export in shell instead

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | `./stop_services.sh` or `lsof -i :8081` to find the process |
| Services won't start | Check `.env` has at least one LLM API key set |
| `uv sync` fails | Ensure Python >= 3.14, try `uv self update` |
| Docs frontend won't start | Run `npm --prefix docs install` first |
| Import errors after pull | Run `uv sync` to update dependencies |
