# Reflexio
Description: Enable AI agent to self-improve through user interactions

## Main Components

| Directory | Description | Details |
|-----------|-------------|---------|
| `src/server/` | FastAPI backend - processes interactions, generates profiles, extracts playbooks | [README](src/server/README.md) |
| `src/reflexio_lib/` | Core library - `Reflexio` orchestrator connecting API to services | `reflexio_lib.py` |
| `src/reflexio_client/` | Python SDK for interacting with Reflexio API | [README](src/README.md) |
| `src/reflexio_commons/` | Shared schemas and configuration models | [README](src/README.md) |
| `src/website/` | Next.js frontend - profiles, interactions, playbooks, evaluations, account, auth UI | `app/`, `components/` |
| `demo/` | Conversation simulation demo - scenarios, simulator, and live viewer | [README](demo/readme.md) |
| `docs/` | API reference documentation site (Next.js) | `app/`, `components/`, `lib/` |

## Architecture

```
Client (SDK/Web)
  -> FastAPI (server/api.py)
    -> get_reflexio() (server/cache/)
      -> Reflexio (reflexio_lib/)
        -> GenerationService (server/services/)
          ├─> ProfileGenerationService -> ProfileExtractor(s) -> Storage
          ├─> PlaybookGenerationService -> PlaybookExtractor(s) -> Storage
          └─> GroupEvaluationScheduler (deferred 10 min) -> Evaluator(s) -> Storage
```

## Prerequisites

| Tool | Version | Purpose | Install |
|------|---------|---------|---------|
| uv | latest | Python dependency management | [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Node.js + npm | >= 18 | Frontend and docs build | [nodejs.org](https://nodejs.org/) |
| Biome | latest | TypeScript/JavaScript lint & format | `npm install --save-dev @biomejs/biome` (per-project) |

## Quick Start

```shell
cp .env.example .env                         # Configure environment (set at least one LLM API key)
uv sync                                      # Install Python dependencies (includes workspace packages)
npm --prefix src/website install         # Install frontend dependencies
npm --prefix src/public_docs install     # Install docs dependencies
./run_services.sh                             # Starts API (8081), Website (8080), Docs (8082)
./stop_services.sh                            # Stop all services
```

**Claude Code users:** Run `/run-services` (in claude code) instead of `./run_services.sh` (in bash) — it auto-installs missing dependencies, health-checks services, and diagnoses/fixes/retries on failure.

## Development

**Code Quality:**
- **Python:** Ruff (lint + format) and Pyright (type check)
- **TypeScript/JavaScript:** Biome (lint + format) and tsc (type check)

**Testing:**
```python
import reflexio
client = reflexio.ReflexioClient(api_key="your-api-key", url_endpoint="http://127.0.0.1:8081/")
```
See `notebooks/reflexio_cookbook.ipynb` and `src/tests/readme.md`

## Publishing

```shell
# Update versions in pyproject.toml files first
cd src/reflexio_commons && uv build && uv publish
cd src/reflexio_client && uv build && uv publish
```

## Key Rules

**Reflexio**:
- **NEVER instantiate `Reflexio()` directly** in API endpoints
- **ALWAYS use** `get_reflexio()` from `server/cache/`

**Storage**:
- **NEVER import storage implementations directly**
- **ALWAYS use** `request_context.storage` (type: BaseStorage)

**LLM**:
- **NEVER import OpenAIClient/ClaudeClient directly**
- **ALWAYS use** `LiteLLMClient` (uses LiteLLM for multi-provider support)

**Prompts**:
- **NEVER hardcode prompts**
- **ALWAYS use** `request_context.prompt_manager.render_prompt(prompt_id, variables)`

**Config**:
- **`tool_can_use` lives at root `Config` level** - Shared across success evaluation and playbook extraction (NOT per-`AgentSuccessConfig`)
