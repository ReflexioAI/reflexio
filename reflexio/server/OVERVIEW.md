# Reflexio
Description: Enable AI agent to self-improve through user interactions

## Main Components

| Directory | Description | Details |
|-----------|-------------|---------|
| `reflexio/server/` | FastAPI backend - processes interactions, generates profiles, extracts playbooks | [README](README.md) |
| `reflexio/lib/` | Core library - `Reflexio` orchestrator connecting API to services | `reflexio_lib.py` |
| `reflexio/client/` | Python SDK for interacting with Reflexio API | `client.py` |
| `reflexio/models/` | Shared schemas and configuration models | `api_schema/`, `config_schema.py` |
| `reflexio/cli/` | `reflexio` command-line interface, including `services start` | `run_services.py` |
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
          └─> agent_success_evaluation/scheduler.py:GroupEvaluationScheduler (deferred 10 min) -> agent_success_evaluation/runner.py:run_group_evaluation -> agent_success_evaluation/service.py -> agent_success_evaluation/components/evaluator.py -> Storage
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
npm --prefix docs install                    # Install docs frontend dependencies
./run_services.sh                             # Starts backend (8061) and Docs (8062)
./stop_services.sh                            # Stop all services
```

## Development

**Code Quality:**
- **Python:** Ruff (lint + format) and Pyright (type check)
- **TypeScript/JavaScript:** Biome (lint + format) and tsc (type check)

**Testing:**
```python
import reflexio
client = reflexio.ReflexioClient(api_key="your-api-key", url_endpoint="http://127.0.0.1:8061/")
```
See `notebooks/00_quickstart.ipynb` and the Testing section of [developer.md](../../developer.md)

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
