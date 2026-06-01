# Reflexio Code Map
Describe the code structure and component dependencies for source code of reflexio

## Table of Contents

- [Overview](#overview)
- [models and client](#models-and-client)
- [cli](#cli)
- [reflexio_lib](#reflexio_lib)
- [server](#server)
- [data](#data)
- [See Also](#see-also)

## Overview
Reflexio is a user profiling and agent playbook system with three main access patterns:

1. **Remote API Access** (`client`) - Applications use Python SDK to call REST API
2. **Local Library Access** (`reflexio_lib`) - Direct synchronous access without HTTP layer
3. **CLI Access** (`cli`) - Local command-line workflows for services, publishing, search, auth, config, and diagnostics

**Core Flow**: User Interactions → Server Processing → Profile/Playbook/Evaluation → Storage

**Shared Components**:
- `models` - API and internal schemas shared by client, CLI, and server
- `server` - FastAPI backend with LLM-based processing services
- `data` - Bundled configs and local fixtures
- `docs` - Next.js API documentation site

## models and client
Description: Shared data contracts and the Python SDK used by external applications, the CLI, and server endpoint helpers

### models
**Path**: `models/`

#### Main Entry Points
- **API Schemas**: `models/api_schema/` - Pydantic request/response models for public API surfaces
- **Internal Schemas**: `models/api_schema/internal_schema.py` - Storage-facing profile, playbook, request, evaluation, and agent-run models
- **Validators**: `models/api_schema/validators.py` - Cross-schema validation helpers

#### Purpose
Provides type-safe data contracts between client and server:
1. **Service Schemas** - Interactions, requests, profiles, user playbooks, agent playbooks, evaluations, and stall-state records
2. **Retriever Schemas** - Search/get/set requests and responses
3. **Login/Auth Schemas** - Credentials, API tokens, feature flags, and organization/account responses
4. **Config Schema** - YAML/API configuration structure (`tool_can_use` at root `Config` level, shared across services)

### client
**Path**: `client/`

Description: Python SDK for interacting with Reflexio API remotely

#### Main Entry Point
- **Client**: `client.py` - `ReflexioClient` class

#### Purpose
Remote API client for applications to:
1. **Publish interactions** - Send user interactions to server for processing
2. **Search/retrieve data** - Query profiles, interactions, playbooks, evaluations, and context
3. **Manage profiles/playbooks** - Delete, regenerate, and update status where supported by API endpoints
4. **Configure** - Set/get organization configuration

#### Architecture Pattern
Async HTTP client wrapping typed models from `models/api_schema/`. Automatically handles authentication via Bearer tokens.

## cli
Description: Command-line entry point for operating Reflexio locally and against a running server

### Main Entry Points
- **CLI app**: `cli/` - Typer command groups for services, publish/search/context, auth, config, status, and diagnostics
- **Reference**: `cli/README.md` - Command map and common workflows

### Purpose
Local operator interface to:
1. **Run services** - Start/stop backend, docs, and optional embedding service
2. **Publish interactions** - Send JSON, JSONL, stdin, or quick single-turn payloads
3. **Search context** - Query profiles, user playbooks, and agent playbooks
4. **Inspect/manage data** - List/delete/regenerate profiles and playbooks
5. **Configure/authenticate** - Manage API keys, server URL, and configuration

### Architecture Pattern
Thin Typer layer over the Python client and local service manager. Use `uv run reflexio --help` to inspect command groups.

## reflexio_lib
Description: Local Python library interface for direct (non-API) access to Reflexio functionality

### Main Entry Point
- **Library**: `reflexio_lib.py` - `Reflexio` class

### Purpose
Direct programmatic access without HTTP/API layer:
1. **Same interface as client** - Mirror of `ReflexioClient` but synchronous
2. **Local execution** - Runs services directly (no network calls)
3. **Testing/debugging** - Useful for local development and testing

### Architecture Pattern
Creates `RequestContext` and directly calls `GenerationService` - bypasses FastAPI layer. Methods are **synchronous** unlike `ReflexioClient`.

## server
Description: FastAPI backend server that processes user interactions to generate profiles, extract playbooks, and evaluate agent success

**Detailed Documentation**: See [`reflexio/server/README.md`](server/README.md) for component details, including the [Prompt Bank](server/prompt/prompt_bank/README.md), [Playbook Service](server/services/playbook/README.md), and [Site Variables](server/site_var/README.md)

### Main Entry Points
- **API**: `api.py` - FastAPI routes
- **Endpoint Helpers**: `api_endpoints/` - Request handlers calling `Reflexio` (reflexio_lib)
- **Core Service**: `services/generation_service.py` - Main orchestrator

### Purpose
Receives user interactions from clients and processes them to:
1. **Generate user profiles** - Extract and maintain user preferences/traits from behavior
2. **Extract playbooks** - Identify issues and improvement opportunities for developers
3. **Evaluate agent success** - Determine if agent successfully fulfilled user's needs

### Component Relationships
```
client (Python SDK)
  -> api.py (FastAPI routes)
    -> api_endpoints/ (request handlers)
      -> reflexio_lib.Reflexio (main entry)
        -> services/generation_service.py (orchestrator)
          ├─> services/profile/ -> storage (BaseStorage)
          ├─> services/playbook/ (playbook extraction) -> storage (BaseStorage)
          └─> services/agent_success_evaluation/ -> storage (BaseStorage)
```

### Key Components
- **`api_endpoints/`**: Request handling, `RequestContext` (bundles storage/config/prompts), auth
- **`db/`**: Auth & config storage only (SQLite) - NOT for profiles/interactions
- **`llm/`**: Unified LLM client (auto-detects OpenAI/Claude from model name)
- **`prompt/`**: Versioned prompt templates in `prompt_bank/`
- **`services/`**: Core business logic
  - `generation_service.py` - Orchestrator (runs profile/playbook/success services)
  - `base_generation_service.py` - Abstract base for parallel actor execution
  - `profile/` - Profile extraction & updates
  - `playbook/` - Playbook extraction, consolidation, and aggregation
  - `agent_success_evaluation/` - Success evaluation
  - `reflection/` - Post-horizon reflection extraction
  - `extraction/` - Resumable async extraction agent infrastructure
  - `shadow_comparison/` - Per-turn regular vs shadow verdict judge
  - `evaluation_overview/` - Evaluation-page aggregates and hero metrics
  - `playbook_optimizer/` - Scenario-based playbook optimization experiments
  - `braintrust/` - Braintrust eval export/sync support
  - `storage/` - Abstract layer (SQLite prod, LocalJSON test)
  - `pre_retrieval/` - Query rewriting and document expansion helpers
  - `configurator/` - YAML config loader
- **`site_var/`**: Global settings singleton

### Architecture Patterns

**Service Pattern** (BaseGenerationService):
1. Load configs from YAML -> 2. Create actors from configs -> 3. Run actors in parallel (ThreadPoolExecutor) -> 4. Save results to storage

**Actor Pattern**: Multiple actors (extractors/evaluators) run in parallel, each processing interactions independently, results aggregated

**Storage Abstraction**: All access via `BaseStorage` interface, implementation selected by configurator, supports vector similarity search

**Data Flow**: `User Interaction -> Storage (save) -> Services (parallel: LLM + Prompts) -> Results -> Storage (save)`


## data
Description: Local storage directory for configuration files and SQLite databases

### Main Entry Points
- **Configs**: `configs/` - YAML configuration files for extractors and evaluators
- **Database**: `sql_app.db` - SQLite database for auth and config storage
- **JSON Storage**: `user_profiles_*.json` - Local JSON files for testing

### Purpose
Local data storage for:
1. **Configuration files** - YAML configs defining extraction/evaluation behavior
2. **Authentication database** - User credentials and API tokens (SQLite/Postgres)
3. **Test data** - LocalJsonStorage files for development/testing

### Architecture Pattern
Referenced by `SimpleConfigurator` for loading configs and by database operations for auth/config persistence. Not directly accessed by application code.

## See Also

- [Server README](server/README.md) -- detailed component documentation for the FastAPI backend
- [Prompt Bank README](server/prompt/prompt_bank/README.md) -- versioned prompt template system
- [Playbook Service README](server/services/playbook/README.md) -- playbook extraction, aggregation, and deduplication pipeline
- [Site Variables README](server/site_var/README.md) -- global configuration and feature flags
- [Retrieval Latency Benchmarks](benchmarks/retrieval_latency/README.md) -- search performance benchmarking
- [OpenClaw Integration](integrations/openclaw/README.md) -- federated OpenClaw plugin setup and behavior
