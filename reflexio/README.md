# Reflexio Code Map
Describe the code structure and component dependencies for source code of reflexio

## Overview
Reflexio is a user profiling and agent playbook system with three main access patterns:

1. **Remote API Access** (`reflexio_client`) - Applications use Python SDK to call REST API
2. **Local Library Access** (`reflexio_lib`) - Direct synchronous access without HTTP layer
3. **Web UI** (`website`) - Next.js dashboard for viewing profiles, interactions, playbooks, evaluations, and settings

**Core Flow**: User Interactions → Server Processing → Profile/Playbook/Evaluation → Storage

**Shared Components**:
- `reflexio_commons` - Data schemas and config models (client/server contract)
- `server` - FastAPI backend with LLM-based processing services
- `data` - Local storage for configs and databases

## reflexio_commons and reflexio_client
These are local packages in the main repository:
- `reflexio_commons` source: `reflexio/reflexio_commons/reflexio_commons/`
- `reflexio_client` source: `reflexio/reflexio_client/reflexio/`

### reflexio_commons
**Path**: `reflexio/reflexio_commons/`
**Installed via**: Poetry path dependency with `develop = true` (editable mode)

Description: Shared schemas and configuration models used across client and server

#### Main Entry Points
- **API Schemas**: `reflexio_commons/api_schema/` - Pydantic models for requests/responses
- **Config Schema**: `reflexio_commons/config_schema.py` - Configuration data models

#### Purpose
Provides type-safe data contracts between client and server:
1. **Service Schemas** - User interactions (with `ToolUsed`), profiles, playbooks (with `BlockingIssue`), evaluation results
2. **Retriever Schemas** - Search/get/set requests and responses
3. **Login Schemas** - Authentication tokens and credentials
4. **Config Schema** - YAML configuration structure (`tool_can_use` at root `Config` level, shared across services)

### reflexio_client
**Path**: `reflexio/reflexio_client/`

Description: Python SDK for interacting with Reflexio API remotely

#### Main Entry Point
- **Client**: `reflexio/client.py` - `ReflexioClient` class

#### Purpose
Remote API client for applications to:
1. **Publish interactions** - Send user interactions to server for processing
2. **Search/retrieve data** - Query profiles, interactions, playbooks
3. **Manage profiles** - Delete profiles, view change logs
4. **Configure** - Set/get organization configuration

#### Architecture Pattern
All methods are **async** and return typed Pydantic responses. Automatically handles authentication via Bearer tokens.

## website
Description: Next.js frontend for viewing profiles, interactions, playbooks, and evaluations

### Main Entry Points
- **Profiles**: `app/profiles/page.tsx` - View and search user profiles
- **Interactions**: `app/interactions/page.tsx` - View conversation history
- **Playbooks**: `app/feedbacks/page.tsx` - View and manage agent playbooks
- **Evaluations**: `app/evaluations/page.tsx` - View agent success evaluation results
- **Settings**: `app/settings/page.tsx` - Configuration and settings management

### Purpose
Web-based interface to:
1. **View profiles** - Browse user profiles with search functionality
2. **View interactions** - Inspect user conversation history
3. **Manage playbooks** - View and manage extracted playbooks
4. **Monitor evaluations** - Track agent success metrics and failure analysis
5. **Configure settings** - Manage application configuration

### Architecture Pattern
Built with Next.js App Router and ShadCN UI components. Communicates with FastAPI backend at `http://0.0.0.0:8081`.

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

**Detailed Documentation**: See `reflexio/server/README.md` for component details

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
reflexio_client (Python SDK)
  -> api.py (FastAPI routes)
    -> api_endpoints/ (request handlers)
      -> reflexio_lib.Reflexio (main entry)
        -> services/generation_service.py (orchestrator)
          ├─> services/profile/ -> storage (BaseStorage)
          ├─> services/feedback/ (playbook extraction) -> storage (BaseStorage)
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
  - `feedback/` - Playbook extraction & aggregation
  - `agent_success_evaluation/` - Success evaluation
  - `storage/` - Abstract layer (SQLite prod, LocalJSON test)
  - `retriever/` - Semantic search
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
