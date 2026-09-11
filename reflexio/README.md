# /reflexio
Description: Shared Python package for learning user profiles and agent playbooks from interactions, with SDK, CLI, local-library, and HTTP entry points.

Paths below are relative to this package. The repository's [public README](../README.md) covers installation and examples.

## Main Entry Points


| Path | When to modify it | Component map |
|------|-------------------|---------------|
| `__init__.py` | Public Python exports | |
| `client/client.py` | Typed HTTP SDK, Bearer authentication, sync calls and explicit async methods | [SDK reference](../client_dist/README.md) |
| `models/api_schema/` | Shared API and storage-facing contracts; domain entities live in `models/api_schema/domain/` | |
| `models/config_schema.py` | Shared `Config` and extractor/storage configuration | |
| `models/profile_id.py` | Canonical UUIDv4 profile identities | |
| `lib/reflexio_lib.py` | `Reflexio` local-library facade; focused `_*.py` mixins own domain operations | |
| `lib/generation_client.py` | Generation client protocol used by callers | |
| `cli/app.py`, `cli/commands/` | Typer command registration and handlers | [CLI](cli/README.md) |
| `server/api.py`, `server/routes/` | FastAPI composition and domain routes | [Server](server/README.md) |
| `server/api_endpoints/` | Request context, shared handlers, and clarification routes | [Endpoint helpers](server/api_endpoints/README.md) |
| `server/services/` | Extraction, aggregation, evaluation, retrieval, and storage | [Services](server/services/README.md) |
| `server/extensions.py` | Optional capability, hook, and typed runtime-service registration | |
| `server/prompt/prompt_bank/` | Versioned LLM prompts | [Prompts](server/prompt/prompt_bank/README.md) |
| `server/site_var/` | Model settings, retrieval defaults, and feature flags | [Site variables](server/site_var/README.md) |
| `mem0/` | Optional hosted mem0 wrappers and scoped cleanup | [mem0](mem0/README.md) |
| `integrations/` | External agent integrations | [OpenClaw](integrations/openclaw/README.md), [embedded OpenClaw](integrations/openclaw-embedded/README.md) |
| `benchmarks/retrieval_latency/` | Storage/library retrieval timing | [Benchmark](benchmarks/retrieval_latency/README.md) |
| `test_support/` | Shared test fixtures and helpers | |

## Purpose


1. **Remote access** — `ReflexioClient` sends typed API requests; the CLI reuses it for publishing, search, and configuration.
2. **Local access** — `Reflexio` runs services directly without HTTP; LLM/provider calls can still use the network.
3. **Learning and retrieval** — Shared services produce profiles and playbooks, evaluate sessions, and retrieve relevant learning.
4. **Deployment reuse** — Optional capabilities extend the same server without importing enterprise implementation code.

## Architecture Pattern


```text
CLI / external application -> client/client.py -> server/routes/
Local application ----------------------------> lib/reflexio_lib.py
server/routes/ -> RequestContext + get_reflexio() -> lib/reflexio_lib.py
  -> services/generation_service.py
     -> durable_learning/ (admission -> user lease -> frozen windows -> fenced commit)
        -> profile/ and playbook/ -> BaseStorage
     -> agent_success_evaluation/ (deferred session evaluation)
  -> unified_search_service.py -> pre_retrieval/ + storage/ + retrieval/
```

- **Generation actors** load configuration, run extractors/evaluators, and persist results through `BaseStorage`.
- **Automatic extraction** uses durable streams with independent project/kind cursors under one user lease. Legacy `learning_jobs` remains only for reconciliation.
- **Paused extraction** uses `services/extraction/` to persist human clarification and resume/finalize idempotently.
- **Aggregation** is fenced per agent version; agent-playbook successors retain the approval workflow.
- **Storage selection** belongs to the configurator. OSS defaults to SQLite; deployment extensions supply other factories.
- **Runtime data** is outside this package (normally under `~/.reflexio`, resolved through `defaults.py` and `cli/paths.py`); there is no checked-in `reflexio/data/` directory.

## Key Endpoints / Commands / Contracts


- **HTTP routes**: the [server route map](server/README.md#api-endpoints) groups the complete publish, search, lifecycle, evaluation, experiment, and clarification surface.
- **CLI commands**: the [CLI command map](cli/README.md) covers services, publish, search/context, data management, config, auth, and diagnostics.
- **Deferred learning**: `publish_interaction(..., wait_for_response=False)` returns after durable admission; `get_learning_status(request_id)` reports progress.
- **Shared schemas**: client, CLI, and server use `models/api_schema/`; enterprise-only schemas belong in the consuming extension.
- **Documentation and tests**: [interactive docs](../docs/README.md), [notebooks](../notebooks/README.md), and `../tests/` sit at repository level.

## Requirements / Problems to Avoid


- **API handlers use `get_reflexio()`**, never fresh `Reflexio()` instances.
- **Use `request_context.storage`**, not concrete storage imports in business logic.
- **Use `LiteLLMClient` and `prompt_manager.render_prompt()`** for model calls and prompts.
- **Keep `tool_can_use` at root `Config` level**; extraction and success evaluation share it.
- **Preserve governance, lineage, and fencing contracts** when changing persistence; see the [service requirements](server/services/README.md#requirements--problems-to-avoid).
- **Keep OSS independent**; register optional providers through `server/extensions.py`.
