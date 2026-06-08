# server/api_endpoints
Description: Bridge between FastAPI routes and business logic — builds `RequestContext`, validates requests, and delegates into `Reflexio`. Most endpoints are registered on the `core_router` in `../api.py`; the files here are the handlers/helpers it calls.

> For the complete endpoint list (publish, retrieval, search, profile/playbook lifecycle, evaluation, Braintrust, operations), see the parent [server README](../README.md#api-endpoints).

## Files

| File | Purpose |
|------|---------|
| `request_context.py` | `RequestContext` — bundles `org_id`, `storage`, `configurator`, `prompt_manager`. Built per request via `get_request_context()` (FastAPI `Depends`). The one object every handler reads storage/config/prompts through. |
| `publisher_api.py` | Publishing + direct CRUD helpers: `add_user_interaction/profile/playbook`, `update_*`, and the full family of single / by-ids / bulk delete helpers for interactions, profiles, playbooks, requests, and sessions; plus `run_playbook_aggregation()` and `clear_user_data()`. |
| `account_api.py` | Identity/config helpers behind `/api/whoami`, `/api/my_config`. |
| `health_api.py` | `GET /`, `/health`, `/healthz`, `/healthz/eval`; `install()` adds response-time + liveness tracking. |
| `pending_tool_call_api.py` | Router for resumable-extraction human clarification: list/get, `resolve`, `answer`, `not_applicable`, `cancel`; HMAC signature verify + migration-retry helpers. |
| `stall_state_api.py` | `GET /api/stall_state`, `POST /api/stall_state/notified` — extraction-agent waiting state. |
| `precondition_checks.py` | `precondition_checks()` — shared request validation. |

## Architecture Pattern

```
api.py (core_router + sub-routers)
  -> Depends(get_request_context) -> RequestContext(org_id, storage, configurator, prompt_manager)
    -> get_reflexio(org_id) -> Reflexio (reflexio_lib) -> services/
```

- **`RequestContext` is the context-passing contract** — handlers receive it via `Depends` and never reach for storage/config/prompts globally.
- **Route handlers call `Reflexio` through `get_reflexio(org_id)`** — these helper files do **not** instantiate `Reflexio()` directly.
- **Auth is injected, not implemented here** — the OS app uses `default_get_org_id` / `DEFAULT_ORG_ID`; the enterprise extension swaps in authenticated org resolution (see `reflexio_ext/server/api_endpoints/`).

## Requirements / Problems to Avoid

- **NEVER instantiate `Reflexio()` in a handler** — use `get_reflexio(org_id)` from `server/cache/`.
- **Keep business logic in `services/`** — endpoints validate, build context, and delegate; they don't embed extraction/evaluation logic.
- **Pending-tool-call writes can race** — use the migration-retry + HMAC-verify helpers already in `pending_tool_call_api.py` rather than writing raw.
