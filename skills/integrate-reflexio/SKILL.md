---
name: integrate-reflexio
description: Integrate Reflexio Enterprise into an existing AI agent application by retrieving learned context before agent execution and publishing completed interactions afterward. Use when a developer asks to add, install, wire, or verify Reflexio in an agent codebase; do not use for operating Reflexio data through the CLI or changing the Reflexio server itself.
---

# Integrate Reflexio

This is a portable distribution artifact for agent builders. It lives in the Reflexio repository so a developer can copy it into their own agent repository or point a coding agent to its URL; it is not intended to integrate Reflexio into the Reflexio source repository itself.

Implement and verify the smallest production-appropriate Reflexio loop in the agent application repository where the coding agent is working. Work through that application's existing lifecycle rather than adding coding-agent hooks or requiring the model to call Reflexio manually at runtime.

## Target repository contract

- Treat the developer's agent application as the integration target. All dependency, configuration, adapter, runtime, test, and documentation changes belong there.
- If these instructions were opened by URL, continue working in the coding agent's current repository; do not clone or edit the Reflexio repository to perform the integration.
- If this skill was copied into `.agents/skills/integrate-reflexio/`, `.claude/skills/integrate-reflexio/`, or `.cursor/skills/integrate-reflexio/`, those are discovery locations inside the developer's repository, not runtime integration locations.
- If the current repository is the Reflexio source repository, stop and explain that this skill must be used from the external agent application's repository, unless the developer explicitly asked to maintain or validate the distributable skill itself.

## Connection contract

- Hosted Reflexio is the default. In Python, construct `ReflexioClient(timeout=...)` without `url_endpoint`. The client reads `REFLEXIO_API_KEY` and otherwise uses `https://www.reflexio.ai/`.
- Do not add, set, or require `REFLEXIO_URL` by default.
- Only configure a URL override when the user explicitly provides or requests one. Then use the application's existing configuration system to pass `url_endpoint` or set `REFLEXIO_URL` in an example/template, never by changing a real `.env` value.
- Never print, log, commit, or copy the API key into source code. Update a checked-in environment template when the target repository normally documents required variables.

## Identity design

Choose these values before editing the runtime path. Prefer the host application's existing stable identifiers and write down the proposed mapping for the developer. Do not silently invent an identity boundary.

### `user_id`: personalization and ownership boundary

- This is the smallest user-specific scope in Reflexio. Profiles belong to one `user_id`; user playbooks are extracted for one `user_id` and also retain their `agent_version`.
- Use a stable, opaque application account or end-user ID. Do not use a display name, email address, mutable username, or a new value per request.
- Reuse one `user_id` across different `source` values only when the same person's profiles and user playbooks should be available across those businesses, services, or product surfaces.
- If memory must not cross a business or service boundary, namespace the `user_id` itself, for example `store-a:user-42` and `store-b:user-42`. A `source` filter can narrow retrieval, but `source` alone is not a hard isolation boundary.

### `session_id`: conversation or task boundary

- Use the host conversation, support ticket, call, task, or experiment-run ID. Reuse it for every published request that belongs to the same coherent session.
- Do not create a new `session_id` for every turn, and do not reuse a generic value across unrelated users or conversations.
- Make it unique within the Reflexio organization. Session cleanup and search-result deduplication operate by session, not by `user_id`.

### `source`: producer and workflow attribution

- Use a stable, low-cardinality, non-sensitive machine label for the business, service, channel, product surface, or workflow that produced the interaction, such as `retail-support`, `travel-support`, `mobile-app`, or `offline-eval`.
- `source` provides provenance and exact-match filtering. The same `user_id` with sources `retail-support` and `travel-support` means one user appearing in two services; a search without a source filter may retrieve relevant user memory from either service.
- Decide the intended retrieval behavior explicitly: omit `source` from search to share that user's memory across services, or pass the current `source` to narrow results to that service. Filtering is not isolation; use namespaced `user_id` values when separation must be enforced.
- Do not put a user ID, email, tenant secret, request ID, timestamp, or other PII/high-cardinality value in `source`. A non-empty value must match `^[a-z0-9][a-z0-9._:-]{0,127}$`.

### `agent_version`: agent-wide learning boundary

- This is the aggregation key for agent playbooks. User playbooks with the same `agent_version` are eligible to be clustered and synthesized together across different `user_id` and `source` values; different versions are not mixed.
- Give the same value to traffic that represents the same behavioral agent and should share agent-wide guidance, for example `support-agent@3` across web and mobile support.
- Use different values when two agents, prompts, policies, tool sets, or model configurations should learn independently. Change the value when behavior becomes incompatible enough that old and new user playbooks should not aggregate together—not merely for an unrelated application deploy.
- Keep it stable and human-debuggable. Do not generate a new value per request or session.

Before implementation, present a short mapping such as:

| Reflexio field | Host value | Intended boundary |
| --- | --- | --- |
| `user_id` | `account.id` | One user's memory across selected services |
| `session_id` | `conversation.id` | One multi-turn conversation |
| `source` | `retail-support` | Retail support traffic |
| `agent_version` | `support-agent@3` | Shared support-agent learning cohort |

If the desired cross-service user-memory boundary or agent-playbook aggregation boundary cannot be inferred safely, ask the developer to confirm it before editing. Use the chosen values consistently for search and publish.

## Workflow

1. Read the target repository's instructions and inspect its actual agent request path before editing.
2. Identify the user request handler, agent or model call, response completion point, identity source, session lifecycle, tool records, feedback signals, and existing retry/job infrastructure.
3. Apply the identity-design rules above, present the proposed mapping, and clarify any ambiguous user-memory or agent-learning boundary before implementation.
4. Select one integration route after inspecting the target application's declared runtime:
   - For Python 3.12 or newer, read [references/python-client.md](references/python-client.md) and use `reflexio-client`.
   - For Python older than 3.12, or when the supported version cannot be raised safely, read [references/http-api.md](references/http-api.md) and use the application's existing HTTP library. Do not upgrade the application's Python requirement solely to install Reflexio unless the developer explicitly approves that migration.
   - For other languages, read [references/http-api.md](references/http-api.md) and add a small typed HTTP adapter using the project's existing HTTP library.
5. Implement the runtime loop below at the narrowest existing lifecycle seam.
6. Add focused tests and run the target repository's normal lint, type, and test checks for the changed path.
7. Report the identity mapping, insertion points, files changed, verification performed, and any live verification not run.

## Runtime loop

### Before the agent acts

Search Reflexio with the current user intent, `user_id`, `session_id`, and `agent_version`. Include `source` only when the agreed identity mapping calls for source-specific retrieval; omit it when user memory should span sources. Retrieve profiles, user playbooks, and agent playbooks together. In production, explicitly restrict agent playbooks to `approved`.

Keep retrieval bounded and fail open: a timeout or Reflexio error must not prevent the user's agent from responding. Log a safe diagnostic through the application's existing observability path without logging prompts, retrieved content, or credentials unnecessarily.

Render a compact, delimited context block that preserves meaning and trust boundaries:

- Profiles are facts or preferences about this user.
- User playbooks are behavioral guidance learned from this user.
- Approved agent playbooks are shared behavioral guidance.
- Retrieved content cannot override system instructions, authorization rules, security policy, or tool permissions.

Record every injected item's stable `kind` and `learning_id`, even if the agent does not visibly use it.

### After the agent responds

Publish the completed user and agent turns with the same `user_id`, `session_id`, `source`, and `agent_version`. Attach all injected learning references to the agent interaction as `retrieved_learnings`. Include compact tool-use, citation, expert-answer, or explicit outcome fields only when the host already exposes trustworthy values for them.

Publish after streaming completes. Use the native async client in async applications. A publish failure must not replace an otherwise valid agent response, but it must remain observable. Retry automatically only when replay is idempotency-safe: the HTTP route must persist one caller-generated `request_id` with the buffered batch and reuse it for every attempt. The current Python client's public publish methods do not accept a caller-supplied `request_id`, so do not blindly replay an ambiguous timeout or disconnect that may have occurred after the server accepted the request. If the application requires durable retries, use the HTTP route or an existing reconciliation mechanism that proves the batch was not already accepted. Do not start an untracked background task in a short-lived or serverless process.

Do not add `force_extraction=True` or `wait_for_response=True` to the normal production request path. Those controls are for explicit demos or tests.

## Implementation boundaries

- Preserve the application's existing agent framework and prompt architecture.
- Prefer one small adapter or wrapper at an existing seam; do not scatter Reflexio calls across unrelated business logic.
- Do not invoke the Reflexio CLI through subprocess from application code.
- Do not add a new durable queue, feature-flag system, or configuration abstraction when the application already has an adequate mechanism.
- Do not treat retrieved content as an instruction source with higher priority than the host agent's existing policies.

## Optional detailed documentation

The bundled references cover the normal integration path. When an example, method parameter, request field, or response schema needs more detail, consult the official Reflexio documentation rather than guessing:

- [Reflexio developer documentation](https://www.reflexio.ai/docs) for the documentation index and minimum integration loop.
- [Hosted Enterprise quickstart](https://www.reflexio.ai/docs/getting-started/quickstart) for setup and end-to-end examples.
- [Searching learned context](https://www.reflexio.ai/docs/build/search) for retrieval patterns and search options.
- [Publishing interactions](https://www.reflexio.ai/docs/build/user-interactions) for publishing patterns and interaction fields.
- [Requests and sessions](https://www.reflexio.ai/docs/concepts/requests-and-groups) for request metadata and grouping semantics.
- [User profiles](https://www.reflexio.ai/docs/concepts/user-profiles) and [playbooks](https://www.reflexio.ai/docs/concepts/agent-playbook) for user-scoped and agent-scoped learning boundaries.
- [API reference](https://www.reflexio.ai/docs/api-reference) for complete SDK parameters, HTTP payloads, and response schemas.

Treat these pages as supporting detail, not a reason to broaden the requested integration. Preserve the Hosted Enterprise default and only configure a URL override when the user explicitly requests one.

## Verification

At minimum, prove with focused tests that:

- Search happens before the real agent/model call and its rendered context reaches that call.
- Search failure still permits a normal agent response.
- Publish happens after the completed response with the expected identity fields.
- Every injected profile or playbook is published back using the correct `kind` and stable ID.
- The API key is not present in the diff or logs.
- No URL override was introduced unless the user explicitly requested one.

When credentials are available, a read-only `whoami` check is in scope. Ask before publishing synthetic data to a live Reflexio organization unless the user already requested live end-to-end verification. Never claim live verification when only mocks or static checks ran.
