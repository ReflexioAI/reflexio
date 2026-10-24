# Reflexio TypeScript SDK

The official TypeScript client for [Reflexio](https://www.reflexio.ai/), the learning platform for AI agents. Publish what your agent did; search what Reflexio learned from it (user profiles, user playbooks and agent playbooks) before the next turn.

It covers the agent-facing calls of the [Python client](https://pypi.org/project/reflexio-client/) under the same names, defaults and environment variables, with request and response types generated from the Reflexio server's OpenAPI schema. It has no runtime dependencies and uses the platform `fetch`, so it runs on Node.js 20.3+, Bun, Deno and edge runtimes.

## Installation

```bash
npm install @reflexioai/client
```

## Quick start

```ts
import { ReflexioClient } from "@reflexioai/client";

const client = new ReflexioClient(); // reads REFLEXIO_API_KEY

// 1. Send what happened.
await client.publishInteraction({
  user_id: "u_123",
  session_id: "s_1",
  interactions: [
    { role: "User", content: "I see a charge I do not recognize" },
    { role: "Agent", content: "Let me pull up your recent charges." },
  ],
});

// 2. Search memory from processed history.
//    New users may have no results until enough history has been processed.
const context = await client.search({ query: "refund request", user_id: "u_123", session_id: "s_1" });
```

Method names are camelCase; arguments use the API's own field names (`user_id`, `top_k`), so the [REST reference](https://www.reflexio.ai/docs/api-reference) and Python examples translate line for line.

| Python | TypeScript |
| --- | --- |
| `ReflexioClient()` | `new ReflexioClient()` |
| `client.for_session(session_id)` | `client.forSession(sessionId)` |
| `client.publish_interaction(user_id=..., session_id=..., interactions=...)` | `await client.publishInteraction({ user_id, session_id, interactions })` |
| `client.search(query=..., user_id=..., session_id=...)` | `await client.search({ query, user_id, session_id })` |
| `client.finalize_session(user_id, session_id)` | `await client.finalizeSession({ user_id, session_id })` |
| `client.get_learning_status(request_id)` | `(await client.getLearningStatus(requestId)).status` |
| `client.get_interactions(user_id=..., top_k=...)` | `await client.getInteractions({ user_id, top_k })` |
| `client.get_profiles(...)`, `get_user_playbooks(...)`, `get_agent_playbooks(...)` | `getProfiles(...)`, `getUserPlaybooks(...)`, `getAgentPlaybooks(...)` |
| `client.grade_on_demand(session_id=..., agent_version=...)` | `await client.gradeOnDemand({ session_id, agent_version })` |
| `client.whoami()` | `await client.whoami()` |

## Configuration

```ts
const client = new ReflexioClient({
  apiKey: process.env.MY_REFLEXIO_KEY, // default: REFLEXIO_API_KEY
  url: "http://localhost:8061/",      // default: REFLEXIO_URL, then https://www.reflexio.ai/
  timeoutMs: 5_000,                    // default: 300000, as in Python
});
```

Every call also takes `{ timeoutMs, signal }` as a second argument.

The API key is a secret. Use the client on a server, never in browser code.

## Working in a session

Bind the session once and make every call for that conversation through the result. The search that surfaced a learning and the publish that recorded the turn then always agree on the session, which is what lets Reflexio attribute an outcome to what it showed your agent.

```ts
const session = client.forSession(conversationId);

const results = await session.search({
  query: userMessage,
  user_id: userId,
  agent_version: "support-agent@3",
  entity_types: ["profiles", "user_playbooks", "agent_playbooks"],
  agent_playbook_status_filter: ["approved"],
  top_k: 3,
});

// ... your agent answers, using the results ...

await session.publishInteraction({
  user_id: userId,
  agent_version: "support-agent@3",
  retrieval_experiment_id: results.experiment?.experiment_id ?? null,
  retrieval_experiment_arm: results.experiment?.arm ?? null,
  interactions: [
    { role: "User", content: userMessage },
    {
      role: "Agent",
      content: agentReply,
      retrieved_learnings: [
        ...(results.profiles ?? []).map((p) => ({ kind: "profile" as const, learning_id: p.profile_id })),
        ...(results.user_playbooks ?? []).map((p) => ({
          kind: "user_playbook" as const,
          learning_id: String(p.user_playbook_id),
        })),
        ...(results.agent_playbooks ?? []).map((p) => ({
          kind: "agent_playbook" as const,
          learning_id: String(p.agent_playbook_id),
        })),
      ],
    },
  ],
});
```

## Responses and errors

Responses are the server's JSON, typed. Two kinds of failure:

- **The call failed**: network error, timeout, HTTP 4xx/5xx, a redirect, or a 2xx that is not JSON (usually `REFLEXIO_URL` pointing at the wrong host). The promise rejects with a `ReflexioError` carrying `status` and the decoded `body` when the server answered.
- **The call succeeded but the operation did not**: the response has `success: false`, as in the Python client. Check it.

```ts
import { ReflexioError } from "@reflexioai/client";

try {
  const results = await client.search({ query, user_id: userId, session_id: sessionId }, { timeoutMs: 2_000 });
  if (!results.success) console.warn("reflexio search failed:", results.msg);
} catch (err) {
  if (err instanceof ReflexioError) console.warn("reflexio unavailable:", err.status, err.message);
}
```

Treat Reflexio as best effort: a failed search or publish should never block your agent's reply.

`publishInteraction` returns once the server has durably accepted the turns; extraction runs afterwards. To follow it, poll `getLearningStatus(response.request_id)`. Pass `wait_for_response: true` only when a caller needs extraction finished before the call returns.

Publishing takes no caller-supplied `request_id`, as in the Python client. After a timeout the server may already have accepted the batch, so do not replay a publish automatically.

## Types

The package exports names for the common schemas (`Interaction`, `SearchResponse`, `PublishInteractionResponse`, `Profile`, `UserPlaybook`, `AgentPlaybook`, ...) and every server schema through `components`:

```ts
import type { components } from "@reflexioai/client";
type ToolUsed = components["schemas"]["ToolUsed"];
```

## Development

```bash
npm ci
npm run typecheck && npm test
npm run generate         # regenerate src/generated/openapi.ts after a server API change
npm run check-generated  # what CI runs: fails if the generated types are stale
```

`generate` reads the OpenAPI document from the Python server in this repository, so run `uv sync` at the repository root first.

Releasing: bump `version` in `package.json`, then `npm publish --access public` from this directory (`prepack` builds `dist/`).
