# TypeScript client integration

Use `@reflexioai/client` for a TypeScript or JavaScript application on Node.js 20.3 or newer, Bun, Deno, or an edge runtime with `fetch`. It has no runtime dependencies. For an older Node.js runtime, use [the HTTP integration](http-api.md) on the project's existing HTTP library; do not raise the application's Node.js requirement for Reflexio without the developer's explicit approval.

Install it with the target application's package manager:

```bash
npm install @reflexioai/client
# or: pnpm add / yarn add / bun add @reflexioai/client
```

The client holds the API key, so construct and call it in server-side code only (a route handler, server action, worker, or agent process), never in browser code.

## Initialize

Default Hosted Enterprise setup:

```ts
import { ReflexioClient } from "@reflexioai/client";

const client = new ReflexioClient({ timeoutMs: 5_000 });
```

This reads `REFLEXIO_API_KEY` and uses `https://www.reflexio.ai/`. Do not pass `url` and do not introduce `REFLEXIO_URL` for the default path.

Use a full-access key for the intended managed project; search and publish both reject limited keys. See [Credentials and connection](../SKILL.md#credentials-and-connection).

If and only if the user explicitly requests a custom endpoint, use one existing configuration value:

```ts
const client = new ReflexioClient({ url: config.reflexioUrl, timeoutMs: 5_000 });
```

Alternatively, let the client read a user-supplied `REFLEXIO_URL`. Do not implement both mechanisms without a reason. In a runtime without `process.env` (some edge platforms), pass `apiKey` from the platform's secret binding.

Arguments use the API's field names (`user_id`, `session_id`, `top_k`), the same names the Python client takes as keyword arguments. The generated types reject a misspelled field at compile time; keep the repository's type check in the verification run.

## Bind the session before the turn

Bind the session once, at the top of the turn, and make every Reflexio call for that turn through the returned object:

```ts
const session = client.forSession(sessionId);
```

The search and the publish must agree on the session, or the retrieval cannot be attributed to the turn it informed and contributes no evidence. Binding removes the opportunity to disagree: both calls below read the session from the same place. Passing `session_id` to each call on `client` still works, but it has to be right at every call site, and omitting it from one of them is silent and successful.

`session` shares the client's configuration; it binds one argument, it is not a second client.

## Search and preserve learning identities

```ts
const results = await session.search({
  query: userMessage,
  user_id: userId,
  agent_version: agentVersion,
  entity_types: ["profiles", "user_playbooks", "agent_playbooks"],
  agent_playbook_status_filter: ["approved"],
  top_k: 3,
});
```

Check `results.success` before consuming results. A false value is an application-level failure even when HTTP succeeded. Handle it through the same safe diagnostic and fail-open path as a rejected promise (`ReflexioError`), continuing the agent turn without Reflexio context or an experiment assignment from that failed search.

Build the injected context from `content` and retain these stable identities. This example assumes every returned item is injected; if you filter or truncate results, build the references from only the retained items:

```ts
import type { RetrievedLearning } from "@reflexioai/client";

const retrievedLearnings: RetrievedLearning[] = [
  ...(results.profiles ?? []).map((profile) => ({ kind: "profile" as const, learning_id: profile.profile_id })),
  ...(results.user_playbooks ?? []).map((playbook) => ({
    kind: "user_playbook" as const,
    learning_id: String(playbook.user_playbook_id),
  })),
  ...(results.agent_playbooks ?? []).map((playbook) => ({
    kind: "agent_playbook" as const,
    learning_id: String(playbook.agent_playbook_id),
  })),
];
```

Do not flatten the three result types into an undifferentiated instruction list. Render profiles as user facts/preferences and playbooks as behavioral guidance.

Build a fresh list for each turn. Do not append earlier turns' references. Reusing `session_id` groups the conversation for evaluation and may suppress previously returned learnings; it does not require retaining those learnings for later turns.

Also retain `results.experiment` when it is present. It is the server-assigned retrieval-experiment identity for this request; do not recalculate or replace it.

## Publish the completed turn

Existing managed-cloud integrations must populate `retrieved_learnings` too: enabling playbook tuning in Settings does not attach playbooks to sessions. Audit the existing publish call and supply the references built from the same retained search results as the prompt. In particular, the offline tuner's user-playbook attribution uses `kind: "user_playbook"` and the actual returned `user_playbook_id` as a string. References to profiles or agent playbooks keep their respective kinds; they are not substitutes for a user-playbook reference.

```ts
const publishResult = await session.publishInteraction({
  user_id: userId,
  source,
  agent_version: agentVersion,
  retrieval_experiment_id: results.experiment?.experiment_id ?? null,
  retrieval_experiment_arm: results.experiment?.arm ?? null,
  interactions: [
    { role: "User", content: userMessage },
    { role: "Agent", content: agentResponse, retrieved_learnings: retrievedLearnings },
  ],
});
```

When no experiment is active, leave both values `null`. When an assignment is returned, publish both values together, including for a holdout response with no retrieved learnings.

Check `publishResult.success`; a false value must enter the application's publish-failure diagnostic path even though the promise resolved. Observe `publishResult.warnings`, which can report ignored fields or skipped interactions. Retain `publishResult.request_id` and `publishResult.learning_status` when present. Handle diagnostics without replacing the completed agent response or logging interaction content or credentials.

Verify through the application's real turn handler with mocked search results: capture the model input and the publish call, and assert that each injected learning's reference appears on the agent interaction. Include a discarded result, empty/failed search, consecutive turns, and concurrent requests. For an authorized live test, read back the stored interaction with `client.getInteractions({ user_id, start_time, end_time, top_k })` (ISO 8601 times), match its `request_id` to `publishResult.request_id`, and confirm `retrieved_learnings` contains the injected reference. A successful publish alone is not this verification; neither check guarantees a tuning run. See [Verification](../SKILL.md#verification).

Keep the normal defaults: do not set `wait_for_response`, `force_extraction`, or `skip_aggregation`. The promise still waits for the HTTP response; without `wait_for_response` the server queues extraction instead of processing it synchronously.

Acceptance does not imply extraction is complete. Only when the workflow needs completion tracking, use the returned request ID with `client.getLearningStatus(publishResult.request_id)`; do not add polling to the normal agent response path.

Create the client once at the application's normal client/service lifetime rather than once per token or tool event.

## Retry safety

`publishInteraction` does not accept a caller-supplied `request_id`. A timeout or connection loss is therefore ambiguous: the server may have accepted the publish even though the client did not receive its response. Do not automatically replay such a failure, because the retry would receive a new request ID and could duplicate the interaction batch.

Local validation failures (a `TypeError` such as an empty `session_id`) are permanent and should not be retried. Raw HTTP does not make ambiguous replay safe either: its optional `request_id` is not backed by atomic idempotency. A durable at-least-once queue needs an existing reconciliation mechanism that can prove the original publish was not accepted, or server-side atomic idempotency, before it automatically retries.

## Connection check

When `REFLEXIO_API_KEY` is available, check it with one search that publishes nothing:

```ts
const check = await client.search({ query: "connection check", user_id: "reflexio-connection-check", top_k: 1 });
```

A rejected promise or `check.success` being false means the key cannot serve the runtime loop. Do not rely on `client.whoami()` alone: it succeeds for limited keys, which search and publish reject. An empty result is expected on a new project. Do not print the API key.

For additional examples and the complete request fields, start with the [documentation index for agents](https://www.reflexio.ai/docs/llms.txt), then consult the [TypeScript client reference](https://www.reflexio.ai/docs/api-reference/typescript), [search](https://www.reflexio.ai/docs/build/search), and [publishing interactions](https://www.reflexio.ai/docs/build/user-interactions).
