# Integrating an AI Agent with Reflexio

This guide is written for an AI coding agent that has been asked to add
Reflexio to another AI agent, editor plugin, CLI assistant, or agent framework.
Follow it as an implementation checklist.

For a portable coding-agent workflow, use the canonical
[`integrate-reflexio` skill](skills/integrate-reflexio/SKILL.md). The skill is a
distribution artifact for agent builders; it is not meant to run against this
Reflexio source repository. Its runtime and verification contract is normative;
this longer guide supplies supporting concepts and background.

Use either handoff:

1. Point a coding agent working in the agent application's repository to
   `https://github.com/ReflexioAI/reflexio/blob/main/skills/integrate-reflexio/SKILL.md`
   with this prompt:

   ```text
   Integrate Reflexio Enterprise into this agent repository. Follow
   https://github.com/ReflexioAI/reflexio/blob/main/skills/integrate-reflexio/SKILL.md
   and its linked references. Treat this repository as the integration target;
   do not modify the Reflexio source repository.
   ```

2. Copy the complete `skills/integrate-reflexio/` directory into the agent
   application's project-level skill location:

| Coding agent | Destination in the agent application | Explicit invocation |
| --- | --- | --- |
| Codex | `.agents/skills/integrate-reflexio/` | `$integrate-reflexio` |
| Claude Code | `.claude/skills/integrate-reflexio/` | `/integrate-reflexio` |
| Cursor | `.agents/skills/integrate-reflexio/` | `/integrate-reflexio` |

Copy the whole directory so `SKILL.md`, `agents/openai.yaml`, and `references/`
remain together. The application integration itself belongs in the agent
builder's repository; the copied skill is only the coding-agent workflow.

Reflexio integration has two jobs:

1. Publish useful interaction history so Reflexio can extract user profiles and
   playbooks.
2. Retrieve relevant profiles and playbooks before the agent acts, then inject
   them into the agent context.

The recommended pattern is to capture every turn through host lifecycle hooks,
buffer locally so the agent never blocks on Reflexio availability, publish in the
background, and inject only query-relevant learnings before the next response or
tool action.

## Integration Checklist

Complete these steps in order:

1. Choose the identity model:
   - Set `user_id` to the boundary where private facts, preferences, and
     user-specific playbooks should stay isolated.
   - Set `agent_version` to the boundary where generalized playbooks should
     transfer across users.
2. Add Reflexio configuration loading:
   - Read `REFLEXIO_API_KEY` for Hosted Enterprise.
   - Use the client's Hosted Enterprise endpoint by default.
   - Only read or set `REFLEXIO_URL` when the user explicitly requests a
     custom or self-hosted endpoint.
3. Add a Reflexio client wrapper:
   - Use short timeouts on interactive paths.
   - Catch exceptions and return neutral values.
   - Never let Reflexio availability break the agent.
4. Add a durable local buffer:
   - Store user turns.
   - Store assistant turns.
   - Store compact tool calls when the host exposes tool events.
   - Track a publish high-water mark.
5. Publish completed interaction batches:
   - Call `publish_interaction` with `user_id`, `agent_version`, `session_id`,
     and `source`.
   - Use `skip_aggregation=False` when user-level playbooks should roll up into
     shared agent-level playbooks.
   - Advance the high-water mark only after publish succeeds.
6. Retrieve before the agent acts:
   - Search with the current `user_id`.
   - Search with the current `agent_version`.
   - Retrieve profiles, user playbooks, and agent playbooks together.
7. Inject compact context:
   - Render user-scoped profiles as preferences.
   - Render user playbooks as user/project-specific rules.
   - Render agent playbooks as shared agent rules.
   - Keep injected context short and query-relevant.
8. Track retrieval and influence separately:
   - Publish every injected item as `retrieved_learnings`, whether or not it
     visibly influenced the answer.
   - Publish `citations` only when an item materially influenced the answer.
9. Add flush and retry paths:
   - Retry unpublished buffers on session start or before the next prompt only
     when the retry reuses a persisted caller-generated `request_id`.
   - Flush before transcript compaction/reset if the host supports it.
   - Flush on session end.
10. Add a manual learn-now path:
    - Let the user or host mark the latest turn as high-signal.
    - Publish immediately with `force_extraction=True`.
11. Verify the integration:
    - Confirm user-scoped data stays scoped to `user_id`.
    - Confirm playbooks aggregate and transfer across users with the same
      `agent_version`.
    - Confirm Reflexio-down behavior does not break the agent.

## Target Architecture

```text
Agent host hooks
  -> local durable buffer
  -> Reflexio publish_interaction
  -> profile + playbook extraction
  -> Reflexio search
  -> injected context before future responses/tools
```

Do not make the model call Reflexio manually during normal operation. Reflexio
should run through the host integration layer: hooks, callbacks, middleware, or
an equivalent wrapper around the agent loop.

## Step 1: Choose Identity and Transfer Boundaries

Before writing code, define these identifiers. They determine what Reflexio
learns, what stays private to a user scope, and what transfers to other users.

| Identifier | Use | Recommendation |
| --- | --- | --- |
| `user_id` | Scope for profiles and user playbooks | Use the human user, tenant, workspace, repo, or project whose preferences should be isolated. For example, use a project id when repo-specific rules should not leak into unrelated repos. |
| `agent_version` | Scope for shared agent playbooks | Use a stable agent name plus major behavior version, for example `my-agent-v1`. Keep it stable if learnings should transfer across users/projects. If you omit it, the SDK uses `DEFAULT_AGENT_VERSION` (`"agent-v0"`) — fine for a single agent, but set an explicit value before you run more than one. |
| `session_id` | Group turns for one conversation | Use the host session/conversation id. Generate a UUID if the host does not provide one. |
| `source` | Producer/workflow label | Use a non-sensitive machine label such as `support-agent:v2`. Where the API permits an empty value, empty means the source is absent. Every non-empty value must match `^[a-z0-9][a-z0-9._:-]{0,127}$` and is limited to 128 ASCII characters. Do not include user identifiers, email addresses, or other PII. |

`user_id` and `agent_version` work together:

- Profiles are scoped to `user_id`. They should not transfer to other users.
- User playbooks are first extracted under `user_id`. They represent rules
  learned from that user's interactions.
- User playbooks can be aggregated into agent playbooks under `agent_version`.
- Agent playbooks transfer across different `user_id` values that use the same
  `agent_version`.

Example:

```text
user_id = "alice"
agent_version = "support-agent-v1"
```

If Alice corrects the agent, Reflexio can extract a user playbook for Alice.
With aggregation enabled, that user-level playbook can become an agent playbook
for `support-agent-v1`. Later, Bob can use:

```text
user_id = "bob"
agent_version = "support-agent-v1"
```

Bob will not receive Alice's private profiles, but Bob can receive the shared
agent playbook because he is using the same `agent_version`.

Use a new `agent_version` when shared playbooks should not transfer, for example
after a major system-prompt rewrite, a product-domain split, or a behavior
change that makes older playbooks unsafe.

Changing `user_id` or `agent_version` after launch changes retrieval behavior, so
make this choice explicit in code and tests.

## Install and Configure Reflexio

For Hosted Enterprise integrations on Python 3.12 or newer, install the
lightweight client. On an older Python runtime, use the HTTP API rather than
raising the application's Python requirement without the developer's approval:

```shell
pip install reflexio-client
```

Then set only the API key. `ReflexioClient()` defaults to
`https://www.reflexio.ai/`:

```shell
REFLEXIO_API_KEY="..."
```

Do not add `REFLEXIO_URL` for this default path. If the user explicitly requests
a custom or self-hosted endpoint, set the override through the host
application's existing configuration system:

```shell
REFLEXIO_URL="http://localhost:8061/"
```

For Local OSS development, install the full package and start the backend:

```shell
pip install reflexio-ai
reflexio services start
```

Implementation rule: if Reflexio is unavailable, the agent must continue
normally. Treat Reflexio as a best-effort learning layer, not as a dependency
that can break user work.

## Configure an LLM Provider

Reflexio's extraction (profiles and playbooks), aggregation, and query
reformulation are LLM-powered. **A self-hosted OSS backend needs a provider
key, or it will accept publishes and extract nothing** — publishes still
succeed, but no profiles or playbooks are ever produced, which looks like a
silent no-op during integration.

Reflexio uses LiteLLM, so it supports many providers (OpenAI, Anthropic,
OpenRouter, Gemini, MiniMax, DeepSeek, xAI, and custom endpoints). Provide a
key one of two ways:

1. Environment variable picked up by LiteLLM (simplest for local dev), in the
   shell or `~/.reflexio/.env`:

   ```shell
   OPENAI_API_KEY="sk-..."        # or ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...
   ```

2. Persisted in the backend `Config` under `api_key_config` (survives restarts).
   Each provider is a nested object — for OpenAI, set `api_key_config.openai.api_key`.
   Set it with the SDK:

   ```python
   client.update_config({"api_key_config": {"openai": {"api_key": "sk-..."}}})
   ```

If you publish a clear correction and `search` returns nothing, an unset or
invalid provider key is the most common cause — check this before debugging the
integration itself.

## Fastest Path: Verify With the CLI

Before wiring SDK hooks, confirm the publish → extract → search loop works
end to end using the bundled `reflexio` CLI. This is the quickest onboarding
smoke test:

```shell
reflexio services start                        # backend on :8061 (+ docs), SQLite storage

reflexio publish --user-id alice --wait --data '{
  "interactions": [
    {"role": "user",      "content": "Deploy the new service."},
    {"role": "assistant", "content": "Deploying to us-east-1..."},
    {"role": "user",      "content": "No — we never deploy production to us-east-1. Always use us-west-2."},
    {"role": "assistant", "content": "Understood. Switching to us-west-2."}
  ]
}'

reflexio search "deployment region"            # should surface the learned rule
```

`--wait` runs extraction synchronously so the result is visible immediately
(see "Extraction is gated" below for why this matters). Once this loop works
from the CLI, replicate it from the SDK in your host hooks.

## Capture Interactions

Publish multi-turn conversation records to Reflexio. A publish request has
top-level metadata plus an interaction list. In the Python SDK, the interaction
list parameter is named `interactions`; in the HTTP request model, it is
`interaction_data_list`.

Publish request fields:

| Field | Required | What to put there | Example |
| --- | --- | --- | --- |
| `user_id` | Yes | The user, tenant, workspace, repo, or project scope whose private profiles and user playbooks should be isolated. | `"alice"`, `"tenant-acme"`, `"repo-reflexio"` |
| `interactions` / `interaction_data_list` | Yes | Ordered conversation turns to publish. Include at least one turn; multi-turn correction examples are best for learning. | `[{"role": "User", "content": "Use pnpm here."}, {"role": "Assistant", "content": "Got it, I will use pnpm."}]` |
| `source` | No | Non-sensitive producer/workflow machine label for debugging and filtering. Empty means the source is absent where the API permits it. Every non-empty value must match `^[a-z0-9][a-z0-9._:-]{0,127}$` and is limited to 128 ASCII characters. Do not include user identifiers, email addresses, or other PII. | `"support-agent:v2"` |
| `agent_version` | Strongly recommended | The shared-agent learning boundary. Use the same value when playbooks should transfer across users. Change it when old playbooks should not transfer. | `"support-agent-v1"`, `"coding-agent-2026-05"` |
| `session_id` | Yes | Host conversation/session id. Generate a UUID if the host has no session id, and reuse it for all turns in that conversation. | `"sess_01HX8Y..."`, `"3f02b7f8-..."` |
| `skip_aggregation` | No | `False` when user playbooks should be eligible to roll up into shared agent playbooks. `True` when you want user-level extraction only. | `false` |
| `force_extraction` | No | `False` for normal background publishing. `True` for manual learn-now, tests, or final flushes where you intentionally want extraction to run immediately. | `false` |
| `wait_for_response` | SDK/query option | `False` on interactive paths. `True` only when the caller is prepared to wait for extraction results. | `false` |

The source contract applies to new publish and filter inputs. Read responses may
return historical source values created before this contract; Reflexio preserves
those values exactly because source participates in session-outcome identity.

Each interaction row should resemble Reflexio's `InteractionData` shape:

```json
{
  "role": "User",
  "content": "Always run tests with --run in this repo."
}
```

Interaction row fields:

| Field | Required | What to put there | Example |
| --- | --- | --- | --- |
| `created_at` | No | Unix timestamp for the turn. Omit it if the publish time is good enough. | `1716249600` |
| `role` | Yes | Speaker role. Use `User` and `Assistant` unless your adapter has a clear host-specific mapping. | `"User"`, `"Assistant"` |
| `content` | Yes | Text the user or assistant actually saw. Keep it faithful to the conversation. | `"Wait, never deploy production to us-east-1. Use us-west-2."` |
| `shadow_content` | No | Alternate or shadow-agent answer for A/B comparison. | `"I would deploy to us-east-1."` |
| `expert_content` | No | Human expert's ideal answer when available. Use this to teach the agent from expert corrections. | `"Production deploys must target us-west-2 after checking the release window."` |
| `user_action` | No | Explicit UI/action signal if the host has one. Valid values include `none`, `click`, `scroll`, and `type`. | `"none"`, `"click"` |
| `user_action_description` | No | Plain-language description of the action or feedback. Use this for thumbs-down, retry, approval, or rejection details. | `"User rejected the plan and said the region was wrong."` |
| `interacted_image_url` | No | URL of an image the user interacted with, if relevant and safe to store. | `"https://example.com/screenshot.png"` |
| `image_encoding` | No | Base64-encoded image data when the image is part of the interaction. Prefer URLs or omit images unless needed. | `"iVBORw0KGgoAAA..."` |
| `tools_used` | No | On assistant turns, compact metadata for tools the assistant used. Avoid raw huge outputs. | `[{"tool_name": "Bash", "tool_data": {"input": "pnpm test -- --run", "output": "passed"}}]` |
| `retrieved_learnings` | No | Every Reflexio item injected into the assistant context, including items that did not visibly influence the answer. | `[{"kind": "agent_playbook", "learning_id": "42"}]` |
| `citations` | No | Reflexio profile/playbook ids that materially influenced the assistant answer. Use ids from the context-injection registry. | `[{"kind": "agent_playbook", "real_id": "42", "tag": "s1", "title": "Use pnpm in this repo"}]` |

Do not publish secrets, raw huge files, or unbounded tool output. Redact or
truncate before buffering.

## Publish Pattern

Use a durable local buffer between the agent host and Reflexio:

1. Append user turns when the user submits them.
2. Append tool calls as they happen.
3. Append the assistant turn when the assistant finishes.
4. Convert unpublished records into Reflexio interactions.
5. Call `publish_interaction`.
6. Mark a high-water point only after a successful publish.
7. Retry unpublished records only when every attempt reuses the batch's
   persisted caller-generated `request_id`.

Python SDK example:

```python
from __future__ import annotations

import logging
from typing import Literal

from pydantic import ValidationError
from reflexio import ReflexioClient


def reflexio_client() -> ReflexioClient:
    return ReflexioClient(timeout=5)


def publish_turns(
    *,
    session_id: str,
    user_id: str,
    agent_version: str,
    interactions: list[dict],
) -> Literal["advance", "quarantine"]:
    if not interactions:
        return "advance"

    client = reflexio_client()
    try:
        response = client.publish_interaction(
            user_id=user_id,
            interactions=interactions,
            source="support-agent:v2",
            agent_version=agent_version,
            session_id=session_id,
            wait_for_response=False,
            force_extraction=False,
            skip_aggregation=False,
        )
    except ValidationError as exc:
        # The payload is invalid, so retrying can never succeed. Advance the
        # high-water mark and stop re-sending it — see "A rejected
        # batch is not a retryable failure" below. Log the exception TYPE, not
        # the exception: a ValidationError renders the offending input, and
        # logging.exception would additionally capture frame locals holding the
        # whole payload.
        logging.error(
            "dropping unpublishable batch of %d (%s)",
            len(interactions),
            type(exc).__name__,
        )
        return "advance"
    except Exception as exc:
        # The request may have reached the server before the response was lost.
        # Keep the batch for reconciliation, but do not put it back into an
        # automatic retry loop: the public SDK cannot reuse a request_id.
        logging.error(
            "quarantining batch with ambiguous publish outcome (%s)",
            type(exc).__name__,
        )
        return "quarantine"

    # Outside the try above, and defensive about the shape — see
    # "Read `warnings`" below for why both matter.
    warnings = getattr(response, "warnings", None)
    if isinstance(warnings, list):
        for warning in warnings:
            logging.warning("reflexio dropped part of the payload: %s", warning)

    return "advance"
```

Advance the buffer watermark only for `"advance"`. Preserve a `"quarantine"`
batch for reconciliation without automatically resending it. If the host needs
automatic durable retries, use the HTTP route with a persisted `request_id`
instead of this SDK pattern.

Use `wait_for_response=False` on interactive paths. Use `force_extraction=True`
only for explicit "learn now", session-final flushes, tests, or workflows where
the caller intentionally waits for extraction.

Keep `skip_aggregation=False` if playbooks learned from one `user_id` should be
eligible to roll up into `agent_version`-scoped agent playbooks. Set it to
`True` only when you intentionally want user-level extraction without cross-user
transfer.

`publish_interaction` always blocks on the HTTP round-trip (so you see 4xx/5xx
and network errors), but with `wait_for_response=False` the server returns in
~100 ms after queuing extraction as a background task — fast enough for
interactive hooks. If you need a truly non-blocking call, a library user can
submit through the client's `_fire_and_forget(self._publish_interaction_async,
...)` path directly.

### Build the Wire Payload With an Allowlist

Convert buffer records to interactions by keeping **only** the fields
`InteractionData` declares — never by excluding the bookkeeping keys you happen
to know about today. A denylist silently widens every time you add a field to
your buffer, and the server discards unrecognised keys rather than rejecting
them — so nothing breaks, and the payload just quietly carries junk.

Both first-party plugins pin their allowlist against the real model in a test,
which fails if `InteractionData` grows a field the plugin would drop. Copy that
pattern rather than importing the model at runtime if your hooks must survive
Reflexio being uninstalled.

One field deserves special care: **do not send `created_at`.** Extraction is
bookmarked on interaction `created_at` (`created_at >= last_processed`), so a
batch recovered from an offline buffer and stamped with its original event time
is stored and then never extracted — permanent, silent loss on exactly the path
the buffer exists to protect. Let the server stamp drain time.

### Read `warnings` — a 200 Does Not Mean Everything Bound

`PublishUserInteractionResponse.warnings` lists what the server accepted but
could not use: unrecognised field names (per interaction, with the caller's own
index) and interactions skipped for carrying no content. A correctly-shaped
payload produces no *payload* warnings, so anything of that kind is real drift
worth logging.

This channel exists because it is otherwise possible to publish 50 interactions,
receive `200 OK`, and store 50 rows with `content = ''` — which happened, and
cost hours to diagnose. A mis-keyed `Content` binds nothing and `content`
defaults to `""`.

Four caveats:

- **The list is bounded** — at most 5 field names per interaction and 20 entries
  overall, each with a `+N more` suffix. On a large broken batch treat it as a
  sample, not an inventory.
- **`user_id` and `session_id` are stripped without a warning.** They are
  request-level fields that callers routinely repeat on every interaction, so
  reporting them would be noise. Their absence from the list is not evidence
  they bound.
- **The same field also carries extraction warnings**, including the
  provider-stall notice — but only when `wait_for_response=True`. The deferred
  path returns payload warnings alone.
- **`_fire_and_forget(self._publish_interaction_async, ...)` reports nothing.**
  That path does not merge local warnings, and the SDK strips unknown keys
  before they reach the server, so the whole channel is silent there. Use
  `publish_interaction` if you want the diagnostic.

Read the warnings **outside** whatever `try` guards the publish call. The
publish already succeeded at that point; if reading diagnostics could raise into
a handler that returns "failed", a successful batch would never be marked
published and would be re-sent on every subsequent hook — an infinite duplicate
loop caused purely by the code meant to improve observability. For the same
reason, keep the read itself total: tolerate a missing or oddly-shaped
`warnings` value instead of assuming it is a list.

### A Rejected Batch Is Not a Retryable Failure

Only *some* shape problems are warnings. A payload where **every** interaction
is contentless is rejected, as are two contradictions: a `user_action` other
than `none` with no `user_action_description`, and `interacted_image_url` and
`image_encoding` both set. Over raw HTTP that is a `422`. **Through the Python
SDK it is a local `pydantic.ValidationError`, raised before any HTTP request** —
so there is no status code to inspect and no response object.

This matters for the buffer pattern above. The natural `except Exception:
return False` treats rejection exactly like a network error: the high-water mark
never advances, and the next hook re-sends the same batch, which is rejected
again, forever. Catch `ValidationError` separately and either advance past the
batch or quarantine it — a payload the client refuses to send will not become
valid on the next attempt.

The example only covers the client-side half. If the server is stricter than
your pinned client — a newer validation rule, or a payload built as raw JSON
rather than through the SDK — rejection arrives as an HTTP `422` inside your
transport's error type, not as a `ValidationError`, and falls straight into the
generic handler. Treat **any** `4xx` other than `408`/`429` the same way you
treat a `ValidationError`: it is a permanent rejection, not a retryable failure.
A `5xx`, timeout, or connection loss is potentially transient, but it is also
ambiguous: the server may have accepted the publish before the response was
lost. Retry only with a caller-generated `request_id` that was persisted with
the buffered batch and reused for every attempt. The current Python client's
public publish methods do not accept that value, so do not blindly replay an
ambiguous SDK failure; use the raw HTTP route when durable at-least-once retry
is required.

### Extraction Is Gated — Don't Expect a Result From One Publish

Normal background publishing does **not** run extraction on every turn. Two
gates stand between a publish and a new profile/playbook:

- **Sliding window / stride** (`window_size` / `stride_size`, default `10` / `8`
  in `Config`): extraction fires once enough new turns have accumulated, not on
  every publish.
- **`should_run` pre-filter**: a cheap check (and an LLM gate) can decide a batch
  carries no durable learning and skip it.

So a single publish — even a clear correction — may legitimately produce
nothing yet. To force extraction immediately (manual "learn now", tests, the
verification smoke test), publish with `force_extraction=True`, which bypasses
both gates. Use this for explicit learn-now and final flushes, **not** for every
interactive turn — that would run an LLM extraction on every message.

## Retrieve and Inject Context

Before the agent plans or edits, search Reflexio using the current task text.
Use unified search so profiles, user playbooks, and shared agent playbooks are
retrieved together:

```python
def search_reflexio(user_id: str, agent_version: str, query: str):
    client = reflexio_client()
    return client.search(
        query=query,
        user_id=user_id,
        agent_version=agent_version,
        entity_types=["profiles", "user_playbooks", "agent_playbooks"],
        agent_playbook_status_filter=["approved"],
        enable_agent_answer=False,
        top_k=3,
        search_mode="hybrid",
    )
```

`top_k` and `threshold` are per entity type; if omitted they default to `5` and
`0.3`. Keep `top_k` small on interactive paths so injected context stays short.

This search shape gives the current user their own profiles and user playbooks,
plus shared agent playbooks generated for the same `agent_version`.

`search` returns a `UnifiedSearchViewResponse` with one list per entity type.
These are the fields you need to render context and build the retrieval registry:

| Result list | Stable ID | `retrieved_learnings.kind` | Title/text field |
| --- | --- | --- | --- |
| `profiles` (`ProfileView`) | `profile_id` | `"profile"` | `content` |
| `user_playbooks` (`UserPlaybookView`) | `user_playbook_id` | `"user_playbook"` | `playbook_name` |
| `agent_playbooks` (`AgentPlaybookView`) | `agent_playbook_id` | `"agent_playbook"` | `playbook_name` |

Publish every injected identity back as `retrieved_learnings` using its canonical
kind and stable ID. When you assign a short tag (`[p1]`, `[r1]`, `[s1]`) to an
injected item, you may also store `tag -> (kind, real_id, title)` for a `citation`
when the item materially influences the response.

Inject the results as short, instruction-like context. Keep the model-facing
format compact and auditable:

```text
Relevant Reflexio learnings:

Project rules:
- [r1] When editing this repo, run `npm test -- --run`.

Shared agent rules:
- [s1] Before changing release pins, verify the package version exists upstream.

User/project preferences:
- [p1] The user prefers concise root READMEs and detailed implementation docs elsewhere.
```

Save a registry from `[r1]`, `[s1]`, and `[p1]` to the real Reflexio ids. Publish
all three as `retrieved_learnings`; when the assistant response materially
follows `[s1]`, also publish that item as a `citation`:

```json
{
  "role": "Assistant",
  "content": "Implemented the focused docs change.",
  "retrieved_learnings": [
    {"kind": "user_playbook", "learning_id": "stored-user-playbook-id"},
    {"kind": "agent_playbook", "learning_id": "stored-agent-playbook-id"},
    {"kind": "profile", "learning_id": "stored-profile-id"}
  ],
  "citations": [
    {
      "kind": "agent_playbook",
      "real_id": "stored-agent-playbook-id",
      "tag": "s1",
      "title": "Verify release pins upstream"
    }
  ]
}
```

This distinction lets Reflexio compare what was shown with what actually
influenced the response.

## Hook Mapping

Map your host's lifecycle to these responsibilities. The exact hook names differ
by agent framework.

| Required moment | What to do |
| --- | --- |
| Setup/install | For Hosted Enterprise, install dependencies and ensure `REFLEXIO_API_KEY` can be resolved. Default Local OSS does not require a Hosted credential. Configure `REFLEXIO_URL` only for an explicitly requested override. |
| Session start | Start or health-check the local backend if using local Reflexio. Retry old unpublished buffers only when replay is idempotency-safe. |
| Before prompt/plan | Search Reflexio with the user's task and inject compact relevant context. |
| Before tool use | If the host supports it, search with the tool command/edit target and inject tool-specific rules. This is useful before file edits or shell commands. |
| After tool use | Buffer the tool name plus compact input/output/result metadata. |
| Assistant stop/message sent | Buffer the assistant response and publish the completed turn batch. |
| Before compaction/reset | Flush unpublished records so transcript loss does not lose learning data. |
| Session end | Final flush. Use `force_extraction=True` if the host can tolerate waiting. |
| Manual learn command | Mark the last turn as a correction and publish immediately with `force_extraction=True`. |

If the host only exposes a single "message completed" callback, implement that
first: buffer user + assistant turns, publish, then add retrieval before the next
message.

## What Reflexio Learns

Reflexio extracts different artifact types. Preserve this distinction in naming
and injection:

| Reflexio artifact | Meaning | How to use it |
| --- | --- | --- |
| Profiles | Facts, preferences, and context scoped to one `user_id` | Inject only for that user scope. Do not treat profiles as cross-user knowledge. |
| User playbooks | Behavioral rules first learned from one `user_id` | Inject for that user scope and allow aggregation when the rule may generalize. |
| Agent playbooks | Aggregated rules scoped to one `agent_version` | Inject for any user running that same `agent_version`. This is how playbooks transfer among users. |

Good learning signals include user corrections, rejected plans with comments,
manual "learn this" commands, successful multi-step workflows, and expert ideal
answers. Avoid extracting from ambiguous chatter or single isolated facts as a
behavioral rule.

The transfer path is:

```text
interactions for user_id
  -> profiles for that user_id
  -> user playbooks for that user_id
  -> aggregation
  -> agent playbooks for agent_version
  -> retrieval by other users on the same agent_version
```

Profiles remain user-scoped. Playbooks are the artifact designed to generalize.
If two users should benefit from the same learned behavior, publish their
interactions with the same `agent_version` and keep aggregation enabled.

## Reliability Requirements

Follow these rules for production agent integrations:

- Keep hook latency bounded. Use short HTTP timeouts on interactive hooks.
- Never let Reflexio exceptions fail the user's agent turn.
- Buffer locally before network calls.
- Advance publish watermarks only after success.
- Retry failed publishes later only with a persisted caller-generated
  `request_id`; do not blindly replay ambiguous SDK failures.
- Truncate large tool fields before publishing.
- Do not run extraction synchronously on every turn.
- Make retrieval query-aware; do not dump all memory into every prompt.
- Keep injected context short enough that the model can follow it.
- Log failures with enough context to debug, but do not log secrets.

## Direct HTTP Fallback

Use the SDK on Python 3.12 or newer when caller-controlled idempotency is not
required. For older Python runtimes, other languages, or durable retry queues,
call the HTTP API directly.

1. Discover routes from `GET /openapi.json` instead of guessing endpoint paths.
2. Include `Content-Type: application/json`.
3. Include a normal `User-Agent`, for example `User-Agent: my-agent-reflexio`.
4. Include `Authorization: Bearer <REFLEXIO_API_KEY>` when using managed
   Reflexio.
5. Generate one `request_id` per publish batch, persist it with the batch, and
   reuse it for every retry.

Core routes:

| Operation | Route |
| --- | --- |
| Publish interactions | `POST /api/publish_interaction` |
| Unified search | `POST /api/search` |
| Read config | `GET /api/get_config` |
| Update config | SDK preferred; if using HTTP, discover the route from `/openapi.json`. |

## Verification Checklist

Run these checks before considering the integration complete:

1. With Reflexio running (and a provider key configured), publish a conversation
   containing a clear correction. Use `force_extraction=True` (or the CLI
   `--wait`) so extraction runs immediately instead of waiting for the
   window/stride gate.
2. Search for the corrected behavior and confirm a profile or playbook appears.
   Empty results usually mean no provider key, or extraction was gated — retry
   with `force_extraction=True`.
3. Start a new agent session and confirm the relevant learning is injected.
4. Stop Reflexio, run a normal agent turn, and confirm the agent still works.
5. Restart Reflexio and confirm an idempotency-safe buffered turn is retried
   with its original `request_id` and published once.
6. Confirm no secrets or oversized tool outputs are stored in the buffer.
7. Confirm changing `user_id` hides user-scoped profiles/playbooks.
8. Confirm shared agent playbooks are filtered by `agent_version`.
9. Confirm manual "learn now" forces extraction for the latest correction.
10. Confirm tests isolate Reflexio-related environment variables.

## Common Mistakes

- Publishing without a stable `session_id`; new publishes require a non-empty
  value, and unstable one-off ids make later auditing harder.
- Using a global `user_id` when project/user isolation is required.
- Changing `agent_version` on every build and accidentally hiding shared
  playbooks from future searches.
- Waiting for extraction on every interactive turn.
- Losing data during transcript compaction because there is no pre-compaction
  flush.
- Injecting every stored learning instead of query-relevant search results.
- Letting Reflexio outages break the host agent.
- Forgetting `retrieved_learnings`, which hides what context was injected, or
  forgetting citations for the subset that materially influenced the response.

## Final Shape

A complete integration has this shape:

- User prompts are appended to a durable session buffer.
- Tool calls are buffered as compact `tools_used` records when the host exposes
  them.
- Assistant completion or session-end hooks publish unpublished records.
- Failed publishes leave the watermark unchanged only when a later replay can
  reuse the persisted caller-generated `request_id`; ambiguous SDK failures are
  reconciled rather than blindly replayed.
- Prompt or planning hooks search Reflexio and inject compact context.
- Tool hooks search Reflexio before risky or mutating actions when supported.
- User-scoped learnings use the chosen `user_id` boundary.
- Shared learnings use `agent_version` as the aggregation boundary.
- A manual learn command publishes immediately with forced extraction when the
  host supports commands or tools.
