# Integrating an AI Agent with Reflexio

This guide is written for an AI coding agent that has been asked to add
Reflexio to another AI agent, editor plugin, CLI assistant, or agent framework.
Follow it as an implementation checklist.

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
   - Read `REFLEXIO_URL`.
   - Read `REFLEXIO_API_KEY` for managed Reflexio.
   - Keep local Reflexio as the no-key default when appropriate.
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
8. Track citations:
   - Record which injected items were shown to the agent.
   - Publish citations on assistant turns when those items influenced the
     answer.
9. Add flush and retry paths:
   - Retry unpublished buffers on session start or before the next prompt.
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
| `agent_version` | Scope for shared agent playbooks | Use a stable agent name plus major behavior version, for example `my-agent-v1`. Keep it stable if learnings should transfer across users/projects. |
| `session_id` | Group turns for one conversation | Use the host session/conversation id. Generate a UUID if the host does not provide one. |
| `source` | Audit label | Use the integration name, for example `my-agent-plugin`. |

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

Prefer the Python SDK for integrations written in Python:

```shell
pip install reflexio-ai
```

For local development, start the backend:

```shell
reflexio services start
```

The local API defaults to `http://localhost:8081/`. If you need a different
backend, read these values from environment or `~/.reflexio/.env`:

```shell
REFLEXIO_URL="http://localhost:8081/"
REFLEXIO_API_KEY=""
```

For managed Reflexio, set both:

```shell
REFLEXIO_URL="https://www.reflexio.ai/"
REFLEXIO_API_KEY="..."
```

Implementation rule: if Reflexio is unavailable, the agent must continue
normally. Treat Reflexio as a best-effort learning layer, not as a dependency
that can break user work.

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
| `source` | No | Integration label for debugging and filtering. Use a stable name for the plugin, framework, or adapter. | `"my-agent-plugin"`, `"vscode-assistant"`, `"support-chatbot"` |
| `agent_version` | Strongly recommended | The shared-agent learning boundary. Use the same value when playbooks should transfer across users. Change it when old playbooks should not transfer. | `"support-agent-v1"`, `"coding-agent-2026-05"` |
| `session_id` | Yes | Host conversation/session id. Generate a UUID if the host has no session id, and reuse it for all turns in that conversation. | `"sess_01HX8Y..."`, `"3f02b7f8-..."` |
| `skip_aggregation` | No | `False` when user playbooks should be eligible to roll up into shared agent playbooks. `True` when you want user-level extraction only. | `false` |
| `force_extraction` | No | `False` for normal background publishing. `True` for manual learn-now, tests, or final flushes where you intentionally want extraction to run immediately. | `false` |
| `wait_for_response` | SDK/query option | `False` on interactive paths. `True` only when the caller is prepared to wait for extraction results. | `false` |

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
| `citations` | No | Reflexio profile/playbook ids that influenced the assistant answer. Use ids from the context-injection registry. | `[{"kind": "playbook", "real_id": "42", "tag": "s1", "title": "Use pnpm in this repo"}]` |

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
7. Retry unpublished records on the next hook/session if publish fails.

Python SDK example:

```python
from __future__ import annotations

import os
from reflexio import ReflexioClient


def reflexio_client() -> ReflexioClient:
    return ReflexioClient(
        url_endpoint=os.environ.get("REFLEXIO_URL", "http://localhost:8081/"),
        api_key=os.environ.get("REFLEXIO_API_KEY", ""),
        timeout=5,
    )


def publish_turns(
    *,
    session_id: str,
    user_id: str,
    agent_version: str,
    interactions: list[dict],
) -> bool:
    if not interactions:
        return True

    client = reflexio_client()
    try:
        client.publish_interaction(
            user_id=user_id,
            interactions=interactions,
            source="my-agent-plugin",
            agent_version=agent_version,
            session_id=session_id,
            wait_for_response=False,
            force_extraction=False,
            skip_aggregation=False,
        )
    except Exception:
        return False

    return True
```

Use `wait_for_response=False` on interactive paths. Use `force_extraction=True`
only for explicit "learn now", session-final flushes, tests, or workflows where
the caller intentionally waits for extraction.

Keep `skip_aggregation=False` if playbooks learned from one `user_id` should be
eligible to roll up into `agent_version`-scoped agent playbooks. Set it to
`True` only when you intentionally want user-level extraction without cross-user
transfer.

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
        agent_playbook_status_filter=["pending", "approved"],
        enable_agent_answer=False,
        top_k=3,
        search_mode="hybrid",
    )
```

This search shape gives the current user their own profiles and user playbooks,
plus shared agent playbooks generated for the same `agent_version`.

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

Save a registry from `[r1]`, `[s1]`, and `[p1]` to the real Reflexio ids. When
the assistant response cites or materially follows those items, publish the
assistant turn with `citations`:

```json
{
  "role": "Assistant",
  "content": "Implemented the focused docs change.",
  "citations": [
    {
      "kind": "playbook",
      "real_id": "stored-playbook-id",
      "tag": "s1",
      "title": "Verify release pins upstream"
    }
  ]
}
```

This feedback loop lets Reflexio evaluate whether injected learnings were useful
or need revision.

## Hook Mapping

Map your host's lifecycle to these responsibilities. The exact hook names differ
by agent framework.

| Required moment | What to do |
| --- | --- |
| Setup/install | Install dependencies, create config, and ensure `REFLEXIO_URL` / `REFLEXIO_API_KEY` can be resolved. |
| Session start | Start or health-check the local backend if using local Reflexio. Retry old unpublished buffers. |
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
- Retry failed publishes later.
- Truncate large tool fields before publishing.
- Do not run extraction synchronously on every turn.
- Make retrieval query-aware; do not dump all memory into every prompt.
- Keep injected context short enough that the model can follow it.
- Log failures with enough context to debug, but do not log secrets.

## Direct HTTP Fallback

Use the SDK when possible. If the integration language cannot use the SDK, call
the HTTP API directly.

1. Discover routes from `GET /openapi.json` instead of guessing endpoint paths.
2. Include `Content-Type: application/json`.
3. Include a normal `User-Agent`, for example `User-Agent: my-agent-reflexio`.
4. Include `Authorization: Bearer <REFLEXIO_API_KEY>` when using managed
   Reflexio.

Core routes:

| Operation | Route |
| --- | --- |
| Publish interactions | `POST /api/publish_interaction` |
| Unified search | `POST /api/search` |
| Read config | `GET /api/get_config` |
| Update config | SDK preferred; if using HTTP, discover the route from `/openapi.json`. |

## Verification Checklist

Run these checks before considering the integration complete:

1. With Reflexio running, publish a conversation containing a clear correction.
2. Search for the corrected behavior and confirm a profile or playbook appears.
3. Start a new agent session and confirm the relevant learning is injected.
4. Stop Reflexio, run a normal agent turn, and confirm the agent still works.
5. Restart Reflexio and confirm the buffered turn is retried and published.
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
- Forgetting citations, which makes it harder for Reflexio to learn whether old
  guidance was useful.

## Final Shape

A complete integration has this shape:

- User prompts are appended to a durable session buffer.
- Tool calls are buffered as compact `tools_used` records when the host exposes
  them.
- Assistant completion or session-end hooks publish unpublished records.
- Failed publishes leave the watermark unchanged for retry.
- Prompt or planning hooks search Reflexio and inject compact context.
- Tool hooks search Reflexio before risky or mutating actions when supported.
- User-scoped learnings use the chosen `user_id` boundary.
- Shared learnings use `agent_version` as the aggregation boundary.
- A manual learn command publishes immediately with forced extraction when the
  host supports commands or tools.
