# `wrap_llm_client` - auto-publish LLM turns to Reflexio

`wrap_llm_client` is a transparent wrapper around the LLM client you already use.
Your application still calls OpenAI, Anthropic, litellm, OpenRouter, or another
compatible client in the normal way. The wrapper watches completion calls and
publishes the resulting conversation turn to Reflexio in the background.

For users, this means Reflexio learns from the actual conversations they have with
your agent. For developers, this means you do not need to remember a separate
`publish_interaction(...)` call after every LLM response.

## The Mental Model

Think of the wrapper as a recorder attached to your LLM client:

1. Your app builds the prompt and calls the LLM exactly as before.
2. The wrapper removes the special `reflexio={...}` argument before the LLM sees it.
3. The LLM response is returned unchanged to your application.
4. The wrapper converts the clean user utterance and assistant response into Reflexio
   `InteractionData`.
5. Reflexio receives the turn in the background, grouped by `session_id`.

The most important distinction:

- **LLM prompt**: whatever you send to the model, often including system text,
  templates, retrieved context, tool state, or hidden instructions.
- **Reflexio user turn**: the clean thing the user actually asked or said.

The wrapper never assumes those are the same. You supply the clean user turn with
`reflexio={"user_content": ...}`.

## Before And After

Without the wrapper:

```python
from reflexio import InteractionData, ReflexioClient

resp = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)

reflexio_client = ReflexioClient()
reflexio_client.publish_interaction(
    user_id="alice",
    session_id="session-001",
    interactions=[
        InteractionData(role="User", content="what's the weather?"),
        InteractionData(
            role="Assistant",
            content=resp.choices[0].message.content or "",
        ),
    ],
)
```

With the wrapper:

```python
from openai import OpenAI
from reflexio import wrap_llm_client

client = wrap_llm_client(OpenAI())

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    reflexio={
        "user_id": "alice",
        "session_id": "session-001",
        "user_content": "what's the weather?",
    },
)
```

`resp` is the same object the original SDK would have returned. The only added
behavior is the background publish.

## Import Paths

All of these import paths resolve to the same wrapper:

```python
from reflexio import wrap_llm_client
from reflexio.client import wrap_llm_client
from reflexio.client.llm import wrap_llm_client
```

Useful types are also exported:

```python
from reflexio.client.llm import (
    AnthropicAdapter,
    BaseAdapter,
    LLMCallContext,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ReflexioParams,
)
```

## Supported Clients

Built-in adapters are active at the same time and are matched by call path. One
wrapped OpenAI client can therefore record both Chat Completions and Responses API
calls.

| Client style | Intercepted call |
| --- | --- |
| OpenAI Chat Completions | `client.chat.completions.create(...)` |
| OpenAI Responses API | `client.responses.create(...)` |
| Azure OpenAI / OpenRouter through the OpenAI SDK | `client.chat.completions.create(...)` / `client.responses.create(...)` |
| litellm module | `litellm.completion(...)` / `litellm.acompletion(...)` |
| litellm bare callable | `wrap_llm_client(litellm.completion)(...)` |
| Anthropic Messages | `client.messages.create(...)` |

Async clients and streaming calls are supported:

- `AsyncOpenAI`
- `litellm.acompletion`
- `AsyncAnthropic`
- `stream=True` for supported call shapes

Custom clients can be supported with `adapters=[...]`.

## Quick Starts

### OpenAI Chat Completions

```python
from openai import OpenAI
from reflexio import wrap_llm_client

client = wrap_llm_client(OpenAI(), reflexio={"source": "support-agent"})

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": engineered_prompt},
    ],
    reflexio={
        "user_id": "user_123",
        "session_id": "chat_456",
        "user_content": "Can you help me change my shipping address?",
    },
)
```

### OpenAI Responses API

```python
from openai import OpenAI
from reflexio import wrap_llm_client

client = wrap_llm_client(OpenAI())

resp = client.responses.create(
    model="gpt-4.1",
    input="Summarize this order issue and suggest the next action.",
    reflexio={
        "user_id": "user_123",
        "session_id": "case_789",
        "user_content": "My order still has the wrong shipping address.",
    },
)
```

### Anthropic

```python
from anthropic import Anthropic
from reflexio import wrap_llm_client

client = wrap_llm_client(Anthropic(), reflexio={"source": "concierge"})

resp = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": engineered_prompt}],
    reflexio={
        "user_id": "user_123",
        "session_id": "trip_planning_001",
        "user_content": "Plan a three-day Tokyo itinerary for my family.",
    },
)
```

### litellm

```python
import litellm
from reflexio import wrap_llm_client

litellm_client = wrap_llm_client(litellm)

resp = litellm_client.completion(
    model="openrouter/openai/gpt-4o-mini",
    messages=[{"role": "user", "content": engineered_prompt}],
    reflexio={
        "user_id": "user_123",
        "session_id": "session_abc",
        "user_content": "Rewrite this note to be more concise.",
    },
)
```

You can also wrap a bare callable:

```python
completion = wrap_llm_client(litellm.completion)

resp = completion(
    model="openrouter/openai/gpt-4o-mini",
    messages=[{"role": "user", "content": engineered_prompt}],
    reflexio={
        "user_id": "user_123",
        "session_id": "session_abc",
        "user_content": "Rewrite this note to be more concise.",
    },
)
```

### Async

```python
from openai import AsyncOpenAI
from reflexio import wrap_llm_client

client = wrap_llm_client(AsyncOpenAI())

resp = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": engineered_prompt}],
    reflexio={
        "user_id": "user_123",
        "session_id": "session_async",
        "user_content": "What changed in my account?",
    },
)
```

The publish still runs off the caller's critical path. The event loop is not blocked
by the synchronous Reflexio HTTP client.

## The `reflexio` Params

The wrapper accepts a special `reflexio={...}` argument at wrap time and per call.
The LLM provider never receives it.

```python
client = wrap_llm_client(
    OpenAI(),
    reflexio={"source": "billing-agent", "agent_version": "v2"},
)

client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    reflexio={
        "user_id": "alice",
        "session_id": "billing-thread-1",
        "user_content": "Why was I charged twice?",
    },
)
```

Wrap-time params are defaults. Per-call params override them key by key.

You can pass either a plain dict or a `ReflexioParams` model:

```python
from reflexio import ReflexioParams

params = ReflexioParams(
    user_id="alice",
    session_id="billing-thread-1",
    user_content="Why was I charged twice?",
)

client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    reflexio=params,
)
```

| Field | Type | Required when publishing | Meaning |
| --- | --- | --- | --- |
| `user_id` | `str` | yes | Stable ID for the end user. |
| `session_id` | `str` | yes | Stable conversation/session ID. Turns with the same value form one Reflexio conversation. |
| `user_content` | `str | None` | no | Clean user utterance to publish as the User turn. |
| `source` | `str` | no | Source label such as app, workflow, or agent name. |
| `agent_version` | `str` | no | Agent version used for playbook aggregation. |
| `publish` | `bool` | no | Default `True`. Set `False` to skip publishing this call. |
| `publish_partial_stream` | `bool` | no | Default `False`. Opt into publishing partial streamed content if the stream is not fully consumed. |
| `skip_aggregation` | `bool` | no | Forwarded to `publish_interaction`. |
| `force_extraction` | `bool` | no | Forwarded to `publish_interaction`. |
| `evaluation_only` | `bool` | no | Store for evaluation and exclude from profile/playbook extraction. |

Validation happens synchronously, before the LLM call:

- Unknown key -> `TypeError`
- Missing `user_id` or `session_id` while publishing -> `ValueError`
- `evaluation_only=True` and `force_extraction=True` together -> `ValueError`

Use `publish=False` for calls that should not be published and therefore do not have
a user/session identity:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=internal_planner_messages,
    reflexio={"publish": False},
)
```

## Choosing `user_id` And `session_id`

For product developers, these two fields are the main integration decision.

Use `user_id` for the person whose preferences and feedback Reflexio should learn
from. This is usually your application's internal user ID, not their display name or
email address.

Use `session_id` for one conversation or task thread. Multiple LLM calls under the
same `session_id` are grouped together server-side:

```text
session_id="checkout-help-42"
  User: I need to change my shipping address.
  Assistant: I can help with that.
  Assistant: (tool call)
  Assistant: Your address is updated.
```

If your agent uses subagents, internal planners, or background analysis, either skip
publishing those calls or give them a different `session_id`/`source` so they do not
look like user-facing conversation turns.

## User-Facing Behavior

A user does not see anything new because of this wrapper. The LLM response still
returns to your app normally. What changes is that Reflexio can learn from the
interaction afterward:

- Profiles can capture durable user facts and preferences.
- User playbooks can capture corrections or behavior requests from this user.
- Agent playbooks can aggregate recurring behavior improvements across users.
- Evaluation workflows can inspect what happened in a session.

The wrapper records only the turn you explicitly identify with `user_id` and
`session_id`. It does not scrape all local chat history, diff previous messages, or
deduplicate a client-side transcript.

## Clean User Turn vs Engineered Prompt

Many agents transform the user's question before sending it to the LLM:

```python
raw_user_question = "Can you help me change my shipping address?"

engineered_prompt = f"""
User question:
{raw_user_question}

Retrieved account context:
{account_context}

Instructions:
- Be concise.
- Use policy citations.
"""
```

Send the engineered prompt to the model, but publish the clean user question:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": engineered_prompt}],
    reflexio={
        "user_id": user_id,
        "session_id": session_id,
        "user_content": raw_user_question,
    },
)
```

This keeps Reflexio's learning data close to what the user actually said, without
polluting profiles or playbooks with RAG context, prompt templates, or hidden system
instructions.

If you have a simple app where the last LLM `user` message really is the clean user
utterance, you can use an extractor:

```python
client = wrap_llm_client(
    OpenAI(),
    user_content_extractor=lambda ctx: ctx.adapter.default_user_content(ctx),
)
```

Explicit `reflexio={"user_content": ...}` still wins over the extractor.

## Tool And Function Calls

Tool/function-call responses are published as Assistant turns with `tools_used`
populated. If the model returns no assistant text and only a tool call, the wrapper
uses the placeholder content `"(tool call)"`.

That placeholder is intentional. Reflexio's history renderer emits tool metadata only
when an interaction has non-empty content, so an empty string would make the tool call
invisible to extraction.

Normal user turn that triggers a tool call:

```text
User: What's the weather in San Francisco?
Assistant: (tool call) [tools_used=lookup_weather(...)]
```

Tool/function-call continuation with no new user utterance:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=history_with_tool_result,
    reflexio={"user_id": user_id, "session_id": session_id},
)
```

This publishes only the assistant turn derived from the response.

In v1, the wrapper records the tool call the model requested. It does not capture tool
results as separate Reflexio interactions unless your later LLM response includes them
or your application publishes them another way.

## Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True,
    reflexio={
        "user_id": "alice",
        "session_id": "stream-001",
        "user_content": "Explain this error.",
    },
)

for chunk in stream:
    print(chunk)
```

Chunks pass through unchanged. The wrapper accumulates them and publishes the
reassembled Assistant turn only when the stream is fully consumed.

By default, early stream termination does not publish:

```python
for chunk in stream:
    if enough_for_preview(chunk):
        break  # no Reflexio publish by default
```

This prevents partial assistant responses from becoming learning evidence. If you
really want partial content recorded, opt in:

```python
client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    stream=True,
    reflexio={
        "user_id": "alice",
        "session_id": "stream-001",
        "user_content": "Explain this error.",
        "publish_partial_stream": True,
    },
)
```

## Internal Calls, Planners, And Subagents

By default, every intercepted completion call with publishable Reflexio params is
recorded. That is correct for user-facing agent responses, but usually wrong for
internal reasoning calls.

Use one of these patterns:

```python
# 1. Per-call skip
client.chat.completions.create(
    model="gpt-4o",
    messages=planner_messages,
    reflexio={"publish": False},
)
```

```python
# 2. Programmatic filter
client = wrap_llm_client(
    OpenAI(),
    publish_filter=lambda ctx: ctx.source != "internal-planner",
)
```

```python
# 3. Separate clients
user_facing_client = wrap_llm_client(OpenAI())
internal_client = OpenAI()
```

For most agent builders, the simplest rule is: wrap the top-level responder client,
and use an unwrapped client for subagents unless you intentionally want those
subagent turns in Reflexio.

## Using Your Own Reflexio Client

By default, the wrapper constructs `ReflexioClient()` and reads `REFLEXIO_API_KEY` /
`REFLEXIO_URL` from the environment. You can pass your own client:

```python
from reflexio import ReflexioClient, wrap_llm_client

reflexio_client = ReflexioClient(
    api_key="...",
    url_endpoint="http://localhost:8061",
)

client = wrap_llm_client(
    OpenAI(),
    reflexio_client=reflexio_client,
)
```

This is useful in tests, local development, multi-tenant deployments, or services that
already manage Reflexio client construction centrally.

## Error Handling And Guarantees

The wrapper is designed to stay out of the way of your LLM call:

- Provider arguments are forwarded after removing only `reflexio=...`.
- The provider response object is returned unchanged.
- Background publish failures are logged and swallowed.
- Response parsing failures are logged and swallowed.
- `user_content_extractor` failures are logged; the wrapper falls back to no User turn.
- `publish_filter` failures are logged; the wrapper skips publishing.

Configuration errors in `reflexio={...}` are different: they raise before the provider
call because they indicate your integration code is wrong.

## Transparent Proxy Caveats

The wrapper is intentionally lightweight and does not subclass provider SDK classes.

```python
wrapped = wrap_llm_client(OpenAI())
isinstance(wrapped, OpenAI)  # False
```

If a framework requires the concrete SDK class, pass it the unwrapped client or wrap
only the call site you control.

The wrapper also intercepts only declared completion call paths. Plain attributes such
as `client.api_key` pass through unchanged. Unknown methods are not wrapped.

## Custom Adapters

Use a custom adapter when your client has a different call path or response shape.
Adapters are prepended to the built-ins, so they win on overlapping paths.

```python
from reflexio.client.llm import BaseAdapter, wrap_llm_client

class MyAdapter(BaseAdapter):
    namespace_prefixes = frozenset({("chat",), ("chat", "completions")})

    def is_completion_call(self, path, attr) -> bool:
        return path[-3:] == ("chat", "completions", "create")

    def _assistant_from_response(self, response):
        return response.text, []

    def _assistant_from_chunks(self, chunks):
        return "".join(chunk.text for chunk in chunks), []

client = wrap_llm_client(my_client, adapters=[MyAdapter()])
```

Each adapter returns `(assistant_text, tools_used)` from responses or stream chunks.
`BaseAdapter` handles:

- prepending the User turn when `ctx.user_content` exists;
- using exact Reflexio role casing (`"User"` / `"Assistant"`);
- adding the `"(tool call)"` placeholder for tool-only assistant turns;
- dropping truly empty assistant responses.

## Testing Patterns

For unit tests, pass a fake Reflexio client whose thread pool runs inline:

```python
class InlinePool:
    def submit(self, fn):
        fn()

class FakeReflexioClient:
    def __init__(self):
        self._thread_pool = InlinePool()
        self.calls = []

    def publish_interaction(self, user_id, interactions, **kwargs):
        self.calls.append((user_id, list(interactions), kwargs))

client = wrap_llm_client(
    fake_llm_client,
    reflexio_client=FakeReflexioClient(),
)
```

Useful assertions:

- the LLM response object is returned by identity;
- `reflexio` is stripped before the provider call;
- `user_content` is used instead of the engineered prompt;
- tool-only turns publish `"(tool call)"`;
- `publish=False` suppresses publishing;
- invalid Reflexio params raise before the fake provider is invoked.

## Reference

```python
wrap_llm_client(
    client,
    reflexio=None,
    *,
    reflexio_client=None,
    adapters=None,
    user_content_extractor=None,
    publish_filter=None,
)
```

Exported types:

- `ReflexioParams`
- `LLMCallContext`
- `LLMAdapter`
- `BaseAdapter`
- `OpenAIChatAdapter`
- `OpenAIResponsesAdapter`
- `AnthropicAdapter`

