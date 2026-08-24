# Python client integration

Install the lightweight Hosted Enterprise client in the target application:

```bash
uv add reflexio-client
# or: pip install reflexio-client
```

The package imports as `reflexio`.

## Initialize

Default Hosted Enterprise setup:

```python
from reflexio import ReflexioClient

client = ReflexioClient(timeout=5)
```

This reads `REFLEXIO_API_KEY` and uses `https://www.reflexio.ai/`. Do not pass `url_endpoint` and do not introduce `REFLEXIO_URL` for the default path.

If and only if the user explicitly requests a custom endpoint, use one existing configuration value:

```python
client = ReflexioClient(url_endpoint=settings.reflexio_url, timeout=5)
```

Alternatively, let the client read a user-supplied `REFLEXIO_URL`. Do not implement both mechanisms without a reason.

## Search and preserve learning identities

Use the synchronous method for a synchronous application:

```python
results = client.search(
    query=user_message,
    user_id=user_id,
    session_id=session_id,
    agent_version=agent_version,
    entity_types=["profiles", "user_playbooks", "agent_playbooks"],
    agent_playbook_status_filter=["approved"],
    top_k=3,
)
```

In an async application, use the native async equivalent with the same arguments:

```python
results = await client.search_async(...)
```

Build the injected context from `content` and retain these stable identities:

```python
retrieved_learnings = [
    *(
        {"kind": "profile", "learning_id": profile.profile_id}
        for profile in results.profiles
    ),
    *(
        {
            "kind": "user_playbook",
            "learning_id": str(playbook.user_playbook_id),
        }
        for playbook in results.user_playbooks
    ),
    *(
        {
            "kind": "agent_playbook",
            "learning_id": str(playbook.agent_playbook_id),
        }
        for playbook in results.agent_playbooks
    ),
]
```

Do not flatten the three result types into an undifferentiated instruction list. Render profiles as user facts/preferences and playbooks as behavioral guidance.

## Publish the completed turn

Use the synchronous method for a synchronous application:

```python
from reflexio import InteractionData

client.publish_interaction(
    user_id=user_id,
    session_id=session_id,
    source=source,
    agent_version=agent_version,
    interactions=[
        InteractionData(role="User", content=user_message),
        InteractionData(
            role="Agent",
            content=agent_response,
            retrieved_learnings=retrieved_learnings,
        ),
    ],
)
```

In an async application, use the native async equivalent with the same arguments:

```python
await client.publish_interaction_async(...)
```

Keep the normal defaults `wait_for_response=False`, `force_extraction=False`, and `skip_aggregation=False`. The call still waits for the HTTP response; `wait_for_response=False` means the server queues extraction instead of processing it synchronously.

Create the client once at the application's normal client/service lifetime rather than once per token or tool event.

## Connection check

When `REFLEXIO_API_KEY` is available, this is a read-only connection check:

```python
identity = client.whoami()
```

Inspect the returned identity or the raised exception. Do not print the API key.

For additional examples and complete method parameters, consult the [Reflexio developer documentation](https://www.reflexio.ai/docs), especially [search](https://www.reflexio.ai/docs/build/search), [publishing interactions](https://www.reflexio.ai/docs/build/user-interactions), and the [API reference](https://www.reflexio.ai/docs/api-reference).
