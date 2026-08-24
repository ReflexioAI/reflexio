# HTTP integration

Use this route only when the target application cannot use the Python client. Reuse the target project's existing HTTP library and add one small typed adapter.

## Connection

Default base URL:

```text
https://www.reflexio.ai
```

Do not add a URL setting for the default path. Only use a different base URL when the user explicitly supplies or requests an override.

For Hosted Enterprise or an authenticated custom endpoint, send these headers:

```http
Authorization: Bearer <REFLEXIO_API_KEY>
Content-Type: application/json
User-Agent: <application-name>-reflexio
```

If the developer explicitly requested a default unauthenticated Local OSS endpoint, omit only the `Authorization` header.

Keep a bounded timeout. Inspect the HTTP status and raw response body before diagnosing authentication, routing, or schema failures. Never log the bearer token.

## Search before agent execution

```http
POST /api/search
```

```json
{
  "query": "the user's current request",
  "user_id": "stable-user-id",
  "session_id": "stable-session-id",
  "agent_version": "support-agent@2",
  "entity_types": ["profiles", "user_playbooks", "agent_playbooks"],
  "agent_playbook_status_filter": ["approved"],
  "top_k": 3
}
```

The successful response contains `profiles`, `user_playbooks`, and `agent_playbooks`. Render each type separately and retain these IDs:

| Result | Stable ID | Publish kind |
| --- | --- | --- |
| Profile | `profile_id` | `profile` |
| User playbook | `user_playbook_id` | `user_playbook` |
| Agent playbook | `agent_playbook_id` | `agent_playbook` |

Convert numeric playbook IDs to strings when creating `retrieved_learnings`.

## Publish after the response completes

```http
POST /api/publish_interaction
```

```json
{
  "request_id": "stable-id-for-this-publish-batch",
  "user_id": "stable-user-id",
  "session_id": "stable-session-id",
  "source": "support-agent:v2",
  "agent_version": "support-agent@2",
  "interaction_data_list": [
    {
      "role": "User",
      "content": "the user's current request"
    },
    {
      "role": "Agent",
      "content": "the completed agent response",
      "retrieved_learnings": [
        {"kind": "profile", "learning_id": "profile-id"},
        {"kind": "agent_playbook", "learning_id": "42"}
      ]
    }
  ]
}
```

Generate `request_id` once when the publish batch is added to the host's durable buffer, persist it with that batch, and reuse the exact value for every attempt. Never generate a new request ID during a retry; the stable ID is what prevents an ambiguous timeout from creating a second stored request.

Use the same identity values as search. Treat permanent `4xx` validation failures differently from potentially retryable timeouts, connection failures, `429`, and `5xx` responses, and retry those transient classes only with the original `request_id`. Keep failures observable without replacing a valid agent response.

## Read-only connection check

```http
GET /api/whoami
```

Use the same headers. A successful identity response verifies the endpoint and API key without publishing customer or synthetic interaction data.

For additional examples and complete request and response fields, consult the [Reflexio developer documentation](https://www.reflexio.ai/docs), especially [search](https://www.reflexio.ai/docs/build/search), [publishing interactions](https://www.reflexio.ai/docs/build/user-interactions), and the [API reference](https://www.reflexio.ai/docs/api-reference).
