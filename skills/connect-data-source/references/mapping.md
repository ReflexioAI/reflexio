# Mapping sampled Braintrust records

Fetch `mapping_schema` from setup context; it is authoritative for the installed server. PUT a complete `definition`, not a partial patch. Construct rules with all nine fields explicitly present. Server defaults on omitted fields are not evidence; do not rely on them.

| Field | Meaning |
| --- | --- |
| `user` | Stable end-user identity; never a merchant, task label, trace ID or invented anonymous user. |
| `session` | Stable conversation ID shared by its turns; never a generic operation name. |
| `input` | Actual user-message text, not a system prompt or a message container. |
| `output` | User-facing answer text, not classification, routing or tool output. |
| `timestamp` | Source event time as timezone-aware ISO timestamp or positive Unix seconds. |
| `completion` | Positive end timestamp, true completed flag, or explicit completed status. |
| `version` | Agent/application learning version, not telemetry or model version. Ask before assigning a source/version constant. |
| `source` | Application request-source/environment label used by learning settings. |
| `references` | Learning references actually injected into this answer; absent differs from an explicit empty array. |

For an unmapped field use `{ "scope": "current", "path": "", "fallback": "", "transform": "none", "constant": "" }`. For a mapped field replace `path` with its observed JSON Pointer; choose `root` only when the root is actually resolved. Escape `~` as `~0` and `/` as `~1`. Arrays use actual numeric indices; do not assume every trace has the same message position.

A response rule has a unique `name`, `when` predicates (e.g. `{ "scope":"current", "path":"/span_attributes/name", "op":"eq", "values":["respond"] }`), and `fields` containing all nine entries. Use the existing `span` layout for one request/answer span. More complex message/history layouts require the server schema's explicit contracts; do not improvise them. Exactly one rule must match each intended answer. Use `unmatched:"held"`, `evaluation_only:false`, and `allow_user_only:false` for normal conversation setup. Preview before validating.

Current/root lookup uses raw `span_id` and `root_span_id`: a root must be uniquely present in the retained sample. `is_root:true` by itself is insufficient. Sibling spans, absent parents, arbitrary joins, and remote fetching of missing relatives are unsupported in setup version 1. Explain that limitation instead of copying a path from an unrelated record. Ingestion also has bounded context, so a sampled root does not guarantee every future response will have it available; unresolved records remain held.

Keep a non-secret local mapping artifact plus evidence record IDs and chosen traffic/history settings if helpful for resuming. Do not include raw credentials or unnecessary message content. The server's preview is the authority for coverage and required-field errors. Missing user/session/message/completion fields require clarification. Missing version/source can affect learning eligibility; inspect the returned eligibility diagnostics. Missing attribution alone does not block dialogue import.

## Offer a retrieved-learning logging patch

After approval, inspect the application's actual Reflexio search, context injection, and Braintrust response tracing path. Log on the same response span:

```json
{
  "metadata": {
    "reflexio": {
      "retrieved_learnings": [
        { "kind": "profile", "learning_id": "actual-injected-profile-id" }
      ]
    }
  }
}
```

Map `/metadata/reflexio/retrieved_learnings`. Supported kinds: `profile`, `user_playbook`, `agent_playbook`. Read IDs from the actual search results (`profile_id`, `user_playbook_id`, `agent_playbook_id`) and convert them to strings. Include only items actually injected into that response's context, not all retrieval candidates. Reset attribution per response; preserve request-local association under concurrent requests. Write `[]` when instrumentation knows that no learning was injected; leave absent if unknown. Never infer IDs from citations or manufacture references for historical traces.

If the application does not yet retrieve Reflexio learning, explain that logging alone cannot create attribution; ask before adding retrieval/context integration. Reuse its existing tracing SDK/version and merge metadata without discarding existing fields. Do not add `publish_interaction`; the connector handles import after frontend activation. Test actual IDs, explicit empty capture, concurrent requests, and unchanged user responses before claiming instrumentation works.

## Offer a patch for required response fields

If the developer confirms that required fields are not logged on the response, offer
to enrich its existing trace. Use real variables from that response's request context,
not placeholders or inferred IDs. Preserve existing metadata and concurrent request
isolation. For example, adapt the application's existing span logging call:

```python
span.log(
    input=current_user_message,
    output=final_user_facing_answer,
    metadata={
        **existing_metadata,
        "user_id": actual_end_user_id,
        "session_id": actual_conversation_id,
    },
)
```

Verify these variables' meanings with the developer, retain existing span completion
instrumentation, and inspect a new trace after testing the patch. A visitor identifier
can be appropriate if it really represents the end user; a merchant or task identifier
cannot. An absent root is unavailable evidence, not proof of missing logging. Do not
copy an earlier serialized chat history into `input` to make validation pass. Existing
historical records remain held until their required evidence is resolved; this patch
only improves newly recorded traces. Never introduce `publish_interaction` or enable
M2 fetching/joins as a workaround.
