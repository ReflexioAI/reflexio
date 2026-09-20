# Mapping sampled Braintrust records

Fetch `mapping_schema` from setup context; it is authoritative for the installed server. PUT a complete `definition`, not a partial patch. Construct rules with all nine fields explicitly present. Server defaults on omitted fields are not evidence; do not rely on them.

| Field | Meaning |
| --- | --- |
| `user` | Stable end-user identity that recurs *across* conversations; never a merchant, task label, trace ID, conversation/session ID, or invented anonymous user. It must not read the same path as `session`, nor hold the same value as `session` on every record: that makes each conversation its own user, so profiles and playbooks never accumulate. Preview reports this as `identity_path_collision` / `identity_equals_session`. An identity is often on a sibling span rather than the answer — check related spans before concluding there is none, and ask the developer rather than choosing the nearest available ID. |
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

Current/root lookup uses raw `span_id` and `root_span_id`: a root must be uniquely present in retained or expanded evidence. `is_root:true` alone is insufficient. Check setup-context capabilities; older servers without `related_span_fetch` support only current/root mappings. Do not send a version-2 mapping to those servers.

## Related fields (mapping version 2)

Sampling automatically expands traces when `related_span_fetch` is true. Each anchor record has `source_project_id` and `related_context: {status, code?, records}`. Each related record has its own `id`, `source_project_id`, and `raw` Braintrust span. Only `status:"complete"` means the bounded lookup was exhausted; it does not promise no future spans will arrive. `partial`, `failed`, and `pending` cannot establish uniqueness. Inspect codes and refresh the sample when appropriate; do not work around limits by guessing.

Set `definition.version:2`, keep all nine field entries, and add `related_sources` to each applicable `span` rule. Example **only when these names and shared values are observed**:

```json
{
  "question": {
    "when": [{"scope":"current", "path":"/span_attributes/name", "op":"eq", "values":["follow-up-question"]}],
    "correlations": [
      {"current_path":"/metadata/message_id", "related_path":"/metadata/message_id"},
      {"current_path":"/metadata/session_id", "related_path":"/metadata/session_id"}
    ]
  }
}
```

The object above is the rule's `related_sources`. Its predicates inspect the candidate span; correlation `current_path` inspects the answer and `related_path` inspects the candidate. Map the question using `{"scope":"related.question","path":"/metadata/latest_user_message","fallback":"","transform":"none","constant":""}`. Keep fields already on the answer in `current` scope. For a verified `core-q&a` layout, those may include `/metadata/visitor_id`, `/metadata/session_id`, `/output`, `/created`, and `/metrics/end`. Names and paths are examples, never defaults to copy without checking actual values.

Relationships always remain within the same source project and `root_span_id`. Use actual message/turn identifiers, plus session correlation where available. Exactly one candidate must satisfy the selector and every correlation; missing correlation values never match each other. A question helper can be logged after the answer, so do not impose a before-answer time condition. Multiple candidates are held; refine the selector or correlations rather than selecting the newest span.

Only the anchor answer counts as an importable interaction. Traffic filters and historical bounds select anchors, not supporting spans. Expanded context may legitimately fall outside that window. Preview returns field provenance with related span IDs and structured held reasons. Save, preview, and validate through the same endpoints as version 1; compare sample-only rule coverage and missing/ambiguous results. Never submit arbitrary queries, executable joins, cross-trace joins, or unverified paths.

Missing required related fields stay held. Optional attribution remains nonblocking. A related `references` mapping also requires `reference_response_id_path` on that related span to equal the answer's `span_id`; a shared session alone is not proof that learnings were injected into this answer. Prefer the existing response-span logging convention below.


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

Map the `references` field to `/metadata/reflexio/retrieved_learnings` once a fresh sample confirms that path on the response span. Supported kinds: `profile`, `user_playbook`, `agent_playbook`. Read IDs from the actual search results (`profile_id`, `user_playbook_id`, `agent_playbook_id`) and convert them to strings. Include only items actually injected into that response's context, not all retrieval candidates. Reset attribution per response; preserve request-local association under concurrent requests. Write `[]` when instrumentation knows that no learning was injected; leave absent if unknown. Never infer IDs from citations or manufacture references for historical traces.

Re-read source status first. If already active, obtain explicit authorization for the separate mapping update before changing its saved mapping. Save the updated mapping, preview the fresh sample, and validate using its `sample_id`, the preview's `sample_digest`, and the saved mapping revision as `expected_revision`. For a draft source, return the updated review link for frontend activation. If already active, explain that saving and validating a draft leaves the active mapping unchanged. Only after explicit authorization to update that active source, re-read `/streams/{id}/status` and PUT `/streams/{id}/active-mapping` with `{ "revision": <status revision>, "mapping_revision": <validated mapping revision> }`, then verify status reports the selected mapping revision. This is a separate lifecycle operation, not part of initial setup; do not call `/activate` or promise retroactive attribution for imported records.

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
only improves newly recorded traces. Never introduce `publish_interaction`. Use supported related-span mapping to recover existing historical evidence before proposing new logging.
