# Data-source setup API, version 1

All requests go to the chosen Reflexio data-plane endpoint. Send:

```http
Authorization: Bearer <REFLEXIO_API_KEY>
Content-Type: application/json
```

Use the developer application's HTTP library or a short local Python/JavaScript HTTP script. Read keys from environment/secure input, use bounded timeouts, and inspect safe error codes. Never echo credential payloads. No CLI or SDK dependency is required.

Start with `GET /api/data-sources/setup-context`. Base path: the returned `api_base_path` (currently `/api/data-sources`). The server derives the destination from the authenticated key. Do not send `X-Reflexio-Project` or ask for a Reflexio project ID. The returned `project_id` and `review_path` are server-generated context, not inputs the user must supply.

Browser sessions retain `/api/projects/{project_id}/data-sources` with a matching `X-Reflexio-Project` header. This compatibility route is not needed for API-key setup. A conflicting header/path cannot redirect a key to another project. The key-scoped route rejects session tokens, limited keys, and revoked keys. A 404 at the initial endpoint on an older server means an upgrade is required; do not fall back to asking for an internal project ID.

| Method and relative path | Request / result |
| --- | --- |
| GET `/setup-context` | Protocol version, project binding, mapping JSON schema, supported scopes, existing connection/stream IDs, relative frontend `review_path`. Require `setup_version = 1`. |
| GET base | Connection and stream configuration plus sample summaries. Sample summaries omit records; fetch sample detail to inspect them. |
| POST `/connections` | `{ "name": "Support traces", "api_key": "<Braintrust key>" }`; returns redacted connection and revision. |
| GET `/connections/{id}/projects` | `{ "projects": [...], "truncated": false }`; entries contain `id`, `name`, and `workspace_id`. Only explicit `truncated:false` establishes completeness. `truncated:true` means the server's bounded pagination stopped early; this endpoint exposes no continuation cursor. If the flag is missing, completeness is unknown. Do not infer absence or unique name matches from an incomplete list, invent pagination parameters, or repeatedly fetch the same bounded list; explain that discovery must be completed before resolving those choices. |
| PUT `/connections/{id}/stream` | `{ "revision": <connection revision>, "external_project_id": "<Braintrust project ID>", "filters": [{ "path": "span_attributes.name", "op": "eq", "values": ["<observed response span name>"] }]  }`; returns stream. Re-read connection after selection because its revision advances. |
| PUT `/streams/{id}/history-draft` | `{ "revision": <stream revision>, "history": true, "history_start": <Unix seconds>, "history_end": <Unix seconds> }`. End is exclusive; omit end to continue history through activation. For new-only collection send `history:false` with no bounds. |
| POST `/streams/{id}/samples` | `{ "revision": <stream revision>, "sample_window": { "start": "2026-08-12T00:00:00Z", "end": "2026-08-20T00:00:00Z" } }`; returns sample ID, HTTP 202. |
| GET `/samples/{id}` | Poll queued/running at a bounded interval (start at 2 seconds); complete includes records. Stop and explain failed/expired status. |
| GET `/streams/{id}/mapping` | Saved draft, revision, validated revision and automatic-mapping provenance if present. |
| PUT `/streams/{id}/mapping` | `{ "expected_revision": <current revision>, "definition": <complete mapping> }`; save returns next revision. |
| POST `/streams/{id}/preview` | `{ "sample_id": "…", "expected_revision": <mapping revision> }`; normalized records, ready/held/excluded counts, structured `blockers`, `rule_coverage`, and sample digest. Does not import. |
| POST `/streams/{id}/mapping/validate` | Same as preview plus `sample_digest` from that preview. Requires at least one eligible conversation or supported replay; does not activate. |
| GET `/streams/{id}/status` | Lifecycle, saved `activation_draft`, coverage and counters. Draft means not importing. |

Traffic filters use `{ "path": "span_attributes.name", "op": "eq", "values": ["respond"] }`. Other supported operations: `in`, `exists` (empty values); supported paths: `metadata.*`, `span_attributes.name`, `tags`. All filters must match. Offer filters based on application tracing code and observed traffic; clarify choices not already agreed to. The placeholder above must be replaced with an observed name, never sent literally. Use `filters: []` only for an intentional all-traffic selection. After PUT, GET the inventory to verify the saved filters, then sample and validate that selection. Preserve agreed filters on resume; an empty sample is not permission to broaden them. When `traffic_filter_pushdown` is true, the same saved filters drive source-side sampling, backfill, ongoing collection, and progress counts. Stream `filter_execution` reports `before_download`, `partial`, or `after_download`. Existence checks and numeric/boolean string coercion stay local; metadata paths also retain local validation for array traversal. If every predicate requires local evaluation, execution is `after_download`. Partial queries return candidate records, so counts can exceed final imports. On older servers, do not promise before-download filtering. Filters select answer anchors. When setup context advertises `related_span_fetch`, sampling separately expands each retained trace without applying these filters to its supporting spans.

`rule_coverage` includes matched/ready/held/excluded counts per rule and `conflicted`
for overlapping selectors. Overlapping records count in each implicated rule; these
counts are not additive across rules. Resolve conflicts before validation.

Use integer revisions exactly as returned. Keep mapping revision separate from connection/stream/lifecycle revisions. HTTP 401 means invalid Reflexio credential, 403 insufficient role or wrong project, 409 stale/conflicting state, 410 expired evidence, 422 invalid configuration/provider credentials, and 429 provider rate limiting. Follow the response's safe `detail.code` / `detail.message`; do not log raw provider payloads.

The activation API remains available to Reflexio's frontend. **This setup guide stops at a validated draft and review link. Do not call `/activate`.** Source management does not grant governance/erase permission.

## Historical sample coverage and recovery

Newer protocol-1 servers advertise `sampling` and `field_guidance` in setup context.
A sample can include `buckets` (`start`, `end`, `status`, `retained`, `code`) and
`example_ids`. Status distinguishes `sampled`, `empty` and `unread`. Read the full
sample response, not only the summary inventory. Windowed samples longer than a day
cover up to seven buckets within shared retention limits (50 records, 1 MB).
These are sample counts, never estimated provider totals. Partial reads are retained;
rate limits stop subsequent windows. Wait for cooldown before retrying; do not loop.

Servers advertising `automatic_read_retries` and `shared_connection_cooldown`
automatically resume queued sample jobs after transient Braintrust failures.
`retry_at` is a Unix timestamp: poll the same job, respecting that time rather
than creating replacement jobs. Successfully read buckets and expanded traces
are retained. `/status` exposes `provider_read.waiting`, `retry_at`, and `code`;
active sources also expose `pending_context`, `held_for_review`, `history_position`
and `history_pages`. A provider cooldown is not a missing-field finding. HTTP
errors may include `detail.retry_after_seconds` and `Retry-After`; honor them.
Older servers without these capabilities still require an explicit new sample.
`historical_read_windows:checkpointed_utc_days` means imports resume date-scoped
pagination without a total historical page ceiling; sample limits are unchanged.

Prefer the suggested examples, but inspect other retained answer layouts when coverage
is incomplete. Preview `field_guidance` includes candidate identity paths and evidence
IDs, unavailable-root explanations and next steps. For a newly proposed or changed identity mapping, clarify uncertain semantics with
the developer. When resuming a draft, preserve its existing identity mappings unless
the evidence contradicts them; do not add a redundant confirmation gate for fields
already mapped and resolving correctly. Missing or conflicting required fields still
need clarification. Before validation, resolve overlapping rule conflicts and ensure
the preview contains at least one eligible conversation or supported replay. Never
copy an unrelated helper field onto an answer.

The optional Reflexio model endpoint can return `mapping_timeout`,
`mapping_provider_unavailable`, `mapping_rate_limited`, or `mapping_invalid_output`.
If using it, retry a timeout/connection failure at most once with the same revision.
Do not automatically retry invalid output, conflicts or rate limits. Agent-inferred
mappings still need no Reflexio model request.

## Related-span capabilities and evidence

Setup context advertises `mapping_versions:[1,2]`, `mapping_scopes` including `related.NAME`, `related_span_fetch:true`, `sibling_joins:true`, `related_span_layouts:["span"]`, and `related_span_correlations:"explicit_equality"` only on servers implementing the same resolver for preview and ingestion. Follow `related_span_limits`; currently 100 spans / five pages / 1 MB per trace and 8 MB per expanded sample. No separate expansion endpoint or raw Braintrust query is needed.

The existing POST sample job and GET sample endpoints return anchors with `related_context`. Inspect `status` (`complete`, `partial`, `failed`, `pending`), optional safe `code`, and `records`. Use the mapping reference's version-2 fields with the existing mapping PUT, preview POST, and validation POST. A failed or truncated lookup is unavailable evidence, not proof of absent logging. The same mapping and evidence appear in the frontend review link. Setup must never call Activate.

## Historical message allowance

Servers advertising `backfill_limit` start new activations with a 5,000-message
historical allowance. One interaction means one individual message, not one trace
or conversation. Complete imports are atomic: the next import may require a
higher total even before all 5,000 slots are used. Saved filters and history dates
continue to bound the selection; ongoing traffic is collected independently.

Status `backfill` contains `message_limit` (null means all), `imported_messages`,
`required_total`, and `state`. Distinguish `limit_reached` / `needs_higher_limit`
from `complete`. Source-record scan estimates are not message counts.

After activation, and only on an explicit user request to extend history, PUT
`/streams/{id}/backfill-limit` with `{ "revision": <lifecycle revision>,
"message_limit": <new cumulative total> }`; send explicit null for all matching
history in the saved range. Omission is invalid. A finite total must increase the
existing allowance. Re-read status on conflict. This does not resume a paused
source, change filters/dates, or reopen explicitly cancelled history. During setup,
return the draft review link and leave this choice to the user in Overview.
