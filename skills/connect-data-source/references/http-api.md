# Data-source setup API, version 1

All requests go to the chosen Reflexio data-plane endpoint. Send:

```http
Authorization: Bearer <REFLEXIO_API_KEY>
X-Reflexio-Project: <destination-project-id>
Content-Type: application/json
```

Use the developer application's HTTP library or a short local Python/JavaScript HTTP script. Read keys from environment/secure input, use bounded timeouts, and inspect safe error codes. Never echo credential payloads. No CLI or SDK dependency is required.

Base path: `/api/projects/{destination-project-id}/data-sources`.

| Method and relative path | Request / result |
| --- | --- |
| GET `/setup-context` | Protocol version, project binding, mapping JSON schema, supported scopes, existing connection/stream IDs, relative frontend `review_path`. Require `setup_version = 1`. |
| GET base | Connection and stream configuration plus sample summaries. Sample summaries omit records; fetch sample detail to inspect them. |
| POST `/connections` | `{ "name": "Support traces", "api_key": "<Braintrust key>" }`; returns redacted connection and revision. |
| GET `/connections/{id}/projects` | Available Braintrust project IDs, names and workspace IDs; check truncation. |
| PUT `/connections/{id}/stream` | `{ "revision": <connection revision>, "external_project_id": "<Braintrust project ID>", "filters": [] }`; returns stream. Re-read connection after selection because its revision advances. |
| PUT `/streams/{id}/history-draft` | `{ "revision": <stream revision>, "history": true, "history_start": <Unix seconds>, "history_end": <Unix seconds> }`. End is exclusive; omit end to continue history through activation. For new-only collection send `history:false` with no bounds. |
| POST `/streams/{id}/samples` | `{ "revision": <stream revision>, "sample_window": { "start": "2026-08-12T00:00:00Z", "end": "2026-08-20T00:00:00Z" } }`; returns sample ID, HTTP 202. |
| GET `/samples/{id}` | Poll queued/running at a bounded interval (start at 2 seconds); complete includes records. Stop and explain failed/expired status. |
| GET `/streams/{id}/mapping` | Saved draft, revision, validated revision and automatic-mapping provenance if present. |
| PUT `/streams/{id}/mapping` | `{ "expected_revision": <current revision>, "definition": <complete mapping> }`; save returns next revision. |
| POST `/streams/{id}/preview` | `{ "sample_id": "…", "expected_revision": <mapping revision> }`; normalized records, ready/held/excluded counts, structured `blockers`, `rule_coverage`, and sample digest. Does not import. |
| POST `/streams/{id}/mapping/validate` | Same as preview plus `sample_digest` from that preview. Requires at least one eligible conversation or supported replay; does not activate. |
| GET `/streams/{id}/status` | Lifecycle, saved `activation_draft`, coverage and counters. Draft means not importing. |

Traffic filters use `{ "path": "span_attributes.name", "op": "eq", "values": ["respond"] }`. Other supported operations: `in`, `exists` (empty values); supported paths: `metadata.*`, `span_attributes.name`, `tags`. All filters must match. Offer filters based on observed traffic; ask before applying them. Filtering does not fetch related spans.

`rule_coverage` includes matched/ready/held/excluded counts per rule and `conflicted`
for overlapping selectors. Overlapping records count in each implicated rule; these
counts are not additive across rules. Resolve conflicts before validation.

Use integer revisions exactly as returned. Keep mapping revision separate from connection/stream/lifecycle revisions. HTTP 401 means invalid Reflexio credential, 403 insufficient role or wrong project, 409 stale/conflicting state, 410 expired evidence, 422 invalid configuration/provider credentials, and 429 provider rate limiting. Follow the response's safe `detail.code` / `detail.message`; do not log raw provider payloads.

The activation API remains available to Reflexio's frontend. **This setup guide stops at a validated draft and review link. Do not call `/activate`.** Source management does not grant governance/erase permission.
