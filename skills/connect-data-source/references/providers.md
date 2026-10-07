# Provider setup

Read the row and flow for the provider selected by the developer. Start with key-scoped setup context; check its advertised providers/regions. Hosted pull connectors do not accept private/custom provider URLs. The copied prompt names a source, not the Reflexio destination: the Reflexio API key binds the latter.

## Pull connectors

POST `/connections` with `name`, explicit `provider` and `region`, and the credential fields below. Read secrets from environment/secure input; placeholder strings here must never be sent literally.

| Provider | Region | Credential payload | Source selection |
| --- | --- | --- | --- |
| `braintrust` | `us` | `api_key`: Braintrust read key | Discover project name/ID. |
| `langfuse` | `us`, `eu`, `jp`, `hipaa-us` | `api_key`: secret key; `public_key`: public key | Keys identify the project. Verify the supplied name/ID against discovery; do not add `workspace_id`. |
| `langsmith` | `us`, `eu` | `api_key`: API key; optional `workspace_id` if required by the key | Discover project name/ID. LangSmith calls a project a session in its API; this is not end-user conversation identity. |
| `phoenix` | `us` | `api_key`: API key; `space_name`: the segment after `/s/` in the Phoenix Cloud URL | Space is required: 1–128 ASCII letters, digits, hyphens or underscores. Discover project name/ID within that space. Private Phoenix is unsupported. |

Choose the advertised region hosting the source, not the user's location. Omit irrelevant credential fields. Resolve the supplied project against GET `/connections/{id}/projects`, observing completeness rules in [the API reference](http-api.md). PUT `/connections/{id}/stream` with the current connection revision, the discovered `external_project_id` and agreed `filters`. A connection credential's LangSmith workspace hint and the discovered stream workspace are separate fields; use workspace identity to disambiguate discovery, but omit `workspace_id` from stream PUT. The server binds the stream to the discovered workspace internally and rejects extra request fields.

Re-read connection-specific setup context after selecting the connection. Follow the shared traffic, time, sample, mapping, preview, and validation steps. Braintrust supports filter pushdown with local checks; Langfuse, LangSmith, and Phoenix filter after download. Cloud samples use bounded pages rather than Braintrust's time buckets. Explain sample completeness accurately and never promise root/related support or optimized reads without the installed connection's capabilities.

## OpenTelemetry: direct receiver

This is a push source, not a provider-history connector. The supplied application/service name identifies the receiver connection; do not ask for a provider project or provider API key.

1. Resume a matching draft receiver, or generate a cryptographically random URL-safe bearer token (32–256 characters using ASCII letters, digits, hyphens or underscores), save it through the developer's environment/secret manager, and POST `/connections` with `provider:"otel"`, `region:"us"`, `name:<service name>`, `api_key:<receiver token>`. This token is distinct from the Reflexio API key. Never print it or include it in the review link. When resuming, ask for the securely stored token; do not rotate credentials automatically.
2. GET `/connections/{id}/projects`. The server supplies one synthetic direct stream; use its returned `id` as `external_project_id` in the stream PUT with the current revision and agreed filters. Read `ingest_path` from the returned stream; resolve it against the supplied Reflexio endpoint. Never construct org/project/stream IDs yourself.
3. Configure the application's existing tracing/exporter with the developer's agreement. Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to that exact full URL, `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`, and the bearer header securely (`OTEL_EXPORTER_OTLP_TRACES_HEADERS` uses `Authorization=Bearer%20<TOKEN>`). The generic `OTEL_EXPORTER_OTLP_ENDPOINT` appends `/v1/traces`; do not put the full per-signal URL there. Preserve existing tracing destinations unless the developer chooses to replace them. Do not add a separate Reflexio interaction-publish flow.
4. Send a representative trace using the existing instrumentation, or ask the developer to exercise the application. The receiver accepts OTLP/HTTP traces as JSON or protobuf, optionally gzip; not gRPC, metrics, or logs. Use HTTPS for network bearer-token traffic. Limits are 1 MB uncompressed and 100 spans per request. Traces sent while draft are encrypted inspection samples only, capped at 50 records, and are not imported.
5. GET the source inventory to obtain its completed sample ID, then GET `/samples/{id}`. Do not POST a pull sample job or request provider history. If no sample exists yet, return the draft review link and explain which test trace is needed. Re-read `/setup-context?connection_id=<id>`: receiver mappings support only `current`, with no root/related fetching or provider backfill. Map observed user identity, conversation identity, input/output and completion using the shared schema. Stable `user.id`/`session.id` attributes and `input.value`/`output.value` are useful instrumentation conventions, not proof that the actual application uses them.
6. Save a new-only history draft (`history:false`, no history bounds) using the current stream revision. Save, preview, and validate the mapping through the shared endpoints. Return this connection's review link and stop before activation. Explain: **review and activate in Reflexio, then resend any draft examples you want imported**. Older traces can be pushed after activation; there is no provider backfill. An accepted batch is durable inbox admission, not proof every record was imported; mapping blockers may still hold records.

Continue with the shared blocker, draft-summary, and optional attribution steps. Missing required identity blocks the draft; absent learning attribution does not. Apply optional instrumentation changes only with the developer's agreement.
