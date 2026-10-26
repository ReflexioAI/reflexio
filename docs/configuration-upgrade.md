# Canonical environment names

Legacy configuration spellings are no longer accepted. The first environment
loader rejects their presence before dotenv loading or blank-value pruning;
direct policy readers reject their obsolete names too. An error names the old
and replacement settings without printing values. Remove old assignments even
when a canonical assignment is present. No automatic copying or precedence
between old and new names remains.

| Removed name | Canonical name / effective policy |
| --- | --- |
| PUBLIC_SCHEMA_AUDIT_ENABLED | PUBLIC_SCHEMA_AUDIT_INTERVAL_SECONDS; zero disables periodic audits, positive seconds enable them |
| REFLEXIO_SEARCH_RECENCY_ENABLED | REFLEXIO_SEARCH_RECENCY_POOL_SIZE; zero disables, positive enables scored recency retrieval and ranking |
| DATA_SUPABASE_DB_URL, POSTGRES_DB_URL | DATA_DB_URL; preserve the exact database target |
| WEB_CONCURRENCY | Explicit server `--workers N`; `REFLEXIO_SERVER_WORKERS` is diagnostic metadata, not a Uvicorn worker selector |
| INTERACTION_CLEANUP_THRESHOLD | REFLEXIO_ROW_LIMIT_INTERACTIONS |
| REFLEXIO_EMBED_TORCH_THREADS | REFLEXIO_NATIVE_COMPUTE_THREADS |
| REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED | REFLEXIO_MISSING_VECTOR_BACKFILL_CAP; zero disables, positive caps each org/tick |
| REFLEXIO_PUBLISH_TIMING_ENABLED, REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS | REFLEXIO_PUBLISH_TIMING_MIN_MS; off disables, nonnegative milliseconds enable |
| CONTROL_PLANE_INGEST_URL | CONTROL_PLANE_URL |
| TOPUP_RECONCILE_ENABLED, TOPUP_RECONCILE_INTERVAL_SECONDS | CREDIT_INVOICE_RECONCILE_INTERVAL_SECONDS; zero disables, positive seconds enable |
| BILLING_METERING_SWEEPS_ENABLED, BILLING_METERING_SWEEP_INTERVAL_SECONDS | BILLING_METERING_MAINTENANCE_INTERVAL_SECONDS; zero disables, positive seconds opt in |
| BILLING_PERIOD_CLOSE_ENABLED, PERIOD_CLOSE_INTERVAL_SECONDS | BILLING_PERIOD_CLOSE_INTERVAL_SECONDS; zero disables, positive seconds enable |

The shared distribution guard includes enterprise-owned names so generic OSS
imports cannot discard obsolete blank assignments before enterprise startup.
This also rejects obsolete enterprise variables inherited by a standalone OSS
process. Update the shell environment and dotenv files before upgrading.

Do not rename a boolean to a numeric setting verbatim. Preserve the former
effective policy: missing-vector repair defaults to 200; publish diagnostics
default off (former true without a threshold meant 2000 ms); metering defaults
off (former true without a cadence meant 3600 seconds); reconciliation defaults
300 seconds; period close defaults 3600 seconds. A blank former period-close
enable switch meant disabled and must become canonical zero. Publish timing
zero enables every-request diagnostics, so a former false must become off.

Independent throttles, grace periods, concurrency, and customer retention caps
remain configurable. No schema/data migration is required. AWS secret JSON
fields may keep their physical names when explicitly injected under canonical
runtime environment names; the runtime does not resolve old names.

When `PYTHON_DOTENV_DISABLED` is set to python-dotenv's truthy values
(`1`, `true`, `t`, `yes`, `y`, case-insensitive), disabled files are neither loaded
nor checked/pruned. Removed names in the process environment still fail. This
preserves isolated tool execution without accepting aliases as runtime settings.

For direct Uvicorn starts that previously used `WEB_CONCURRENCY=N`, pass
`--workers N` explicitly before removing the environment assignment. The Reflexio
launcher also accepts `--workers N` (with `--no-reload` for multiple workers).
Renaming the old input to `REFLEXIO_SERVER_WORKERS` alone does not preserve the
worker count: the latter records diagnostics and the launcher overwrites it.

Recency admission belongs to the candidate-pool budget. Remove the old enable
assignment even if blank or a canonical pool is present. Former false maps to
zero; true/unset retains a valid positive custom pool or the unchanged default
20. Former invalid zero/negative pool values were clamped to one while enabled;
explicitly choose one to preserve that old effective policy. The fractional and
logit penalty scales remain independent and configurable. Zero penalties do not
disable a positive candidate pool. Temporal query signals, session deduplication
and relevance-floor budgets retain their own policies.

Enterprise schema-audit admission belongs to its cadence. Remove the old
`PUBLIC_SCHEMA_AUDIT_ENABLED` assignment, including blanks or coexistence with
the canonical setting. Former false becomes zero; true/unset preserves a
positive custom cadence or the unchanged 86400-second default. Role restrictions
and UTC scheduling remain independent. An explicit manual audit with periodic
scheduling disabled must supply its existing positive `interval_seconds`
argument; no hidden manual cadence is substituted. Adopt the matched enterprise
implementation before selecting zero; the preceding enterprise reader accepts
only positive intervals.

## Inference numerical-library budget

`REFLEXIO_NATIVE_COMPUTE_THREADS` is the single positive integer CPU library
budget (default 2). It owns Torch, BLAS/OpenMP and ONNX intra-op session limits
inside the inference service. ONNX inter-op remains one for its sequential graph
execution. These are per-backend limits, not an aggregate process CPU quota.
`TOKENIZERS_PARALLELISM` still controls tokenizer fork safety independently.

Remove old inference inputs `REFLEXIO_INFERENCE_TORCH_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`. Unlike the strictly rejected aliases
above, blank/default-equivalent `2` library assignments can remain during rollout
but do not select a limit. Any custom legacy assignment requires an explicit new
budget before model startup; no value is inferred from retired inputs. If old
limits differ, choose the intended shared limit explicitly. Once chosen, the new
budget is authoritative even alongside old assignments. Native library variables
are then derived outputs for cold imports, and loaded pools are configured at
model construction. API/client processes do not configure native pools merely
by importing this module.

Default behavior changes for unmanaged Nomic: leaving its old Torch input unset
previously preserved Torch's detected pool size; it now uses 2. Set the new budget
to the required positive size to retain that capacity. Managed E5 and MiniLM keep
their previous default 2; managed query processes retain 1 through explicit role
configuration. This requires the matched enterprise launcher/image update.

For a direct service launch, set the canonical budget before importing models,
or execute `python -m reflexio.native_compute -- python -m uvicorn
reflexio.server.llm.embedding_service:app`. The wrapper execs the command with
prepared library environment and keeps its PID/signal ownership. No environment
file is rewritten, and the retired `REFLEXIO_EMBED_TORCH_THREADS` alias remains
strictly rejected, including blank assignments.
