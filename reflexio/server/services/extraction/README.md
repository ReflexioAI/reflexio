# services/extraction

Shared async extraction runtime for profile and playbook pipelines.

This package is intentionally domain-neutral. Profile and playbook modules own
their prompts, schemas, extractor semantics, and storage decisions; this package
owns the reusable runtime needed when extraction pauses, waits for more
information, and resumes outside the request path.

## Files

| File | Responsibility |
|------|----------------|
| `resumable_agent.py` | Runs extraction agents, records agent usage, injects resolved pending-tool context, and exposes pending tool-call helpers. |
| `pending_tool_call_dispatch.py` | Implements the `ask_human` and pending-info tool dispatch flow. |
| `prior_answer_search.py` | Finds and formats previous human answers for async extraction context. |
| `agent_run_records.py` | Builds durable extraction-agent run records and source interaction identity. |
| `resume_scheduler.py` | Discovers and schedules due paused/finalization work in a background singleton. |
| `resume_worker.py` | Resumes paused runs, rebuilds request context, and records retry state. |
| `outcome.py` | Provides the generic extraction outcome wrapper used by callers. |

## Boundary Rules

- Keep profile-specific and playbook-specific extraction behavior in their own
  modules; call this package only for shared async runtime concerns.
- Add a new file here when the behavior is shared by more than one extraction
  caller or is part of the resumable runtime itself.
- Split into subpackages only when a responsibility grows large enough that a
  flat package stops being easier to scan.
- Do not recreate removed legacy modules such as `tools.py`, `plan.py`, or
  `invariants.py`; use the current focused files above.
- New durable extraction runs require a non-empty `user_id`. Nullable stored
  bindings remain readable for backward compatibility; playbook resume derives
  an in-memory owner only from complete, unanimous persisted source evidence.

## Resume discovery

When an `org_id_provider` is installed, the scheduler calls it on every tick and
treats its actionable org list as authoritative for cross-ref discovery. Without
a provider, local storage discovery remains the fallback. Each provider result
selects a current org for reading scheduler configuration, so an org retained
from an earlier tick cannot block later discovery if it becomes stale. If the
provider itself raises, the scheduler still attempts the last bootstrap org for
that tick. Before draining each discovered org context, it expires pending tool
calls on that context's storage ref; separate refs have separate queues.
