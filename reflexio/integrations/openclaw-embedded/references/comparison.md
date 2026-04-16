# openclaw-embedded vs integrations/openclaw

Two Openclaw integrations ship in the same repo. This doc explains when to pick which.

| Aspect | `integrations/openclaw/` (federated) | `integrations/openclaw-embedded/` (this plugin) |
|---|---|---|
| Reflexio server required | Yes | No |
| Storage | Reflexio server (SQLite / Supabase / pgvector) | Openclaw's native memory (`.md` files at `.reflexio/`) |
| Extraction LLM | Server-side (uses Reflexio's prompt bank on the server) | Openclaw's agent LLM + sub-agents (via `api.runtime.subagent.run()`) |
| Dedup | Server-side batch + cross-instance aggregation | Per-write shallow + daily full-sweep |
| Multi-user / multi-agent | Yes (each `agentId` → distinct `user_id`) | No (one instance = one user) |
| Cross-instance playbook sharing | Yes (aggregation → agent playbooks) | No (v1 out of scope) |
| External dependencies | `reflexio` CLI, LLM API key on the server, running server | Openclaw only (agent's own LLM) |
| CLI integration | Shells out to `reflexio search / publish / aggregate` | None — pure in-Openclaw |
| Target user | Teams with many agent instances behind a shared server | Solo user, no infrastructure |

## Pick `openclaw-embedded` if:

- You don't want to run a Reflexio server.
- You only have one agent instance, or memory sharing across instances doesn't matter.
- You prefer fewer moving parts.

## Pick `integrations/openclaw/` (federated) if:

- You have or want to run a Reflexio server.
- You have multiple agent instances and want cross-instance playbook sharing via aggregation.
- You want multi-user support (each human treated as a distinct Reflexio user).

## Can both be installed?

Yes — no conflict. Different hook names, different skill names, different cron jobs, different extraPaths. But installing both is pointless: they serve the same purpose by different means. Pick one.
