---
paths:
  - "reflexio/**/*.py"
---

# Reflexio Architecture Guardrails

## Reflexio Instance
- **NEVER** instantiate `Reflexio()` directly in API endpoints
- **ALWAYS** use `get_reflexio()` from `server/cache/`

## Storage
- **NEVER** import storage implementations directly
- **ALWAYS** use `request_context.storage` (type: `BaseStorage`)

## LLM
- **NEVER** import `OpenAIClient` or `ClaudeClient` directly
- **ALWAYS** use `LiteLLMClient` (uses LiteLLM for multi-provider support)

## Prompts
- **NEVER** hardcode prompts
- **ALWAYS** use `request_context.prompt_manager.render_prompt(prompt_id, variables)`

## Config
- `tool_can_use` lives at root `Config` level — shared across success evaluation and feedback extraction (NOT per-`AgentSuccessConfig`)

## SQLite storage: lineage events & atomicity (one connection = one transaction)
- The `sqlite3` connection is shared and `autocommit=False`, so `conn.commit()` **anywhere flushes the entire pending transaction**, not just the adjacent statement. The FTS/vec helpers (`_fts_*`, `_vec_*` in `sqlite_storage/_base.py`) **self-commit** internally.
- **NEVER** interleave a self-committing helper between two writes you need atomic. E.g. `emit lineage event → _fts_delete()/_vec_delete() → DELETE row → commit` looks atomic ("one `with self._lock:` block, one commit at the end") but the helper's commit durably writes the audit event **before** the row is deleted — so a crash or a no-op delete leaves a phantom `hard_delete`/`status_change` event for a row that still exists. This bit a whole family of B1 delete/update methods.
- **ALWAYS** emit a lineage event only **after** (or in the same `conn.commit()` as) the mutation it attests to, guarded on `cur.rowcount > 0`; run `_fts_*`/`_vec_*` cleanup **after** that commit (index maintenance, not the audited fact). The existence/eligibility check must use the **same predicate** (including `user_id` scope) as the mutation — otherwise you audit erasures that never happened, including for another user's row in the same org.
