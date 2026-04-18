# Future Work

Items explicitly deferred from v1. Each has a rationale for why it's not in scope yet.

## Convert to code-plugin (`definePluginEntry` + `registerCommand`)

**Rationale:** V1 uses the legacy "Hook-only / Non-capability" plugin pattern (matches the `self-improving-agent` example). Converting to a code-plugin would gain:
- Real slash commands (`/reflexio-consolidate` instead of `/skill reflexio-consolidate`)
- `api.runtime.config.write` for programmatic config changes (currently we use `exec` + `openclaw config set`)
- Potentially cleaner cron registration (if the plugin SDK exposes it)

**Cost:** TypeScript plugin scaffolding, SDK version pinning, build pipeline.

## Cross-instance playbook sharing

**Rationale:** V1 treats one Openclaw instance as one user. Sharing playbooks across instances would require infrastructure (sync service, shared extraPath on network storage, or federation via Reflexio server). V1 punts this to `integrations/openclaw/` (federated) which already solves it.

**Option for v2:** Lightweight git-based sync — checkpoint `.reflexio/` to a shared git remote; other instances pull periodically.

## Participation in Openclaw's native dreaming

**Rationale:** Openclaw's memory-core plugin has a built-in "dreaming" consolidation system (opt-in, runs daily at 3am). Our full-sweep consolidation is a parallel structure, not a participant. Participating would require a plugin API not yet exposed by memory-core.

**Watch for:** an `api.runtime.memoryCore.dreaming.registerConsolidator()` or similar in future Openclaw releases.

## Ported incremental / expert / should-generate prompt variants

**Rationale:** Reflexio's prompt_bank has `profile_update_instruction_incremental`, `playbook_extraction_main_expert`, and `playbook_should_generate`. V1 skips these as YAGNI:
- Incremental extraction assumes a standing session with bounded batches; our Flow C extracts full transcripts.
- Expert-mode is for advanced scenarios beyond v1.
- `should_generate` is a cost-saving pre-check; extraction returning an empty list achieves the same at small additional cost.

## Removed frontmatter fields

- `source_sessions` — dead-pointer problem; LLM has no reliable way to recover session keys long after the fact.
- `confirmation_kind` — no v1 consumer.
- `confidence` — LLM self-assigned confidence is unreliable; omit rather than hallucinate precision.
- `tags` — agent-assigned, not normalized; semantic search on body content suffices.

Add any of these in v2 only if a concrete consumer materializes.

## Native Windows support

**Rationale:** V1 used shell scripts that assumed Unix. V1.1 moved to registered tools (`reflexio_write_profile`, `reflexio_write_playbook`), which are platform-agnostic. This item is resolved.

## Playbook TTL / expiration

**Rationale:** The 99% case is "playbooks never expire". V1 drops the field; occasional task-specific playbooks encode time bounds in the `## When` text. V2 could add structured playbook expiration if a concrete use case materializes.
