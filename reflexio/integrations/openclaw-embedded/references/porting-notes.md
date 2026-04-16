# Porting Notes: Reflexio prompt_bank → openclaw-embedded prompts

Track every deviation from source prompts so maintainers can re-apply adaptations when upstream versions bump.

## profile_extraction.md

**Source:** `open_source/reflexio/reflexio/server/prompt/prompt_bank/profile_update_instruction_start/v1.0.0.prompt.md`

**Initial port:** 2026-04-16

**Changes from source:**

- **Output format**: `StructuredProfilesOutput` JSON with `profiles: list[ProfileAddItem{content, ttl, metadata}]` → array of `{topic_kebab, content, ttl}` objects that drive `./scripts/reflexio-write.sh profile <topic_kebab> <ttl>` invocations per item.
- **Dropped fields**: `custom_features` dict, `metadata` field. Our frontmatter doesn't carry these.
- **Added guidance**: slug generation rules — kebab-case, ≤48 chars, `[a-z0-9][a-z0-9-]*`.
- **Kept verbatim**: TTL enum semantics, "do NOT re-extract existing profiles" constraint, extraction criteria (what counts as a profile signal).
- **Variable substitution**: `existing_profiles` in source was populated by Reflexio server from SQL. In our port, the Flow C sub-agent runs `memory_search(top_k=10, filter={type: profile})` and injects results into the `{existing_profiles_context}` slot.

**On upstream upgrade (e.g., to v2.0.0):** diff the source file, re-apply the output-format, dropped-fields, added-guidance deltas.

## playbook_extraction.md

**Source:** `open_source/reflexio/reflexio/server/prompt/prompt_bank/playbook_extraction_context/v2.0.0.prompt.md`

**Initial port:** 2026-04-16

**Changes from source:**

- **Output schema**: 6-field (`trigger`, `instruction`, `pitfall`, `rationale`, `blocking_issue`, `content`) → 3-field (`When`, `What`, `Why`).
  - **Rationale (preserved in design spec section 9):** forcing DO+DON'T symmetry (instruction + pitfall) often leads the LLM to hallucinate a symmetric "don't" when only a "do" was actually observed. Collapsing to `## What` lets the content carry whichever was observed.
- **Autoregressive ordering adapted**: source generates rationale first, then structured fields, then content. Our port generates `Why → What → When` internally but emits `When → What → Why` in the document.
- **Dropped**: expert-mode branches (separate `playbook_extraction_context_expert/` source); `existing_feedbacks` variable (upstream v2 already removed it).
- **Strengthened**: explicit "no confirmation → no playbook" gate. Our skill and prompt both enforce that a correction without a following confirmation is NOT grounds for writing a playbook.
- **Variable substitution**: `{transcript}` is passed by the Flow C sub-agent; no other runtime variables.

**On upstream upgrade:** diff the source file, re-apply field collapse, ordering adaptation, expert-branch drop, confirmation gate.

## shallow_dedup_pairwise.md

**Source:** New file; informed by `profile_deduplication/*.prompt.md` and `playbook_deduplication/*.prompt.md`.

- Simplified to strictly pairwise (candidate vs. top-1 neighbor). Reflexio's native dedup is N-way group-based.
- Output schema: `{decision: enum, merged_content?, merged_slug?, rationale}`.
- Our full-sweep cron handles N-way; shallow stays in the hot path.

## full_consolidation.md

**Source:** New file; informed by `playbook_aggregation/v1.0.0.prompt.md`.

- Adapted for single-instance n-way clustering; Reflexio's aggregation is cross-instance.
- Output schema: `{action: enum, merged_content?, merged_slug?, ids_merged_in, ids_kept_separate, rationale}`.
- Cluster size capped at 10.
