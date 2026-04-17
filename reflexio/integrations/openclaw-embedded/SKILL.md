---
name: reflexio-embedded
description: "Captures user facts and procedural corrections into .reflexio/ so the agent learns across sessions. Use when: (1) user states a preference, fact, config, or constraint; (2) user corrects the agent and confirms the fix with an explicit 'good'/'perfect' or by moving on without re-correcting for 1-2 turns; (3) at start of a user turn, to retrieve relevant facts and playbooks from past sessions."
metadata:
---

# Reflexio Embedded Skill

Captures user facts (profiles) and procedural corrections (playbooks) into `.reflexio/`, so the agent learns across sessions. All memory lives in Openclaw's native primitives — no external service required.

## First-time setup per agent

If `.reflexio/.setup_complete_<agentId>` does NOT exist (where `<agentId>` is your current agent id), perform this one-time check. The setup step runs probing commands via `exec` and asks for approval before making changes.

**Steps:**

1. Probe current config:
   - `openclaw config get plugins.entries.active-memory.config.agents`
   - `openclaw config get agents.defaults.memorySearch.extraPaths`
   - `openclaw memory status --deep`

2. If active-memory is not targeting this agent:
   Ask user: *"To auto-inject relevant facts into each turn, I can enable active-memory for this agent. OK if I run `openclaw config set plugins.entries.active-memory.config.agents '[\"<agentId>\"]' --strict-json`?"*
   On approval, run the command.

3. If `.reflexio/` is not registered as an extraPath:
   Ask user: *"I need to register .reflexio/ as a memory path. OK if I run `openclaw config set agents.defaults.memorySearch.extraPaths '[\".reflexio/\"]' --strict-json`?"*
   On approval, run the command.

4. If no embedding provider is configured (FTS-only mode):
   Tell user: *"Vector search requires an embedding API key (OpenAI, Gemini, Voyage, or Mistral). The plugin works without one but retrieval quality drops. Would you like guidance on adding one?"*
   If yes, guide them through `openclaw config set` or `openclaw configure`.

5. On each decline, note the degraded mode but do not block:
   - No active-memory → you must run `openclaw memory search` via exec at turn start (see "Retrieval" section below).
   - No extraPath → WARN the user the plugin cannot function without this step.
   - No embedding → continue with FTS-only.

6. When all checks resolved (approved or accepted with warning): create the marker:
   ```bash
   mkdir -p .reflexio
   touch .reflexio/.setup_complete_<agentId>
   ```

**If exec is not available** (strict admin policy): fall back to telling the user the exact commands to run manually.

## First-Use Initialisation

Before any write, ensure `.reflexio/` and its subdirectories exist. This is idempotent — safe to run every session:

```bash
mkdir -p .reflexio/profiles .reflexio/playbooks
```

Never overwrite existing files. Never write secrets, tokens, private keys, environment variables, or credentials into `.reflexio/` files. When capturing a fact involves a user-pasted snippet that contains credentials, redact first.

## Quick Reference

| Situation                                                 | Action                                     |
|-----------------------------------------------------------|--------------------------------------------|
| User states preference, fact, config, or constraint       | Write profile via `reflexio-write.sh`      |
| User correction → you adjust → user confirms              | Write playbook via `reflexio-write.sh`     |
| Start of user turn, no Active Memory injection appeared   | Run `openclaw memory search` via exec (see below) |
| Unsure whether to capture                                 | Skip; batch pass at session-end has a second shot |

## Detection Triggers

### Profile signals (write immediately, same turn)

- **Preferences**: "I prefer X", "I like Y", "I don't like Z", "I always do Q"
- **Facts about self**: "I'm a [role]", "my timezone is X", "I've been doing Y for Z years"
- **Config**: "use X", "our team uses Y", "the repo is at Z"
- **Constraints**: "I'm vegetarian", "no dairy", "I can't X", "don't use Y"

For each such signal, invoke `reflexio-write.sh` with a kebab-case topic slug and an appropriate TTL. See "TTL Selection" below.

### Playbook signals (write AFTER confirmation)

Playbooks require a specific multi-turn pattern:

1. **Correction**: *"No, that's wrong"*, *"Actually..."*, *"Don't do X"*, *"Not like that"*, *"We don't use X here"*.
2. **You adjust**: you redo the work per the correction.
3. **Confirmation** (required — without this, do NOT write a playbook):
   - Explicit: *"good"*, *"perfect"*, *"yes that's right"*, *"correct"*.
   - Implicit: the user moves to an unrelated topic without re-correcting for 1-2 more turns.

**Explicit don't-write rule**: if you see a correction without subsequent confirmation, do not write a playbook. The fix may be wrong; let the batch pass at session end re-evaluate.

## Retrieval

### When Active Memory is enabled

Your turn context may already contain Reflexio-prefixed entries injected by Active Memory. Incorporate them before responding. No tool call needed.

### Fallback when Active Memory is absent

At the start of each user turn, preprocess the user's message (see **Query Preprocessing** below) then search via exec:

```bash
openclaw memory search "<preprocessed query from user's message>" --json --max-results 5
```

The result is a JSON object with a `results` array. Each entry has `path`, `score`, and `snippet` fields. Incorporate any `.reflexio/`-sourced results before responding. Skip if the user's message is trivial (greeting, acknowledgment).

**Important:** Do NOT use the `memory_search` tool — it returns memory engine config, not search results. Always use `openclaw memory search` via exec.

## File Format

**Do NOT construct filenames or frontmatter by hand.** Use `./scripts/reflexio-write.sh` (via the `exec` tool). The script generates IDs, enforces the frontmatter schema, and writes atomically.

### Profile template (for mental model — the script emits this)

```markdown
---
type: profile
id: prof_<nanoid>
created: <ISO timestamp>
ttl: <enum>
expires: <ISO date or "never">
supersedes: [<old_id>]   # optional, only after a merge
---

<1-3 sentences, one fact per file>
```

### Playbook template

```markdown
---
type: playbook
id: pbk_<nanoid>
created: <ISO timestamp>
supersedes: [<old_id>]   # optional
---

## When
<1-sentence trigger — this is the search anchor; make it a noun phrase>

## What
<2-3 sentences of the procedural rule; DO / DON'T as actually observed>

## Why
<rationale, can be longer — reference only, not recall content>
```

### How to invoke `reflexio-write.sh`

**Profile:**

```bash
echo "User is vegetarian — no meat or fish." | \
  ./scripts/reflexio-write.sh profile diet-vegetarian one_year
```

**Playbook:**

```bash
./scripts/reflexio-write.sh playbook commit-no-ai-attribution --body "$(cat <<'EOF'
## When
Composing a git commit message on this project.

## What
Write conventional, scope-prefixed messages. Do not add AI-attribution trailers.

## Why
On <date> the user corrected commits that included Co-Authored-By trailers. Project's git-conventions rule prohibits them. Correction stuck across subsequent commits.
EOF
)"
```

## TTL Selection (profiles only)

- `infinity` — durable, non-perishable facts (diet, name, permanent preferences)
- `one_year` — stable but could plausibly change (address, role, team)
- `one_quarter` — current focus (active project, sprint theme)
- `one_month` — short-term context
- `one_week` / `one_day` — transient (today's agenda, this week's priorities)

Pick the most generous TTL that still reflects reality. When in doubt, prefer `infinity` — let dedup handle later contradictions via supersession.

## Query Preprocessing

Before calling `openclaw memory search`, rewrite the raw text into a clean search query. Raw user messages are often too conversational for embedding similarity, and too noisy for FTS keyword matching.

**Rewrite instruction (apply mentally — no extra tool call):**

> Rewrite into a single, descriptive sentence that captures the core fact or topic. Expand with 2-3 important synonyms or related technical terms to improve matching. Remove conversational filler (apologies, hedging, corrections, "by the way"). Return ONLY the rewritten text.

**Examples:**

| Raw text | Rewritten search query |
|---|---|
| "Oh, sorry I typed it wrong, I do like apple juice" | `"User preference for apple juice. Related: fruit juice, beverage, drink preference"` |
| "Actually I'm not vegetarian anymore, I eat everything" | `"Dietary preference update, no longer vegetarian. Related: omnivore, diet change, food restrictions"` |
| "By the way my timezone is PST" | `"User timezone Pacific Standard Time. Related: time zone, PST, America/Los_Angeles"` |
| "No wait, don't use pnpm, we use yarn on this project" | `"Package manager preference yarn over pnpm. Related: node package manager, dependency tool, npm alternative"` |
| "I changed my mind — I prefer dark mode now" | `"User display preference dark mode. Related: theme, appearance, light mode, UI preference"` |

This produces queries that work well for both vector similarity (descriptive sentence captures semantic intent) and BM25 keyword matching (synonym expansion hits related terms).

## Shallow Dedup (in-session writes only)

Before writing a profile or playbook, check whether a similar or contradictory one already exists:

1. Preprocess the query (see **Query Preprocessing** above), then search via exec:
   ```bash
   openclaw memory search "<preprocessed search query>" --json --max-results 5
   ```
2. If no results or `results[0].score < 0.4`: write normally, no dedup needed.
3. If `results[0].score >= 0.4`: a near-duplicate or contradiction may exist. Decide:

### Contradiction (user changed their mind)

If the user's new statement **directly contradicts** an existing file (e.g., "I'm NOT vegetarian anymore" vs an existing "User is vegetarian" profile), this is a **supersession**. Always handle it immediately — don't defer to batch.

**Steps:**
1. Note the existing file's `id` and `path` from the search result's `snippet` (contains frontmatter with `id:`) and `path` field.
2. Write the new file with `--supersedes`:
   ```bash
   echo "User is not vegetarian. Likes beef, tuna, and shrimp." | \
     ./scripts/reflexio-write.sh profile diet-not-vegetarian infinity \
       --supersedes "prof_3ecg"
   ```
3. Delete the old file:
   ```bash
   rm .reflexio/profiles/diet-vegetarian-3ecg.md
   ```

The `--supersedes` flag records the lineage in the new file's frontmatter. The `rm` removes the contradicted file so retrieval never returns stale facts.

### Near-duplicate (same fact, minor rewording)

If the existing file covers the same fact with minor wording differences (e.g., "User prefers dark mode" vs "User likes dark mode"), **skip the write**. The existing file is sufficient.

### Genuinely distinct (related topic, different facts)

If the existing file covers a related but different fact (e.g., existing: "User is vegetarian" vs new: "User's favorite cuisine is Italian"), **write normally** without supersedes. They're complementary, not contradictory.

### When in doubt

If you're unsure whether something is a contradiction, near-duplicate, or distinct: **write the new file without supersedes and without deleting the old**. The daily consolidation cron will cluster and merge them. Err on the side of preserving information.

## Safety

- **Never write secrets.** No API keys, tokens, access tokens, private keys, environment variables, OAuth secrets, auth headers. If the user's message contains any of these, redact them before writing.
- **Redact pasted code.** User-pasted snippets often contain credentials. Strip them first.
- **PII.** Do not capture PII beyond what's operationally useful (name, timezone, role are fine; government IDs, addresses, phone numbers only if explicitly relevant).

## Best Practices

1. **Write immediately** on a clear signal. Don't queue to session-end — that's Flow C's job; you have a different role.
2. **One fact per profile file.** Multi-fact files are harder to dedupe and easier to contradict.
3. **Trigger phrase = search anchor.** Write `## When` as a noun phrase describing the situation, not a sentence. Retrieval hits on semantic similarity to this field.
4. **Skip writing when uncertain.** Flow C has a second pass over the full transcript. It's better to let it handle ambiguous cases.
5. **Prefer shorter TTL for transient facts.** Don't let "working on project X" accumulate as infinity-TTL cruft.

## Opt-in Hook

This skill works standalone — your in-session Flow A (profile) and Flow B (playbook) writes populate `.reflexio/` without any hook.

The optional hook (`hook/` directory of this plugin) adds two capabilities:

1. **TTL sweep at session start**: deletes expired profiles before Active Memory runs.
2. **Session-end batch extraction (Flow C)**: on `session:compact:before`, `command:stop`, or `command:reset`, spawns a `reflexio-extractor` sub-agent that extracts profiles/playbooks from the full transcript and runs shallow pairwise dedup.

See this plugin's `README.md` for install instructions (runs via `./scripts/install.sh`). If the hook is not installed, Flows A+B still work.
