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
| User states preference, fact, config, or constraint       | `reflexio.ts write-profile`                |
| User correction → you adjust → user confirms              | `reflexio.ts write-playbook`               |
| Start of user turn, need context                          | `reflexio.ts search`                       |
| Unsure whether to capture                                 | Skip; batch pass at session-end has a second shot |

## Detection Triggers

### Profile signals (write immediately, same turn)

- **Preferences**: "I prefer X", "I like Y", "I don't like Z", "I always do Q"
- **Facts about self**: "I'm a [role]", "my timezone is X", "I've been doing Y for Z years"
- **Config**: "use X", "our team uses Y", "the repo is at Z"
- **Constraints**: "I'm vegetarian", "no dairy", "I can't X", "don't use Y"

For each such signal, call `reflexio.ts write-profile` with a kebab-case topic slug and an appropriate TTL. See "TTL Selection" below.

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

At the start of each user turn, run via exec:

```bash
npx tsx ./scripts/reflexio.ts search --query "<user's message>"
```

The script handles query preprocessing and memory search internally. Incorporate any results into your response. Skip if the user's message is trivial (greeting, acknowledgment).

**Important:** Do NOT use the `memory_search` tool — it returns memory engine config, not search results.

## File Format

**Do NOT construct filenames or frontmatter by hand.** Use `reflexio.ts` (via the `exec` tool). The script generates IDs, enforces the frontmatter schema, and writes atomically.

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

### How to invoke

**Profile:**

```bash
npx tsx ./scripts/reflexio.ts write-profile \
  --slug diet-vegan --ttl infinity \
  --body "User is vegan. No meat, no fish, no dairy, no eggs."
```

**Playbook:**

```bash
npx tsx ./scripts/reflexio.ts write-playbook \
  --slug commit-no-ai-attribution \
  --body "## When
Composing a git commit message on this project.

## What
Write conventional, scope-prefixed messages. Do not add AI-attribution trailers.

## Why
On <date> the user corrected commits that included Co-Authored-By trailers."
```

**Retrieve context:**

```bash
npx tsx ./scripts/reflexio.ts search --query "user's question here"
```

The script handles everything automatically: query preprocessing for better search results, memory search, contradiction detection against existing entries, atomic file creation, and old-file cleanup on supersession. You only need to detect the signal, compose the content, and call the command.

## TTL Selection (profiles only)

- `infinity` — durable, non-perishable facts (diet, name, permanent preferences)
- `one_year` — stable but could plausibly change (address, role, team)
- `one_quarter` — current focus (active project, sprint theme)
- `one_month` — short-term context
- `one_week` / `one_day` — transient (today's agenda, this week's priorities)

Pick the most generous TTL that still reflects reality. When in doubt, prefer `infinity` — let dedup handle later contradictions via supersession.

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
