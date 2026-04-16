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
   - No active-memory → you must run `memory_search` explicitly at turn start (see "Retrieval" section below).
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
| Start of user turn, no Active Memory injection appeared   | Run `memory_search` fallback (see below)   |
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

At the start of each user turn, call:

```
memory_search(query=<user's current message>, filter={type: profile|playbook})
```

Incorporate any `.reflexio/`-sourced results before responding. Skip if the user's message is trivial (greeting, acknowledgment).
