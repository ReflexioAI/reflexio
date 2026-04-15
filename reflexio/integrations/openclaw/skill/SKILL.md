---
name: reflexio
description: "Self-improving OpenClaw agents via Reflexio cross-session memory: the agent learns from every correction, tool failure, and stated preference so it stops repeating the same mistakes. Searches past playbooks before each task, and — when invoked — extracts new learnings from the current conversation and upserts them directly into Reflexio via CRUD. The hook is hard-pinned to a local Reflexio server at 127.0.0.1:8081 (no remote endpoints). No LLM provider API key is required: extraction runs in your own agent session, not on the server."
---

# Reflexio: Learn from Past Sessions

Reflexio remembers corrections and preferences from past conversations with this user. Use it to retrieve task-specific guidance before working, and to record new learnings as you go.

**This skill triggers on TWO types of messages:**
1. **Task requests** — user asks you to do something → search Reflexio first
2. **Corrections, preferences, and completed steps** → run `/reflexio-extract` to persist the learning

The `/reflexio-extract` slash command performs extraction in your own context (using the v3.0.0 extraction rubric) and writes playbooks to Reflexio via direct CRUD. Nothing about this integration requires an LLM provider API key on the Reflexio server.

---

## Privacy & Data Collection

**Read this before enabling the skill.** Reflexio reads from and writes to a local Reflexio server on your machine. Treat the following as material privacy information, not incidental detail.

### Single network hop — localhost only

The hook is hard-pinned to `http://127.0.0.1:8081`. It communicates via native `fetch()` with no configuration knobs; the destination is a hardcoded constant in `handler.js`. It reads no environment variables and no dotfiles. This hop cannot leave your machine.

The `/reflexio-extract` slash command, when invoked, sends playbook CRUD calls (search / add / update) to the same local server. Those calls carry the trigger / instruction / pitfall / content fields that you (the agent) extracted from the current conversation — not the raw conversation transcript.

**The Reflexio server does not make outbound LLM calls for this integration.** Playbook extraction runs in your own agent session, which is why you don't need to configure an LLM provider API key in `~/.reflexio/.env` to make this integration work. The server just does CRUD + semantic search against its local storage.

### No automatic capture

The hook does not buffer conversation turns, write to SQLite, or POST to `/api/publish_interaction`. It only reads from Reflexio to inject past-session context. Persisting new learnings is an **explicit** action — you (the agent) run `/reflexio-extract` when there's something worth saving.

This split is important for consent: the user sees `/reflexio-extract` happen in their terminal. There is no silent session-end upload.

### What gets written to the local Reflexio store

Only the fields you produce during `/reflexio-extract`:
- `content` — a concise natural-language summary of one learning
- `trigger` — when the rule applies
- `instruction` / `pitfall` / `rationale` — structured fields of the rule

Raw conversation transcripts, tool outputs, and file paths do NOT end up in the local store unless you quote them verbatim into one of the fields above. If you work on sensitive tasks, omit sensitive strings from the extraction, or skip running `/reflexio-extract` entirely for that session.

### How to disable

- **Per-session opt-out:** `openclaw hooks disable reflexio-context` — stops context injection immediately. Manual `reflexio` CLI calls still work.
- **Full uninstall:** `reflexio setup openclaw --uninstall` — removes the hook, slash commands, and workspace rule.
- **Wipe stored data:** delete `~/.reflexio/` (the local store, including all extracted playbooks and profiles).
- **Sensitive-task-only opt-out:** tell the agent at the start of the task. The workspace rule instructs it to honor the objection — skip search, skip extract, skip local server start — for the rest of the session.

### Transparency expectations

- On the first turn of a session, the agent should briefly tell you that Reflexio is active — it retrieves past-session memory from a local server, and writes new learnings only when you run `/reflexio-extract`. One or two sentences.
- If the agent needs to start the local Reflexio server in the background, it should announce that before launching the process.
- If you see a `REFLEXIO_CONTEXT.md` block in the agent's context, that's injected past-session memory driving the response. You can ask the agent to ignore it.

These expectations are enforced by the workspace rule at `~/.openclaw/workspace/reflexio.md`. If a deployment wants stricter silence or stricter disclosure, edit that file.

---

## Step-by-Step: When User Gives a Task

Follow these steps IN ORDER:

**Step 1 — Ensure the local Reflexio server is running:**
This integration always talks to the local Reflexio server at `127.0.0.1:8081`. Check that it's running:
```bash
reflexio status check
```
If this fails with a connection error, tell the user you're starting the local Reflexio server in the background, then run:
```bash
nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
sleep 5 && reflexio status check
```
Then continue to Step 2 immediately.

**Step 2 — Search for relevant corrections:**
```bash
reflexio search "<the user's request or task description>"
```
Use the user's actual request as the query — not keywords. Different tasks return different playbooks. The server auto-scopes results to the current agent via OpenClaw's session key.

**Step 3 — Apply results and do the task:**
- If search returned playbooks → follow the instructions, avoid the pitfalls
- If search returned profiles → adapt your approach to the user's preferences
- If search returned nothing or failed → proceed normally

---

## Step-by-Step: When to Persist a Learning

### Scenario 1: User Corrects You

When the user corrects your approach or states a preference:

**Step 1 — Apply the correction** to your work first.

**Step 2 — Wait for enough context.** Don't run `/reflexio-extract` immediately after the first correction message. Continue working until the correction is fully resolved and you have the full arc:
- The original request
- Your initial approach (including any self-corrections you wrote out loud)
- The user's correction (their exact words)
- Your corrected approach and outcome

**Step 3 — Run `/reflexio-extract`.** The command applies the v3.0.0 extraction rubric in your own context, produces one or more playbook entries, and for each one runs `reflexio user-playbooks search` first — if a similar entry already exists it is updated (merging new evidence into existing content); otherwise a new entry is added via `reflexio user-playbooks add`. Nothing about this requires a Reflexio-server LLM.

**Detect correction patterns:**

_Verbal corrections:_
- "No, use X instead of Y"
- "Don't do X, always do Y"
- "I prefer X", "Always use X in this project"
- "That's wrong, the correct approach is..."

_Non-verbal / implicit corrections (also persist these):_
- **Tool-call rejection** — user rejected a tool use mid-response.
- **Self-correction written out loud** — you realized mid-response you were doing the wrong thing and said so. Preserve the self-correction sentence verbatim when extracting.
- **Repeated tool failure with user intervention** — you failed the same operation 2+ times and the user redirected.

**Key principle:** wait for sufficient context before extracting. A simple one-line correction ("always use type hints") can be extracted immediately. A multi-turn correction (user corrects, explains why, adds exceptions) should be extracted once the full chain is resolved.

### Scenario 2: After Completing a Key Step

After completing a meaningful milestone — a key step, sub-task, or the full task — reflect on what you learned and run `/reflexio-extract`. Good signals to persist:

- Non-obvious discoveries about this project or environment
- Dead ends and tool quirks encountered
- Successful recipes worth replaying — specific formulas, tool sequences, parameter values, computed answers
- User preferences revealed through the work

Don't wait until the entire task is done — extract at natural milestones.

---

## Multi-User Architecture

Each OpenClaw agent instance is a unique Reflexio user, identified by its `agentId`. This means:

- **User playbooks** — corrections and recipes specific to this agent instance's history
- **`user_id`** is auto-derived from OpenClaw's session key (the `agent:<id>:...` prefix). You don't need to set it manually.

`reflexio search` returns both user playbooks (instance-specific) and any agent playbooks (shared across instances) that exist in the store. This integration does not produce new agent playbooks — cross-instance aggregation is a server-side LLM operation that was intentionally dropped to keep the integration LLM-free. Teams that want cross-instance playbook sharing can use managed Reflexio or the Claude Code integration instead.

---

## What Reflexio Stores

**User Profiles** — stable facts about the user:
- Expertise, background, role
- Communication style and preferences
- Technology stack and project conventions

Profiles are populated by whatever tooling produced them previously; this integration reads them via search but does not write new ones.

**User Playbooks** — per-instance behavioral rules and recipes:
- **trigger**: when does this rule apply?
- **instruction**: what to do (< 20 words for Correction SOPs, up to 80 for Success Path Recipes)
- **pitfall**: what to avoid
- **rationale**: why the correction matters
- **content**: concise standalone insight (SOP) or actionable recipe (Recipe) with concrete values

Written by `/reflexio-extract` via direct CRUD.

---

## Server Management

This integration always talks to the local Reflexio server at `http://127.0.0.1:8081`. There is no remote-server mode — the hook is hard-pinned to loopback at the code level.

- **Check:** `reflexio status check`
- **Start (background):** `nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &`
- **Before starting it, tell the user briefly.** One sentence is enough: "Starting the local Reflexio server in the background so I can fetch your past-session memory." Do not launch processes on the user's machine without telling them first.
- If the user objects, skip the server start and proceed without Reflexio for this session.
- If `reflexio` is not found, ask the user to install it: `pipx install reflexio-ai` (or `pip install --user reflexio-ai`)

---

## Command Reference

| Command | Purpose | When |
|---------|---------|------|
| `reflexio search "<task>"` | Task-specific playbooks + profiles | Before every task |
| `reflexio user-playbooks search "<query>" --agent-version openclaw-agent` | Find existing playbook before writing | Inside `/reflexio-extract` |
| `reflexio user-playbooks add --agent-version openclaw-agent ...` | Add a new playbook | Inside `/reflexio-extract` when no match |
| `reflexio user-playbooks update --playbook-id <id> --content ...` | Merge new evidence into an existing playbook | Inside `/reflexio-extract` on a match |
| `reflexio user-playbooks list --agent-version openclaw-agent` | Review stored playbooks | Debugging, verification |
| `reflexio status check` | Check server | First use, or if commands fail |
| `/reflexio-extract` | Apply v3.0.0 rubric + upsert playbooks | After corrections or key steps |

---

## Tips

- **Use the user's actual request as the search query** — not keywords
- **Preserve the user's exact words** in extracted content
- **Include evidence** — tool failures, error messages, self-correction sentences. Without evidence, extracted rules are vague; with evidence, they are precise
- **Search before you add.** `/reflexio-extract` does this for you, but if you're running `reflexio user-playbooks add` directly for some reason, still run `search` first to avoid duplicates
- **If Reflexio is unreachable, proceed normally** — it enhances but never blocks
- **Tell the user Reflexio is active at session start** (see Privacy & Data Collection above). Cross-session memory is not something to leave implicit.
- **Honor sensitive-task objections** — if the user says "don't log this," stop all Reflexio calls (search, extract, server start) for the rest of the session
- **Suggest `/reflexio-extract`** if a session had many corrections or a notable successful recipe
