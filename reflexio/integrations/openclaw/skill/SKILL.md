---
name: reflexio
description: "Self-improving OpenClaw agents via Reflexio cross-session memory: the agent learns from every correction, tool failure, and stated preference so it stops repeating the same mistakes. Searches past playbooks before each task, captures new learnings after — and cross-instance aggregation turns one agent's correction into shared knowledge for every other agent. The hook is hard-pinned to a local Reflexio server at 127.0.0.1:8081 (no remote endpoints). That local server performs LLM-based extraction using whatever provider you configure in ~/.reflexio/.env, so extracted summaries may reach third-party APIs — see Privacy section before enabling."
---

# Reflexio: Learn from Past Sessions

Reflexio remembers corrections and preferences from past conversations with this user. Use it to retrieve task-specific guidance before working, and to record new learnings as you go.

**This skill triggers on TWO types of messages:**
1. **Task requests** — user asks you to do something → search Reflexio first
2. **Corrections, preferences, and completed steps** → publish to Reflexio

The user can also run `/reflexio-extract` for comprehensive extraction of all session learnings, or `/reflexio-aggregate` to consolidate learnings across all agent instances.

---

## Privacy & Data Collection

**Read this before enabling the skill.** Reflexio causes the agent to automatically capture conversations and forward them to a local Reflexio server for extraction and storage. Treat the following as material privacy information, not incidental detail.

### Credential requirement (not declared in skill metadata — important)

The skill itself declares no required environment variables and the hook reads none. **But the end-to-end system does require an LLM provider API key**, and you will be asked for one during First-Use Setup:

- `reflexio setup openclaw` runs an interactive wizard that prompts you to choose an LLM provider (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, MiniMax, DashScope, xAI, Moonshot, ZAI, or any local provider via LiteLLM) and paste an API key.
- The key is written to `~/.reflexio/.env`. The local Reflexio server reads it when extracting playbooks and profiles from captured conversations.
- **The hook itself never reads this key or sees the contents of `~/.reflexio/.env`.** That's a property of the hook code, which has no filesystem config reads and no environment variable access at all. But the *server* the hook POSTs to does read the key, so from a "what credentials will I end up providing to make this work" perspective, treat the LLM provider key as a required dependency of the skill.
- **If you want fully offline operation, point the wizard at a local LLM** (Ollama at `http://127.0.0.1:11434`, LM Studio, vLLM, etc.) instead of a hosted provider. The wizard accepts any LiteLLM-compatible base URL.

This credential is not declared in the skill's registry metadata because metadata-level `requires.env` only applies to the hook's own runtime reads, and the hook is deliberately stateless. But you should know it's coming before you install.

### Two distinct network hops — know the difference

1. **The hook → the local Reflexio server (always localhost).**
   The hook is hard-pinned to `http://127.0.0.1:8081`. It communicates via native `fetch()` with no configuration knobs; the destination is a hardcoded constant in `handler.js`. It reads no environment variables and no dotfiles. This hop cannot leave your machine.

2. **The local Reflexio server → an LLM provider (may leave your machine).**
   The server uses an LLM provider (configured via `~/.reflexio/.env`) to extract playbooks and profiles from captured conversations. That provider is whichever one you set — OpenAI, Anthropic, Gemini, DeepSeek, etc. If you configured an external provider, **excerpts of your conversations will be sent to that provider** as part of extraction. The primary conversation text is stored locally in SQLite at `~/.reflexio/` and is not sent to the provider directly, but the extracted summaries, trigger texts, and sample content are.

If you want a fully offline setup, point `~/.reflexio/.env` at a local LLM (Ollama, LM Studio, vLLM, etc.) before enabling the hook. If you haven't audited that configuration, assume extraction forwards sensitive content to a third-party API.

**Localhost-only is a property of the hook, not the full system.** Earlier drafts of this documentation framed the integration as fully local. That framing was incomplete — it described the hook's network boundary correctly but ignored that the local Reflexio server then makes its own outbound LLM calls during extraction. The accurate statement is: the *hook* has no off-host destination, but the *server behind it* forwards excerpts to whichever LLM provider you configured.

If you want remote Reflexio (managed or self-hosted) from OpenClaw, this integration is not the one to use — it is deliberately crippled to localhost at the hook level. Use the Claude Code integration instead, which supports remote servers via `REFLEXIO_URL`.

**What is captured (locally)**

- Full message content — every user turn and every assistant turn in the session
- Every tool call, its inputs, and its outputs — including **failed tool calls and the exact error strings**
- Self-corrections written out loud mid-response ("actually, this isn't quite right because…") — these are preserved verbatim because they're the most valuable learning signal, but they also surface internal reasoning
- User profile signals the local extraction pipeline infers from the conversation: expertise, working style, project conventions

None of this is scrubbed for PII, credentials, file paths, stack traces, or API outputs. Anything that appears in the conversation ends up in the local database. If you work on sensitive tasks, disable the hook before starting them, or tell the agent mid-task to stop logging.

**How to disable**

- **Per-session opt-out:** `openclaw hooks disable reflexio-context` — stops automatic capture immediately. Skill-driven search and publish commands still work until the agent stops calling them.
- **Full uninstall:** `reflexio setup openclaw --uninstall` — removes the hook, slash commands, and workspace rule.
- **Search-only mode** (keep retrieval, stop auto-capture): edit `hook/handler.js` to remove the `command:stop` event handler, then reinstall. The `message:received` injection continues working without buffering turns to the database.
- **Wipe stored data:** delete `~/.reflexio/sessions.db` (buffered turns) and `~/.reflexio/` (full local store including extracted playbooks).
- **Sensitive-task-only opt-out:** tell the agent at the start of the task. The workspace rule instructs it to honor the objection — skip search, skip capture, skip local server start — for the rest of the session.

**Transparency expectations**

- On the first turn of a session, the agent should briefly tell you that Reflexio is active and that it captures conversations into a local SQLite database on your machine. One or two sentences.
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
- If search returned playbooks → follow the content guidance in each playbook
- If search returned profiles → adapt your approach to the user's preferences
- If search returned nothing or failed → proceed normally

---

## Step-by-Step: When to Publish

### Scenario 1: User Corrects You

When the user corrects your approach or states a preference:

**Step 1 — Apply the correction** to your work first.

**Step 2 — Wait for enough context.** Don't publish immediately after the first correction message. Continue working until the correction is fully resolved and you have enough context to write a rich summary:
- The original request
- Your initial approach (including any self-corrections you wrote out loud)
- The user's correction (their exact words)
- Your corrected approach and outcome

**Step 3 — Build a JSON summary and publish:**

```bash
cat > /tmp/reflexio-summary.json << 'SUMMARY_EOF'
{
  "user_id": "<your-agent-id>",
  "agent_version": "openclaw-agent",
  "source": "openclaw",
  "interactions": [
    {"role": "user", "content": "<original request>"},
    {
      "role": "assistant",
      "content": "<initial approach — preserve any self-correction text verbatim, e.g. 'this isn't quite right because...'>",
      "tools_used": [
        {"tool_name": "<tool>", "tool_data": {"input": "<params> — FAILED: <exact error>"}}
      ]
    },
    {"role": "user", "content": "<user's correction — preserve their exact words>"},
    {"role": "assistant", "content": "<corrected approach and outcome>"}
  ]
}
SUMMARY_EOF
reflexio publish --agent-version openclaw-agent --source openclaw --skip-aggregation --force-extraction --file /tmp/reflexio-summary.json && rm -f /tmp/reflexio-summary.json
```

`tools_used` is **required** whenever the original approach involved a failed or rejected tool call — the error string is the evidence Reflexio needs to extract a precise behavioral rule. For pure-text corrections, the field can be omitted.

**Detect correction patterns:**

_Verbal corrections:_
- "No, use X instead of Y"
- "Don't do X, always do Y"
- "I prefer X", "Always use X in this project"
- "That's wrong, the correct approach is..."

_Non-verbal / implicit corrections (also publish these):_
- **Tool-call rejection** — user rejected a tool use mid-response. Record it in `tools_used` with `— REJECTED BY USER` and write `[rejected tool use — see tools_used above]` in the following user turn's `content`.
- **Self-correction written out loud** — you realized mid-response you were doing the wrong thing and said so. Preserve the self-correction sentence verbatim in the assistant turn's `content`.
- **Repeated tool failure with user intervention** — you failed the same operation 2+ times and the user redirected. List every failed attempt under `tools_used` on the original assistant turn.

**Key principle:** Wait for sufficient context before publishing. A simple one-line correction ("always use type hints") can be published immediately. A multi-turn correction (user corrects, explains why, adds exceptions) should be published once the full chain is resolved.

### Scenario 2: After Completing a Key Step

After completing a meaningful milestone — a key step, sub-task, or the full task — reflect on what you learned and publish:

- Non-obvious discoveries about this project or environment
- Dead ends and tool quirks encountered
- User preferences revealed through the work
- Patterns that would help future sessions

Don't wait until the entire task is done — publish at natural milestones. Build the same JSON summary format as above and publish with the same command.

---

## Multi-User and Agent Playbooks

Each OpenClaw agent instance is a unique Reflexio user, identified by its `agentId`. This means:

- **User playbooks** — corrections specific to this agent instance's interactions
- **Agent playbooks** — shared corrections aggregated from ALL instances of this agent

`reflexio search` returns both user playbooks (instance-specific) and agent playbooks (shared across all instances) — so every agent instance benefits from the collective learning.

The `user_id` field in publish payloads is auto-derived from OpenClaw's session key (the `agentId` prefix). You don't need to set it manually.

---

## What Reflexio Stores

**User Profiles** — stable facts learned from conversations:
- Expertise, background, role
- Communication style and preferences
- Technology stack and project conventions

**User Playbooks** — per-instance behavioral corrections:
- **trigger**: when does this rule apply?
- **content**: the actionable behavioral rule
- **rationale**: why the correction matters
- **blocking_issue**: what prevents correct behavior (if applicable)

**Agent Playbooks** — shared corrections aggregated from all instances:
- Same structure as user playbooks
- Produced by `reflexio agent-playbooks aggregate`
- Returned alongside user playbooks in every `reflexio search`

The Reflexio server LLM analyzes your published summary and determines what gets extracted — you don't decide the structure.

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
| `reflexio search "<task>"` | Task-specific playbooks | Before every task |
| `reflexio user-profiles search "<query>"` | User preferences | When personalizing |
| `reflexio publish --force-extraction --file ...` | Publish corrections/learnings | After corrections or key steps |
| `reflexio agent-playbooks aggregate` | Consolidate across instances | After corrections, or on schedule |
| `reflexio agent-playbooks list` | View shared playbooks | Debugging, review |
| `reflexio status check` | Check server | First use, or if commands fail |
| `/reflexio-extract` | Comprehensive extraction | High-signal sessions |
| `/reflexio-aggregate` | Manual aggregation | Consolidate learnings |

---

## Tips

- **Use the user's actual request as the search query** — not keywords
- **Preserve the user's exact words** in correction summaries
- **Include evidence** — tool failures, error messages, self-correction sentences. Without evidence, Reflexio extracts vague profile entries; with evidence, it extracts precise playbook rules
- **If Reflexio is unreachable, proceed normally** — it enhances but never blocks
- **Tell the user Reflexio is active at session start** (see Privacy & Data Collection above). Cross-session logging is not something to leave implicit.
- **Honor sensitive-task objections** — if the user says "don't log this," stop all Reflexio calls (search, publish, server start) for the rest of the session
- **Suggest `/reflexio-extract`** if a session had many corrections or learnings
