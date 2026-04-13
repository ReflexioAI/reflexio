---
name: reflexio
description: "Search Reflexio for task-specific playbooks before working, AND publish corrections, preferences, and learnings to Reflexio. Retrieves behavioral rules so you follow them from the start. Publishes when users correct you, state preferences, or you complete key steps with learnings. Use on ANY task."
---

# Reflexio: Learn from Past Sessions

Reflexio remembers corrections and preferences from past conversations with this user. Use it to retrieve task-specific guidance before working, and to record new learnings as you go.

**This skill triggers on TWO types of messages:**
1. **Task requests** — user asks you to do something → search Reflexio first
2. **Corrections, preferences, and completed steps** → publish to Reflexio

The user can also run `/reflexio-extract` for comprehensive extraction of all session learnings, or `/reflexio-aggregate` to consolidate learnings across all agent instances.

---

## Step-by-Step: When User Gives a Task

Follow these steps IN ORDER:

**Step 1 — Ensure server is running (local server only):**
Check the `REFLEXIO_URL` environment variable. If it points to a remote server (anything other than `localhost` or `127.0.0.1`), **skip this step entirely** — managed Reflexio servers are always running. Go directly to Step 2.

If `REFLEXIO_URL` is unset or points to localhost/127.0.0.1:
```bash
reflexio status check
```
If this fails with a connection error, start the server in the background:
```bash
nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
sleep 5 && reflexio status check
```
Then continue to Step 2 immediately.

**Step 2 — Search for relevant corrections:**
```bash
reflexio search "<the user's request or task description>" --user-id $REFLEXIO_USER_ID
```
Use the user's actual request as the query — not keywords. Different tasks return different playbooks.

**Step 3 — Apply results and do the task:**
- If search returned playbooks → follow the instructions, avoid the pitfalls
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
reflexio publish --user-id $REFLEXIO_USER_ID --agent-version openclaw-agent --source openclaw --skip-aggregation --force-extraction --file /tmp/reflexio-summary.json && rm -f /tmp/reflexio-summary.json
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

The `user_id` field in publish commands is auto-derived from OpenClaw's `agentId`. Use `$REFLEXIO_USER_ID` or the agentId directly.

---

## What Reflexio Stores

**User Profiles** — stable facts learned from conversations:
- Expertise, background, role
- Communication style and preferences
- Technology stack and project conventions

**User Playbooks** — per-instance behavioral corrections:
- **trigger**: when does this rule apply?
- **instruction**: what to do instead
- **pitfall**: what to avoid
- **rationale**: why the correction matters

**Agent Playbooks** — shared corrections aggregated from all instances:
- Same structure as user playbooks
- Produced by `reflexio agent-playbooks aggregate`
- Returned alongside user playbooks in every `reflexio search`

The Reflexio server LLM analyzes your published summary and determines what gets extracted — you don't decide the structure.

---

## Server Management

**Managed Reflexio (remote server):** If `REFLEXIO_URL` points to a non-localhost URL, the server is always running. Do NOT attempt to start it. If a command fails with a connection error, report it to the user — it's a network or configuration issue.

**Local server** (`REFLEXIO_URL` unset or pointing to localhost/127.0.0.1):
- **Check:** `reflexio status check`
- **Start (background):** `nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &`
- Never ask the user to start the server — handle it silently
- If `reflexio` is not found, ask the user to install it: `pip install reflexio-ai`

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
- **Don't mention Reflexio to the user** unless they ask
- **Suggest `/reflexio-extract`** if a session had many corrections or learnings
