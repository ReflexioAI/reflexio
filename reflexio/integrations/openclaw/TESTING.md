# Manual Testing Guide — Reflexio × OpenClaw Integration

Step-by-step guide for manually testing the integration end-to-end. Each phase builds on the previous one.

## Prerequisites

- OpenClaw installed and running (`openclaw --version`)
- Reflexio CLI installed (`pip install reflexio` or `uv pip install reflexio`)
- An LLM API key (e.g., `OPENAI_API_KEY`) for local server mode

## Phase 1: Install & Verify

### 1.1 Run the setup wizard

```bash
reflexio setup openclaw
```

Follow the prompts to configure LLM provider and storage backend.

### 1.2 Verify all components are installed

```bash
# Hook registered?
openclaw hooks list
# Should show: ✓ ready │ 🧠 reflexio-context

# Skill installed?
ls ~/.openclaw/skills/reflexio/SKILL.md

# Rule installed?
ls ~/.openclaw/workspace/reflexio.md

# Commands installed?
ls ~/.openclaw/skills/reflexio-extract/SKILL.md
ls ~/.openclaw/skills/reflexio-aggregate/SKILL.md
```

### 1.3 Verify Reflexio server

```bash
reflexio status check
```

If using local server and it's not running:

```bash
reflexio services start --only backend &
sleep 5
reflexio status check
```

---

## Phase 2: Search (Cold Start)

On a fresh install, there are no playbooks yet. Verify the search hook doesn't break anything.

### 2.1 Start a conversation with an OpenClaw agent

Send a task message:

```
Help me write a Python function to parse CSV files
```

**What to check:**
- The agent responds normally (no errors, no mention of Reflexio)
- In the agent logs or stderr, you should see: `[reflexio] Search failed: ...` or search results (even if empty)
- The hook should NOT block the response — if the server is slow, the 5-second timeout kicks in

### 2.2 Verify the agent doesn't mention Reflexio

The rule file says "never mention Reflexio to the user." Confirm the agent doesn't say anything about Reflexio, playbooks, or search results.

---

## Phase 3: Capture & Publish

### 3.1 Create a correction scenario

Send a task, then correct the agent:

```
User: Set up a new Express server with TypeScript
Agent: [starts setting up with npm]
User: No, use pnpm instead of npm. We always use pnpm in this project.
Agent: [switches to pnpm]
```

**What to check:**
- The skill should detect the correction and publish it (look for `reflexio publish` in logs)
- The agent should apply the correction immediately

### 3.2 Complete the task and trigger step-completion publish

Continue the conversation until the agent completes a key step:

```
User: Now add a health check endpoint at /health
Agent: [implements the endpoint]
```

**What to check:**
- After completing the step, the skill may publish learnings (e.g., project conventions discovered)

### 3.3 End the session

Close the OpenClaw session (Ctrl+C, `/stop`, or however your agent handles it).

**What to check:**
- The hook's `command:stop` handler flushes remaining turns to Reflexio
- In stderr/logs: `[reflexio] Queued N interactions for publish`

### 3.4 Verify playbooks were extracted

Wait ~30 seconds for server-side extraction, then:

```bash
reflexio user-playbooks list --limit 10
```

You should see at least one playbook related to "pnpm" with content like "use pnpm instead of npm."

```bash
reflexio user-profiles list --limit 10
```

You may see a profile entry about project conventions.

---

## Phase 4: Retrieval (Warm Start)

### 4.1 Start a new session and trigger a relevant task

```
User: Install lodash as a dependency
```

**What to check:**
- The `message:received` hook runs `reflexio search "Install lodash as a dependency"`
- The agent should receive the "use pnpm" playbook as injected context
- The agent should use `pnpm add lodash` (not `npm install lodash`) **without being told again**

### 4.2 Try an unrelated task

```
User: Write unit tests for the health check endpoint
```

**What to check:**
- Search returns different (or no) playbooks — the pnpm correction is not relevant to testing
- The agent works normally

---

## Phase 5: Manual Commands

### 5.1 Test `/reflexio-extract`

In a conversation with corrections or learnings, run:

```
/reflexio-extract
```

**What to check:**
- The agent reviews the conversation and builds a JSON summary
- It publishes via `reflexio publish --force-extraction`
- It reports what was published
- You can verify with `reflexio user-playbooks list`

### 5.2 Test `/reflexio-aggregate`

After accumulating some playbooks:

```
/reflexio-aggregate
```

**What to check:**
- The agent runs `reflexio agent-playbooks aggregate --wait`
- It reports how many playbooks were created/updated
- You can verify with `reflexio agent-playbooks list`

---

## Phase 6: Multi-User (Multiple Agent Instances)

This phase tests that different OpenClaw agents get isolated user playbooks but share agent playbooks.

### 6.1 Run two different agents

If you have multiple OpenClaw agents configured (e.g., `main` and `work`):

**Agent "main":**
```
User: Format this code
Agent: [uses prettier]
User: Use biome, not prettier
```

**Agent "work":**
```
User: Set up the database
Agent: [creates PostgreSQL]
User: Use SQLite for development
```

### 6.2 Verify user playbook isolation

```bash
# Check what each agent sees
reflexio user-playbooks list --user-id main
reflexio user-playbooks list --user-id work
```

- "main" should have the biome playbook, NOT the SQLite one
- "work" should have the SQLite playbook, NOT the biome one

### 6.3 Aggregate and verify shared playbooks

```bash
reflexio agent-playbooks aggregate --agent-version openclaw-agent --wait
reflexio agent-playbooks list --agent-version openclaw-agent
```

- Agent playbooks should contain both corrections (biome + SQLite)
- Both agents should see these shared playbooks via `reflexio search`

---

## Phase 7: Graceful Degradation

### 7.1 Stop the server and verify agent still works

```bash
reflexio services stop
```

Start a new OpenClaw session and send a task:

```
User: Write a sorting algorithm
```

**What to check:**
- The agent responds normally (no crashes, no errors visible to user)
- In stderr: `[reflexio] Search failed: connect ECONNREFUSED` (or similar)
- The hook buffers turns to local SQLite — they'll be published when the server is back

### 7.2 Restart server and verify buffered turns are retried

```bash
reflexio services start --only backend &
```

Start a new session (triggers `agent:bootstrap`):

**What to check:**
- In stderr: `[reflexio] Retrying N unpublished session(s)` — the bootstrap handler retries buffered turns from Phase 7.1

---

## Phase 8: Uninstall

### 8.1 Uninstall the integration

```bash
reflexio setup openclaw --uninstall
```

### 8.2 Verify all components are removed

```bash
openclaw hooks list
# Should NOT show reflexio-context

ls ~/.openclaw/skills/reflexio 2>/dev/null && echo "STILL EXISTS" || echo "Removed"
ls ~/.openclaw/skills/reflexio-extract 2>/dev/null && echo "STILL EXISTS" || echo "Removed"
ls ~/.openclaw/skills/reflexio-aggregate 2>/dev/null && echo "STILL EXISTS" || echo "Removed"
ls ~/.openclaw/workspace/reflexio.md 2>/dev/null && echo "STILL EXISTS" || echo "Removed"
```

All should print "Removed."

### 8.3 Verify agent works without Reflexio

Start a new session and confirm the agent works normally with no Reflexio-related errors.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Agent doesn't follow past corrections | Search hook timeout or no playbooks yet | Check `reflexio user-playbooks list`; verify server is running |
| `[reflexio] Search failed` in every message | Server not running | `reflexio services start --only backend` |
| Playbooks not extracted after session | Batch interval not met (need 5+ interactions) | Use `/reflexio-extract` for manual extraction |
| Agent mentions Reflexio to user | Rule not installed | Check `~/.openclaw/workspace/reflexio.md` exists |
| Wrong user_id in playbooks | REFLEXIO_USER_ID env override | Unset the env var; let auto-detection work |
| Aggregation never runs | Flag file stuck | Remove `~/.reflexio/logs/.aggregation-running` |
