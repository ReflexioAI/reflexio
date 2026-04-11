---
name: reflexio
description: "Use Reflexio to retrieve task-specific playbooks before working. Search with your current task to get behavioral corrections learned from past sessions. Conversations are captured automatically by the hook."
---

# Reflexio Agent Memory

Reflexio stores behavioral corrections and user profiles from past sessions. Use it to retrieve task-specific guidance before working, so you don't repeat past mistakes.

## First-Time Setup

Before using Reflexio, check if it's configured:

```
reflexio status check
```

If you see connection errors, guide the user through setup:

### Missing API Key

If `reflexio status check` reports no API key or connection failure:

1. Ask the user: "Reflexio needs an LLM API key for playbook extraction. Do you have an OPENAI_API_KEY or ANTHROPIC_API_KEY set in your environment?"
2. If yes: configure the connection:
   ```
   reflexio auth login --api-key <KEY> --url http://localhost:8081
   ```
3. If no: ask the user to provide one:
   - "Please set your API key: `export OPENAI_API_KEY=sk-...`"
   - Or: "Please set your API key: `export ANTHROPIC_API_KEY=sk-ant-...`"
4. Then configure: `reflexio auth login --api-key <KEY>`

### Remote Server

If the user has a hosted Reflexio server:

1. Ask for the server URL and API key:
   - "What is your Reflexio server URL? (e.g., https://api.reflexio.ai)"
   - "What is your Reflexio API key? (starts with rflx-)"
2. Configure:
   ```
   reflexio auth login --url "https://api.reflexio.ai" --api-key "rflx-xxxxxxxxxxxx"
   ```
3. Verify: `reflexio status check`

### Local Server (Default)

For local usage:

1. Set your LLM API key: `export OPENAI_API_KEY=sk-...`
2. Configure: `reflexio auth login --url http://localhost:8081`
3. Set server config: `reflexio config set --data '{"batch_size": 10}'`
4. Start the server: `reflexio services start`
5. Verify: `reflexio status check`

## How Reflexio Playbooks Work

Each playbook has a **feedback_content** field — a freeform natural-language summary that is the primary human-readable description of what was learned. This is the core behavioral guideline you should follow.

Playbooks also include optional structured fields for richer context:
- **trigger**: The situation/task type where this rule applies (used for semantic search matching)
- **instruction** (optional): The correct behavior to follow
- **pitfall** (optional): The mistake to avoid
- **rationale** (optional): Why the old behavior was wrong and how the new policy prevents recurrence

When you search, Reflexio matches your query against trigger fields using semantic search. Only playbooks whose triggers are relevant to your current task are returned.

## When to Use Reflexio

### Search Before Every Task (Primary Usage)

**Trigger**: Before starting ANY task the user gives you. Use the user's actual request as the search query.

```
reflexio search "<the user's actual request or task description>"
```

The search matches your query against playbook trigger fields. Results include behavioral corrections from past sessions formatted as:
```
- <freeform summary — this is the primary behavioral guideline>
  Trigger: <the situation this applies to>
  Instruction: <the correct behavior>
  Pitfall: <the mistake to avoid>
  Rationale: <why the old behavior was wrong>
```

Read the playbook content first — it's a standalone guideline. The structured fields below it provide additional context. If the Trigger matches your current task, follow the guidance.

Examples:
- User asks to deploy: `reflexio search "deploying to staging"`
- User asks to write tests: `reflexio search "writing tests for API endpoints"`
- User asks to set up a service: `reflexio search "setting up a new microservice"`

Different tasks return different playbooks — a deployment task gets deployment corrections, a testing task gets testing corrections.

### Check User Profiles (When Personalizing)

**Trigger**: When you want to tailor your communication or approach based on who the user is.

```
reflexio user-profiles search "<what you want to know>"
```

Examples:
- `reflexio user-profiles search "expertise level and background"`
- `reflexio user-profiles search "communication preferences"`
- `reflexio user-profiles search "technology stack preferences"`

### Add Explicit Playbook (Rare — For Non-Obvious Learnings)

**Trigger**: When YOU discover something non-obvious during your work that the automatic pipeline might miss. This is for high-confidence, immediate learnings — not for user corrections (those are detected automatically by Reflexio's server-side LLM pipeline when conversations are captured).

Use this when:
- You discover a non-obvious tool behavior or workaround
- You find a project-specific convention not documented elsewhere
- You identify a pattern that would help future sessions

Do NOT use this for:
- User corrections ("No, use pnpm not npm") — the automatic pipeline handles these
- General knowledge — only project/user-specific learnings

```
reflexio user-playbooks add \
  --content "<freeform summary of the learning>" \
  --trigger "<the situation/task type this applies to>" \
  --instruction "<what to do>" \
  --pitfall "<what to avoid>"
```

### Conversation Capture (Automatic)

The hook automatically buffers each turn during the session and flushes the full conversation to Reflexio at session end. Reflexio's server then:

1. Runs a "should generate" check — LLM analyzes the conversation for learning signals (corrections, criticism, re-steering, friction)
2. If signals found, extracts playbooks (freeform content + optional trigger/instruction/pitfall/rationale) via LLM
3. Also extracts user profiles (preferences, expertise, communication style)
4. Stores everything with vector embeddings for future semantic search

You do NOT need to manually capture conversations or detect corrections. This is fully automatic.

## Command Reference

| Command | Purpose | When |
|---------|---------|------|
| `reflexio search "<task>"` | Find task-specific playbooks | Before every task (primary usage) |
| `reflexio user-profiles search "<query>"` | Find user preferences | When personalizing responses |
| `reflexio user-playbooks add ...` | Record a non-obvious learning | Rare: only for learnings the auto-pipeline would miss |
| `reflexio status check` | Check server connectivity | First, or if commands fail |
| `reflexio auth login --api-key KEY` | Configure credentials | First-time setup |

## Tips

- **Search query = the user's task**: Describe what you're about to do, not keywords. The system matches against playbook trigger fields semantically.
- **Different tasks, different playbooks**: A "deploy" search returns deployment corrections; a "test" search returns testing corrections. Always search with the specific task.
- **Don't manually detect corrections**: The server-side LLM pipeline handles correction detection automatically when conversations are captured at session end.
- **If Reflexio is unreachable, proceed normally**: It enhances but doesn't block your work.
- **Use --json for machine-readable output**: All commands support `--json` for structured JSON envelope output.
