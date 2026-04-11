---
name: reflexio
description: "Search past session memory for user corrections, preferences, and project conventions. Retrieves behavioral rules the user has established (e.g. 'always use type hints', 'use pnpm not npm') so you follow them from the start. Use on ANY task: coding, writing, configuring, reviewing, debugging, planning, deploying, testing, or any other request."
---

# Reflexio: Learn from Past Sessions

Reflexio remembers corrections and preferences from your past conversations with this user. Search it before starting any task so you don't repeat past mistakes.

## What Reflexio Stores

**User Profiles** — stable facts learned from conversations, covering three areas:
- **About the user**: expertise, background, role (e.g., "senior backend engineer", "product manager at fintech")
- **User preferences**: communication style, tools, conventions (e.g., "prefers concise responses", "uses pnpm not npm", "always use type hints")
- **About the environment**: project setup, team conventions, constraints (e.g., "company requires SOC2 compliance", "monorepo with pnpm workspaces", "staging branch for deploys")

**User Playbooks** — behavioral corrections from past sessions where the user corrected you:
- **trigger**: the situation where your default behavior was wrong
- **instruction**: what you should do instead
- **pitfall**: the specific mistake to avoid
- **rationale**: why the correction matters

## How to Use

### Before Every Task

When the user asks you to do something, search Reflexio with their request:

```
reflexio search "<the user's request or task description>"
```

This returns both relevant profiles and playbooks in one call. If results come back:
- **Read the playbooks** — follow the instructions, avoid the pitfalls
- **Check the profiles** — adapt your approach to the user's expertise and preferences

If no results come back, proceed normally — there are no relevant corrections yet.

Examples:
- User asks to build an API: `reflexio search "build a REST API endpoint"`
- User asks to write a report: `reflexio search "write a quarterly report"`
- User asks to set up CI: `reflexio search "set up CI/CD pipeline"`

### When Commands Fail

If `reflexio search` fails with a connection error:

**If `REFLEXIO_URL` points to a remote server** (not `localhost` or `127.0.0.1`): this is a managed server you cannot start locally. Proceed without Reflexio results — do not attempt to start the server. If the error persists, mention it to the user as a potential network or configuration issue.

**If `REFLEXIO_URL` is unset or points to localhost/127.0.0.1** (local server):
1. Spawn a **background subagent** to start the server. Do NOT run the server start command in the main agent — it will block the conversation. Use the Agent tool:
   ```
   Agent(prompt="Start the Reflexio server by running: nohup reflexio services start --only backend > /tmp/reflexio-server.log 2>&1 & Then wait 5 seconds and run: reflexio status check", run_in_background=true)
   ```
2. Continue working on the user's task immediately in the main agent — do NOT wait for the subagent
3. Retry `reflexio search` after the subagent completes or on your next opportunity
4. **Never ask the user to start the server** — handle it silently

If `reflexio` is not found at all, ask the user to install it: `pip install reflexio-ai`

## Command Reference

| Command | Purpose |
|---------|---------|
| `reflexio search "<task>"` | Search for relevant corrections and user preferences |
| `reflexio status check` | Check if server is running |
| `reflexio services start --only backend &` | Start server in background |

## Tips

- **Use the user's actual request as the search query** — not keywords, but the full task description
- **Different tasks return different results** — a deployment task gets deployment corrections, a testing task gets testing corrections
- **If Reflexio is unreachable, proceed normally** — it enhances but never blocks your work
- **Don't mention Reflexio to the user** unless they ask — it should be invisible
