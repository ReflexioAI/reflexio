# Reflexio OpenClaw Integration

Connect [OpenClaw](https://openclaw.ai) agents to [Reflexio](https://github.com/reflexio-ai/reflexio) for automatic self-improvement. Conversations are captured automatically; task-specific playbooks are retrieved on-demand via the `reflexio` CLI.

## How It Works

The integration has two independent mechanisms:

### 1. Capture (Hook — automatic, runs every session)

```
Each Turn (message:sent)
  └── Buffer (user message, agent response) → local JSONL file

Session End (command:stop)
  └── reflexio interactions publish --file → publish full conversation to Reflexio server
      └── Server automatically:
          1. Detects learning signals (corrections, friction, re-steering)
          2. Extracts playbooks: freeform content summary + optional structured fields (trigger/instruction/pitfall/rationale)
          3. Extracts user profiles (preferences, expertise, communication style)
          4. Stores everything with vector embeddings for semantic search
```

Correction detection happens **server-side via LLM** — the agent does not need to detect corrections itself.

### 2. Retrieve (Skill — on-demand, per-task)

```
Agent receives task from user
  └── reflexio search "<the user's actual task>"
      └── Semantic search matches query against playbook trigger fields
      └── Returns only playbooks relevant to THIS specific task
      └── Each result has a freeform summary (primary) + optional structured fields (trigger/instruction/pitfall)

Agent needs to personalize response
  └── reflexio user-profiles search "<what to know about the user>"
      └── Vector + FTS hybrid search on profile content
      └── Returns relevant user preferences, expertise, communication style
```

Playbook retrieval is **per-task, not per-session**. Different tasks return different playbooks. The query is matched against the `trigger` field of stored playbooks using semantic search.

### Why Per-Task Search (Not Session-Start Bulk Loading)

Reflexio's LangChain integration (`ReflexioChatModel`) wraps every LLM call with a per-turn search. The OpenClaw integration follows the same principle:

- **Task-specific**: `reflexio search "deploying to staging"` returns deployment corrections — not package management or testing corrections
- **Token-efficient**: Only inject playbooks relevant to the current task, not everything
- **Fresh**: Each search returns the latest playbooks, not stale bootstrap context
- **Matching mechanism**: The search query is matched against the `trigger` field (the situation where the agent's default behavior was wrong). Only playbooks whose trigger semantically matches your task are returned.

## Prerequisites

- [OpenClaw](https://openclaw.ai) installed and running
- The `reflexio` CLI on PATH: `pip install reflexio`
- A running Reflexio server (local or remote)
- For cloud/Supabase mode: `REFLEXIO_API_KEY` environment variable set

**Reflexio server also requires an LLM API key** (e.g., `OPENAI_API_KEY`) in the `.env` file for playbook extraction. Supported providers: OpenAI, Anthropic, Google Gemini, DeepSeek, OpenRouter, MiniMax, DashScope, xAI, Moonshot, ZAI.

> Run `reflexio setup openclaw` to automate LLM provider selection, storage configuration, and hook installation.

## Installation

### Hook (automatic capture)

```bash
# Install the hook
openclaw hooks install /path/to/reflexio/integrations/openclaw/hook --link
openclaw hooks enable reflexio-context
openclaw gateway restart

# Verify
openclaw hooks list
# Should show: ✓ ready │ 🧠 reflexio-context
```

### Skill (on-demand search)

```bash
# Copy or symlink into OpenClaw workspace
cp -r /path/to/reflexio/integrations/openclaw/skill ~/.openclaw/skills/reflexio
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REFLEXIO_API_KEY` | — | Required for cloud/Supabase mode. Not needed for local/SQLite. |
| `REFLEXIO_URL` | `http://127.0.0.1:8081` | Reflexio server URL |
| `REFLEXIO_USER_ID` | `openclaw` | User ID for profile and playbook scoping |
| `REFLEXIO_AGENT_VERSION` | `openclaw-agent` | A label identifying your agent version. Playbooks are scoped by this tag. |

## What to Expect

**Session 1 (cold start):** No playbooks exist yet. The agent works normally. At session end, the hook captures the full conversation. Reflexio's server-side LLM pipeline analyzes it and extracts any corrections or user preferences.

**Session 2+:** Before each task, the agent runs `reflexio search "<task>"` and gets task-specific corrections from past sessions. Over time:

- Mistakes made once are not repeated (corrections match by trigger similarity)
- User preferences are remembered (profiles extracted automatically)
- The agent adapts its approach per-task based on accumulated playbooks

**The learning loop:**
1. Agent works on a task → user corrects a mistake
2. Session ends → hook captures full conversation → server extracts playbooks (freeform content + structured fields)
3. Next session, similar task → agent searches → gets the correction → applies the behavioral guideline
4. Mistake not repeated

## File Structure

```
openclaw/
├── README.md           ← This file
├── hook/               ← Automatic: capture conversations at session end
│   ├── handler.js      ← Event handlers: bootstrap (profiles), message:sent, command:stop
│   ├── HOOK.md         ← Hook metadata (events, requirements)
│   └── package.json    ← npm package manifest
└── skill/              ← On-demand: search for task-specific playbooks
    └── SKILL.md        ← Teaches agent when/how to use reflexio CLI
```

## Comparison with LangChain Integration

| Aspect | OpenClaw | LangChain |
|--------|----------|-----------|
| Integration method | CLI commands + hooks | Python SDK + callbacks |
| Context retrieval | Per-task via `reflexio search` (agent-initiated) | Per-LLM-call via middleware (automatic) |
| Conversation capture | Hook buffers + flushes at session end | Callback captures per chain run |
| Agent teaching | SKILL.md (natural language) | Tool definition (structured) |
| Dependencies | `reflexio` CLI only | `langchain-core >= 0.3.0` |

Both integrations connect to the same Reflexio server and share the same playbook/profile data.
