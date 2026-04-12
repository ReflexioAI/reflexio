<p align="center">
  <a href="https://github.com/reflexio-ai/reflexio">
    <img src="docs/images/banner.png" width="800px" alt="Reflexio - Make Your Agents Improve Themselves">
  </a>
</p>
<div align="center">

[![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/reflexio-client)](https://pypi.org/project/reflexio-client/)

[Quick Start](#quick-start) · [Features](#features) · [SDK](#sdk-usage) · [CLI](reflexio/cli/README.md) · [Architecture](#architecture) · [Docs](https://www.reflexio.ai/docs) · [Contributing](#contributing)

</div>

---

## What is Reflexio?
The moat for AI agents isn't the model — it's what your agent learns from every interaction it handles.

Reflexio is a **self-improvement platform** for AI agents. It turns every conversation your AI agent has into a learning opportunity — automatically extracting user preferences and behavioral playbooks so your agent continuously improves itself without manual tuning.

```mermaid
flowchart LR
    A[AI Agent] -->|conversations| B[Reflexio]
    G[Human Expert] -->|ideal responses| B
    B --> C[User Profiles]
    B --> D[Playbook Extraction]
    D --> E[Playbook Aggregation]
    B --> F[Success Evaluation]
```

Publish conversations from your agent, and Reflexio closes the self-improvement loop:

- **Never Repeat the Same Mistake**: Transforms user corrections and interaction signals into improved decision-making processes — so agents adapt their behavior and avoid repeating the same mistakes.
- **Lock In What Works**: Persists successful strategies and workflows so your agent reuses proven paths instead of starting from scratch.
- **Correct in Real Time**: Retrieves personalization and operational signals to fix agent behavior live — no retraining required.
- **Learn from Human Experts**: Publish expert-provided ideal responses alongside agent responses — Reflexio automatically extracts actionable playbooks from the differences.
- **Personal & Global Improvements**: Separates individual user preferences from system-wide agent improvements.
- **AI First Self-Optimization**: Agents autonomously reflect, learn, and improve — less human-in-the-loop, more compounding gains.

> **For developers**: See [developer.md](developer.md) for project structure, environment setup, testing, and coding guidelines.

## Demo

<p align="center">
  <img src="docs/images/reflexio_example.gif" width="800px" alt="Reflexio example experience">
</p>

## Quick Start

### Prerequisites

| Tool | Description |
| --- | --- |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python package manager |
| [Node.js](https://nodejs.org/) >= 18 | Frontend runtime |

<p align="center">
  <img src="docs/images/demo.gif" width="800px" alt="Reflexio Demo">
</p>

### Setup

```shell
# 1. Clone and configure
git clone https://github.com/reflexio-ai/reflexio.git
cd reflexio
cp .env.example .env          # Set at least one LLM API key (OpenAI, Anthropic, etc.)

# 2. Install dependencies
uv sync                                   # Python (includes workspace packages)
npm --prefix docs install                  # API docs

# 3. Start services (--storage sqlite is the default)
uv run reflexio services start                    # API (8081), Docs (8082), SQLite storage
uv run reflexio services stop                     # Stop all services
```

> Alternative: `python -m reflexio.cli services start` or `./run_services.sh`

Once running, open **[http://localhost:8082](http://localhost:8082)** to interactively browse and try out the API.
<p align="center">
  <img src="docs/images/doc_website.png" width="800px" alt="Reflexio Doc Website">
</p>

### Try it in 30 seconds (CLI)

Reflexio ships a first-class CLI — the fastest way to see the loop end-to-end with no code. Publish a real multi-turn conversation where the user **corrects** the agent (that's the signal Reflexio learns from), then search for what was extracted:

```shell
uv run reflexio publish --user-id alice --wait --data '{
  "interactions": [
    {"role": "user",      "content": "Deploy the new service."},
    {"role": "assistant", "content": "Starting deployment to us-east-1..."},
    {"role": "user",      "content": "Wait — we never deploy production to us-east-1. Always use us-west-2."},
    {"role": "assistant", "content": "Understood. Switching to us-west-2."}
  ]
}'

# Search the extracted profiles and playbooks
uv run reflexio search "deployment region"
```

One conversation, two artifacts: a user profile (`production region is us-west-2`) and an agent playbook (`confirm region before deploying`). See the [CLI reference](reflexio/cli/README.md) for all input modes (inline JSON, `--file`, `--stdin`) and the full command list.

### Integrate with the Python SDK

```python
import reflexio

client = reflexio.ReflexioClient(
    url_endpoint="http://localhost:8081/"
)

# Publish a multi-turn conversation where the user corrects the agent —
# Reflexio extracts a profile ("prod region = us-west-2") and a playbook
# ("confirm region before deploying").
client.publish_interaction(
    request_id="req-001",
    user_id="alice",
    interactions=[
        reflexio.Interaction(role="user",      content="Deploy the new service."),
        reflexio.Interaction(role="assistant", content="Starting deployment to us-east-1..."),
        reflexio.Interaction(role="user",      content="Wait — we never deploy production to us-east-1. Always use us-west-2."),
        reflexio.Interaction(role="assistant", content="Understood. Switching to us-west-2."),
    ],
)
```

Reflexio will automatically generate profiles and extract playbooks in the background.

## Features

### Profile Generation

- Extracts behavioral profiles from conversations using configurable extractors
- Supports versioning (current → pending → archived) with upgrade/downgrade workflows
- Multiple extractors run in parallel with independent windows and strides

[Read more about user profiles →](https://www.reflexio.ai/docs/concepts/user-profiles)

### Playbook Extraction & Aggregation

- Extracts playbooks from user behavior patterns
- Clusters similar entries and aggregates with LLM (with change detection to skip unchanged clusters)
- Approval workflow: review and approve/reject agent playbooks

[Read more about agent playbooks →](https://www.reflexio.ai/docs/concepts/agent-playbook)

### Expert Learning

- Publish human-expert ideal responses alongside agent responses via the `expert_content` field
- Reflexio automatically compares agent vs. expert responses, focusing on substantive differences (missing info, incorrect approach, reasoning gaps) while ignoring stylistic ones
- Generates actionable playbooks as trigger/instruction/pitfall SOPs that teach the agent what to do differently

[Read more about interactions & expert content →](https://www.reflexio.ai/docs/concepts/interactions#5-expert-content-for-learning-from-experts)

### Agent Success Evaluation

- Session-level evaluation triggered automatically (10 min after last request)
- Shadow comparison mode: A/B test regular vs shadow agent responses
- Tool usage analysis for blocking issue detection

[Read more about evaluation →](https://www.reflexio.ai/docs/examples/agent-evaluation)

### Search & Retrieval

- Hybrid search (vector + full-text) across profiles and playbooks
- LLM-powered query rewriting for improved recall
- Unified search across all entity types in parallel

### Multi-Provider LLM Support

- OpenAI, Anthropic, Google Gemini, OpenRouter, Azure, MiniMax, and custom endpoints
- Powered by LiteLLM — configure your preferred provider via API keys or custom endpoints

## SDK Usage

For detailed API documentation, see the [full API reference](https://www.reflexio.ai/docs/api-reference).

Install the client:

```shell
pip install reflexio-client
```

### Basic usage

```python
import reflexio

client = reflexio.ReflexioClient(
    url_endpoint="http://localhost:8081/"
)

# Publish interactions
await client.publish_interaction(
    request_id="req-001",
    user_id="user-123",
    interactions=[...],
    agent_version="v1",       # optional: track agent versions
    session_id="session-abc", # optional: group requests into sessions
)

# Search profiles
profiles = await client.search_profiles(
    reflexio.SearchUserProfileRequest(query="deployment region preference")
)

# Search agent playbooks
playbooks = await client.get_agent_playbooks(
    reflexio.GetAgentPlaybooksRequest(agent_version="v1")
)
```

### Configuration

```python
# Update org configuration
await client.set_config(reflexio.SetConfigRequest(
    config=reflexio.Config(
        api_key_config=reflexio.APIKeyConfig(openai="sk-..."),
        profile_extractor_configs=[...],
        playbook_configs=[reflexio.PlaybookConfig(...)],
    )
))
```

## Architecture

```
Client (SDK / Web UI)
  → FastAPI Backend
    → Reflexio Orchestrator
      → GenerationService
        ├─ ProfileGenerationService  → Extractor(s) → Deduplicator → Storage
        ├─ PlaybookGenerationService → Extractor(s) → Deduplicator → Storage
        └─ GroupEvaluationScheduler  → Evaluator(s) → Storage (deferred 10 min)
```

See [developer.md](developer.md) for project structure, supported LLM providers, and development setup.

## Documentation

For comprehensive guides, examples, and API reference, visit the **[Reflexio Documentation](https://www.reflexio.ai/docs)**.

## Contributing

We welcome contributions! Please see [developer.md](developer.md) for guidelines.

## License

This project is currently licensed under [Apache License 2.0](LICENSE).
