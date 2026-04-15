# Reflexio Python SDK

[![PyPI](https://img.shields.io/pypi/v/reflexio-ai)](https://pypi.org/project/reflexio-ai/)
[![Python >= 3.12](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/ReflexioAI/reflexio/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/reflexio-ai/month)](https://pepy.tech/project/reflexio-ai)

The official Python SDK for [Reflexio](https://www.reflexio.ai/) — the adaptive memory layer for AI agents. Reflexio automatically extracts user profiles, generates playbooks, and evaluates agent performance from conversation data. This client provides type-safe, sync-first access to the full Reflexio API. For source code and contributions, see the [GitHub repository](https://github.com/ReflexioAI/reflexio).

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Publishing Interactions](#publishing-interactions)
- [Profiles](#profiles)
- [Interactions](#interactions)
- [Requests & Sessions](#requests--sessions)
- [Playbooks](#playbooks)
  - [User Playbooks](#user-playbooks-extracted-from-interactions)
  - [Agent Playbooks](#agent-playbooks-clustered-insights)
- [Unified Search](#unified-search)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Bulk Delete Operations](#bulk-delete-operations)
- [Fire-and-Forget vs Blocking](#fire-and-forget-vs-blocking)
- [API Reference](#api-reference)
- [Requirements](#requirements)

## Installation

```bash
pip install reflexio-client
```

With LangChain integration:

```bash
pip install reflexio-client[langchain]
```

## Quick Start

```python
from reflexio import ReflexioClient

# API key from constructor or REFLEXIO_API_KEY env var
client = ReflexioClient(api_key="your-api-key")

# Or connect to a self-hosted instance
client = ReflexioClient(
    api_key="your-api-key",
    url_endpoint="http://localhost:8081",
)
```

## Authentication

The client authenticates via Bearer token. Provide your API key in one of two ways:

1. **Constructor**: `ReflexioClient(api_key="your-key")`
2. **Environment variable**: Set `REFLEXIO_API_KEY` (auto-detected)

The base URL defaults to `https://www.reflexio.ai/` and can be overridden with `url_endpoint` or the `REFLEXIO_API_URL` env var.

## Publishing Interactions

Publish user interactions to trigger profile extraction, playbook generation, and evaluation:

```python
# Server-async mode: HTTP round-trip blocks, but server returns as soon as
# it has registered the background extraction (~100 ms).
response = client.publish_interaction(
    user_id="user-123",
    interactions=[
        {"role": "user",      "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Go to Settings > Security > Reset Password."},
    ],
    source="support-bot",
    agent_version="v2.1",
    session_id="session-abc",
)

# Wait for the server to finish processing before returning.
response = client.publish_interaction(
    user_id="user-123",
    interactions=[
        {"role": "user",      "content": "Thanks, that worked!"},
        {"role": "assistant", "content": "Glad I could help!"},
    ],
    agent_version="v2.1",
    wait_for_response=True,
)
print(response.success, response.message)
```

## Profiles

```python
# Semantic search for profiles
results = client.search_profiles(user_id="user-123", query="password preferences")
for profile in results.profiles:
    print(profile.profile_name, profile.profile_content)

# Get all profiles for a user
profiles = client.get_profiles(user_id="user-123")

# Filter by status
from reflexio import Status
profiles = client.get_profiles(user_id="user-123", status_filter=[Status.CURRENT])

# Get all profiles across all users
all_profiles = client.get_all_profiles(limit=50)

# Delete a specific profile
client.delete_profile(user_id="user-123", profile_id="prof-456", wait_for_response=True)

# Get profile change history
changelog = client.get_profile_change_log()

# Rerun profile generation from existing interactions
response = client.rerun_profile_generation(
    user_id="user-123",
    extractor_names=["preferences"],
    wait_for_response=True,
)
print(f"Generated {response.profiles_generated} profiles")
```

## Interactions

```python
# Semantic search
results = client.search_interactions(user_id="user-123", query="password reset")

# List interactions for a user
interactions = client.get_interactions(user_id="user-123", top_k=50)

# Get all interactions across all users
all_interactions = client.get_all_interactions(limit=100)

# Delete a specific interaction
client.delete_interaction(
    user_id="user-123", interaction_id="int-789", wait_for_response=True
)
```

## Requests & Sessions

```python
# Get requests grouped by session
requests = client.get_requests(user_id="user-123")

# Delete a request and its interactions
client.delete_request(request_id="req-001", wait_for_response=True)

# Delete all requests in a session
client.delete_session(session_id="session-abc", wait_for_response=True)
```

## Playbooks

### User Playbooks (extracted from interactions)

```python
from reflexio import UserPlaybook

# Get user playbooks
playbooks = client.get_user_playbooks(playbook_name="usability", limit=50)

# Search user playbooks
results = client.search_user_playbooks(query="slow response", agent_version="v2.1")

# Add a user playbook directly
client.add_user_playbook(user_playbooks=[
    UserPlaybook(
        agent_version="v2.1",
        request_id="req-001",
        content="User found the response helpful",
        playbook_name="satisfaction",
    )
])

# Rerun playbook generation
client.rerun_playbook_generation(
    agent_version="v2.1",
    playbook_name="usability",
    wait_for_response=True,
)
```

### Agent Playbooks (clustered insights)

```python
from reflexio import AgentPlaybook, PlaybookStatus

# Get agent playbooks
agent_playbooks = client.get_agent_playbooks(
    playbook_name="usability",
    playbook_status_filter=PlaybookStatus.APPROVED,
)

# Search agent playbooks
results = client.search_agent_playbooks(query="response quality", agent_version="v2.1")

# Add an agent playbook directly
client.add_agent_playbooks(agent_playbooks=[
    AgentPlaybook(
        agent_version="v2.1",
        content="Users prefer concise answers",
        playbook_status=PlaybookStatus.APPROVED,
        playbook_metadata="Aggregated from 15 user playbooks",
        playbook_name="style",
    )
])

# Run playbook aggregation
client.run_playbook_aggregation(
    agent_version="v2.1",
    playbook_name="usability",
    wait_for_response=True,
)
```

## Unified Search

Search across profiles, agent playbooks, user playbooks, and skills in one call:

```python
from reflexio import ConversationTurn

results = client.search(
    query="user prefers dark mode",
    top_k=5,
    agent_version="v2.1",
    user_id="user-123",
    enable_reformulation=True,
    conversation_history=[
        ConversationTurn(role="user", content="What themes are available?"),
        ConversationTurn(role="assistant", content="We support light and dark themes."),
    ],
)

print(results.profiles)
print(results.feedbacks)       # agent playbooks
print(results.raw_feedbacks)   # user playbooks
```

## Evaluation

```python
# Get agent success evaluation results
results = client.get_agent_success_evaluation_results(
    agent_version="v2.1",
    limit=50,
)
```

## Configuration

```python
from reflexio import Config

# Get current config
config = client.get_config()
print(config)

# Update config
client.set_config(Config(
    profile_extractor_config=[...],
    playbook_config=[...],
))
```

## Bulk Delete Operations

```python
# Delete by IDs
client.delete_requests_by_ids(["req-001", "req-002"])
client.delete_profiles_by_ids(["prof-001", "prof-002"])
client.delete_agent_playbooks_by_ids([1, 2, 3])
client.delete_user_playbooks_by_ids([4, 5, 6])

# Delete all
client.delete_all_interactions()
client.delete_all_profiles()
client.delete_all_playbooks()
```

## Fire-and-Forget vs Blocking

Most write/delete methods support `wait_for_response`:

- **`wait_for_response=False`** (default): Non-blocking fire-and-forget. Returns `None`. Efficient for high-throughput pipelines.
- **`wait_for_response=True`**: Blocks until the server finishes processing. Returns the full response.

In async contexts (e.g., FastAPI), fire-and-forget uses the existing event loop. In sync contexts, it uses a shared thread pool.

## API Reference

### Interactions

| Method | Description |
|--------|-------------|
| `publish_interaction()` | Publish interactions (triggers profile/playbook/evaluation) |
| `search_interactions()` | Semantic search for interactions |
| `get_interactions()` | Get interactions for a user |
| `get_all_interactions()` | Get all interactions across users |
| `delete_interaction()` | Delete a specific interaction |
| `delete_all_interactions()` | Delete all interactions |

### Profiles

| Method | Description |
|--------|-------------|
| `search_profiles()` | Semantic search for profiles |
| `get_profiles()` | Get profiles for a user |
| `get_all_profiles()` | Get all profiles across users |
| `delete_profile()` | Delete profiles by ID or search query |
| `delete_profiles_by_ids()` | Bulk delete profiles by ID |
| `delete_all_profiles()` | Delete all profiles |
| `get_profile_change_log()` | Get profile change history |
| `rerun_profile_generation()` | Regenerate profiles from interactions |
| `manual_profile_generation()` | Trigger profile generation with window-sized interactions |

### Agent Playbooks

| Method | Description |
|--------|-------------|
| `get_agent_playbooks()` | Get agent playbooks |
| `search_agent_playbooks()` | Search agent playbooks |
| `add_agent_playbooks()` | Add agent playbooks directly |
| `run_playbook_aggregation()` | Cluster user playbooks into agent playbooks |
| `delete_agent_playbooks_by_ids()` | Bulk delete agent playbooks |
| `delete_all_playbooks()` | Delete all playbooks |

### User Playbooks

| Method | Description |
|--------|-------------|
| `get_user_playbooks()` | Get user playbooks |
| `search_user_playbooks()` | Search user playbooks |
| `add_user_playbook()` | Add user playbook directly |
| `rerun_playbook_generation()` | Regenerate playbooks for an agent version |
| `manual_playbook_generation()` | Trigger playbook generation with window-sized interactions |
| `delete_user_playbooks_by_ids()` | Bulk delete user playbooks |

### Requests & Sessions

| Method | Description |
|--------|-------------|
| `get_requests()` | Get requests grouped by session |
| `delete_request()` | Delete a request and its interactions |
| `delete_session()` | Delete all requests in a session |
| `delete_requests_by_ids()` | Bulk delete requests by ID |

### Search

| Method | Description |
|--------|-------------|
| `search()` | Unified search across all entity types |

### Evaluation

| Method | Description |
|--------|-------------|
| `get_agent_success_evaluation_results()` | Get evaluation results |

### Configuration

| Method | Description |
|--------|-------------|
| `get_config()` | Get current configuration |
| `set_config()` | Update org configuration |

## Requirements

- Python >= 3.12
- `pydantic >= 2.0.0`
- `requests >= 2.25.0`
- `aiohttp >= 3.12.9`
- `python-dateutil >= 2.8.0`
- `python-dotenv >= 0.19.0`
