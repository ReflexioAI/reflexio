# Reflexio Client

Python SDK for interacting with the [Reflexio](https://www.reflexio.ai/) API. Provides type-safe, sync-first access to user profiles, interactions, playbooks, evaluations, and configuration.

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
from reflexio import InteractionData

# Fire-and-forget (non-blocking, returns None)
client.publish_interaction(
    user_id="user-123",
    interactions=[
        InteractionData(
            request_id="req-001",
            user_request="How do I reset my password?",
            agent_response="Go to Settings > Security > Reset Password.",
        )
    ],
    source="support-bot",
    agent_version="v2.1",
    session_id="session-abc",
)

# Wait for server to finish processing
response = client.publish_interaction(
    user_id="user-123",
    interactions=[
        InteractionData(
            request_id="req-002",
            user_request="Thanks, that worked!",
            agent_response="Glad I could help!",
        )
    ],
    agent_version="v2.1",
    wait_for_response=True,
)
print(response.success, response.msg)
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
# Get user playbooks
raw = client.get_raw_feedbacks(feedback_name="usability", limit=50)

# Search user playbooks
results = client.search_raw_feedbacks(query="slow response", agent_version="v2.1")

# Add user playbook directly
from reflexio import RawFeedback
client.add_raw_feedback(raw_feedbacks=[
    RawFeedback(
        agent_version="v2.1",
        request_id="req-001",
        feedback_content="User found the response helpful",
        feedback_name="satisfaction",
    )
])

# Rerun playbook generation
client.rerun_feedback_generation(
    agent_version="v2.1",
    feedback_name="usability",
    wait_for_response=True,
)
```

### Agent Playbooks (clustered insights)

```python
# Get agent playbooks
from reflexio import PlaybookStatus
agent_playbooks = client.get_agent_playbooks(
    playbook_name="usability",
    playbook_status_filter=PlaybookStatus.APPROVED,
)

# Search agent playbooks
results = client.search_agent_playbooks(query="response quality", agent_version="v2.1")

# Add agent playbook directly
from reflexio import AgentPlaybook
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

## Skills

```python
# Search skills
skills = client.search_skills(request={"query": "data export", "agent_version": "v2.1"})

# Get skills
skills = client.get_skills(request={"agent_version": "v2.1"})
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
client.delete_feedbacks_by_ids([1, 2, 3])
client.delete_raw_feedbacks_by_ids([4, 5, 6])

# Delete all
client.delete_all_interactions()
client.delete_all_profiles()
client.delete_all_feedbacks()
```

## Fire-and-Forget vs Blocking

Most write/delete methods support `wait_for_response`:

- **`wait_for_response=False`** (default): Non-blocking fire-and-forget. Returns `None`. Efficient for high-throughput pipelines.
- **`wait_for_response=True`**: Blocks until the server finishes processing. Returns the full response.

In async contexts (e.g., FastAPI), fire-and-forget uses the existing event loop. In sync contexts, it uses a shared thread pool.

## API Reference

| Method | Description |
|--------|-------------|
| `publish_interaction()` | Publish interactions (triggers profile/playbook/evaluation) |
| `search_profiles()` | Semantic search for profiles |
| `get_profiles()` | Get profiles for a user |
| `get_all_profiles()` | Get all profiles across users |
| `delete_profile()` | Delete profiles by ID or search query |
| `get_profile_change_log()` | Get profile change history |
| `rerun_profile_generation()` | Regenerate profiles from interactions |
| `manual_profile_generation()` | Trigger profile generation with window-sized interactions |
| `search_interactions()` | Semantic search for interactions |
| `get_interactions()` | Get interactions for a user |
| `get_all_interactions()` | Get all interactions across users |
| `delete_interaction()` | Delete a specific interaction |
| `get_requests()` | Get requests grouped by session |
| `delete_request()` | Delete a request and its interactions |
| `delete_session()` | Delete all requests in a session |
| `get_raw_feedbacks()` | Get user playbooks |
| `search_raw_feedbacks()` | Search user playbooks |
| `add_raw_feedback()` | Add user playbook directly |
| `get_feedbacks()` | Get agent playbooks |
| `search_feedbacks()` | Search agent playbooks |
| `add_feedbacks()` | Add agent playbook directly |
| `rerun_feedback_generation()` | Regenerate playbooks for an agent version |
| `manual_feedback_generation()` | Trigger playbook generation with window-sized interactions |
| `run_feedback_aggregation()` | Cluster user playbooks into agent playbooks |
| `search()` | Unified search across all entity types |
| `get_agent_success_evaluation_results()` | Get evaluation results |
| `search_skills()` | Search skills |
| `get_skills()` | Get skills |
| `set_config()` | Update org configuration |
| `get_config()` | Get current configuration |
| `delete_requests_by_ids()` | Bulk delete requests |
| `delete_profiles_by_ids()` | Bulk delete profiles |
| `delete_feedbacks_by_ids()` | Bulk delete agent playbooks |
| `delete_raw_feedbacks_by_ids()` | Bulk delete user playbooks |
| `delete_all_interactions()` | Delete all interactions |
| `delete_all_profiles()` | Delete all profiles |
| `delete_all_feedbacks()` | Delete all playbooks |

## Requirements

- Python >= 3.11
- `pydantic >= 2.0.0`
- `requests >= 2.25.0`
- `aiohttp >= 3.12.9`
- `python-dateutil >= 2.8.0`
- `python-dotenv >= 0.19.0`
