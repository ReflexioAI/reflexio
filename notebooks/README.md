# Reflexio Notebooks

Interactive tutorials for learning Reflexio, from your first workflow to advanced production patterns.

| # | Notebook | Level | Time | Description |
|---|----------|-------|------|-------------|
| 00 | [Quickstart](00_quickstart.ipynb) | Beginner | 5 min | Your first Reflexio workflow |
| 01 | [Interactions](01_interactions.ipynb) | Beginner | 12 min | Publishing & searching interactions |
| 02 | [Profiles](02_profiles.ipynb) | Beginner | 12 min | User profiles & memory |
| 03 | [Playbooks](03_playbook.ipynb) | Intermediate | 15 min | Agent playbooks & improvement |
| 04 | [Configuration](04_configuration.ipynb) | Intermediate | 15 min | Custom extraction configs |
| 05 | [Concurrent Sessions](05_concurrent_sessions.ipynb) | Advanced | 15 min | Multi-user load & data isolation |
| 06 | [Simulation](06_real_world_simulation.ipynb) | Advanced | 20 min | Watch Reflexio learn from conversations |
| 07 | [LangChain](07_langchain_integration.ipynb) | Intermediate | 15 min | LangChain integration |

## Prerequisites

- Reflexio server running (`uv run reflexio services start --only backend`)
- `OPENAI_API_KEY` set in your `.env` file
- **Storage:** SQLite is used by default — no database setup needed

## Quick Start

```bash
pip install reflexio-client
uv run reflexio services start --only backend   # start the server
jupyter notebook notebooks/00_quickstart.ipynb
```

## Shared Utilities

`_display_helpers.py` provides consistent output formatting across all notebooks. It is imported automatically — you don't need to install it separately.
