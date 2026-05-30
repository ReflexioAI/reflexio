# Reflexio Notebooks

Interactive tutorials for learning Reflexio, from your first workflow to advanced production patterns.

> **Start Here:** New to Reflexio? Begin with notebook 00 (Quickstart) for a 5-minute end-to-end walkthrough, then continue with 01 (Interactions) to learn the core publish-and-search loop.

| # | Notebook | Level | Time | Description |
|---|----------|-------|------|-------------|
| 00 | [Quickstart](00_quickstart.ipynb) | Beginner | 5 min | End-to-end setup and first publish/search cycle |
| 01 | [Interactions](01_interactions.ipynb) | Beginner | 12 min | Publish conversations and search extracted data |
| 02 | [Profiles](02_profiles.ipynb) | Beginner | 12 min | Explore how Reflexio builds persistent user profiles from interactions |
| 03 | [Playbooks](03_playbook.ipynb) | Intermediate | 15 min | Create, aggregate, and govern agent playbooks |
| 04 | [Configuration](04_configuration.ipynb) | Intermediate | 15 min | Customize extraction prompts, models, and pipeline behavior |
| 05 | [Concurrent Sessions](05_concurrent_sessions.ipynb) | Advanced | 15 min | Simulate multi-user load and verify data isolation |
| 06 | [Simulation](06_real_world_simulation.ipynb) | Advanced | 20 min | Run a multi-turn simulation and watch profiles evolve over time |

## Prerequisites

- **Reflexio server running** — all notebooks call the backend API, so start it first: `uv run reflexio services start --only backend` (see the [root README](../README.md) Quick Start section for full setup instructions)
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
