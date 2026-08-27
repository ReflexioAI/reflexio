# Reflexio Notebooks

Interactive tutorials for learning Reflexio, from your first workflow to advanced production patterns.

> **Start Here:** New to Reflexio? Begin with notebook 00 (Quickstart) for a 5-minute end-to-end walkthrough, then continue with 01 (Interactions) to learn the core publish-and-search loop.

| # | Notebook | Level | Time | Description |
|---|----------|-------|------|-------------|
| 00 | [Quickstart](00_quickstart.ipynb) | Beginner | 5 min | Retrieve, inject, publish learning IDs, and inspect quality |
| 01 | [Interactions](01_interactions.ipynb) | Beginner | 12 min | Publish turns, tools, and retrieved-learning attribution |
| 02 | [Profiles](02_profiles.ipynb) | Beginner | 12 min | Explore user profiles and per-profile relevance/impact |
| 03 | [Playbooks](03_playbook.ipynb) | Intermediate | 15 min | Create, aggregate, and evaluate retrieved playbooks |
| 04 | [Configuration](04_configuration.ipynb) | Intermediate | 15 min | Customize extraction prompts, models, and pipeline behavior |
| 05 | [Concurrent Sessions](05_concurrent_sessions.ipynb) | Advanced | 15 min | Simulate multi-user load and verify data isolation |
| 06 | [Simulation](06_real_world_simulation.ipynb) | Advanced | 20 min | Generate context-aware turns and inspect learning impact |
| 07 | [mem0 Drop-In Wrapper](07_mem0_dropin_wrapper.ipynb) | Beginner | 10 min | Keep your mem0 code, add Reflexio learning with a one-line import change (needs `MEM0_API_KEY` and `pip install 'reflexio-ai[mem0]'`) |

## Prerequisites

- **Reflexio server running** — all notebooks call the backend API, so start it first: `uv run reflexio services start --only backend` (see the [root README](../README.md) Quick Start section for full setup instructions)
- `OPENAI_API_KEY` set in your `.env` file
- **Storage:** SQLite is used by default — no database setup needed

## Quick Start

```bash
pip install 'reflexio-ai[notebooks]' jupyter
uv run reflexio services start --only backend   # start the server
jupyter notebook notebooks/00_quickstart.ipynb
```


## Recommended Retrieval and Quality Loop

**Retrieve → inject → publish `retrieved_learnings` → grade → inspect.**
Attach every injected profile or playbook's stable `{kind, learning_id}` to the
assistant turn, including context it did not cite. Build the prompt and references
from the same retained subset. Seed turns and turns without injected context omit
the field. Never invent IDs or report discarded search results.

Quickstart, Interactions, Profiles, Playbooks, and Simulation demonstrate
`grade_on_demand` followed by `get_retrieved_learning_evaluation_results`, including
status, relevance/impact reasons, and coverage counts. Configuration explains
`retrieved_learning_sampling_rate` for automatic monitoring. Null verdicts are
ungraded, not negative; empty results require checking status and attribution.
The dashboard groups learning verdicts by response, so its percentages differ
from per-learning counts.

Generation and on-demand grading call LLMs and may incur cost. Use a disposable
demo organization/database: some cleanup cells delete all connected data.
See [evaluation guidance](https://www.reflexio.ai/docs/build/agent-evaluation#read-learning-verdicts)
and [dashboard metrics](https://www.reflexio.ai/docs/portal/measuring-reflexio-impact#retrieved-learning-effects).

## Shared Utilities

`_display_helpers.py` provides consistent output formatting across all notebooks. It is imported automatically — you don't need to install it separately.
