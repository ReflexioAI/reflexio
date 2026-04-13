---
name: reflexio-aggregate
description: "Aggregate user playbooks from all OpenClaw instances into shared agent playbooks. Use when you want to consolidate learnings across all agents."
---

# Aggregate Learnings into Shared Agent Playbooks

Aggregate user playbooks collected across all OpenClaw instances into shared agent playbooks that every agent can search.

## What Aggregation Does

Each time a conversation is extracted (via `/reflexio-extract` or the automatic hook), the learnings are stored as **user playbooks** — behavioral rules tied to a specific user and agent instance. Aggregation goes one step further: it clusters similar user playbooks from all instances, deduplicates overlapping rules, and produces **agent playbooks** — shared behavioral guidelines available to every OpenClaw agent, regardless of which instance or user originally generated the learning.

This means a correction one agent learned from one user can benefit every agent going forward.

## How to Aggregate

1. Ensure the server is running **(local server only)**:

Check the `REFLEXIO_URL` environment variable. If it points to a remote server (not `localhost` or `127.0.0.1`), skip this step — managed Reflexio servers are always running.

For local servers:
```bash
reflexio status check
```
If not running, start it:
```bash
nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
sleep 5
```

2. Run aggregation and wait for results:

```bash
reflexio agent-playbooks aggregate --agent-version openclaw-agent --wait
```

The `--wait` flag blocks until the aggregation job completes and prints a summary of how many playbooks were created or updated. Aggregation runs an LLM clustering pass over all user playbooks for the `openclaw-agent` version, so it may take 30–60 seconds depending on volume.

3. Report results to the user: how many agent playbooks were created or updated, and what themes emerged (if the output includes them).

4. Suggest reviewing the new agent playbooks:

```bash
reflexio agent-playbooks list --agent-version openclaw-agent
```

This shows all shared behavioral rules now available to every OpenClaw agent on the next `reflexio search` call.

## Setting Up Regular Aggregation

For teams running OpenClaw continuously, set up a cron job to aggregate periodically so shared playbooks stay fresh:

```bash
# Run aggregation daily at 3am
0 3 * * * reflexio agent-playbooks aggregate --agent-version openclaw-agent --wait >> ~/.reflexio/logs/aggregate.log 2>&1
```

Alternatively, configure aggregation to run automatically after each publish by setting `auto_aggregate: true` in your Reflexio server config:

```bash
reflexio config set --data '{"auto_aggregate": true}'
```

With `auto_aggregate` enabled, you only need to run this command manually when you want an immediate consolidation — for example, after a batch of high-friction sessions where you want the learnings available right away.
