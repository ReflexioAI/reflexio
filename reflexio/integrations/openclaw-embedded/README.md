# Reflexio OpenClaw-Embedded Plugin

A lightweight Openclaw plugin that delivers Reflexio-style user profile and playbook capabilities entirely within Openclaw's native primitives — no Reflexio server required.

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [First-use Setup](#first-use-setup)
- [Configuration](#configuration)
- [Comparison with Other Reflexio Integrations](#comparison-with-other-reflexio-integrations)
- [Uninstall](#uninstall)
- [Further Reading](#further-reading)

## How It Works

The plugin captures two kinds of memory:

- **Profiles** — durable user facts (diet, preferences, timezone, role). Stored as `.md` files under `.reflexio/profiles/` with a TTL.
- **Playbooks** — procedural rules learned from corrections (user corrects → agent adjusts → user confirms → rule written). Stored under `.reflexio/playbooks/`.

Three flows capture memory at different moments:

- **Flow A (in-session profile)**: agent detects a preference/fact/config in the user message and writes immediately.
- **Flow B (in-session playbook)**: agent recognizes correction+confirmation multi-turn pattern and writes the rule.
- **Flow C (session-end batch)**: hook fires on `session:compact:before` / `command:stop` / `command:reset`; spawns a sub-agent that extracts from the full transcript, runs shallow pairwise dedup, and writes/deletes `.md` files.

A daily 3am cron job runs full-sweep consolidation (n-way cluster merges) across all files.

All retrieval is via Openclaw's memory engine — vector + FTS + MMR + temporal decay. When Active Memory is enabled, relevant profiles/playbooks are auto-injected into each turn.

## Prerequisites

- [OpenClaw](https://openclaw.ai) installed and `openclaw` CLI on PATH
- Node.js (for the plugin runtime)
- macOS or Linux (Windows via WSL)
- A bash-compatible shell (install/uninstall scripts use `#!/usr/bin/env bash`)
- Strongly recommended:
  - An embedding provider API key (OpenAI, Gemini, Voyage, or Mistral) for vector search
  - The `active-memory` plugin enabled (auto-retrieval into turns)

The plugin works without active-memory and without an embedding key — with degraded retrieval quality. See `references/architecture.md` for degradation modes.

## Installation

```bash
# From the plugin directory:
./scripts/install.sh
```

What it does:
1. Installs and links the `plugin/` directory as an Openclaw plugin
2. Copies SKILL.md, consolidate skill, and agent definitions to workspace
3. Copies prompts to workspace
4. Enables the `active-memory` plugin and configures agent targeting + extraPath
5. Registers a daily 3am consolidation cron
6. Restarts the Openclaw gateway
7. Prints verification commands

## First-use Setup

The first time an agent invokes the `reflexio-embedded` skill, it runs a one-time bootstrap:

1. Probes current config via `openclaw config get` + `openclaw memory status --deep`.
2. For any missing prereq, asks the user for approval before running `openclaw config set` via the `exec` tool.
3. On success, creates `.reflexio/.setup_complete_<agentId>` marker — subsequent sessions skip.

This guarantees zero manual `openclaw.json` editing. If `exec` is denied by admin policy, the skill prints the exact commands for the user to run manually.

## Configuration

Defaults live in `config.json`. To override, use one of:

1. Edit `config.json` directly
2. Use `openclaw config` for overrides persisted at the Openclaw layer

(env var overrides are planned for v2; see `references/future-work.md`)

Tunables:

| Knob | Default | What it controls |
|---|---|---|
| `dedup.shallow_threshold` | 0.7 | Similarity above which in-session writes trigger pairwise dedup |
| `dedup.full_threshold` | 0.75 | Similarity cluster-member cutoff in daily consolidation |
| `dedup.top_k` | 5 | How many neighbors to consider |
| `ttl_sweep.on_bootstrap` | `true` | Whether to sweep expired profiles on each agent bootstrap |
| `consolidation.cron` | `"0 3 * * *"` | Daily consolidation schedule |
| `extraction.subagent_timeout_seconds` | 120 | Flow C sub-agent timeout |

### Tuning guidance

| Symptom | Likely cause | Knob |
|---|---|---|
| Duplicate `.md` files accumulating between cron runs | Shallow threshold too high | Lower `shallow_threshold` (e.g., 0.65) |
| Good-but-distinct entries getting merged | Thresholds too low | Raise both thresholds (e.g., 0.8) |
| Daily consolidation takes too long | Too many / too broad clusters | Raise `full_threshold`, cap cluster size |
| Session-end latency slightly noticeable | Too many shallow dedup LLM calls | Lower `top_k` to 3 |

## Comparison with Other Reflexio Integrations

See `references/comparison.md` for a full matrix.

- **`integrations/openclaw-embedded/`** (this plugin): self-contained; no Reflexio server; single-user.
- **`integrations/openclaw/`** (federated): requires running Reflexio server; multi-user; cross-instance aggregation.

Both can coexist in the same Openclaw instance, but installing both serves no purpose — pick one.

## Uninstall

```bash
./scripts/uninstall.sh           # preserves .reflexio/ user data
./scripts/uninstall.sh --purge   # also deletes .reflexio/ user data
```

## Further Reading

- [Design spec](../../../../docs/superpowers/specs/2026-04-16-reflexio-openclaw-embedded-plugin-design.md)
- [Implementation plan](../../../../docs/superpowers/plans/2026-04-16-reflexio-openclaw-embedded-plugin.md)
- [Architecture deep-dive](references/architecture.md)
- [Prompt porting notes](references/porting-notes.md)
- [Future work / v2 deferrals](references/future-work.md)
- [Manual testing guide](TESTING.md)
