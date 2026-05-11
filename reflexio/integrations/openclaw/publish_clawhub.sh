#!/usr/bin/env bash
# Publishes the reflexio openclaw integration as a single ClawHub skill.
#
# Two modes:
#   1) CLI publish — stages into a temp dir, generates top-level SKILL.md,
#      runs `bun clawhub skill publish`. Requires bun + an authed session.
#   2) Stage-only — same staging + SKILL.md generation, but skips publish
#      and prints the staged path so you can drop it onto the web form at
#      https://clawhub.ai/publish-skill.
#
# Usage:
#   ./publish_clawhub.sh <semver> [changelog]            # CLI publish
#   ./publish_clawhub.sh --stage-only                    # produce folder for web upload
#
# Prereqs:
#   - CLI mode: `bun clawhub login` done, `bun` on PATH
#   - Stage-only mode: just `rsync` and `python3` on PATH

set -euo pipefail

STAGE_ONLY=0
if [[ "${1:-}" == "--stage-only" ]]; then
  STAGE_ONLY=1
  VERSION="0.0.0-staged"
  CHANGELOG=""
else
  VERSION="${1:?usage: publish_clawhub.sh <semver> [changelog]  |  --stage-only}"
  if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]; then
    echo "error: version must be semver (got: $VERSION)" >&2
    exit 1
  fi
  CHANGELOG="${2:-Release $VERSION}"
fi

SRC="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d -t reflexio-clawhub-XXXXXX)"
if (( STAGE_ONLY == 0 )); then
  trap 'rm -rf "$STAGE"' EXIT
fi

if (( STAGE_ONLY == 0 )); then
  if ! command -v bun >/dev/null 2>&1; then
    echo "error: bun not found on PATH (install from https://bun.sh)" >&2
    exit 1
  fi

  if ! bun clawhub whoami >/dev/null 2>&1; then
    echo "error: not logged in to clawhub. run: bun clawhub login" >&2
    exit 1
  fi
fi

echo "staging bundle → $STAGE"
# Prune everything that shouldn't be in the published bundle. The web
# uploader at clawhub.ai/publish-skill does NOT read .clawhubignore
# (that's a CLI-only feature), so we pre-filter here and the same stage
# dir works for both CLI publish and web drag-drop.
rsync -a \
  --exclude='node_modules/' \
  --exclude='package-lock.json' \
  --exclude='.DS_Store' \
  --exclude='eval/' \
  --exclude='TESTING.md' \
  --exclude='publish_clawhub.sh' \
  --exclude='.clawhubignore' \
  "$SRC"/ "$STAGE"/

echo "generating top-level SKILL.md from skill/SKILL.md"
python3 - "$STAGE" <<'PY'
import re
import sys
from pathlib import Path

stage = Path(sys.argv[1])
src_skill = stage / "skill" / "SKILL.md"
body = src_skill.read_text()

m = re.match(r"^---\n(.*?)\n---\n(.*)$", body, re.DOTALL)
if not m:
    sys.exit(f"error: {src_skill} has no YAML frontmatter")
orig_frontmatter, orig_body = m.group(1), m.group(2)

# Pull the original description verbatim (multi-line safe).
desc_match = re.search(
    r'^description:\s*"((?:[^"\\]|\\.)*)"', orig_frontmatter, re.MULTILINE
)
if not desc_match:
    sys.exit("error: could not find description in skill/SKILL.md frontmatter")
description = desc_match.group(1)

frontmatter = (
    "---\n"
    "name: reflexio\n"
    f'description: "{description}"\n'
    "metadata:\n"
    "  openclaw:\n"
    "    homepage: https://github.com/reflexio-ai/reflexio/tree/main/reflexio/integrations/openclaw\n"
    "    emoji: 🧠\n"
    "    requires:\n"
    "      bins: [reflexio]\n"
    "---\n"
)

privacy = """
## Privacy & Data Collection — Read This First

**This skill causes the agent to automatically capture conversations and
forward them to a local Reflexio server for LLM-based extraction.** Read this
before enabling — there are two distinct network hops and you need to
understand both.

### Credential requirement (not declared in skill metadata)

The skill's registry metadata declares no required environment variables and
the hook reads none. **But the end-to-end system does require an LLM provider
API key**, and you WILL be asked for one during First-Use Setup.

- Step 2 of First-Use Setup below runs `reflexio setup openclaw`, which opens
  an **interactive wizard** prompting you to choose an LLM provider (OpenAI,
  Anthropic, Gemini, DeepSeek, OpenRouter, and several others) and paste an
  API key.
- The key is stored in `~/.reflexio/.env` and read by the local Reflexio
  server during extraction. **The hook itself never reads the key** (the hook
  has no environment variable access and no filesystem config reads — both
  enforced in `handler.js`).
- If you want fully offline operation, point the wizard at a local LLM
  (Ollama at `http://127.0.0.1:11434`, LM Studio, vLLM, etc.) instead of a
  hosted provider — the wizard accepts any LiteLLM-compatible base URL.

**Why the metadata doesn't declare it:** ClawHub's `metadata.openclaw.requires.env`
describes environment variables the hook's own code path reads. The hook is
deliberately stateless at the credential level, so listing anything there
would be inaccurate. The dependency is at the *backend server* level, one
hop away. This disclosure is here instead of in the metadata because prose
is the right place to explain the distinction.

### Network hops — two of them

**Hop 1: the hook → the local Reflexio server (always localhost).**
The hook is hard-pinned to `http://127.0.0.1:8081`. It communicates via native
`fetch()` with no configuration knobs; the destination is a hardcoded constant
in `handler.js`. It reads zero environment variables and zero configuration
files. This hop cannot leave your machine.

**Hop 2: the local Reflexio server → an LLM provider (may leave your
machine).** The server uses an LLM provider (OpenAI, Anthropic, Gemini,
DeepSeek, etc.) to extract playbooks and profiles from captured conversations.
That provider is configured in `~/.reflexio/.env`. **If you configured an
external provider, excerpts of your conversations will be sent to that
provider** as part of extraction — trigger text, sample content, and enough
context for the extractor to produce a useful summary. The primary full
conversation text stays in your local SQLite database at `~/.reflexio/`, but
the extracted summaries and illustrative excerpts traverse whatever the LLM
provider's network path is.

**If you want fully offline operation**, configure the local server to use a
local LLM (Ollama, LM Studio, vLLM, etc.) before enabling this skill. Do not
rely on the hook's localhost pinning as a privacy guarantee for the system as
a whole — that only bounds the hook, not the server behind it.

**What is captured:** full user and assistant messages; every tool call, input,
and output (including failed tool calls and exact error strings); self-correction
text from the assistant's own output. None of this is scrubbed for PII,
credentials, file paths, or API outputs. If you work on sensitive tasks,
disable the hook before starting them, or tell the agent mid-task.

**How to disable:**

- Per-session, stop automatic capture: `openclaw hooks disable reflexio-context`
- Remove the integration entirely: `reflexio setup openclaw --uninstall`
- Search-only mode (keep retrieval, drop auto-publish): edit `hook/handler.js`,
  remove the `command:stop` handler, and reinstall
- Sensitive task: tell the agent "don't log this" — the workspace rule instructs
  it to stop all capture and search calls for the rest of the session

**Transparency contract:** the agent is instructed to tell you Reflexio is
active on the first turn of each session, disclose what is captured AND
mention that the local server may forward excerpts to an LLM provider, and
honor session-level opt-outs. This is enforced by
`~/.openclaw/workspace/reflexio.md`. Edit that file if your deployment needs
different rules (require explicit approval before every publish, offline-only,
etc).

Full data-handling details live in the **Privacy & Data Collection** section
further down in this SKILL.md.

---

"""

first_use = """
## First-Use Setup

Run these commands the first time this skill is invoked. They are idempotent —
skip if `reflexio` is already on PATH, the local Reflexio server is responding
at `http://127.0.0.1:8081/health`, and `openclaw hooks list` shows
`reflexio-context` as `ready`. **Only run them after reading the privacy
disclosure above, and be aware that step 2 is interactive — it will prompt
you to pick an LLM provider and paste an API key for the local server's
extraction pipeline.**

```bash
# 1. Install the reflexio CLI (pipx preferred; pip as fallback).
command -v reflexio >/dev/null 2>&1 \\
  || pipx install reflexio-ai 2>/dev/null \\
  || pip install --user reflexio-ai

# 2. INTERACTIVE: activates the hook, slash commands, and workspace rule in
# OpenClaw. Also prompts you to choose an LLM provider and paste an API key,
# which gets written to ~/.reflexio/.env. That key is what the local server
# uses to extract playbooks from captured conversations. Point at a local LLM
# (Ollama / LM Studio / vLLM) here if you want fully offline operation.
# Before running this, tell the user what will be asked and why.
reflexio setup openclaw

# 3. Start the local Reflexio backend in the background. The hook talks to
# this server via HTTP on 127.0.0.1:8081 — it will NOT start the server for
# you. Tell the user you are doing this before running it.
curl -sf --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1 \\
  || (nohup reflexio services start --only backend --no-reload \\
        >> ~/.reflexio/logs/server.log 2>&1 & \\
      sleep 5)
```

This installs:

- `reflexio-context` hook — HTTP-only capture + per-message playbook
  injection (communicates only with `http://127.0.0.1:8081`)
- `/reflexio-extract` slash command — publish session learnings mid-session
- `/reflexio-aggregate` slash command — consolidate user playbooks into
  shared agent playbooks
- `~/.openclaw/workspace/reflexio.md` — always-active behavioral rule
  (transparency + opt-out handling)

If the local server is unreachable, the hook logs the error and the agent
continues the user's task without Reflexio context that session.

---

"""

(stage / "SKILL.md").write_text(frontmatter + privacy + first_use + orig_body)
print(f"wrote {stage / 'SKILL.md'}")
PY

if (( STAGE_ONLY == 1 )); then
  cat <<EOF

─────────────────────────────────────────────────────────────
Stage ready. Upload via web at https://clawhub.ai/publish-skill

Drag this folder onto the drop zone:
  $STAGE

Form values to use:
  Slug:        reflexio
  Display name: Reflexio
  Version:     <pick a semver, e.g. 1.0.0>
  Tags:        latest
  Changelog:   Initial release

When done, delete the temp folder: rm -rf $STAGE
─────────────────────────────────────────────────────────────
EOF
  exit 0
fi

echo "publishing to clawhub: slug=reflexio version=$VERSION"
(
  cd "$STAGE"
  bun clawhub skill publish . \
    --slug reflexio \
    --version "$VERSION" \
    --tags latest \
    --changelog "$CHANGELOG"
)

echo "done. verify at https://clawhub.ai/skill/reflexio"
