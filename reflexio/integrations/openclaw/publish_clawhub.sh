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

**This skill retrieves cross-session memory from a local Reflexio server and
injects it as context before the agent responds. Writing new learnings is
explicit — it only happens when the agent runs `/reflexio-extract`, which
applies the extraction rubric in its own session and upserts playbooks via
direct CRUD.** Read this before enabling.

### No LLM provider API key is required

The Reflexio server does NOT perform LLM-based extraction for this
integration. Playbook extraction runs in your agent's own LLM session
(whatever provider OpenClaw itself uses). The server only performs CRUD and
semantic search against its local store, so `reflexio setup openclaw` does
not prompt for any LLM provider key.

If a previous version of this integration asked you to store an API key in
`~/.reflexio/.env`, that key is no longer consulted by the openclaw code
path. You can leave it in place (other integrations may still use it) or
remove it.

### Single network hop — localhost only

The hook is hard-pinned to `http://127.0.0.1:8081`. It communicates via native
`fetch()` with no configuration knobs; the destination is a hardcoded constant
in `handler.js`. It reads zero environment variables and zero configuration
files. This hop cannot leave your machine.

The `/reflexio-extract` slash command, when invoked, sends playbook CRUD
calls (search / add / update) to the same local server. Those calls carry the
trigger / instruction / pitfall / content fields that the agent extracted
from the current conversation — the raw transcript is not forwarded.

### What gets written to the local store

Only the fields the agent produces during `/reflexio-extract`:

- `content` — concise natural-language summary of one learning
- `trigger` — when the rule applies
- `instruction` / `pitfall` / `rationale` — structured fields

Raw conversation transcripts, tool outputs, and file paths do NOT end up in
the local store unless you quote them verbatim into one of the fields above.
If you work on sensitive tasks, omit those strings from the extraction, or
skip `/reflexio-extract` entirely for that session.

**How to disable:**

- Per-session, stop context injection: `openclaw hooks disable reflexio-context`
- Remove the integration entirely: `reflexio setup openclaw --uninstall`
- Sensitive task: tell the agent "don't log this" — the workspace rule
  instructs it to stop all search and extract calls for the rest of the session

**Transparency contract:** the agent is instructed to tell you Reflexio is
active on the first turn of each session, disclose that it retrieves
past-session memory from a local server, and honor session-level opt-outs.
This is enforced by `~/.openclaw/workspace/reflexio.md`. Edit that file if
your deployment needs different rules.

---

"""

first_use = """
## First-Use Setup

Run these commands the first time this skill is invoked. They are idempotent —
skip if `reflexio` is already on PATH, the local Reflexio server is responding
at `http://127.0.0.1:8081/health`, and `openclaw hooks list` shows
`reflexio-context` as `ready`. No LLM provider API key is required.

```bash
# 1. Install the reflexio CLI (pipx preferred; pip as fallback).
command -v reflexio >/dev/null 2>&1 \\
  || pipx install reflexio-ai 2>/dev/null \\
  || pip install --user reflexio-ai

# 2. Activates the hook, slash command, and workspace rule in OpenClaw.
# Prompts only for a storage backend (local SQLite is the default). Before
# running this, tell the user what will be asked and why.
reflexio setup openclaw

# 3. Start the local Reflexio backend in the background. The hook talks to
# this server via HTTP on 127.0.0.1:8081 — it will NOT start the server for
# you. Tell the user you are doing this before running it.
curl -sf --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1 \\
  || (nohup reflexio services start --only backend \\
        >> ~/.reflexio/logs/server.log 2>&1 & \\
      sleep 5)
```

This installs:

- `reflexio-context` hook — search-only, HTTP-only (communicates with
  `http://127.0.0.1:8081` for bootstrap profile + per-message playbook
  injection). It never buffers conversations.
- `/reflexio-extract` slash command — applies the v3.0.0 rubric in your own
  agent session, searches for existing playbooks, and adds or updates via
  the `reflexio user-playbooks` CLI.
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
