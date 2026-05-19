# learn

**When to use:** The user wants immediate extraction of skills/preferences from the current session — e.g., after a major correction, before context compaction, or to test the loop.

## What to do
Run: `bash "$OPENCLAW_PLUGIN_ROOT/scripts/cli.sh" learn --note "<one-line description of why>"`

## Expected output
- "published N interactions, extraction queued" — success
- "reflexio unreachable" — backend down; suggest `bash "$OPENCLAW_PLUGIN_ROOT/scripts/cli.sh" restart`
