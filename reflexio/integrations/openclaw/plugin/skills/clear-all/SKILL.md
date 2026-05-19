# clear-all

**When to use:** The user explicitly asks to delete all locally-stored skills/preferences. This is destructive and unrecoverable.

## What to do
First confirm with the user: "This will delete ALL locally-learned skills and preferences. Are you sure?" If they confirm with yes/y, run:

`bash "$OPENCLAW_PLUGIN_ROOT/scripts/cli.sh" clear-all --yes`
