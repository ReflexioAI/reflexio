# Manual Testing Guide

End-to-end manual validation of `openclaw-embedded`. Run this before each release.

## Prerequisites

- A clean Openclaw instance (fresh workspace, or separate test workspace)
- Node and openclaw CLI on PATH
- An embedding provider configured (optional but recommended for full coverage)
- A terminal with `bats` installed (for running shell unit tests)

## 1. Unit tests

From the plugin directory:

```bash
bats tests/test_reflexio_write.bats
```

Expected: all tests pass. If any fail, stop — fix before proceeding.

## 2. Hook smoke test

```bash
node hook/smoke-test.js
```

Expected: all PASS lines printed, no FAIL.

## 3. Install plugin

```bash
./scripts/install.sh
```

Expected verification output:
```
  ✓ hook registered
  ✓ cron registered
```

If any ⚠ warnings, investigate before moving on.

## 4. First-use bootstrap

- Open a new Openclaw agent session.
- Say: "test the reflexio-embedded skill setup".
- Expected: the agent invokes the skill, runs probing commands, asks for approval to configure active-memory and extraPath.
- Approve each step.
- Verify `.reflexio/.setup_complete_<agentId>` marker exists.

## 5. Flow A — profile capture

In the agent session:

- Say: "By the way, I'm vegetarian."
- Expected: agent writes `.reflexio/profiles/diet-vegetarian-<nanoid>.md` with:
  - Frontmatter: `type: profile`, `id: prof_*`, `ttl: infinity`, `expires: never`
  - Body: ~1-sentence description

```bash
cat .reflexio/profiles/diet-*.md
```

Verify the file matches expectations.

## 6. Flow B — playbook capture

- Say: "Write a commit message for 'fix auth bug'."
- Expected: agent writes a commit message (may include Co-Authored-By).
- Say: "No, don't add Co-Authored-By trailers."
- Expected: agent rewrites without the trailer.
- Say: "Perfect, commit it."
- Expected: agent writes `.reflexio/playbooks/commit-no-ai-attribution-<nanoid>.md`.

```bash
cat .reflexio/playbooks/commit-*.md
```

Verify frontmatter + `## When` / `## What` / `## Why` sections.

## 7. Flow C — batch extraction at session boundary

- Have a longer conversation (5+ turns) covering facts and corrections.
- Trigger `command:stop` (or let the session compact naturally).
- Expected: the hook fires a `reflexio-extractor` sub-agent.

Inspect via:
```bash
openclaw tasks list --agent reflexio-extractor
```

Expected: a completed task record exists.

Check `.reflexio/profiles/` and `.reflexio/playbooks/` — new files should have appeared corresponding to any facts/corrections the agent missed in-session.

## 8. Retrieval validation

Start a new session. Ask: "What do you know about my diet?"

- With Active Memory enabled: expected answer references "vegetarian" from the captured profile.
- With Active Memory disabled: expected agent calls `memory_search` per SKILL.md fallback, then answers.

## 9. Consolidation (on-demand)

After accumulating 10+ entries across sessions, run:

```
/skill reflexio-consolidate
```

Expected:
- Agent delegates to `reflexio-consolidator` sub-agent.
- Returns a runId.

Inspect:
```bash
openclaw tasks list --agent reflexio-consolidator
```

Check `.reflexio/` before and after — duplicate or overlapping entries should be collapsed, with `supersedes` frontmatter on merged files.

## 10. TTL sweep

- Create a profile with short TTL:
  ```bash
  echo "temp fact" | ./scripts/reflexio-write.sh profile test-temp one_day
  ```
- Manually edit its `expires` to a past date:
  ```bash
  # Edit .reflexio/profiles/test-temp-*.md — set expires: 2020-01-01
  ```
- Restart the agent session (triggers `agent:bootstrap` hook).
- Expected: the expired profile is deleted.

## 11. Degradation: no Active Memory

- Disable active-memory: `openclaw plugins disable active-memory`
- Restart gateway.
- Start new session, ask a question whose answer needs a captured profile.
- Expected: agent calls `memory_search` explicitly (per SKILL.md fallback), then answers.
- Re-enable: `openclaw plugins enable active-memory`.

## 12. Degradation: no embedding provider

- Unset embedding env vars.
- Restart gateway.
- Ask the agent something whose answer needs retrieval.
- Expected: retrieval works via FTS only (quality lower but functional).

## 13. Uninstall

```bash
./scripts/uninstall.sh
```

Verify:
- `openclaw hooks list` does NOT include `reflexio-embedded`.
- `openclaw cron list` does NOT include `reflexio-embedded-consolidate`.
- `~/.openclaw/workspace/skills/reflexio-embedded/` does not exist.
- `.reflexio/` in the workspace is preserved.

Run `./scripts/uninstall.sh --purge` in a test workspace to verify `.reflexio/` is deleted.

## Test report template

When testing for a release:

```
Tested: <date> <tester>
Openclaw version: <version>
Unit tests: <pass/fail>
Hook smoke test: <pass/fail>
Flow A: <pass/fail — notes>
Flow B: <pass/fail — notes>
Flow C: <pass/fail — notes>
Retrieval: <pass/fail — notes>
Consolidation: <pass/fail — notes>
TTL sweep: <pass/fail — notes>
Degradation modes: <pass/fail — notes>
Uninstall: <pass/fail — notes>
```
