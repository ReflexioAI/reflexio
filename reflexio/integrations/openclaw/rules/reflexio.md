# Reflexio: Cross-Session Memory

This agent uses Reflexio to remember corrections and preferences across sessions. **The hook reads past-session memory from a local Reflexio server at `127.0.0.1:8081` and injects it as context.** It does not buffer conversations or upload anything automatically. When you want to persist a learning, run `/reflexio-extract` — that slash command applies the extraction rubric in your own agent session and upserts playbooks into Reflexio via direct CRUD. No LLM call is made by the Reflexio server, and no LLM provider API key is required for this integration.

## How It Works

A `message:received` hook runs `reflexio search` on every message you receive. When relevant past-session playbooks or profiles exist, they are injected as context before your response via a file named `REFLEXIO_CONTEXT.md`. A `agent:bootstrap` hook injects the user profile (if one exists) as `REFLEXIO_USER_PROFILE.md`.

## What You MUST Do

1. **When you see a block titled "REFLEXIO_CONTEXT"** — read it carefully and follow every instruction it contains. These are rules extracted from the user's past sessions. Behavior Corrections tell you what to do differently. User Preferences tell you how the user works. Agent Playbooks (if any exist from other integrations) are shared corrections.

2. **Be transparent about Reflexio.** On the first turn of each session, briefly tell the user: Reflexio cross-session memory is active; it reads past-session playbooks from a local Reflexio server at `127.0.0.1:8081`; new learnings are only written when you run `/reflexio-extract`. Two sentences is plenty. If the user asks to disable it, point them to `openclaw hooks disable reflexio-context` or `reflexio setup openclaw --uninstall`, and stop using Reflexio for the rest of the session.

3. **If no Reflexio context appears for a task** — the hook may not have found relevant results, or the local server may be starting up. Run a manual search before beginning work:
   ```
   reflexio search "<the user's task or request>"
   ```
   If the command fails with a connection error, **tell the user you're starting the local Reflexio server in the background** before running:
   ```bash
   nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
   ```

4. **Persist learnings at the right moments.** After the user corrects you, or after you complete a substantive step with a non-obvious recipe, run `/reflexio-extract`. The command handles the extraction rubric, search, and upsert for you. Do not wait for the end of the session — extract at natural milestones so learnings aren't lost to interruption.

5. **If Reflexio is unavailable, proceed normally** — it enhances but never blocks your work. Never degrade the core agent experience because of Reflexio issues.

6. **Respect user objections.** If the user says they don't want their conversation logged, or that they're working on a sensitive task, stop all Reflexio calls for the remainder of the session. Do not launch the local server, do not run `/reflexio-extract`, and do not run `reflexio search`. Offer to disable the hook entirely if they'd like a permanent opt-out.
