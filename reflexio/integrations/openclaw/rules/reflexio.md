# Reflexio: Cross-Session Memory

This agent uses Reflexio to remember corrections and preferences across sessions.

## How It Works

A `message:received` hook automatically runs `reflexio search` on every message you receive. When relevant corrections or preferences exist, they are injected as context before your response via a file named `REFLEXIO_CONTEXT.md`.

## What You MUST Do

1. **When you see a block titled "REFLEXIO_CONTEXT"** — read it carefully and follow every instruction it contains. These are explicit corrections from the user's past sessions. Behavior Corrections tell you what to do differently. User Preferences tell you how the user works. Agent Playbooks are shared corrections from all instances.

2. **If no Reflexio context appears for a task** — the hook may not have found relevant results, or the server may be starting up. Run a manual search before beginning work:
   ```
   reflexio search "<the user's task or request>"
   ```
   If the command fails with a connection error and `REFLEXIO_URL` is unset or points to localhost/127.0.0.1 (local server), start the server in the background and continue working:
   ```bash
   nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
   ```
   If `REFLEXIO_URL` points to a remote server, do not attempt to start it — proceed without Reflexio results.

3. **Never mention Reflexio to the user** unless they ask about it. It should be invisible.

4. **If Reflexio is unavailable, proceed normally** — it enhances but never blocks your work. Never degrade the core agent experience because of Reflexio issues.

5. **Never ask the user to manage the Reflexio server** — infrastructure is the integration's responsibility. Start the server silently if needed; never surface server status to the user.
