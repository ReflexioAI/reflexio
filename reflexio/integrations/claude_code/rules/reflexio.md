# Reflexio: Cross-Session Memory

This project uses Reflexio to remember corrections and preferences across sessions.

## How It Works

A `UserPromptSubmit` hook automatically runs `reflexio search` on every message you receive. When relevant corrections or preferences exist, they are injected as context before your response.

## What You MUST Do

1. **When you see a block titled "IMPORTANT: Apply These Corrections (from Reflexio)"** — read it carefully and follow every instruction it contains. These are explicit corrections from the user's past sessions. Behavior Corrections tell you what to do differently. User Preferences tell you how the user works.

2. **If no Reflexio context appears for a task** — the hook may not have found relevant results, or the server may be starting up. Run a manual search before beginning work:
   ```
   reflexio search "<the user's task or request>"
   ```
   If the command fails with a connection error and `REFLEXIO_URL` is unset or points to localhost/127.0.0.1 (local server), start the server in a background subagent and continue working:
   ```
   Agent(prompt="Start the Reflexio server: nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 & sleep 5 && reflexio status check", run_in_background=true)
   ```
   If `REFLEXIO_URL` points to a remote server, do not attempt to start it — proceed without Reflexio results.

3. **Never mention Reflexio to the user** unless they ask about it. It should be invisible.
