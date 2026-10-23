# OpenClaw lab

A throwaway OpenClaw + Reflexio environment for developing and verifying the
plugin against a real OpenClaw, on a shared machine.

Everything lives under `$OPENCLAW_LAB_DIR` (default
`${TMPDIR:-/tmp}/reflexio-openclaw-lab`): its own OpenClaw install and state,
Reflexio's HOME and SQLite data, and randomly chosen free ports. It never reads
or writes `~/.openclaw` or `~/.reflexio`, and never uses a shared backend port.

```bash
cd open_source/reflexio   # or the Reflexio repo root
uv sync
reflexio/integrations/openclaw/lab/lab.sh up           # ~1 min first time
uv run --no-sync python reflexio/integrations/openclaw/lab/scenarios.py
reflexio/integrations/openclaw/lab/lab.sh down          # --wipe deletes the lab
```

Needs Node `>=24.22.1` (picked up from nvm when available), `npm`, `uv` and
`curl`. `OPENCLAW_LAB_VERSION` selects the OpenClaw release (default: the
version the plugin was last qualified on).

## What `up` builds

- OpenClaw installed with `npm --prefix` into the lab, state isolated with
  `OPENCLAW_STATE_DIR` / `OPENCLAW_CONFIG_PATH`.
- A Reflexio backend (`reflexio services start --only backend --storage
  sqlite`) with `MOCK_LLM_RESPONSE=true`, run from the lab directory (so no
  developer `.env` is read) with its colocated embedding service on a lab port.
- `stub_llm.py`, an OpenAI-compatible endpoint registered as the agent's model.
  It records every request to `llm-requests.jsonl` and cites any `[oc:…]` tags
  in the latest user message, so tests can see exactly what reached the model.
- The plugin, installed through the **built SDK wheel’s production setup path**,
  outside the source checkout. This verifies bundled source, compilation, host
  configuration migration and actual plugin load together. No npm plugin release
  or in-place source installation is used.

`lab.sh reinstall` rebuilds the SDK wheel and reinstalls after a plugin change.
`lab.sh turn "…"` runs one embedded turn; `lab.sh oc …` runs any `openclaw`
command against the lab; `eval "$(lab.sh env)"` gives you the same from a shell.
`lab.sh gateway [stop]` runs a Gateway, which `session_end` (e.g. `/new`) needs;
while it runs, embedded `--local` turns refuse to start.

## Scenarios

`scenarios.py` asserts on what landed in Reflexio or in the model request,
never on exit codes alone (a fail-open plugin exits 0 whether or not it works):

| Scenario | Checks |
| --- | --- |
| load | `plugins inspect --runtime`: loaded, 3 hooks, `/reflexio` registered |
| publish | the turn is stored under the plugin's user id and `request_id` |
| inject + attribute | a seeded preference reaches the model prompt; the stored reply carries it in `retrieved_learnings` and the citation is counted |
| duplicate publish is recognized | the server's refusal of a re-sent `request_id` matches what the plugin treats as already delivered |
| backend down is harmless | with an unreachable backend the turn is unchanged and the failure is reported |
| recursion guard | `OPENCLAW_SMART_INTERNAL=1` (set by Reflexio's extraction CLI) disables the plugin |
| session_end finalizes | through a Gateway, `/new` calls `finalize_session` |

Not covered end to end (the stub model makes no tool calls and the CLI does
not dispatch plugin commands): `/reflexio` replies, tool loops, steering
messages, compaction, the `/new` race and slow publishes. `plugin/tests/`
covers each of those against recorded shapes. Gateway channels are untested.

## Real model (`up --real [MODEL]`)

Uses a ChatGPT/Codex subscription for both the agent and Reflexio's extraction
(through `openclaw infer`; `MODEL` defaults to `openai/gpt-5.6-luna`). Use a
separate `OPENCLAW_LAB_DIR`: a stub lab must be wiped before switching.

On first `up --real` the lab runs `openclaw models auth login --provider openai
--device-code` against its own state: open the printed URL, enter the code, and
the login lands in the lab's auth store only. It deliberately does **not**
import `~/.codex`: that would share a rotating refresh token with the Codex
CLI, and a refresh inside the lab could sign the CLI out. The login needs a
TTY; under automation wrap it in `script -qfec "lab.sh oc models auth login
--provider openai --device-code" login.log`.
