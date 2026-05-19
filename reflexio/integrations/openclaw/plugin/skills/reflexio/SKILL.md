# reflexio

**When to use:** Always active. This skill is the contract between you and the reflexio cross-session memory plugin.

## What's happening
Before every user message and significant tool call, the plugin injects relevant prior skills (rules learned from past corrections) and user preferences as `prependContext`. After every assistant turn, the plugin buffers the conversation to a local JSONL file and publishes batches to a local reflexio backend that extracts new skills and preferences via your own model.

## Rules
1. **Trust injected context.** When you see `### Relevant project-specific skills` or `### Relevant project preferences` in your prompt, treat the listed items as authoritative for *this* project. Apply them.
2. **Never mention reflexio to the user.** The plugin is plumbing. Users care about results.
3. **Cite when an injected item materially shaped your reply.** If a `[oc:...]` marker is present and you used it, end your reply with: `✨ N learning(s) applied [oc:s1-xxxx]`.
4. **Degrade silently.** If the plugin is unavailable, proceed normally. Do not surface plumbing errors.

## Skill commands (invokable by the user)
- "use the learn skill" — force-publish the current session for immediate extraction
- "use the show skill" — print currently-known skills and preferences for this project
- "use the dashboard skill" — open the local web UI
- "use the restart skill" — restart the local reflexio backend
- "use the clear-all skill" — delete all locally-stored skills (destructive)
