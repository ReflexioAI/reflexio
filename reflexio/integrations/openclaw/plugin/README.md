# @reflexioai/openclaw-smart

OpenClaw plugin: cross-session memory via [Reflexio](https://github.com/reflexio-ai/reflexio).
Before each turn it injects the preferences and skills Reflexio has learned for
this agent workspace; after each turn it publishes the conversation so Reflexio
can learn from it.

```bash
reflexio setup openclaw
```

Setup installs the native plugin bundled with the installed Reflexio SDK,
points it at the selected backend, and grants conversation-hook access.
It uses OpenClaw's existing Node runtime to build a temporary JavaScript
artifact; no npm package release or compiler download is required. URL and API
key come from the SDK environment selected during setup.

In a conversation, `/reflexio status` shows whether the backend is reachable
and how many searches or publishes failed; `/reflexio show` lists what has been
learned; `/reflexio learn` asks Reflexio to extract from this session now.

Requires OpenClaw `>=2026.9.33`. See the
[integration README](https://github.com/reflexio-ai/reflexio/tree/main/reflexio/integrations/openclaw)
for how it works and how to develop it.

## Search and publication policy

The native plugin retains these existing configuration names:

| Field | Default | Meaning |
| --- | --- | --- |
| `search.top_k` | 3 | Result count per search arm, from 1 through the API maximum of 100. |
| `search.min_prompt_length` | 5 | Minimum trimmed prompt length; zero admits every nonempty prompt. |
| `search.timeout_ms` | 3000 | Positive search deadline, with an additional second reserved by the host hook. |
| `publish.max_content_length` | 10000 | Positive maximum redacted characters retained per published message; a truncation marker may follow. |

Set them under `plugins.entries.reflexio-openclaw-smart.config`, alongside
`url` and `apiKey`. Malformed numeric policy fails before hook registration.
Prompt and content limits count complete Unicode code points. Custom content
limits do not truncate the local text used to resolve citations.
The older manifest advertised a 5000ms search default but its Python hook did
not consume these fields. Select 5000ms explicitly if desired; the native
runtime keeps its existing 3000ms default.

Publication and finalization share the hook's absolute deadline. Each request
receives only the remaining time; there is no separate fixed per-request cap.
An unacknowledged publication remains queued for a later event with its original
request identity, rather than spinning through configurable retries or releasing
later turns out of order.

Setup preserves the four supported policies and removes obsolete fields when
reinstalling. For a manual upgrade, remove the obsolete
`publish.batch_size`, `publish.max_retries`, `server` and `extraction` plugin
settings and the separate `reflexio-embedded` plugin entry. The four policy fields
above keep their existing spelling. Extraction model selection belongs to the
backend's configured model (`openclaw/<model>`), not this transport plugin.
Keep an operator copy of the old configuration before removing retired fields.

Numeric defaults and bounds come from the shipped plugin manifest, shared by
registration and runtime validation. The per-message retained prefix is capped
at 999988 code points so its 12-code-point truncation marker fits the backend's
1000000-character field limit. An API-contract test checks both this bound and
the search maximum against the actual route models.
