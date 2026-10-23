# @reflexioai/openclaw-smart

OpenClaw plugin: cross-session memory via [Reflexio](https://github.com/reflexio-ai/reflexio).
Before each turn it injects the preferences and skills Reflexio has learned for
this agent workspace; after each turn it publishes the conversation so Reflexio
can learn from it.

```bash
openclaw plugins install @reflexioai/openclaw-smart --accept-capabilities
openclaw config set plugins.entries.reflexio-openclaw-smart.hooks.allowConversationAccess true
openclaw config set plugins.entries.reflexio-openclaw-smart.config.url http://127.0.0.1:8061/
openclaw gateway restart
```

For managed or self-hosted Reflexio also set `config.apiKey` (a secret
reference such as `'${REFLEXIO_API_KEY}'` keeps the key out of the config file).
`reflexio setup openclaw` does all of this for you.

In a conversation, `/reflexio status` shows whether the backend is reachable
and how many searches or publishes failed; `/reflexio show` lists what has been
learned; `/reflexio learn` asks Reflexio to extract from this session now.

Requires OpenClaw `>=2026.9.33`. See the
[integration README](https://github.com/reflexio-ai/reflexio/tree/main/reflexio/integrations/openclaw)
for how it works and how to develop it.
