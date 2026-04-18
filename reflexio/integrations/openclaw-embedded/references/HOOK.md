---
name: reflexio-embedded
description: "Reflexio Embedded hook: TTL sweep on bootstrap; spawn reflexio-extractor sub-agent at session boundaries."
metadata:
  openclaw:
    emoji: "🧠"
    events:
      - "before_agent_start"
      - "before_compaction"
      - "before_reset"
      - "session_end"
---

> **Note:** This file is no longer discovered by the Openclaw CLI. Hooks are
> now registered programmatically from `../index.ts` via
> `definePluginEntry({ register(api) { api.on(...) } })`. This doc is kept
> only for human reference; see `../index.ts` and `./handler.ts` for the
> actual behaviour.
