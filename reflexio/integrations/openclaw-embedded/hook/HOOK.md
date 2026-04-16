---
name: reflexio-embedded
description: "Reflexio Embedded hook: TTL sweep on bootstrap; spawn reflexio-extractor sub-agent at session boundaries."
metadata:
  openclaw:
    emoji: "🧠"
    events:
      - "agent:bootstrap"
      - "session:compact:before"
      - "command:stop"
      - "command:reset"
---
