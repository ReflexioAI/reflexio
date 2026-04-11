---
paths:
  - "reflexio/**/*.py"
---

# Reflexio Architecture Guardrails

## Reflexio Instance
- **NEVER** instantiate `Reflexio()` directly in API endpoints
- **ALWAYS** use `get_reflexio()` from `server/cache/`

## Storage
- **NEVER** import storage implementations directly
- **ALWAYS** use `request_context.storage` (type: `BaseStorage`)

## LLM
- **NEVER** import `OpenAIClient` or `ClaudeClient` directly
- **ALWAYS** use `LiteLLMClient` (uses LiteLLM for multi-provider support)

## Prompts
- **NEVER** hardcode prompts
- **ALWAYS** use `request_context.prompt_manager.render_prompt(prompt_id, variables)`

## Config
- `tool_can_use` lives at root `Config` level — shared across success evaluation and feedback extraction (NOT per-`AgentSuccessConfig`)
