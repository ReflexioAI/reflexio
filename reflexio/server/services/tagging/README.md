# tagging

Compact post-generation entity tagging capability.

- `service.py` tags profiles, playbooks, and persisted agent-success evaluation
  summaries with their configured tagging prompts. Evaluation tagging receives
  summary fields only; it never reloads transcripts or includes identifiers.
- `tagging_scheduler.py` coalesces tagging by organization, user, and agent
  version, then rebuilds request context for background execution.

This package intentionally does not use `components/`: the service and scheduler are the only module responsibilities.
