# /reflexio/server/services/tagging
Description: Deferred tagging of profiles, playbooks, and agent-success evaluation summaries.

## Main Entry Points


- `service.py` tags profiles, playbooks, and persisted agent-success evaluation
  summaries with their configured tagging prompts. Evaluation tagging receives
  summary fields only; it never reloads transcripts or includes identifiers.
- `tagging_scheduler.py` coalesces tagging by organization, user, and agent
  version, then rebuilds request context for background execution.

## Purpose


Attach configured tags after generation so retrieval and evaluation views can use them.

## Architecture Pattern


`tagging_scheduler.py` coalesces organization/user/version work, rebuilds request context, and invokes `service.py` against persisted entities.

## Requirements / Problems to Avoid


This package intentionally does not use `components/`: the service and scheduler are the only module responsibilities.
