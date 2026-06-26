# tagging

Compact post-generation entity tagging capability.

- `service.py` tags profiles and playbooks with the configured tagging prompts.
- `tagging_scheduler.py` debounces post-publish tagging and rebuilds request context for background execution.

This package intentionally does not use `components/`: the service and scheduler are the only module responsibilities.
