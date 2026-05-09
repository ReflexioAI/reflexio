"""GEPA-driven playbook content optimizer.

Top-level entry points exported from this package:

- ``PlaybookOptimizer.optimize(target)`` — runs one paired-rollout search
  for a single playbook and persists the result.
- ``PlaybookOptimizationScheduler.get_instance()`` — singleton that
  debounces enqueues from upstream services (the aggregator and the
  user-playbook generation service) and dispatches each ``optimize`` call
  on a worker thread.
- ``PlaybookOptimizationTarget`` — the (kind, target_id) pair both APIs
  consume.

The full doc lives at ``docs/playbook_optimizer_assistant_backends.md``.
"""

from .optimizer import PlaybookOptimizationTarget, PlaybookOptimizer
from .scheduler import PlaybookOptimizationScheduler

__all__ = [
    "PlaybookOptimizationScheduler",
    "PlaybookOptimizationTarget",
    "PlaybookOptimizer",
]
