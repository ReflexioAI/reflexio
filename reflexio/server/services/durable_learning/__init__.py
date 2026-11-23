"""Durable execution exports; pure admission imports do not load model workers."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflexio.server.services.durable_learning.scheduler import (
        DurableLearningScheduler,
        maybe_start_durable_learning,
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

__all__ = [
    "DurableLearningScheduler",
    "DurableLearningWorker",
    "maybe_start_durable_learning",
]


def __getattr__(name: str) -> Any:
    if name == "DurableLearningWorker":
        from reflexio.server.services.durable_learning.worker import (
            DurableLearningWorker,
        )

        return DurableLearningWorker
    if name in {"DurableLearningScheduler", "maybe_start_durable_learning"}:
        from reflexio.server.services.durable_learning import scheduler

        return getattr(scheduler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
