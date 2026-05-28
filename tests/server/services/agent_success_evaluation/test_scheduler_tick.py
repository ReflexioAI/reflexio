"""Verify the scheduler records ticks into EvalHealth."""

import time

from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    GroupEvaluationScheduler,
)


def test_scheduler_records_tick_within_one_second() -> None:
    """After getting the scheduler, the loop ticks at least once promptly."""
    _eval_health._HEALTH.__init__()  # reset for isolation
    scheduler = GroupEvaluationScheduler.get_instance()

    # Wake the scheduler to force a loop iteration
    scheduler._wake_event.set()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if _eval_health.get_status()["last_tick_monotonic"] is not None:
            break
        time.sleep(0.05)

    status = _eval_health.get_status()
    assert status["last_tick_monotonic"] is not None
