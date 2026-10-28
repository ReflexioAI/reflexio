from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from functools import partial

from reflexio.models.config_schema import OptimizerSchedulingPolicy
from reflexio.server.callback_executor import submit_callback
from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.work_scope import (
    WorkScope,
    WorkScopeError,
    bind_work_scope,
    current_project_id,
)

from .optimizer import PlaybookOptimizationRunStatus, PlaybookOptimizationTarget

logger = logging.getLogger(__name__)

# Queue coalescing and abort policy share this attribution identity.
ScheduleKey = tuple[str, str | None, str, int]
ScheduledCallback = Callable[[], PlaybookOptimizationRunStatus | None]


class PlaybookOptimizationScheduler:
    """Admit optimizer work to the bounded process callback executor.

    The executor coalesces queued work by org/project/target and serializes
    matching active work. No timer or elapsed-time debounce window is needed.
    Captured failure policy still suppresses new admission during abort cooldown.
    Callbacks retain the executor's existing best-effort overflow behavior;
    this is neither a durable queue nor cross-process deduplication.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PlaybookOptimizationScheduler:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._abort_counts: dict[ScheduleKey, tuple[int, float]] = {}

    def enqueue(
        self,
        *,
        org_id: str,
        target: PlaybookOptimizationTarget,
        callback: ScheduledCallback,
        policy: OptimizerSchedulingPolicy | None = None,
    ) -> None:
        if policy is None:
            policy = OptimizerSchedulingPolicy()
        # Resolve scope on the enqueueing thread, before its request ends.
        key: ScheduleKey = (
            org_id,
            current_project_id(),
            target.kind,
            target.target_id,
        )
        with self._mutex:
            if self._cooldown_remaining_locked(key, time.monotonic()) > 0:
                logger.info(
                    "Skipping playbook optimization enqueue during cooldown: %s", key
                )
                return
        submit_callback(
            f"playbook-opt-{key[2]}-{key[3]}",
            partial(self._run_callback, key, callback, policy),
            coalesce_key=("playbook-optimization", *key),
        )

    def _run_callback(
        self,
        key: ScheduleKey,
        callback: ScheduledCallback,
        policy: OptimizerSchedulingPolicy,
    ) -> None:
        try:
            with bind_work_scope(WorkScope(org_id=key[0], project_id=key[1])):
                status = callback()
            if status == "aborted":
                self._record_abort(key, policy)
            elif status in {"completed", "skipped"}:
                self._clear_abort_state(key)
        except WorkScopeError:
            # NOT an operational failure: the run could not be attributed to a
            # project, so tolerating it would misattribute or silently skip a
            # tenant write. Escalated rather than propagated — see WorkScopeError.
            logger.exception(
                "Playbook optimization scope binding failed for key=%s", key
            )
            capture_anomaly(
                "playbook_optimizer.work_scope_failed",
                level="error",
                org_id=key[0],
                project_id=key[1],
                target_kind=key[2],
            )
            self._record_abort(key, policy)
        except Exception:
            logger.exception("Playbook optimization callback failed for key=%s", key)
            self._record_abort(key, policy)

    def _cooldown_remaining_locked(self, key: ScheduleKey, now: float) -> float:
        state = self._abort_counts.get(key)
        if state is None:
            return 0.0
        count, cooldown_until = state
        if cooldown_until <= 0:
            return 0.0
        remaining = cooldown_until - now
        if remaining > 0:
            return remaining
        del self._abort_counts[key]
        return 0.0

    def _record_abort(
        self, key: ScheduleKey, policy: OptimizerSchedulingPolicy
    ) -> None:
        now = time.monotonic()
        with self._mutex:
            self._cooldown_remaining_locked(key, now)
            count, _ = self._abort_counts.get(key, (0, 0.0))
            count += 1
            if count >= policy.abort_cooldown_threshold:
                cooldown_until = now + policy.cooldown_after_aborts_seconds
                self._abort_counts[key] = (0, cooldown_until)
                logger.warning(
                    "Playbook optimization entered abort cooldown for key=%s seconds=%s",
                    key,
                    policy.cooldown_after_aborts_seconds,
                )
            else:
                self._abort_counts[key] = (count, 0.0)

    def _clear_abort_state(self, key: ScheduleKey) -> None:
        with self._mutex:
            self._abort_counts.pop(key, None)
