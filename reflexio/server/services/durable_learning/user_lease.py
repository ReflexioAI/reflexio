"""One heartbeat/fencing context for claimed extraction user-work leases."""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from reflexio import durable_learning_policy
from reflexio.server.background_work import (
    background_work,
    report_background_failure,
    report_background_success,
)
from reflexio.server.services.durable_learning.worker import (
    _release,
    _reserve,
    _user_ref,
)
from reflexio.server.services.storage.storage_base import AgentRunRecord, BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _UserWorkOwner:
    storage: BaseStorage = field(repr=False)
    org_id: str = field(repr=False)
    user_id: str = field(repr=False)
    token: str = field(repr=False)
    lost: threading.Event = field(default_factory=threading.Event, repr=False)
    uncertain: threading.Event = field(default_factory=threading.Event, repr=False)

    def raise_if_lost(self) -> None:
        if self.lost.is_set():
            raise LeaseLostError("Extraction turn was cancelled or lost its lease")

    def fence(self, storage: BaseStorage) -> None:
        if storage is not self.storage or storage.org_id != self.org_id:
            raise LeaseLostError(
                "Active user work belongs to a different storage target"
            )
        self.raise_if_lost()
        storage.fence_user_extraction(self.user_id, self.token)
        # A timeout can cancel the copied handle while this database call waits.
        self.raise_if_lost()

    def validate_explicit_return(self) -> None:
        self.raise_if_lost()
        if self.uncertain.is_set():
            # A failed renewal request is not proof of replacement. The same
            # resource fence used by writes establishes actual ownership.
            self.fence(self.storage)
            org_id, user_id = self.org_id, self.user_id
            report_background_success(
                f"durable-learning-heartbeat:{org_id}:{_user_ref(user_id)}"
            )


_active_lease: ContextVar[_UserWorkOwner | None] = ContextVar(
    "extraction_user_work_owner", default=None
)


def fence_explicit_extraction(storage: BaseStorage) -> None:
    owner = _active_lease.get()
    if owner is not None:
        owner.fence(storage)


def cancel_extraction_turn(storage: BaseStorage) -> None:
    """Stop this handle and its copied threads, retaining caller-owned cleanup.

    Timeout is not authority to update rows by request identity. The existing
    failure handler still owns the actual lease and its retry/release transaction;
    this signal only prevents the abandoned execution from writing or adopting.
    """
    owner = _active_lease.get()
    if owner is None:
        return
    if owner.storage is not storage or owner.org_id != storage.org_id:
        raise LeaseLostError("Cannot cancel a different storage target's turn")
    owner.lost.set()


def _require_user_work_owner(storage: BaseStorage) -> _UserWorkOwner:
    owner = _active_lease.get()
    if owner is None:
        raise LeaseLostError("Agent execution requires an active user-work owner")
    if owner.storage is not storage or owner.org_id != storage.org_id:
        raise LeaseLostError("Agent execution belongs to a different storage target")
    owner.raise_if_lost()
    return owner


def adopt_agent_run_for_execution(
    storage: BaseStorage, run: AgentRunRecord
) -> AgentRunRecord:
    """Reuse an adopted revision or atomically attach the exact inspected one."""
    owner = _require_user_work_owner(storage)
    with storage.commit_scope(), storage._stream_sql() as db:
        owner.fence(storage)
        try:
            storage._fence_owned_user_work_run(
                db, expected=run, user_id=owner.user_id, token=owner.token
            )
        except LeaseLostError:
            # An unadopted row may acquire this owner. A stale logical revision
            # cannot: adoption rereads and matches its exact inspected witness.
            owner.raise_if_lost()
            claim = storage._adopt_owned_user_work_run(
                db,
                expected=run,
                user_id=owner.user_id,
                token=owner.token,
                worker_id="extraction",
            )
        else:
            owner.raise_if_lost()
            return run
        if claim is None:
            raise LeaseLostError("Agent execution adoption was refused")
        current = storage.get_agent_run(run.id)
        if current is None or (current.claimed_by, current.claimed_at) != claim:
            raise LeaseLostError("Adopted agent revision is unavailable on the writer")
        owner.fence(storage)
        return current


def create_owned_agent_run(storage: BaseStorage, run: AgentRunRecord) -> AgentRunRecord:
    """Create and adopt together; failed adoption leaves no unowned new row."""
    owner = _require_user_work_owner(storage)
    if run.binding.user_id != owner.user_id:
        raise LeaseLostError("New agent execution requires its established user")
    with storage.commit_scope():
        owner.fence(storage)
        return adopt_agent_run_for_execution(storage, storage.create_agent_run(run))


@contextmanager
def agent_run_user_work(storage: BaseStorage, run: AgentRunRecord) -> Iterator[None]:
    """Direct agent entry uses canonical admission; nested execution reuses it."""
    user_id = run.binding.user_id
    if not user_id or not user_id.strip():
        if run.binding.extractor_kind != "playbook":
            raise LeaseLostError("Agent execution has no established user")
        from reflexio.server.services.playbook.review_window import (
            infer_playbook_review_user_id,
        )

        user_id = infer_playbook_review_user_id(
            storage=storage,
            source_interaction_ids=run.binding.source_interaction_ids,
            subject="Agent execution",
        )
    if _active_lease.get() is not None:
        owner = _require_user_work_owner(storage)
        if owner.user_id != user_id:
            raise LeaseLostError("Agent execution belongs to a different user")
        owner.fence(storage)
        yield
    else:
        with user_extraction_lease(storage, user_id):
            yield


@contextmanager
def owned_agent_run_write(storage: BaseStorage, run: AgentRunRecord) -> Iterator[None]:
    """Join an execution write to both owners; never wrap provider work here.

    Historical insertion and independently authorized source callbacks do not
    enter this seam. Production execution must already have adopted its run.
    The enclosing user-work context retains heartbeat, capacity and release.
    Do not clear that lease inside this writer scope.
    """
    owner = _require_user_work_owner(storage)
    with storage.commit_scope(), storage._stream_sql() as db:
        storage._fence_owned_user_work_run(
            db, expected=run, user_id=owner.user_id, token=owner.token
        )
        yield
        # Effects may themselves have waited for resource locks. Refuse their
        # commit if that wait exhausted the actual parent lease.
        owner.fence(storage)


@contextmanager
def owned_user_work_context(
    storage: BaseStorage, user_id: str, token: str, lease_seconds: int
) -> Iterator[_UserWorkOwner]:
    """Renew and bind a claimed lease; its caller owns admission and release.

    Keeping release with the acquiring turn lets its failure handler defer work
    under the still-held token, including a heartbeat thread that cannot start.
    Scheduled turns must not enter the manual admission wrapper a second time.
    """
    current = _active_lease.get()
    if current is not None:
        if (
            current.storage is not storage
            or current.org_id != storage.org_id
            or current.user_id != user_id
            or current.token != token
        ):
            raise LeaseLostError(
                "Nested execution belongs to a different user-work owner"
            )
        current.fence(storage)
        yield current
        return
    owner = _UserWorkOwner(storage, storage.org_id, user_id, token)
    org_id = owner.org_id
    stop = threading.Event()

    @background_work()
    def heartbeat() -> None:
        while not stop.wait(
            durable_learning_policy.renewal_interval_seconds(lease_seconds)
        ):
            if owner.lost.is_set():
                return
            try:
                if not storage.renew_extraction(user_id, token, lease_seconds):
                    owner.lost.set()
                    return
                owner.uncertain.clear()
                report_background_success(
                    f"durable-learning-heartbeat:{org_id}:{_user_ref(user_id)}"
                )
            except Exception as exc:
                owner.uncertain.set()
                report_background_failure(
                    logger,
                    "extraction_heartbeat_failed",
                    exc,
                    scope=f"durable-learning-heartbeat:{org_id}:{_user_ref(user_id)}",
                    org_id=owner.org_id,
                )

    binding = _active_lease.set(owner)
    thread = None
    started = False
    try:
        thread = threading.Thread(
            target=heartbeat, daemon=True, name="reflexio-extraction-heartbeat"
        )
        thread.start()
        started = True
        yield owner
    finally:
        _active_lease.reset(binding)
        stop.set()
        if started and thread is not None:
            thread.join(timeout=1)


@contextmanager
def user_extraction_lease(
    storage: BaseStorage, user_id: str, *, max_wait_seconds: float | None = None
) -> Iterator[_UserWorkOwner]:
    """Acquire through the shared budget; zero wait makes exactly one attempt."""
    lease_seconds = durable_learning_policy.resolve_user_lease_seconds()
    deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds
    token = None
    while token is None:
        if (
            deadline is not None
            and max_wait_seconds != 0
            and time.monotonic() >= deadline
        ):
            raise UserLeaseBusyError("Extraction capacity or user lease is busy")
        if _reserve():
            try:
                token = storage.claim_user_extraction(
                    user_id, "explicit", lease_seconds
                )
            except Exception:
                _release()
                raise
            if token is None:
                _release()
        if token is None:
            if max_wait_seconds == 0:
                raise UserLeaseBusyError("Extraction capacity or user lease is busy")
            time.sleep(0.1)
    try:
        with owned_user_work_context(storage, user_id, token, lease_seconds) as owner:
            yield owner
            owner.validate_explicit_return()
    finally:
        try:
            storage.release_user_extraction(user_id, token)
        finally:
            _release()


class UserLeaseBusyError(RuntimeError):
    """A bounded caller could not acquire capacity and a user lease in time."""
