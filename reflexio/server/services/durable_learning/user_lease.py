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
from reflexio.server.services.storage.storage_base import BaseStorage
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

    @property
    def scope(self) -> str:
        return f"durable-learning-heartbeat:{self.org_id}:{_user_ref(self.user_id)}"

    def fence(self, storage: BaseStorage) -> None:
        if storage is not self.storage or storage.org_id != self.org_id:
            raise LeaseLostError(
                "Active user work belongs to a different storage target"
            )
        storage.fence_user_extraction(self.user_id, self.token)

    def validate_explicit_return(self) -> None:
        if self.lost.is_set():
            raise LeaseLostError("Explicit extraction lease expired")
        if self.uncertain.is_set():
            # A failed renewal request is not proof of replacement. The same
            # resource fence used by writes establishes actual ownership.
            self.fence(self.storage)
            report_background_success(self.scope)


_active_lease: ContextVar[_UserWorkOwner | None] = ContextVar(
    "extraction_user_work_owner", default=None
)


def fence_explicit_extraction(storage: BaseStorage) -> None:
    owner = _active_lease.get()
    if owner is not None:
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
    owner = _UserWorkOwner(storage, storage.org_id, user_id, token)
    stop = threading.Event()

    @background_work()
    def heartbeat() -> None:
        while not stop.wait(
            durable_learning_policy.renewal_interval_seconds(lease_seconds)
        ):
            try:
                if not storage.renew_extraction(user_id, token, lease_seconds):
                    owner.lost.set()
                    return
                owner.uncertain.clear()
                report_background_success(owner.scope)
            except Exception as exc:
                owner.uncertain.set()
                report_background_failure(
                    logger,
                    "extraction_heartbeat_failed",
                    exc,
                    scope=owner.scope,
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
) -> Iterator[None]:
    lease_seconds = durable_learning_policy.resolve_user_lease_seconds()
    deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds
    token = None
    while token is None:
        if deadline is not None and time.monotonic() >= deadline:
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
            time.sleep(0.1)
    try:
        with owned_user_work_context(storage, user_id, token, lease_seconds) as owner:
            yield
            owner.validate_explicit_return()
    finally:
        try:
            storage.release_user_extraction(user_id, token)
        finally:
            _release()


class UserLeaseBusyError(RuntimeError):
    """A bounded caller could not acquire capacity and a user lease in time."""
