"""Neutral tenant-scope seam for deferred work. OSS defines the hole.

Background work is decoupled *in time* from the request that created it: a
durable learning job, a debounced tagging pass, a shadow-comparison judge and a
publish-learning run all execute on a daemon thread long after their request
returned. Such work therefore cannot inherit a request-scoped context variable,
and wrapping the worker in a context-manager scope does not fix it either —
the debounce schedulers deliberately *coalesce across requests*, so by the time
a callback fires there may be several requests behind it.

That is why the scope travels on the **job payload** (``LearningJob.project_id``
and the scheduler keys' project component) rather than in ambient context: the
payload is the only thing that survives coalescing with its attribution intact.

This module supplies the two halves OSS needs to carry a scope it does not
itself understand:

- :func:`current_project_id` — read the scope at *enqueue* time, so it can be
  written into the payload/key.
- :func:`bind_work_scope` — re-establish it at *fire* time, so the deferred
  write is attributed to the project that queued it.

Both are inert without a registered provider, which is the OSS case: a bare
install has one org and no projects, so an absent project is normal, never an
error. Enterprise registers a provider at its composition root and owns the
fail-closed behaviour (see :class:`WorkScopeError`). OSS never imports
``reflexio_ext``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol

from reflexio.server.extensions import ServiceKey, get_service


class WorkScopeError(RuntimeError):
    """Deferred work could not establish the tenant scope it needs.

    **OSS never raises this.** It exists so that a provider (enterprise) can
    signal a scope/attribution failure as something categorically different
    from an ordinary operational error.

    Background workers must not treat it as tolerable. The daemon worker loops
    narrow their handlers so an operational failure is logged and the next job
    proceeds, while a ``WorkScopeError`` is *escalated* through
    ``capture_anomaly``. Escalation — not propagation — is the correct
    behaviour at the top frame of a daemon thread: letting it out of the loop
    would silently shrink a fixed worker pool until deferred work stopped
    running altogether, which is strictly less observable than reporting it.
    """


@dataclass(frozen=True)
class WorkScope:
    """The tenant scope a unit of deferred work must be attributed to.

    "Unset" and "empty" are the SAME state here, and are normalised to ``None``
    on construction. A provider that reads the scope out of a transaction-local
    Postgres GUC cannot tell them apart: on a pooled connection an unset GUC
    reads back as the empty string, not NULL. Without this coercion an empty
    project would form a debounce key distinct from an absent one, and would be
    stored as ``''`` rather than NULL on the job row — two spellings of "no
    project" that no longer compare equal.

    Attributes:
        org_id (str): Owning organisation. Always present.
        project_id (str | None): Owning project, or ``None`` where projects do
            not exist (OSS) or the caller has none. ``None`` is a normal value
            in OSS and must never be treated as an error here.
    """

    org_id: str
    project_id: str | None = None

    def __post_init__(self) -> None:
        if self.project_id == "":
            # frozen dataclass: bypass the setattr guard to normalise.
            object.__setattr__(self, "project_id", None)


class WorkScopeProvider(Protocol):
    """Contract for the registered scope provider. Implemented by enterprise."""

    def current(self) -> WorkScope | None:
        """Return the scope in effect on this thread, or None if unscoped."""
        ...

    def bind(self, scope: WorkScope) -> AbstractContextManager[None]:
        """Return a context manager that makes ``scope`` current for its body."""
        ...


WORK_SCOPE_PROVIDER = ServiceKey[WorkScopeProvider]("work_scope_provider")


def current_work_scope() -> WorkScope | None:
    """Return the scope in effect, or ``None`` when no provider is registered."""
    provider = get_service(WORK_SCOPE_PROVIDER)
    if provider is None:
        return None
    return provider.current()


def current_project_id() -> str | None:
    """Project to stamp onto a job payload/scheduler key at enqueue time.

    Returns ``None`` in OSS (no provider registered) and for an unscoped
    caller. Call this at *enqueue* time, never at fire time — reading it inside
    a debounced callback would resolve whichever request happened to win the
    coalescing race, which is the misattribution this seam exists to prevent.
    """
    scope = current_work_scope()
    if scope is None:
        return None
    # A provider may hand back "" for an unset Postgres GUC; normalise it to
    # None so "unset" and "empty" never form two distinct keys. WorkScope's
    # __post_init__ already does this, but a provider is free to return any
    # object satisfying the protocol, so do not rely on it having done so.
    return scope.project_id or None


@contextmanager
def bind_work_scope(scope: WorkScope | None) -> Iterator[None]:
    """Re-establish ``scope`` for the body, using the registered provider.

    A no-op when no provider is registered (OSS) or ``scope`` is ``None``, so
    every background worker can bind unconditionally.

    Args:
        scope (WorkScope | None): Scope recovered from the job payload/key.

    Raises:
        WorkScopeError: Only from a registered provider that rejects the scope.
    """
    if scope is None:
        yield
        return
    provider = get_service(WORK_SCOPE_PROVIDER)
    binder = nullcontext() if provider is None else provider.bind(scope)
    with binder:
        yield


__all__ = [
    "WORK_SCOPE_PROVIDER",
    "WorkScope",
    "WorkScopeError",
    "WorkScopeProvider",
    "bind_work_scope",
    "current_project_id",
    "current_work_scope",
]
