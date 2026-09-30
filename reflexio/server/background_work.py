"""Explicit classification of deferred work, independent of tenant context.

Storage providers may use this marker to bound background admission. It does
not bind a project, hold a connection, or change an OSS storage implementation.
New worker threads establish their own scope instead of copying request state.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_background: ContextVar[bool] = ContextVar("reflexio_background_work", default=False)


def is_background_work() -> bool:
    """Whether this execution belongs to a deferred worker."""
    return _background.get()


@contextmanager
def background_work() -> Iterator[None]:
    """Classify a worker body, restoring the caller's classification on exit."""
    token = _background.set(True)
    try:
        yield
    finally:
        _background.reset(token)
