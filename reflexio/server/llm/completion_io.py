"""Explicit, context-local interception of admitted physical completions.

Local qualification can record or replay a raw provider response without replacing
client methods or bypassing request admission. Nothing is persisted here. A hook
receives provider parameters (which may contain secrets); its caller owns private
storage, request matching and physical-call accounting. Hooks must not alter
parameters, and live calls must go through ``dispatch`` to retain existing
provider timeout, process-isolation and resource ownership behavior.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

CompletionIO = Callable[[dict[str, Any], float, Callable[[], Any]], Any]
_completion_io: ContextVar[CompletionIO | None] = ContextVar(
    "completion_io", default=None
)


@contextmanager
def completion_io_scope(callback: CompletionIO) -> Iterator[None]:
    """Intercept completions in this context, restoring any outer hook on exit."""
    token = _completion_io.set(callback)
    try:
        yield
    finally:
        _completion_io.reset(token)


def intercept_completion_io[Client](
    method: Callable[[Client, dict[str, Any], float], Any],
) -> Callable[[Client, dict[str, Any], float], Any]:
    """Wrap the existing physical IO boundary; leave its implementation intact."""

    @wraps(method)
    def completion(self: Client, params: dict[str, Any], hard_timeout: float) -> Any:
        callback = _completion_io.get()
        if callback is None:
            return method(self, params, hard_timeout)
        return callback(
            params, hard_timeout, lambda: method(self, params, hard_timeout)
        )

    return completion
