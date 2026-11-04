"""Value-free exception detail for broad ``except Exception`` log lines.

A handler that binds ``exc`` and then logs only a consequence ("passing
through", "lease rebuilds uncached") leaves an operator with no way to tell a
database outage from a code defect. This module renders the two facts that
localise such a failure *without* echoing anything a caller supplied.

**What is deliberately NOT rendered: the exception MESSAGE.** Database and
HTTP clients embed request context in their exception messages verbatim -- a
connection URL can carry ``user:password@`` and a query token -- and
``pydantic`` embeds the offending input, which for a stored configuration can
hold credentials. So the type and the raise coordinate are emitted;
``str(exc)`` is not. A source coordinate is our own data and can never be a
caller's.

Placed at ``reflexio/server/`` rather than inside any one service package
because unrelated subtrees need it, including
``services/offline_tuner/open_world/diagnostics.py``. Stdlib-only, so it
cold-imports anywhere.
"""

from __future__ import annotations

__all__ = ["exception_detail", "raised_at"]


def raised_at(exc: BaseException) -> str | None:
    """Locate where an exception surfaced, as ``module:lineno``.

    Walks to the deepest traceback frame, which is where the exception was
    actually raised rather than where it was caught.

    Renders the dotted module name, never ``tb_frame.f_code.co_filename``: the
    filename is an absolute path inside the container
    (``/app/...``), which is noise in a log line and differs
    between the image and a developer checkout. The dotted name is stable,
    bounded, and is what a reader greps for anyway.

    Args:
        exc (BaseException): Exception carrying a traceback.

    Returns:
        str | None: ``module:lineno``, or None when no traceback is attached
            (an exception constructed but never raised, or one whose traceback
            was stripped).
    """
    traceback = exc.__traceback__
    if traceback is None:
        return None
    deepest = traceback
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    frame = deepest.tb_frame
    module = frame.f_globals.get("__name__")
    if not isinstance(module, str):
        module = frame.f_code.co_filename.rsplit("/", 1)[-1]
    return f"{module}:{deepest.tb_lineno}"


def exception_detail(exc: BaseException) -> str:
    """Render *exc* as a ready-to-interpolate, value-free log fragment.

    Shape: ``exception_type=<Type> raised_at=<module:lineno>``. Key=value so a
    CloudWatch Insights ``parse`` can pull either field out, matching the
    convention the auth provisioning logs already use.

    Args:
        exc (BaseException): The exception a broad handler caught.

    Returns:
        str: ``exception_type=... raised_at=...``; ``raised_at=unknown`` when
            no traceback is attached.
    """
    return (
        f"exception_type={type(exc).__name__} raised_at={raised_at(exc) or 'unknown'}"
    )
