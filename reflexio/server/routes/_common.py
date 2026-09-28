"""Shared route helpers used across domain route modules (Tier3 A2)."""

from collections.abc import Callable

from reflexio.server import search_runtime
from reflexio.server.operation_limiter import (
    OperationName,
    limiter_http_exception,
    run_with_operation_limit,
)


def _run_limited_api[T](
    org_id: str,
    operation: OperationName,
    fn: Callable[[], T],
) -> T:
    try:
        return run_with_operation_limit(
            org_id=org_id,
            operation=operation,
            fn=fn,
        )
    except search_runtime.SearchDeadlineError:
        raise
    except TimeoutError as exc:
        if operation == "search":
            search_runtime.checkpoint()
        http_exc = limiter_http_exception(operation)
        raise http_exc from exc
