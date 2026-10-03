"""A swallowed lineage GC failure must shorten the next tick's wait.

``report_background_failure`` downgrades a transient failure to WARNING and only
escalates one that keeps recurring within its episode gap. A handler that
swallows a failure without ``_record_tick_failure()`` leaves the scheduler on its
daily cadence, so a persistent outage there would stay at WARNING for good.
"""

import ast
import inspect

from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.lineage.gc_scheduler import LineageGCScheduler


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            names.add(
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
    return names


def test_every_reporting_handler_records_the_tick_failure():
    tree = ast.parse(inspect.getsource(gc_scheduler))
    handlers = [
        handler
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        and "report_background_failure" in _calls(handler)
    ]

    assert len(handlers) >= 10  # the scan must still see the scheduler's handlers
    missing = [h.lineno for h in handlers if "_record_tick_failure" not in _calls(h)]
    assert missing == [], f"handlers at lines {missing} swallow without a fast retry"


def test_a_failing_org_gets_the_fast_retry_not_the_daily_cadence():
    def broken_context(_org_id: str):
        raise ConnectionError("server closed the connection unexpectedly")

    scheduler = LineageGCScheduler(
        request_context_factory=broken_context,  # type: ignore[arg-type]
        bootstrap_org_id="org-boot",
    )
    scheduler._tick_had_failure = False

    scheduler._sweep_org("org-1")

    assert scheduler._next_interval(86400) <= gc_scheduler._FAILED_TICK_RETRY_SECONDS
