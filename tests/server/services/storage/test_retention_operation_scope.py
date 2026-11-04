"""Every public retention operation enters its backend execution boundary."""

from contextlib import contextmanager
from unittest.mock import Mock

import pytest


class BackendFailureError(RuntimeError):
    pass


_OPERATIONS = [
    (
        "count_retention_target_rows",
        ("profiles",),
        {},
        "_retention_table_exists",
        False,
    ),
    (
        "estimate_retention_target_rows",
        ("profiles",),
        {},
        "_retention_estimate_rows",
        0,
    ),
    (
        "probe_retention_targets",
        ({"profiles": 0.0},),
        {},
        "_retention_probe_targets",
        {},
    ),
    (
        "delete_oldest_retention_target_rows",
        ("profiles", 1),
        {},
        "_retention_table_exists",
        False,
    ),
    (
        "expire_retention_target_rows",
        ("interactions",),
        {
            "older_than_epoch": 0,
            "budget": 1,
            "batch_size": 1,
            "archiver": None,
            "deadline": float("inf"),
        },
        "_retention_table_exists",
        False,
    ),
    (
        "gc_retired_optimization_jobs",
        (),
        {"older_than_epoch": 0, "stale_before_epoch": 0, "limit": 1},
        "_retention_gc_retired_optimization_jobs",
        0,
    ),
]


@pytest.mark.parametrize("operation,args,kwargs,hook,value", _OPERATIONS)
@pytest.mark.parametrize("fail", [False, True])
def test_retention_boundary_covers_backend_work_and_unwinds(
    storage, monkeypatch, operation, args, kwargs, hook, value, fail
):
    events = []

    @contextmanager
    def execution_scope():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def backend_work(*_args, **_kwargs):
        assert events == ["enter"]
        if fail:
            raise BackendFailureError("backend work failed")
        return value

    monkeypatch.setattr(storage, "_retention_scope", execution_scope, raising=False)
    backend = Mock(side_effect=backend_work)
    monkeypatch.setattr(storage, hook, backend)
    if fail:
        with pytest.raises(BackendFailureError, match="backend work failed"):
            getattr(storage, operation)(*args, **kwargs)
    else:
        getattr(storage, operation)(*args, **kwargs)
    backend.assert_called_once()
    assert events == ["enter", "exit"]
