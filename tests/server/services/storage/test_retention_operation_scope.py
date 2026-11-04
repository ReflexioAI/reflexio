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


@pytest.mark.parametrize("fail", [False, True])
def test_real_nested_probe_restores_outer_and_caller_context(
    storage, monkeypatch, fail
):
    state = ["caller"]
    nested_depths = []

    @contextmanager
    def execution_scope():
        parent = list(state)
        state.append("retention")
        nested_depths.append(len(state))
        try:
            yield
        finally:
            state[:] = parent

    def estimate(_target):
        assert state == ["caller", "retention", "retention"]
        if fail:
            raise BackendFailureError("nested backend failure")
        return

    original_probe = storage._retention_probe_targets

    def real_probe(thresholds):
        assert state == ["caller", "retention"]
        result = original_probe(thresholds)
        # The real per-target path catches and records an inner error. Its
        # execution context must already have returned to this outer scope.
        assert state == ["caller", "retention"]
        return result

    monkeypatch.setattr(storage, "_retention_scope", execution_scope)
    monkeypatch.setattr(storage, "_retention_estimate_rows", estimate)
    monkeypatch.setattr(storage, "_retention_probe_targets", real_probe)
    result = storage.probe_retention_targets({"profiles": 0.0})["profiles"]
    if fail:
        assert isinstance(result.error, BackendFailureError)
    else:
        assert result.error is None and result.exact and result.rows == 0
    assert max(nested_depths) == 3
    assert state == ["caller"]


@pytest.mark.parametrize("operation,args,kwargs,hook,value", _OPERATIONS)
def test_rejected_boundary_never_enters_backend_work(
    storage, monkeypatch, operation, args, kwargs, hook, value
):
    class RejectedScope:
        def __enter__(self):
            raise BackendFailureError("retention entry denied")

        def __exit__(self, *_args):
            return False

    backend = Mock(return_value=value)
    monkeypatch.setattr(storage, "_retention_scope", RejectedScope)
    monkeypatch.setattr(storage, hook, backend)
    with pytest.raises(BackendFailureError, match="retention entry denied"):
        getattr(storage, operation)(*args, **kwargs)
    backend.assert_not_called()
