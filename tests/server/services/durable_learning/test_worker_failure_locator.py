"""A failed extraction says WHERE it failed, and never WHAT was in it.

Prod once emitted ``Extraction window failed kind=profile error=ValueError``
twice in one night. That is the whole signal: no message, no frame, and a
WARNING sits below the error-reporting quota floor, so nothing richer existed anywhere.
The class name is logged *instead of* the message on purpose - an exception
raised on this path can quote the customer's own interaction text, and on the
window handler the value is persisted as well as logged.

Both halves below are load-bearing. The locator half makes the next occurrence
a one-step diagnosis; the privacy half is what stops someone restoring
diagnosability later by reaching for ``str(exc)``.

All three broad handlers in ``worker.py`` are covered - effects delivery,
window execution, and turn setup - because they are one class of defect, not
three coincidences.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning import worker as worker_module
from reflexio.server.services.durable_learning.worker import (
    DurableLearningWorker,
    exception_locator,
)
from reflexio.server.services.storage.storage_base._extraction_stream import Window

SENTINEL = "SENTINEL_CUSTOMER_TEXT"
# ``parent/file.py:12`` - the parent component is absent only for a file at
# the filesystem root, which no real frame has.
_LOCATOR = re.compile(r"(?:[\w.-]+/)?[\w.-]+\.py:\d+")


def _raise_from_a_dependency() -> None:
    """Stand-in for a library raising inside the guarded block.

    This module lives under ``tests/``, not under the ``reflexio`` package
    root, so to ``exception_locator`` it is indistinguishable from a
    dependency - which is exactly the case the two-part locator exists for.
    """
    raise ValueError(SENTINEL)


def _window(window_id: str = "w1") -> Window:
    return Window(
        window_id=window_id,
        user_id="u1",
        kind="profile",
        project_id="",
        predecessor=0,
        end_seq=1,
        manifest=[],
        policy={},
        force=False,
        skip_aggregation=False,
    )


class _Storage:
    """Just enough surface for ``_turn`` to reach each handler in turn."""

    def __init__(self, *, pending_effects: bool = False, claim_raises: bool = False):
        self._pending_effects = pending_effects
        self._claim_raises = claim_raises
        self.retried: list[str] = []
        self.effects_retried: list[str] = []
        self.deferred = 0

    def claim_extraction(self, instance_id: str, lease_seconds: int):
        return ("u1", "tok")

    def pending_extraction_effects(self, *, user_id: str, token: str):
        if self._claim_raises:
            _raise_from_a_dependency()
        if self._pending_effects:
            return iter([(_window("effects-w"), {"billing": {}})])
        return iter(())

    def prepare_extraction(self, user_id: str, token: str) -> Window:
        return _window()

    def renew_extraction(self, user_id: str, token: str, lease_seconds: int) -> bool:
        return True

    def retry_extraction(self, window: Window, token: str, error: str) -> None:
        self.retried.append(error)

    def retry_extraction_effects(self, window: Window) -> None:
        self.effects_retried.append(window.window_id)

    def defer_extraction_setup(self, user_id: str, token: str) -> None:
        self.deferred += 1

    def release_user_extraction(self, user_id: str, token: str) -> None:
        return None


class _Context:
    def __init__(self, storage: _Storage) -> None:
        self.storage = storage


def _run_turn(monkeypatch: pytest.MonkeyPatch, storage: _Storage) -> None:
    class _Executor:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def execute(self, window: Window, token: str) -> None:
            _raise_from_a_dependency()

        def deliver(self, window: Window, effects: Any, *, token: str) -> None:
            _raise_from_a_dependency()

    monkeypatch.setattr(worker_module, "WindowExecutor", _Executor)
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client", lambda _context: None
    )
    # The fakes above implement only the slice of RequestContext/BaseStorage
    # that ``_turn`` touches on the way to each handler.
    factory = cast("Callable[[str], RequestContext]", lambda _org: _Context(storage))
    DurableLearningWorker(factory)._turn("org1", 30)


def _one_record(caplog: pytest.LogCaptureFixture, needle: str) -> str:
    records = [r for r in caplog.records if needle in r.getMessage()]
    assert len(records) == 1, caplog.text
    assert records[0].exc_info is None
    return records[0].getMessage()


def _assert_diagnosable_and_private(emitted: str) -> None:
    """Every locator-bearing warning owes both halves."""
    assert "error=ValueError" in emitted, emitted
    assert _LOCATOR.search(emitted), emitted
    # The raise site, then the first-party frame that led there.
    assert "test_worker_failure_locator.py:" in emitted, emitted
    assert "durable_learning/worker.py:" in emitted, emitted
    assert SENTINEL not in emitted


def test_window_failure_reports_a_locator_but_not_the_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = _Storage()
    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        _run_turn(monkeypatch, storage)

    _assert_diagnosable_and_private(_one_record(caplog, "Extraction window failed"))
    assert SENTINEL not in caplog.text

    # The retry record is persisted per tenant, so it owes the same both halves.
    assert len(storage.retried) == 1
    stored = storage.retried[0]
    assert SENTINEL not in stored
    assert stored.startswith("ValueError@"), stored
    assert _LOCATOR.search(stored), stored


def test_effects_failure_reports_a_locator_but_not_the_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = _Storage(pending_effects=True)
    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        _run_turn(monkeypatch, storage)

    _assert_diagnosable_and_private(
        _one_record(caplog, "Extraction effects will retry")
    )
    assert SENTINEL not in caplog.text
    # Log-only by schema: retry_extraction_effects stores a due time, no error.
    assert storage.effects_retried == ["effects-w"]


def test_setup_failure_reports_a_locator_but_not_the_message(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = _Storage(claim_raises=True)
    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        _run_turn(monkeypatch, storage)

    _assert_diagnosable_and_private(_one_record(caplog, "Extraction setup will retry"))
    assert SENTINEL not in caplog.text
    assert storage.deferred == 1


def test_locator_is_bounded() -> None:
    try:
        _raise_from_a_dependency()
    except ValueError as exc:
        assert len(exception_locator(exc)) <= worker_module._LOCATOR_MAX


def test_first_party_raise_site_is_not_repeated() -> None:
    """When our frame IS the raise site, emit one locator - not ``X<-X``."""
    try:
        # Raised inside the reflexio package: raise site == first-party frame.
        Window.from_row({})
    except Exception as exc:
        where = exception_locator(exc)

    assert "<-" not in where, where
    assert _LOCATOR.fullmatch(where), where


def test_locator_without_a_traceback_is_named_not_empty() -> None:
    assert exception_locator(ValueError(SENTINEL)) == "unknown"


def _raise_through(dependency_path: str, first_party_path: str) -> BaseException:
    """An exception whose frames carry the given ``co_filename`` values.

    ``compile(..., path, ...)`` sets ``co_filename`` for real, so these are
    genuine frames - the locator sees exactly what it would in production.
    """

    def _make(path: str, src: str) -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        exec(compile(src, path, "exec"), namespace)  # noqa: S102
        return namespace

    inner = _make(dependency_path, "def boom():\n    raise ValueError('S')\n")
    outer = _make(first_party_path, "def call(f):\n    f()\n")
    try:
        outer["call"](inner["boom"])
    except ValueError as exc:
        return exc
    raise AssertionError("the helper must raise")


def _package_root() -> str:
    return str(Path(worker_module.__file__).resolve().parents[3])


def test_a_truncated_locator_still_names_a_line() -> None:
    """Overrunning the budget must cost path, never the ``:line``.

    A blind prefix slice cut the appended first-party coordinate mid-filename,
    which is worse than no locator: it reads as diagnosed and points nowhere.
    """
    long_dependency = (
        "/srv/app/.venv/lib/python3.13/site-packages"
        "/vendored_validation_namespace_pkg/generated_model_builder_impl.py"
    )
    long_first_party = str(
        Path(_package_root())
        / "server"
        / "services"
        / "durable_learning_window_delivery"
        / "outcome_plan_decoder_helpers.py"
    )
    where = exception_locator(_raise_through(long_dependency, long_first_party))

    # The case is only meaningful if the untruncated form would overrun.
    assert len(f"{long_dependency}<-{long_first_party}") > worker_module._LOCATOR_MAX
    assert len(where) <= worker_module._LOCATOR_MAX, where

    halves = where.split("<-")
    assert len(halves) == 2, where
    # The invariant first: truncation may cost path, never the coordinate.
    for half in halves:
        assert re.search(r":\d+$", half), half
    for half in halves:
        assert len(half) <= worker_module._POSITION_MAX, half


def test_a_single_component_path_cannot_crowd_out_the_line() -> None:
    """Even a pathological path yields a coordinate that names its line."""
    absurd = "/" + "z" * 400 + "/" + "y" * 400 + ".py"
    position = worker_module._position(absurd, 987654)

    assert position.endswith(":987654"), position
    assert len(position) <= worker_module._POSITION_MAX, position


def test_symlinked_deployment_still_matches_first_party_frames(
    tmp_path: Path,
) -> None:
    """A ``current``-style release symlink must not reject every own frame.

    ``co_filename`` keeps the path a module was imported *as*. Resolving only
    our own side made the prefix test match nothing on such a deployment, so
    no first-party frame was ever found - silently, and for good.
    """
    release = tmp_path / "current"
    release.symlink_to(_package_root())
    symlinked = str(release / "server" / "services" / "durable_learning" / "worker.py")
    real = os.path.realpath(symlinked)

    # The premise: one file, two spellings that genuinely disagree as strings.
    assert symlinked != real
    assert os.path.samefile(symlinked, real)

    # Roots as computed by a deployment that imported THROUGH the symlink.
    roots = worker_module._first_party_roots(symlinked)
    assert symlinked.startswith(roots), roots
    assert real.startswith(roots), roots


def test_symlinked_first_party_frame_is_found_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The symlinked frame reaches the emitted locator, not just the predicate."""
    release = tmp_path / "current"
    release.symlink_to(_package_root())
    symlinked_caller = str(
        release / "server" / "services" / "durable_learning" / "window_executor.py"
    )
    monkeypatch.setattr(
        worker_module,
        "_FIRST_PARTY_ROOTS",
        worker_module._first_party_roots(
            str(release / "server" / "services" / "durable_learning" / "worker.py")
        ),
    )

    where = exception_locator(
        _raise_through(
            "/srv/app/.venv/lib/python3.13/json/decoder.py", symlinked_caller
        )
    )

    assert "<-" in where, where
    assert where.endswith("durable_learning/window_executor.py:2"), where


def test_effects_lease_loss_retains_retry_and_reports_lease_lost(monkeypatch):
    from unittest.mock import Mock

    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
    )

    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    monkeypatch.setattr(
        __name__ + "._raise_from_a_dependency", Mock(side_effect=LeaseLostError())
    )
    storage = _Storage(pending_effects=True)
    _run_turn(monkeypatch, storage)
    assert storage.effects_retried == ["effects-w"]
    attempts = [
        call for call in sink.record.call_args_list if call.args[0] == "worker.attempts"
    ]
    assert len(attempts) == 1
    assert attempts[0].kwargs["attributes"] == {
        "phase": "effects",
        "outcome": "lease_lost",
    }


@pytest.mark.parametrize("scope_error", [False, True])
def test_setup_deferral_counts_scheduled_retry(monkeypatch, scope_error):
    from unittest.mock import Mock

    from reflexio.server.work_scope import WorkScopeError

    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    storage = _Storage(claim_raises=not scope_error)
    if scope_error:
        monkeypatch.setattr(
            worker_module, "bind_work_scope", Mock(side_effect=WorkScopeError())
        )
    _run_turn(monkeypatch, storage)
    assert storage.deferred == 1
    retries = [
        call.kwargs["attributes"]
        for call in sink.record.call_args_list
        if call.args[0] == "worker.retries"
    ]
    assert retries == [{"phase": "prepare"}]


def test_window_retry_redirected_to_effects_has_correct_phase(monkeypatch):
    from unittest.mock import Mock

    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
    )

    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    storage = _Storage()
    monkeypatch.setattr(storage, "retry_extraction", Mock(side_effect=LeaseLostError()))
    _run_turn(monkeypatch, storage)
    assert storage.effects_retried == ["w1"]
    retries = [
        call.kwargs["attributes"]
        for call in sink.record.call_args_list
        if call.args[0] == "worker.retries"
    ]
    assert retries == [{"phase": "effects"}]


@pytest.mark.parametrize(
    "operation", ["pending_extraction_effects", "prepare_extraction"]
)
def test_setup_lease_loss_does_not_schedule_or_count_retry(monkeypatch, operation):
    from unittest.mock import Mock

    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
    )

    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    storage = _Storage()
    monkeypatch.setattr(storage, operation, Mock(side_effect=LeaseLostError()))
    _run_turn(monkeypatch, storage)
    assert storage.deferred == 0
    assert not storage.retried
    assert not storage.effects_retried
    attempts = [
        call.kwargs["attributes"]
        for call in sink.record.call_args_list
        if call.args[0] == "worker.attempts"
    ]
    assert attempts == [{"phase": "prepare", "outcome": "lease_lost"}]
    assert not any(
        call.args[0] == "worker.retries" for call in sink.record.call_args_list
    )


def test_transient_turn_failure_is_a_warning_and_releases_the_slot(
    caplog: pytest.LogCaptureFixture, transient_failure_classifier: type[Exception]
) -> None:
    def factory(_org_id: str) -> RequestContext:
        raise transient_failure_classifier("server closed the connection")

    caplog.set_level(logging.WARNING, logger=worker_module.logger.name)
    worker = DurableLearningWorker(factory)

    assert worker.drain_org("org1", batch_size=2, lease_seconds=30) == 0

    failures = [r for r in caplog.records if "extraction_turn_failed" in r.getMessage()]
    # Both turns ran: the first failure released its capacity slot.
    assert [r.levelno for r in failures] == [logging.WARNING, logging.WARNING]
    assert not failures[0].exc_info

    from reflexio.server import background_work

    # Nothing was claimed, so the failing unit is the org's claim.
    assert background_work._streaks["durable-learning-claim:org1"][0] == 2
    # No storage: nothing was attempted, so nothing is reported either way.
    worker._factory = lambda _org_id: cast(
        RequestContext, SimpleNamespace(storage=None)
    )
    worker.drain_org("org1", batch_size=1, lease_seconds=30)
    assert background_work._streaks["durable-learning-claim:org1"][0] == 2
    # The claim query runs and finds no user: the claim unit succeeded.
    idle = MagicMock()
    idle.claim_extraction.return_value = None
    worker._factory = lambda _org_id: cast(
        RequestContext, SimpleNamespace(storage=idle)
    )
    worker.drain_org("org1", batch_size=1, lease_seconds=30)
    assert background_work._streaks == {}


def _claimed_storage(user_id: str, *, release_fails: Exception | None) -> MagicMock:
    storage = MagicMock()
    storage.claim_extraction.return_value = (user_id, "token")
    storage.pending_extraction_effects.return_value = []
    storage.prepare_extraction.return_value = None
    storage.renew_extraction.return_value = True
    if release_fails is not None:
        storage.release_user_extraction.side_effect = release_fails
    return storage


def test_a_stuck_users_turn_streak_survives_idle_polls_and_other_users(
    transient_failure_classifier: type[Exception],
) -> None:
    """Only the same user's completed turn ends that user's streak."""
    from reflexio.server import background_work
    from reflexio.server.services.durable_learning.worker import _user_ref

    stuck = _claimed_storage(
        "user-a", release_fails=transient_failure_classifier("SSL EOF")
    )
    healthy = _claimed_storage("user-b", release_fails=None)
    idle = MagicMock()
    idle.claim_extraction.return_value = None
    storages = iter([stuck, idle, healthy, stuck])
    worker = DurableLearningWorker(
        lambda _org_id: cast(RequestContext, SimpleNamespace(storage=next(storages)))
    )
    stuck_scope = f"durable-learning-turn:org1:{_user_ref('user-a')}"

    worker.drain_org("org1", batch_size=4, lease_seconds=30)

    assert background_work._streaks[stuck_scope][0] == 2
    assert "user-a" not in " ".join(background_work._streaks)

    # The same user returns but has nothing to prepare: it did not complete.
    worker._factory = lambda _org_id: cast(
        RequestContext,
        SimpleNamespace(storage=_claimed_storage("user-a", release_fails=None)),
    )
    worker.drain_org("org1", batch_size=1, lease_seconds=30)
    assert background_work._streaks[stuck_scope][0] == 2

    # The same user's extraction completes: only now does its streak end.
    worker._factory = lambda _org_id: cast(
        RequestContext, SimpleNamespace(storage=_delivering_storage("user-a"))
    )
    with _executor_stubbed():
        worker.drain_org("org1", batch_size=1, lease_seconds=30)
    assert background_work._streaks == {}


def _delivering_storage(user_id: str) -> MagicMock:
    storage = _claimed_storage(user_id, release_fails=None)
    storage.pending_extraction_effects.return_value = [
        (SimpleNamespace(project_id=None), [])
    ]
    return storage


def _executor_stubbed():
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(patch.object(worker_module, "WindowExecutor"))
    stack.enter_context(
        patch(
            "reflexio.lib._base.create_generation_litellm_client",
            lambda _context: None,
        )
    )
    return stack


def test_a_turn_that_swallows_its_setup_failure_does_not_end_the_streak(
    monkeypatch: pytest.MonkeyPatch,
    transient_failure_classifier: type[Exception],
) -> None:
    """Codex repro: setup keeps failing while deferral alternately fails and
    succeeds. A successful deferral is a retry being scheduled, not a
    completed turn, so the streak must keep growing and escalate."""
    from reflexio.server import background_work

    clock = [1000.0]
    monkeypatch.setattr(background_work, "_monotonic", lambda: clock[0])
    storage = _claimed_storage("user-a", release_fails=None)
    storage.pending_extraction_effects.side_effect = transient_failure_classifier(
        "server closed the connection"
    )
    deferrals = iter([transient_failure_classifier("SSL EOF"), None] * 3)

    def defer(*_args: object) -> None:
        failure = next(deferrals)
        if failure is not None:
            raise failure

    storage.defer_extraction_setup.side_effect = defer
    worker = DurableLearningWorker(
        lambda _org_id: cast(RequestContext, SimpleNamespace(storage=storage))
    )
    levels: list[int] = []
    handler = logging.Handler()
    handler.emit = lambda record: (  # type: ignore[method-assign]
        levels.append(record.levelno)
        if "extraction_turn_failed" in record.getMessage()
        else None
    )
    worker_module.logger.addHandler(handler)
    try:
        for _ in range(5):
            worker.drain_org("org1", batch_size=1, lease_seconds=30)
            clock[0] += 160
    finally:
        worker_module.logger.removeHandler(handler)

    assert levels == [logging.WARNING, logging.WARNING, logging.ERROR]


def test_heartbeat_failure_is_scoped_to_the_leased_user(
    transient_failure_classifier: type[Exception],
) -> None:
    import time as _time

    from reflexio.server import background_work
    from reflexio.server.services.durable_learning.worker import _user_ref

    storage = _claimed_storage("user-a", release_fails=None)
    storage.renew_extraction.side_effect = transient_failure_classifier("SSL EOF")
    storage.prepare_extraction.side_effect = lambda *_args: _time.sleep(0.6)
    worker = DurableLearningWorker(
        lambda _org_id: cast(RequestContext, SimpleNamespace(storage=storage))
    )

    worker.drain_org("org1", batch_size=1, lease_seconds=1)

    assert set(background_work._streaks) == {
        f"durable-learning-heartbeat:org1:{_user_ref('user-a')}"
    }
