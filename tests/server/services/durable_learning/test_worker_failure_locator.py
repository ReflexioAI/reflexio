"""A failed extraction says WHERE it failed, and never WHAT was in it.

Prod once emitted ``Extraction window failed kind=profile error=ValueError``
twice in one night. That is the whole signal: no message, no frame, and a
WARNING sits below the Sentry quota floor, so nothing richer existed anywhere.
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
from typing import Any, cast

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
