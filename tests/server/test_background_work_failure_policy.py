"""Background failures: transient drops are warnings, bugs and outages are errors.

A transient failure extends its scope's streak; the streak escalates to ERROR
once it holds ``_ESCALATE_AFTER_FAILURES`` failures spanning at least
``_ESCALATE_AFTER_SECONDS``, and a success for the same scope ends it.
"""

from __future__ import annotations

import logging
import sys
from uuid import uuid4

import pytest

from reflexio.server import background_work
from reflexio.server.background_work import (
    configure_transient_failure_classifier,
    is_transient_failure,
    report_background_failure,
    report_background_success,
)

_LOGGER = logging.getLogger("tests.background_failure_policy")
_SCOPE = "worker:org-1"


def _raised(exc: BaseException) -> BaseException:
    """Return ``exc`` with a real traceback, as a worker's ``except`` sees it."""
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


def _report(caplog, exc: BaseException, scope: str = _SCOPE, **kwargs):
    caplog.clear()
    report_background_failure(
        _LOGGER, "worker_failed", exc, scope=scope, org_id="o", **kwargs
    )
    records = [r for r in caplog.records if r.name == _LOGGER.name]
    assert len(records) == 1
    return records[0]


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(background_work, "_monotonic", lambda: now[0])
    return now


@pytest.fixture(autouse=True)
def _capture(caplog, monkeypatch):
    # A full-suite run installs DuplicateFilter on shared handlers. Fresh logger
    # identities prevent earlier cases from rewriting this policy assertion.
    logger = logging.getLogger(f"tests.background_failure_policy.{uuid4().hex}")
    monkeypatch.setattr(sys.modules[__name__], "_LOGGER", logger)
    caplog.set_level(logging.DEBUG, logger=logger.name)


# --- rendering --------------------------------------------------------------


def test_transient_failure_is_one_warning_without_traceback(
    caplog, transient_failure_classifier
):
    record = _report(caplog, _raised(transient_failure_classifier("ssl eof\nmore")))

    assert record.levelno == logging.WARNING
    assert not record.exc_info
    message = record.getMessage()
    assert "\n" not in message
    assert (
        "event=worker_failed transient=true consecutive_failures=1 "
        "scope=worker:org-1" in message
    )
    assert "error_class=TransientTestError error=ssl eof more org_id=o" in message


def test_long_transient_message_is_truncated(caplog, transient_failure_classifier):
    record = _report(caplog, _raised(transient_failure_classifier("x" * 5000)))

    assert len(record.getMessage()) < 400


def test_non_transient_failure_is_error_with_traceback_immediately(
    caplog, transient_failure_classifier
):
    record = _report(caplog, _raised(ValueError("bad row")))

    assert record.levelno == logging.ERROR
    assert record.exc_info and record.exc_info[1].args == ("bad row",)
    assert record.getMessage() == "event=worker_failed scope=worker:org-1 org_id=o"
    assert _SCOPE not in background_work._streaks


def test_without_a_classifier_every_failure_is_an_error(caplog):
    configure_transient_failure_classifier(None)

    record = _report(caplog, _raised(ConnectionError("dropped")))

    assert record.levelno == logging.ERROR
    assert record.exc_info


# --- classification: only the deliberate __cause__ chain --------------------


def test_transient_cause_is_transient(caplog, transient_failure_classifier):
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = transient_failure_classifier("dropped")

    assert is_transient_failure(wrapper)
    assert _report(caplog, _raised(wrapper)).levelno == logging.WARNING


def test_transient_only_in_context_is_an_error(caplog, transient_failure_classifier):
    """A bug raised while handling a dropped connection is a bug, not the drop."""
    try:
        try:
            raise transient_failure_classifier("dropped")
        except ConnectionError:
            raise KeyError("bug in the recovery path")  # noqa: B904
    except KeyError as caught:
        exc = caught

    assert isinstance(exc.__context__, transient_failure_classifier)
    assert exc.__cause__ is None
    assert not is_transient_failure(exc)
    record = _report(caplog, exc)
    assert record.levelno == logging.ERROR
    assert record.exc_info


def test_cyclic_chain_terminates(transient_failure_classifier):
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__cause__, second.__cause__ = second, first

    assert not is_transient_failure(first)


def test_raising_classifier_is_not_transient_and_still_reports(caplog):
    def broken(_exc: BaseException) -> bool:
        raise KeyError("classifier bug")

    configure_transient_failure_classifier(broken)
    try:
        record = _report(caplog, _raised(ConnectionError("dropped")))
        assert record.levelno == logging.ERROR
        assert record.exc_info
        assert "transient_failure_classifier_failed" in caplog.text
    finally:
        configure_transient_failure_classifier(None)


# --- the streak -------------------------------------------------------------


def test_one_blip_then_success_is_a_single_warning(
    caplog, clock, transient_failure_classifier
):
    record = _report(caplog, _raised(transient_failure_classifier("x")))
    report_background_success(_SCOPE)

    assert record.levelno == logging.WARNING
    assert _SCOPE not in background_work._streaks


def test_fast_loop_failing_for_a_minute_never_escalates(
    caplog, clock, transient_failure_classifier
):
    """Ten failures over 60s: the count is met, the duration is not."""
    levels = []
    for _ in range(10):
        levels.append(_report(caplog, _raised(transient_failure_classifier("x"))))
        clock[0] += 6.0

    assert [r.levelno for r in levels] == [logging.WARNING] * 10
    assert "consecutive_failures=10" in levels[-1].getMessage()


def test_three_failures_spanning_the_floor_escalate_with_traceback(
    caplog, clock, transient_failure_classifier
):
    first = _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 150
    second = _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 150
    third = _report(caplog, _raised(transient_failure_classifier("x")))

    assert [first.levelno, second.levelno] == [logging.WARNING, logging.WARNING]
    assert third.levelno == logging.ERROR
    assert third.exc_info
    message = third.getMessage()
    assert (
        "event=worker_failed transient=true escalated=true consecutive_failures=3 "
        "streak_seconds=300 tracked=true scope=worker:org-1" in message
    )


def test_two_failures_spanning_the_floor_do_not_escalate(
    caplog, clock, transient_failure_classifier
):
    """The duration is met, the count is not: one slow retry is not an outage."""
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 3600
    record = _report(caplog, _raised(transient_failure_classifier("x")))

    assert record.levelno == logging.WARNING


def test_daily_failures_never_form_a_streak(
    caplog, clock, transient_failure_classifier
):
    """A day apart is past the idle horizon, so each failure starts afresh.

    This is why daily and hourly jobs are NOT put on this policy: they log
    their failures at ERROR directly, and a streak would page days late.
    """
    levels = []
    for _ in range(3):
        levels.append(_report(caplog, _raised(transient_failure_classifier("x"))))
        clock[0] += 86_400

    assert [r.levelno for r in levels] == [logging.WARNING] * 3


def test_success_resets_the_streak(caplog, clock, transient_failure_classifier):
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 200
    _report(caplog, _raised(transient_failure_classifier("x")))
    report_background_success(_SCOPE)
    clock[0] += 200
    record = _report(caplog, _raised(transient_failure_classifier("x")))

    assert record.levelno == logging.WARNING
    assert "consecutive_failures=1 " in record.getMessage()


def test_success_for_another_scope_does_not_reset(
    caplog, clock, transient_failure_classifier
):
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 200
    _report(caplog, _raised(transient_failure_classifier("x")))
    report_background_success("worker:org-2")
    clock[0] += 200
    record = _report(caplog, _raised(transient_failure_classifier("x")))

    assert record.levelno == logging.ERROR


def test_success_for_an_untracked_scope_is_a_no_op():
    report_background_success("never-failed")

    assert "never-failed" not in background_work._streaks


def test_scopes_escalate_independently(caplog, clock, transient_failure_classifier):
    for _ in range(2):
        _report(caplog, _raised(transient_failure_classifier("x")), scope="a")
        clock[0] += 200
    other = _report(caplog, _raised(transient_failure_classifier("x")), scope="b")
    same = _report(caplog, _raised(transient_failure_classifier("x")), scope="a")

    assert other.levelno == logging.WARNING
    assert same.levelno == logging.ERROR


def test_more_failing_scopes_than_tracked_is_an_error(
    caplog, clock, monkeypatch, transient_failure_classifier
):
    monkeypatch.setattr(background_work, "_MAX_TRACKED_SCOPES", 3)
    for scope in ("a", "b", "c"):
        assert (
            _report(caplog, _raised(transient_failure_classifier("x")), scope=scope)
        ).levelno == logging.WARNING

    untracked = _report(caplog, _raised(transient_failure_classifier("x")), scope="d")

    assert untracked.levelno == logging.ERROR
    assert untracked.exc_info
    assert "tracked=false" in untracked.getMessage()
    assert set(background_work._streaks) == {"a", "b", "c"}

    report_background_success("a")  # a success makes room again
    record = _report(caplog, _raised(transient_failure_classifier("x")), scope="d")
    assert record.levelno == logging.WARNING
    assert set(background_work._streaks) == {"b", "c", "d"}


def test_a_streak_idle_for_over_an_hour_is_forgotten(
    caplog, clock, transient_failure_classifier
):
    """Blips days apart on a unit that went idle must not add up to a page."""
    for _ in range(2):
        _report(caplog, _raised(transient_failure_classifier("x")))
        clock[0] += background_work._STREAK_IDLE_RESET_SECONDS + 1

    record = _report(caplog, _raised(transient_failure_classifier("x")))

    assert record.levelno == logging.WARNING
    assert background_work._streaks[_SCOPE][0] == 1


def test_a_streak_failing_within_the_idle_horizon_still_escalates(
    caplog, clock, transient_failure_classifier
):
    levels = []
    for _ in range(3):
        levels.append(
            _report(caplog, _raised(transient_failure_classifier("x"))).levelno
        )
        clock[0] += background_work._STREAK_IDLE_RESET_SECONDS - 1

    assert levels == [logging.WARNING, logging.WARNING, logging.ERROR]


def test_idle_streaks_make_room_at_capacity(
    caplog, clock, monkeypatch, transient_failure_classifier
):
    monkeypatch.setattr(background_work, "_MAX_TRACKED_SCOPES", 2)
    for scope in ("a", "b"):
        _report(caplog, _raised(transient_failure_classifier("x")), scope=scope)
    clock[0] += background_work._STREAK_IDLE_RESET_SECONDS + 1

    record = _report(caplog, _raised(transient_failure_classifier("x")), scope="c")

    assert record.levelno == logging.WARNING
    assert set(background_work._streaks) == {"c"}


def test_default_tracked_scope_bound_is_4096():
    assert background_work._MAX_TRACKED_SCOPES == 4096


# --- detail=: value-free rendering ------------------------------------------

_SECRET = "postgres://admin:hunter2-SECRET@db.internal:5432/x"


def _value_free(exc: BaseException) -> str:
    return f"error_class={type(exc).__name__}"


def _no_record_mentions_secret(caplog) -> bool:
    for record in caplog.records:
        rendered = record.getMessage()
        if record.exc_info:
            rendered += logging.Formatter().formatException(record.exc_info)
        if "hunter2-SECRET" in rendered:
            return False
    return True


@pytest.mark.parametrize("transient", [True, False], ids=["transient", "bug"])
def test_detail_keeps_the_message_out_of_every_record(
    caplog, clock, transient_failure_classifier, transient
):
    def make() -> BaseException:
        if transient:
            return _raised(transient_failure_classifier(_SECRET))
        return _raised(ValueError(_SECRET))

    caplog.clear()
    for _ in range(3):  # the third, if transient, is the escalated ERROR
        report_background_failure(
            _LOGGER, "worker_failed", make(), scope=_SCOPE, detail=_value_free
        )
        clock[0] += 200
    records = [r for r in caplog.records if r.name == _LOGGER.name]

    assert len(records) == 3
    assert _no_record_mentions_secret(caplog)
    assert records[-1].levelno == logging.ERROR
    assert not records[-1].exc_info
    assert "error_class=" in records[-1].getMessage()


def test_capture_does_not_inherit_other_policy_tests_duplicate_history(caplog):
    from reflexio.cli.log_format import DuplicateFilter

    message = "event=worker_failed scope=worker:org-1 org_id=o"
    duplicate_filter = DuplicateFilter(window_seconds=0)
    duplicate_filter._suppressed[
        ("tests.background_failure_policy", "event=%s scope=%s%s%s")
    ] = 6
    _LOGGER.addFilter(duplicate_filter)
    try:
        record = _report(caplog, _raised(ValueError("bad row")))
        assert record.getMessage() == message
        assert record.exc_info and record.exc_info[1].args == ("bad row",)
    finally:
        _LOGGER.removeFilter(duplicate_filter)
