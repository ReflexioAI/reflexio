"""Background failures: transient drops are warnings, bugs and outages are errors."""

from __future__ import annotations

import logging

import pytest

from reflexio.server import background_work
from reflexio.server.background_work import (
    configure_transient_failure_classifier,
    is_transient_failure,
    report_background_failure,
)

_LOGGER = logging.getLogger("tests.background_failure_policy")


def _raised(exc: BaseException) -> BaseException:
    """Return ``exc`` with a real traceback, as a worker's ``except`` sees it."""
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


def _report(caplog, exc: BaseException, scope: str = "worker:org-1"):
    caplog.clear()
    report_background_failure(_LOGGER, "worker_failed", exc, scope=scope, org_id="o")
    records = [r for r in caplog.records if r.name == _LOGGER.name]
    assert len(records) == 1
    return records[0]


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(background_work, "_monotonic", lambda: now[0])
    return now


@pytest.fixture(autouse=True)
def _capture(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER.name)


def test_transient_failure_is_one_warning_without_traceback(
    caplog, transient_failure_classifier
):
    record = _report(caplog, _raised(transient_failure_classifier("ssl eof\nmore")))

    assert record.levelno == logging.WARNING
    assert not record.exc_info
    message = record.getMessage()
    assert "\n" not in message
    assert "event=worker_failed transient=true scope=worker:org-1" in message
    assert "error_class=TransientTestError error=ssl eof more org_id=o" in message


def test_long_transient_message_is_truncated(caplog, transient_failure_classifier):
    record = _report(caplog, _raised(transient_failure_classifier("x" * 5000)))

    assert len(record.getMessage()) < 400


def test_non_transient_failure_is_error_with_traceback(
    caplog, transient_failure_classifier
):
    record = _report(caplog, _raised(ValueError("bad row")))

    assert record.levelno == logging.ERROR
    assert record.exc_info and record.exc_info[1].args == ("bad row",)
    assert record.getMessage() == "event=worker_failed scope=worker:org-1 org_id=o"


def test_without_a_classifier_every_failure_is_an_error(caplog):
    configure_transient_failure_classifier(None)

    record = _report(caplog, _raised(ConnectionError("dropped")))

    assert record.levelno == logging.ERROR
    assert record.exc_info


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_transient_link_anywhere_in_the_chain_is_transient(
    link, caplog, transient_failure_classifier
):
    wrapper = RuntimeError("wrapped")
    setattr(wrapper, link, transient_failure_classifier("dropped"))

    assert is_transient_failure(wrapper)
    assert _report(caplog, _raised(wrapper)).levelno == logging.WARNING


def test_cyclic_chain_terminates(transient_failure_classifier):
    first, second = RuntimeError("a"), RuntimeError("b")
    first.__context__, second.__context__ = second, first

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


def test_transient_failures_persisting_ten_minutes_escalate(
    caplog, clock, transient_failure_classifier
):
    levels = []
    for _ in range(7):  # t = 0, 100, ..., 600 seconds; gaps stay under the gap
        levels.append(_report(caplog, _raised(transient_failure_classifier("x"))))
        clock[0] += 100

    assert [r.levelno for r in levels[:6]] == [logging.WARNING] * 6
    escalated = levels[6]
    assert escalated.levelno == logging.ERROR
    assert escalated.exc_info
    assert "escalated=true transient_for_seconds=600" in escalated.getMessage()


def test_blips_hours_apart_never_escalate(caplog, clock, transient_failure_classifier):
    first = _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 7200
    second = _report(caplog, _raised(transient_failure_classifier("x")))

    assert [first.levelno, second.levelno] == [logging.WARNING, logging.WARNING]


def test_a_gap_longer_than_twenty_minutes_starts_a_new_episode(
    caplog, clock, transient_failure_classifier
):
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 290
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 1201  # gap > 1200: the episode resets at t=1491
    _report(caplog, _raised(transient_failure_classifier("x")))
    clock[0] += 290  # only 290s into the new episode
    record = _report(caplog, _raised(transient_failure_classifier("x")))

    assert record.levelno == logging.WARNING


@pytest.mark.parametrize(
    ("cadence", "escalates_on"),
    [
        (340.0, 3),  # lineage GC / aggregation backoff: 300s plus run time
        (930.0, 2),  # Braintrust: 900s poll plus run time
    ],
    ids=["300s-backoff", "900s-poller"],
)
def test_a_slow_scheduler_still_escalates_a_persistent_outage(
    caplog, clock, transient_failure_classifier, cadence, escalates_on
):
    """Each failure of a slow scheduler must extend the episode, not reset it."""
    levels = []
    for _ in range(escalates_on):
        levels.append(
            _report(caplog, _raised(transient_failure_classifier("x"))).levelno
        )
        clock[0] += cadence

    assert levels[:-1] == [logging.WARNING] * (escalates_on - 1)
    assert levels[-1] == logging.ERROR


def test_scopes_escalate_independently(caplog, clock, transient_failure_classifier):
    for _ in range(6):
        _report(caplog, _raised(transient_failure_classifier("x")), scope="a")
        clock[0] += 100
    other = _report(caplog, _raised(transient_failure_classifier("x")), scope="b")
    same = _report(caplog, _raised(transient_failure_classifier("x")), scope="a")

    assert other.levelno == logging.WARNING
    assert same.levelno == logging.ERROR


def test_tracked_scopes_stay_bounded_and_ended_episodes_make_room(
    caplog, clock, monkeypatch, transient_failure_classifier
):
    monkeypatch.setattr(background_work, "_MAX_TRACKED_SCOPES", 3)
    for scope in ("a", "b", "c", "d"):
        _report(caplog, _raised(transient_failure_classifier("x")), scope=scope)

    assert set(background_work._episodes) == {"a", "b", "c"}

    clock[0] += 1201  # every tracked episode has now ended
    _report(caplog, _raised(transient_failure_classifier("x")), scope="d")

    assert set(background_work._episodes) == {"d"}
