from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pytest

from reflexio.server.error_reporting import (
    ErrorLevel,
    capture_anomaly,
    configure_error_reporter,
    error_tags,
    set_error_tags,
)


class RecordingReporter:
    def __init__(self) -> None:
        self.context_tags: list[dict[str, str]] = []
        self.current_tags: list[dict[str, str]] = []
        self.anomalies: list[tuple[str, ErrorLevel, dict[str, str]]] = []

    @contextmanager
    def error_tags(self, tags: Mapping[str, str]) -> Iterator[None]:
        self.context_tags.append(dict(tags))
        yield

    def set_error_tags(self, tags: Mapping[str, str]) -> None:
        self.current_tags.append(dict(tags))

    def capture_anomaly(
        self,
        message: str,
        *,
        level: ErrorLevel,
        tags: Mapping[str, str],
    ) -> None:
        self.anomalies.append((message, level, dict(tags)))


@pytest.fixture(autouse=True)
def _clear_reporter() -> Iterator[None]:
    configure_error_reporter(None)
    yield
    configure_error_reporter(None)


def test_unconfigured_reporter_is_noop() -> None:
    with error_tags(org_id="org-1"):
        pass
    set_error_tags(org_id="org-1")
    capture_anomaly("demo", org_id="org-1")


def test_facade_normalizes_tags_once_and_omits_none() -> None:
    reporter = RecordingReporter()
    configure_error_reporter(reporter)

    with error_tags(org_id="org-1", attempt=2, missing=None):
        pass
    set_error_tags(active=True, missing=None)
    capture_anomaly("demo", level="error", count=3, missing=None)

    assert reporter.context_tags == [{"org_id": "org-1", "attempt": "2"}]
    assert reporter.current_tags == [{"active": "True"}]
    assert reporter.anomalies == [("demo", "error", {"count": "3"})]


def test_context_uses_reporter_snapshot() -> None:
    first = RecordingReporter()
    second = RecordingReporter()
    configure_error_reporter(first)

    with error_tags(source="first"):
        configure_error_reporter(second)

    assert first.context_tags == [{"source": "first"}]
    assert second.context_tags == []


class _FailingContext:
    def __init__(self, *, fail_enter: bool = False, fail_exit: bool = False) -> None:
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit

    def __enter__(self) -> None:
        if self.fail_enter:
            raise RuntimeError("enter failed")

    def __exit__(self, *_args: object) -> None:
        if self.fail_exit:
            raise RuntimeError("exit failed")


class FailingReporter(RecordingReporter):
    def __init__(self, *, fail_enter: bool = False, fail_exit: bool = False) -> None:
        super().__init__()
        self.fail_enter = fail_enter
        self.fail_exit = fail_exit

    def error_tags(self, tags: Mapping[str, str]) -> _FailingContext:
        del tags
        return _FailingContext(fail_enter=self.fail_enter, fail_exit=self.fail_exit)

    def set_error_tags(self, tags: Mapping[str, str]) -> None:
        del tags
        raise RuntimeError("set failed")

    def capture_anomaly(
        self,
        message: str,
        *,
        level: ErrorLevel,
        tags: Mapping[str, str],
    ) -> None:
        del message, level, tags
        raise RuntimeError("capture failed")


@pytest.mark.parametrize("fail_enter,fail_exit", [(True, False), (False, True)])
def test_reporter_context_failure_does_not_break_product_code(
    fail_enter: bool, fail_exit: bool
) -> None:
    configure_error_reporter(
        FailingReporter(fail_enter=fail_enter, fail_exit=fail_exit)
    )

    with error_tags(operation="demo"):
        pass


def test_reporter_exit_failure_does_not_mask_original_exception() -> None:
    configure_error_reporter(FailingReporter(fail_exit=True))
    original = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt) as raised, error_tags(operation="demo"):
        raise original

    assert raised.value is original


def test_non_context_reporter_failures_are_swallowed() -> None:
    configure_error_reporter(FailingReporter())

    set_error_tags(operation="demo")
    capture_anomaly("demo")
