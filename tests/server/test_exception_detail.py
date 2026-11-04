"""Exception detail names the type and raise site, never the message."""

import inspect

from reflexio.server.exception_detail import exception_detail, raised_at


def _raise_inner() -> None:
    raise ValueError("postgresql://user:secret@host/db")


def _raise_line() -> int:
    source, start = inspect.getsourcelines(_raise_inner)
    return start + next(i for i, line in enumerate(source) if "raise " in line)


def _caught() -> ValueError:
    try:
        _raise_inner()
    except ValueError as exc:
        return exc
    raise AssertionError("unreachable")


def test_raised_at_names_the_deepest_frame_by_module():
    assert raised_at(_caught()) == f"{__name__}:{_raise_line()}"


def test_detail_omits_the_message():
    detail = exception_detail(_caught())

    assert detail == f"exception_type=ValueError raised_at={__name__}:{_raise_line()}"
    assert "secret" not in detail


def test_unraised_exception_has_no_locator():
    exc = RuntimeError("never raised")

    assert raised_at(exc) is None
    assert exception_detail(exc) == "exception_type=RuntimeError raised_at=unknown"
