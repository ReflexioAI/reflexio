"""``configure_logging`` must be callable, idempotent, and never raise a threshold.

This function used to be a module body: ~100 lines that ran as an import side
effect of ``reflexio.server``, with no name, no arguments and no return value.
Nothing could call it, so nothing could test it -- and working out what a
deployed process could actually log meant running a child interpreter against it.
Two measurements in one session got it wrong first, and a test that tried to
observe it instead rebuilt the configuration and asserted against its own work.

Two properties carry real weight here:

**Lower-only.** ``level`` and ``sentry_floor`` may lower a threshold and never
raise one. ``SENTRY_LOGS_LEVEL=error`` -- a plausible "just send me errors" --
used to raise the first-party loggers to ERROR and delete every WARNING line from
stdout, which for a self-host customer is the only channel there is.

**Idempotent.** The old body guarded its console handler against a second
attachment but NOT its two file handlers, which was invisible while it ran once
and stacks on every call now that it is a function. Measured before the guard was
added: three handlers became five, then seven.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from reflexio.server import configure_logging, resolve_log_level

_FIRST_PARTY = ("reflexio", "reflexio_ext")


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Give each test the root logger back, handlers and levels included.

    ``reflexio.server`` configured the root at import, and pytest attaches its
    own handlers, so without this every test would inherit whichever profile ran
    last -- and the order-dependent failures would look like real ones.
    """
    root = logging.getLogger()
    saved_root = (root.level, list(root.handlers))
    saved = {name: logging.getLogger(name).level for name in _FIRST_PARTY}
    try:
        yield
    finally:
        root.setLevel(saved_root[0])
        root.handlers[:] = saved_root[1]
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def _blank_root() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    for name in _FIRST_PARTY:
        logging.getLogger(name).setLevel(logging.NOTSET)
    return root


# --------------------------------------------------------------------------
# The two profiles, and that the production one is unchanged from the body.
# --------------------------------------------------------------------------


def test_the_production_profile_is_a_warning_root_and_one_stdout_handler() -> None:
    """Exactly what the module body produced, so the extraction changed nothing.

    ``reflexio_ext/tests/deployment/test_self_host_observability.py`` measures the
    same shape through the real import path in a child interpreter. If these two
    ever disagree, the extraction has drifted from the thing it replaced.
    """
    _blank_root()

    report = configure_logging(verbose=False)

    assert report.profile == "production"
    assert report.root_level == logging.WARNING
    assert report.handlers == ("StreamHandler:INFO",)


def test_the_verbose_profile_adds_the_developer_file_handlers() -> None:
    """The control. Without it, a function that ignored ``verbose`` would pass above."""
    _blank_root()

    report = configure_logging(verbose=True)

    assert report.profile == "verbose"
    assert report.root_level == logging.DEBUG
    assert report.handlers.count("RotatingFileHandler:DEBUG") == 2


def test_the_report_describes_the_live_root_rather_than_its_arguments() -> None:
    """A report assembled from the inputs would agree with itself and prove nothing."""
    _blank_root()

    report = configure_logging(verbose=False)

    root = logging.getLogger()
    assert report.root_level == root.level
    assert len(report.handlers) == len(root.handlers)


# --------------------------------------------------------------------------
# Idempotency -- the bug the extraction exposed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verbose", [False, True])
def test_calling_it_twice_attaches_no_second_handler(verbose: bool) -> None:
    """Measured without the file-handler guard: 3 handlers became 5, then 7.

    Each extra ``RotatingFileHandler`` writes every record to the same file
    again, so the symptom is duplicated lines rather than an error.
    """
    _blank_root()

    first = configure_logging(verbose=verbose)
    second = configure_logging(verbose=verbose)

    assert second.handlers == first.handlers
    assert second == first, "a second call must be a no-op, report included"


# --------------------------------------------------------------------------
# Lower-only. The money property.
# --------------------------------------------------------------------------


def test_a_sentry_floor_below_the_effective_level_lowers_first_party() -> None:
    """Required: SENTRY_LOGS_LEVEL=info forwards nothing if INFO never fires."""
    _blank_root()

    configure_logging(verbose=False, sentry_floor=logging.INFO)

    for name in _FIRST_PARTY:
        assert logging.getLogger(f"{name}.child").isEnabledFor(logging.INFO), (
            f"{name} was not lowered, so Sentry Logs would receive nothing"
        )


def test_a_sentry_floor_above_the_effective_level_never_raises_it() -> None:
    """The defect that shipped. A WARNING must survive a floor of ERROR.

    Sentry filters its Logs pipeline on its own threshold independently of the
    logger level, so raising here would buy nothing and cost every WARNING line
    on the only channel a self-host operator has.
    """
    _blank_root()

    configure_logging(verbose=False, sentry_floor=logging.ERROR)

    for name in _FIRST_PARTY:
        assert logging.getLogger(f"{name}.child").isEnabledFor(logging.WARNING), (
            f"a floor of ERROR raised {name} and deleted its WARNING records"
        )


def test_a_sentry_floor_does_not_touch_the_root() -> None:
    """Scope preservation: the floor applied to first-party loggers before, too.

    Widening it to the root would push third-party INFO -- litellm, httpx -- into
    a paid pipeline, which is a volume regression rather than a behaviour one and
    would not show up as a failure anywhere else.
    """
    _blank_root()

    report = configure_logging(verbose=False, sentry_floor=logging.DEBUG)

    assert report.root_level == logging.WARNING


def test_the_level_argument_lowers_the_root(  # noqa: D103
) -> None:
    _blank_root()

    report = configure_logging(verbose=False, level=logging.DEBUG)

    assert report.root_level == logging.DEBUG


def test_the_level_argument_cannot_raise_the_root_above_warning() -> None:
    """The control for the test above, and the same asymmetry one level up.

    ``REFLEXIO_LOG_LEVEL=critical`` must not be a way to silence warnings.
    """
    _blank_root()

    report = configure_logging(verbose=False, level=logging.CRITICAL)

    assert report.root_level == logging.WARNING


# --------------------------------------------------------------------------
# Pinning.
# --------------------------------------------------------------------------


def test_info_loggers_are_pinned_individually_and_reported() -> None:
    """The surgical alternative to a global level, and the one to prefer."""
    _blank_root()

    report = configure_logging(verbose=False, info_loggers=("reflexio_ext.billing",))

    assert ("reflexio_ext.billing", logging.INFO) in report.pinned
    assert logging.getLogger("reflexio_ext.billing").isEnabledFor(logging.INFO)
    assert report.root_level == logging.WARNING, (
        "a per-subsystem opt-in must not move the global floor -- that is the "
        "2026-07-04 incident it exists to avoid"
    )


def test_noisy_third_party_loggers_stay_at_warning_in_both_profiles() -> None:
    """Verbose logging is for OUR code; litellm at DEBUG buries it."""
    _blank_root()

    report = configure_logging(verbose=True)

    assert ("litellm", logging.WARNING) in report.pinned
    assert ("httpx", logging.WARNING) in report.pinned


# --------------------------------------------------------------------------
# Level-name resolution.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("debug", logging.DEBUG),
        ("WARNING", logging.WARNING),
        ("  info  ", logging.INFO),
        (None, logging.WARNING),
        ("", logging.WARNING),
        ("verbose", logging.WARNING),
        ("20", logging.WARNING),
    ],
)
def test_resolve_log_level_falls_back_rather_than_raising(
    raw: str | None, expected: int
) -> None:
    """An unusable value must not stop a process booting.

    ``"20"`` is in the table deliberately: a numeric string looks like a level and
    is not one, so accepting it would let ``REFLEXIO_LOG_LEVEL=20`` mean something
    different from ``REFLEXIO_LOG_LEVEL=info`` while looking equivalent.
    """
    assert resolve_log_level(raw, logging.WARNING) == expected


def test_the_report_is_frozen_so_a_caller_cannot_edit_it() -> None:
    """Hashability is the check, and class identity deliberately is not.

    An earlier version asserted ``isinstance(report, LoggingReport)`` and failed
    only in the full suite: ``tests/server/services/storage/test_storage_defaults.py``
    calls ``importlib.reload`` on ``reflexio.server`` five times to re-evaluate its
    module-level globals, and each reload defines a NEW ``LoggingReport`` class
    object. The report was a ``LoggingReport``; it was not the test's
    ``LoggingReport``.

    Class identity across a reload boundary is not a property worth pinning, and
    asserting it made this file order-dependent. Frozen-ness is worth pinning: a
    mutable report could be edited by a caller and then no longer describe what
    the function did.
    """
    _blank_root()

    report = configure_logging(verbose=False)

    hash(report)  # raises TypeError if the dataclass stops being frozen
    with pytest.raises(AttributeError):
        report.root_level = logging.DEBUG  # type: ignore[misc]


# --------------------------------------------------------------------------
# Profile resolution. One name, with the old three kept as aliases.
# --------------------------------------------------------------------------


def _resolve(monkeypatch: pytest.MonkeyPatch, **env: str) -> dict[str, object]:
    """Resolve with a clean slate, so an ambient ~/.reflexio/.env cannot decide."""
    from reflexio.server import resolve_logging_env

    for name in (
        "REFLEXIO_LOG_PROFILE",
        "DEBUG_LOG_TO_CONSOLE",
        "ENVIRONMENT",
        "REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS",
        "REFLEXIO_LOG_LEVEL",
        "REFLEXIO_INFO_LOGGERS",
    ):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return resolve_logging_env()


def test_the_new_profile_name_selects_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve(monkeypatch, REFLEXIO_LOG_PROFILE="verbose")["verbose"] is True


def test_the_default_is_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing set must be the safe branch, not the chatty one."""
    assert _resolve(monkeypatch)["verbose"] is False


def test_the_legacy_flag_still_selects_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-host customer or a dev machine may only set the old name."""
    assert _resolve(monkeypatch, DEBUG_LOG_TO_CONSOLE="true")["verbose"] is True


def test_the_new_name_wins_over_the_legacy_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise an ambient ~/.reflexio/.env would override a deliberate choice."""
    assert (
        _resolve(
            monkeypatch, REFLEXIO_LOG_PROFILE="production", DEBUG_LOG_TO_CONSOLE="true"
        )["verbose"]
        is False
    )


@pytest.mark.parametrize(
    "requested_via", ["REFLEXIO_LOG_PROFILE", "DEBUG_LOG_TO_CONSOLE"]
)
def test_production_refuses_a_verbose_request_from_either_name(
    monkeypatch: pytest.MonkeyPatch, requested_via: str
) -> None:
    """The gate that stops a copied env file making production chatty.

    It applied to the legacy flag only; extending it to the new name is the
    behaviour-preserving generalization, and the parametrization is what stops the
    new name becoming a way around it.
    """
    value = "verbose" if requested_via == "REFLEXIO_LOG_PROFILE" else "true"
    resolved = _resolve(monkeypatch, **{requested_via: value}, ENVIRONMENT="production")

    assert resolved["verbose"] is False


def test_the_break_glass_override_still_re_enables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: a deliberate incident override must work."""
    resolved = _resolve(
        monkeypatch,
        REFLEXIO_LOG_PROFILE="verbose",
        ENVIRONMENT="production",
        REFLEXIO_ALLOW_PRODUCTION_DEBUG_LOGS="true",
    )

    assert resolved["verbose"] is True


def test_an_unrecognised_profile_falls_back_to_the_legacy_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must not silently mean production while the old flag says verbose."""
    resolved = _resolve(
        monkeypatch, REFLEXIO_LOG_PROFILE="cheerful", DEBUG_LOG_TO_CONSOLE="true"
    )

    assert resolved["verbose"] is True


def test_the_log_level_and_info_loggers_are_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve(
        monkeypatch,
        REFLEXIO_LOG_LEVEL="debug",
        REFLEXIO_INFO_LOGGERS=" reflexio_ext.billing , reflexio.search ,, ",
    )

    assert resolved["level"] == logging.DEBUG
    assert resolved["info_loggers"] == ("reflexio_ext.billing", "reflexio.search")
