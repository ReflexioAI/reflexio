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
import logging.handlers
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from reflexio.server import configure_logging, resolve_log_level

_FIRST_PARTY = ("reflexio", "reflexio_ext")


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Give each test the root logger back, handlers and every pinned level.

    ``reflexio.server`` configured the root at import, and pytest attaches its
    own handlers, so without this every test would inherit whichever profile ran
    last -- and the order-dependent failures would look like real ones.

    EVERY existing logger's level is snapshotted, not just the two first-party
    roots. ``configure_logging`` also pins ``litellm``, ``httpx``,
    ``site_var_manager`` and whatever ``info_loggers`` names, so a narrower
    fixture left ``reflexio_ext.billing`` at INFO and leaked it into later tests.
    Raised by CodeRabbit on reflexio#551, and the same order-dependence class that
    made a sibling test in this file pass alone and fail in the full suite.
    """
    root = logging.getLogger()
    saved_root = (root.level, list(root.handlers))
    saved = {
        name: existing.level
        for name, existing in root.manager.loggerDict.items()
        if isinstance(existing, logging.Logger)
    }
    try:
        yield
    finally:
        root.setLevel(saved_root[0])
        root.handlers[:] = saved_root[1]
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
        # Loggers CREATED by a test did not exist at snapshot time, so restoring
        # the snapshot cannot reset them. NOTSET is what they would have had.
        for name in set(root.manager.loggerDict) - set(saved):
            candidate = root.manager.loggerDict[name]
            if isinstance(candidate, logging.Logger):
                candidate.setLevel(logging.NOTSET)


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


# --------------------------------------------------------------------------
# The two defects Codex found on reflexio#551, each with its control.
# --------------------------------------------------------------------------


def test_a_debug_level_actually_reaches_the_stdout_handler() -> None:
    """An accepted value that does nothing is worse than a rejected one.

    ``REFLEXIO_LOG_LEVEL=debug`` lowered the root to DEBUG while the production
    handler stayed pinned at INFO, so every DEBUG record was discarded on its way
    out. The setting was accepted and inert -- the exact class of defect this
    module was rewritten to remove.
    """
    _blank_root()

    report = configure_logging(verbose=False, level=logging.DEBUG)

    assert report.handlers == ("StreamHandler:DEBUG",), (
        "the stdout handler must drop to the resolved root level, or a DEBUG "
        "record never leaves the process"
    )
    handler = logging.getLogger().handlers[0]
    assert handler.level <= logging.DEBUG


def test_the_handler_floor_stays_at_info_by_default() -> None:
    """The control, and the reason the handler is not simply set to the root level.

    INFO is BELOW the production root of WARNING on purpose: a single logger
    raised to INFO must still reach the log driver. A handler pinned to the root's
    WARNING would silence exactly the per-subsystem opt-in that
    ``REFLEXIO_INFO_LOGGERS`` exists to provide.
    """
    _blank_root()

    report = configure_logging(verbose=False)

    assert report.handlers == ("StreamHandler:INFO",)


def test_a_second_call_updates_the_handler_level_rather_than_stacking() -> None:
    """The handler is reused, so its level has to be re-applied, not just set once."""
    _blank_root()

    configure_logging(verbose=False)
    report = configure_logging(verbose=False, level=logging.DEBUG)

    assert report.handlers == ("StreamHandler:DEBUG",)


def test_a_noisy_logger_named_in_info_loggers_stays_suppressed() -> None:
    """Suppression must win over a per-subsystem override, as it did before.

    ``REFLEXIO_INFO_LOGGERS=litellm`` overwrote litellm's WARNING pin and turned
    its INFO traffic on in production -- a volume regression nothing else would
    have caught, and one the ordering in the original module body prevented.
    """
    _blank_root()
    logging.getLogger("litellm").setLevel(logging.NOTSET)

    report = configure_logging(verbose=False, info_loggers=("litellm",))

    assert not logging.getLogger("litellm").isEnabledFor(logging.INFO)
    assert ("litellm", logging.WARNING) in report.pinned, (
        "the report must show the level that actually won, not the one requested"
    )


def test_a_non_noisy_logger_named_in_info_loggers_is_still_raised() -> None:
    """The control. Suppression winning must not mean the override never works."""
    _blank_root()

    configure_logging(verbose=False, info_loggers=("reflexio_ext.billing",))

    assert logging.getLogger("reflexio_ext.billing").isEnabledFor(logging.INFO)


def test_a_handler_this_module_did_not_attach_is_left_alone() -> None:
    """Only OUR stdout handler may have its level adjusted.

    ``_MANAGED_HANDLER_ATTR`` exists for this, and without this test the marker
    could be deleted with nothing failing -- found by mutation: rewriting the reuse
    branch to adjust any ``StreamHandler`` left all 34 other tests green.

    The hazard is concrete rather than theoretical. pytest's own
    ``LogCaptureHandler`` is a ``StreamHandler`` subclass, so a module that
    "reuses the StreamHandler it finds" would silently re-level the test
    framework's capture handler -- and in a deployed process, any library handler
    attached before ours.
    """
    root = _blank_root()
    foreign = logging.StreamHandler()
    foreign.setLevel(logging.CRITICAL)
    root.addHandler(foreign)

    configure_logging(verbose=False, level=logging.DEBUG)

    assert foreign.level == logging.CRITICAL, (
        "configure_logging re-levelled a handler it did not attach"
    )


def test_a_foreign_rotating_handler_does_not_suppress_the_verbose_profile() -> None:
    """The verbose profile must still get its console and both log files.

    ``RotatingFileHandler`` is a ``StreamHandler`` subclass via ``FileHandler``, so
    a class-based guard treated one unrelated rotating handler -- an embedding
    application's own -- as proof that both Reflexio file handlers AND the console
    already existed. The result was a verbose profile with no colored console, no
    ``dev.log`` and no ``llm_io.log``. Raised by Codex on reflexio#551.
    """
    root = _blank_root()
    foreign = logging.handlers.RotatingFileHandler(
        Path(tempfile.gettempdir()) / "foreign_probe.log", maxBytes=1024, backupCount=0
    )
    root.addHandler(foreign)
    try:
        report = configure_logging(verbose=True)

        roles = {
            getattr(h, "_reflexio_managed_handler_role", None) for h in root.handlers
        }
        assert {"console", "devlog", "llmio"} <= roles, (
            f"the verbose profile skipped its own handlers; roles present: {roles}"
        )
        assert report.handlers.count("RotatingFileHandler:DEBUG") == 2
    finally:
        foreign.close()
        root.removeHandler(foreign)


def test_a_foreign_console_still_suppresses_ours() -> None:
    """The control: the foreign-console check must keep working.

    Excluding ``FileHandler`` narrowed that check; it did not remove it. A real
    console attached by an embedding application must still win, or two handlers
    write every record to the same stream.
    """
    root = _blank_root()
    foreign = logging.StreamHandler()
    root.addHandler(foreign)

    configure_logging(verbose=False)

    roles = {getattr(h, "_reflexio_managed_handler_role", None) for h in root.handlers}
    assert "stdout" not in roles, "a genuine foreign console must still suppress ours"
