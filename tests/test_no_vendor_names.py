"""This package must not name a proprietary telemetry vendor.

`reflexio` is published to PyPI and installed by people who have no Sentry
account, no enterprise licence, and no copy of the superproject. It went from
ZERO vendor references to fifteen lines of them in #551, when the lower-only
logger rule was moved out of the enterprise layer into `configure_logging` and
the vendor vocabulary was carried along with it.

The move was right -- one owner for logger levels, instead of two subsystems
fighting over them. The vocabulary was not. What this package legitimately needs
is a SCOPE argument ("set the first-party loggers to at most this level"), which
is a statement about loggers. Why a caller wants it, and which product consumes
the result, is the caller's business.

Nothing functional leaked: no environment variable was read, `sentry_sdk` was
never imported, and it never became a dependency. This guard exists because that
distinction is easy to lose on the next refactor, and because #551 verified a
dozen invariants without verifying this one -- the absence is what let it ship.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy_guard

_PACKAGE = Path(__file__).resolve().parents[1] / "reflexio"

#: PROPRIETARY vendors only, and the exclusions are as load-bearing as the
#: entries -- each was measured against this tree, not guessed:
#:
#: * `opentelemetry` was in the first draft and had to come out.
#:   `reflexio/server/llm/token_accounting.py` cites OTel's semantic convention
#:   for `gen_ai.usage.cache_read.input_tokens`, which is a vendor-neutral open
#:   STANDARD and exactly the kind of thing a published package should reference.
#: * `stripe` cannot go in at all: `LOCK_STRIPES` and "striped locks" appear in
#:   `server/cache/reflexio_cache.py` and `sqlite_storage/_base.py`. Eight hits,
#:   none of them the payment vendor.
#:
#: Both would have landed this guard red on correct code, and a guard that lands
#: red is a guard somebody turns off. Recorded as the fix rather than as an
#: allowlist entry: an exception is how a guard stops being one.
_FORBIDDEN = ("sentry", "datadog", "newrelic", "new relic", "honeycomb", "appdynamics")


def _offenders() -> list[str]:
    hits: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        rel = path.relative_to(_PACKAGE.parent).as_posix()
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            lowered = line.lower()
            if any(name in lowered for name in _FORBIDDEN):
                hits.append(f"{rel}:{number}: {line.strip()[:100]}")
    return hits


def test_the_package_names_no_telemetry_vendor() -> None:
    """A published package must not carry an enterprise vendor's vocabulary."""
    offenders = _offenders()

    assert not offenders, (
        "The published package names a proprietary telemetry vendor:\n  "
        + "\n  ".join(offenders)
        + "\n\nThis layer takes SCOPE arguments -- which loggers, at what level. "
        "The vendor, and the reason a caller wants it, belong in the enterprise "
        "layer that calls in."
    )


def test_the_scan_actually_reads_the_package() -> None:
    """Anti-vacuity: a scan over an empty file set passes and measures nothing.

    The path is built from `parents[1]`, so moving this file one directory
    silently empties the scan -- and the test above would be green forever.
    """
    files = list(_PACKAGE.rglob("*.py"))

    assert _PACKAGE.is_dir(), f"{_PACKAGE} is not a directory"
    assert len(files) > 100, (
        f"only {len(files)} python files under {_PACKAGE}; the scan is measuring "
        "almost nothing"
    )
    assert any(p.name == "__init__.py" and p.parent.name == "server" for p in files), (
        "server/__init__.py is not in the scanned set -- it is the file this guard "
    )
    "was written for"


def test_matching_is_case_insensitive() -> None:
    """The original leak appeared as `Sentry`, `SENTRY_LOGS_LEVEL` and `sentry_floor`.

    A case-sensitive scan would have caught roughly a third of it.
    """
    assert all(name == name.lower() for name in _FORBIDDEN), (
        "_FORBIDDEN is compared against a lowercased line, so an uppercase entry "
        "can never match"
    )
