"""The publish line must not leave two thirds of a publish unattributed.

Measured on prod over 2.5h, 78 publishes: `total_ms` mean 74,136, of which
**50,369 (68%)** fell outside every named phase, and a further 11,656 sat inside
`commit_scope` but outside its named children.

`publisher_api.add_user_interaction`'s own comment names the culprit it cannot
reach: `_safe_coverage` is "two more remote round trips after [`run`] returns".
It is inside the timing scope and in no phase, so a publish that spends seconds
in it reports the time and attributes none of it.

WHAT THIS DRIVES, AND WHY IT MATTERS
------------------------------------
The real `Reflexio.publish_interaction`, NOT a hand-wrapped call. The first
version of this test opened the phase itself around `_safe_coverage` and then
asserted the phase existed -- it passed against unmodified production code,
because it was measuring its own wrapper. A test that supplies the thing it
checks for cannot fail.

The bound is two-sided on purpose: a lower bound alone passes against a wrap
placed around the whole request under a name claiming to be one part of it.
"""

from __future__ import annotations

import tempfile
import time
from unittest.mock import patch

import pytest

from reflexio.server import publish_timing

from .test_generation_service_publish_timing import (  # noqa: TID252
    _publish_request,
    _reflexio,
)


@pytest.fixture(autouse=True)
def _timing_on(monkeypatch):
    """Timing is off unless the flag is set; `collect()` is otherwise a no-op."""
    monkeypatch.setenv(publish_timing.ENV_ENABLED, "true")
    monkeypatch.setenv(publish_timing.ENV_THRESHOLD_MS, "0")
    monkeypatch.setenv(publish_timing.ENV_INTERVAL_SECONDS, "0")
    publish_timing.reset_for_tests()
    yield
    publish_timing.reset_for_tests()


def test_the_coverage_reads_are_attributed_to_their_own_phase() -> None:
    """`_safe_coverage` is two remote round trips on EVERY publish."""
    coverage_s = 0.30
    ingest_s = 0.60

    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        storage = reflexio._get_storage()
        assert storage is not None
        storage_cls = type(storage)
        original_bulk = storage_cls.add_user_interactions_bulk

        def slow_bulk(self: object, *args: object, **kwargs: object) -> object:
            time.sleep(ingest_s)
            return original_bulk(self, *args, **kwargs)  # type: ignore[arg-type]

        def slow_status(_self: object, *_a: object, **_k: object) -> None:
            time.sleep(coverage_s)
            return

        with (
            patch.object(storage_cls, "add_user_interactions_bulk", slow_bulk),
            patch.object(storage_cls, "extraction_status", slow_status),
            publish_timing.collect(),
        ):
            response = reflexio.publish_interaction(
                _publish_request(), defer_learning=True
            )
            snap = publish_timing.snapshot()

    assert response.success, response.message
    assert snap is not None
    assert "coverage_reads_ms" in snap, (
        "the post-commit coverage reads are in no phase, so a publish that "
        f"spends seconds in them attributes none of it: {sorted(snap)}"
    )
    reads_ms = snap["coverage_reads_ms"]
    assert reads_ms >= int(coverage_s * 1000 * 0.75), (
        f"coverage_reads_ms={reads_ms} does not contain the "
        f"{int(coverage_s * 1000)}ms read, so the phase is not around it"
    )
    # Proves the upper bound is not vacuous: the excluded region really was slow.
    assert snap.get("add_interactions_ms", 0) >= int(ingest_s * 1000 * 0.75), (
        f"the contrasting region was not slow, so the bound proves nothing: {snap}"
    )
    assert reads_ms < int((coverage_s + ingest_s * 0.5) * 1000), (
        f"coverage_reads_ms={reads_ms} has swallowed the {int(ingest_s * 1000)}ms "
        "ingest, so the phase is wrapped around more than the coverage reads"
    )


def test_the_duplicate_check_is_attributed_to_its_own_phase() -> None:
    """`get_request` is a remote read on every publish, inside the scope.

    It was part of the 11,656ms that `commit_scope_ms` reported and did not
    attribute to anything it contains.
    """
    dup_s = 0.30
    ingest_s = 0.60

    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        storage = reflexio._get_storage()
        assert storage is not None
        storage_cls = type(storage)
        original_bulk = storage_cls.add_user_interactions_bulk

        def slow_bulk(self: object, *a: object, **k: object) -> object:
            time.sleep(ingest_s)
            return original_bulk(self, *a, **k)  # type: ignore[arg-type]

        def slow_get_request(_self: object, *_a: object, **_k: object) -> None:
            time.sleep(dup_s)
            return

        with (
            patch.object(storage_cls, "add_user_interactions_bulk", slow_bulk),
            patch.object(storage_cls, "get_request", slow_get_request),
            publish_timing.collect(),
        ):
            response = reflexio.publish_interaction(
                _publish_request(), defer_learning=True
            )
            snap = publish_timing.snapshot()

    assert response.success, response.message
    assert snap is not None
    assert "dup_check_ms" in snap, sorted(snap)
    dup_ms = snap["dup_check_ms"]
    # TWO lookups on the HTTP path: a preflight before the scope and the
    # in-scope one. `phase` accumulates by name, so the line must report the
    # total both round trips cost -- reporting only the second would leave the
    # first in the unattributed remainder, which is the whole point of this
    # change. Raised by review on #536.
    assert dup_ms >= int(2 * dup_s * 1000 * 0.75), (
        f"dup_check_ms={dup_ms} does not cover BOTH lookups "
        f"({int(2 * dup_s * 1000)}ms expected); only one is phased"
    )
    assert snap.get("add_interactions_ms", 0) >= int(ingest_s * 1000 * 0.75), snap
    assert dup_ms < int((2 * dup_s + ingest_s * 0.5) * 1000), (
        f"dup_check_ms={dup_ms} swallowed the ingest, so the phase is too wide"
    )


def test_the_new_phases_all_appear_on_an_ordinary_publish() -> None:
    """Every phase added for attribution must actually be reached.

    A phase wrapped around a branch that never runs contributes nothing and
    would leave the remainder just as unexplained, while looking instrumented.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        with publish_timing.collect():
            response = reflexio.publish_interaction(
                _publish_request(), defer_learning=True
            )
            snap = publish_timing.snapshot()

    assert response.success, response.message
    assert snap is not None
    for key in (
        "coverage_reads_ms",
        "dup_check_ms",
        "post_publish_ms",
        "scope_commit_ms",
    ):
        assert key in snap, f"{key} missing from an ordinary publish: {sorted(snap)}"
