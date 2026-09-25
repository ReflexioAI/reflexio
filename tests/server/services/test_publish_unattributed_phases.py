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
    """Every query in the combined coverage report belongs to this phase."""
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

        def slow_status(_self: object, *_a: object, **_k: object):
            time.sleep(coverage_s)
            return (
                {"status": "pending", "reason": "queued"},
                {"profile": 0, "playbook": 0},
            )

        with (
            patch.object(storage_cls, "add_user_interactions_bulk", slow_bulk),
            patch.object(storage_cls, "extraction_report", slow_status),
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


def test_publish_checks_duplicates_only_in_the_atomic_insert() -> None:
    """No preflight read; the measured insertion arbitrates duplicate IDs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        storage = reflexio._get_storage()
        assert storage is not None
        with (
            patch.object(
                type(storage),
                "get_request",
                side_effect=AssertionError("duplicate read"),
            ),
            publish_timing.collect(),
        ):
            response = reflexio.publish_interaction(
                _publish_request(), defer_learning=True
            )
            snap = publish_timing.snapshot()
        duplicate = reflexio.publish_interaction(
            _publish_request(), defer_learning=True
        )
    assert response.success, response.message
    assert not duplicate.success
    assert "already exists" in duplicate.message
    assert snap is not None
    assert "add_request_ms" in snap
    assert "dup_check_ms" not in snap


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
        "add_request_ms",
        "post_publish_ms",
        "scope_commit_ms",
    ):
        assert key in snap, f"{key} missing from an ordinary publish: {sorted(snap)}"
