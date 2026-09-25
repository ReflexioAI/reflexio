"""The publish-timing line must survive every path a publish can take.

`reflexio/server/publish_timing.py` carries its own module-level guards in
`reflexio_ext/tests/test_publish_timing.py`. What those cannot reach is WHERE
the request path calls it from, and the placement is most of the design. There
are exactly two reporting points, and they partition the exits:

* `publisher_api.add_user_interaction` -- the worker's outermost frame --
  reports everything that reached the worker. It is out here rather than in
  `GenerationService.run` because `run` returns BEFORE the post-commit
  coverage reads (`lib/_interactions.py::_safe_coverage`, two remote round
  trips on every publish) and never sees a cold `get_reflexio` fail at all.
* the route reports the admission exits, which never reach the worker: the
  503, and a client disconnect while the request is still queued. That is the
  dominant production outcome: measured from ALB access logs 2026-09-24,
  23:15-23:45 UTC, 59 of 59 publish requests ended `elb=460 target=-`. A 460
  is the client giving up, NOT a lost publish -- 48 of those 59 still
  committed.

And one thing that must NOT be counted: the in-process coverage wait under
`defer_learning=False` is blocking on a background worker, not serving the
publish, so `run` subtracts it.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from starlette.requests import Request

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server import publish_timing
from reflexio.server.api_endpoints import publisher_api
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.rate_limit import limiter
from reflexio.server.services.generation_service import GenerationService

_ORG_ID = "test_org_publish_timing"
_USER_ID = "test_user_publish_timing"
_PUBLISHER_MODULE = publisher_api.__name__


@pytest.fixture(autouse=True)
def _timing_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Report every publish, with no throttle, so nothing is time-dependent."""
    publish_timing.reset_for_tests()
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setenv(publish_timing.ENV_ENABLED, "true")
    monkeypatch.setenv(publish_timing.ENV_THRESHOLD_MS, "0")
    monkeypatch.setenv(publish_timing.ENV_INTERVAL_SECONDS, "0")
    yield
    publish_timing.reset_for_tests()


def _publish_request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        request_id="req-publish-timing",
        user_id=_USER_ID,
        interaction_data_list=[
            InteractionData(
                content="test interaction",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
        ],
        session_id="test_session_id",
    )


def _timing_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == publish_timing.__name__
        and record.levelno == publish_timing.PUBLISH_TIMING_LOG_LEVEL
    ]


def _starlette_request() -> Request:
    """A real `Request`: slowapi's decorator rejects anything else."""
    app = SimpleNamespace(state=SimpleNamespace(limiter=limiter))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/publish_interaction",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
            "app": app,
        }
    )


def _reflexio(temp_dir: str) -> Reflexio:
    """A real `Reflexio` on a temp SQLite dir, standing in for the cache."""
    return Reflexio(org_id=_ORG_ID, storage_base_dir=temp_dir)


def _service(temp_dir: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(org_id=_ORG_ID, storage_base_dir=temp_dir),
    )


def test_a_successful_publish_reports_exactly_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One line, from the outermost worker frame, covering the whole publish."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        with (
            patch(f"{_PUBLISHER_MODULE}.get_reflexio", return_value=reflexio),
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            publish_timing.collect(),
        ):
            response = publisher_api.add_user_interaction(
                org_id=_ORG_ID, request=_publish_request(), defer_learning=True
            )
    assert response.success, response.message
    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    assert "n_interactions=1" in lines[0]


def test_the_line_covers_the_post_commit_coverage_reads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`_safe_coverage` runs AFTER `run()` returns and is part of serving.

    Two remote round trips on every publish. Reporting from inside `run()`
    left them out entirely: a publish whose service portion was under the
    threshold and whose coverage reads stalled produced no line at all.
    """
    slow_read_s = 0.4

    def slow_status(*_args: object, **_kwargs: object) -> dict[str, str]:
        time.sleep(slow_read_s)
        return {"status": "done", "reason": "complete"}

    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        storage = reflexio._get_storage()  # noqa: SLF001 -- patching its class
        with (
            patch(f"{_PUBLISHER_MODULE}.get_reflexio", return_value=reflexio),
            patch.object(type(storage), "extraction_status", slow_status),
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            publish_timing.collect(),
        ):
            publisher_api.add_user_interaction(
                org_id=_ORG_ID, request=_publish_request(), defer_learning=True
            )

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    total_ms = int(lines[0].split("total_ms=")[1].split()[0])
    assert total_ms >= int(slow_read_s * 1000 * 0.75), (
        f"total_ms={total_ms} excludes the {int(slow_read_s * 1000)}ms coverage "
        f"read, so it is not measuring the whole publish: {lines[0]}"
    )


def test_a_failure_before_the_service_starts_still_reports(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cold `get_reflexio` never reaches `GenerationService.run`.

    Config decryption, a storage pool dial, a credential lookup -- all happen
    before `run()` exists, so a backstop inside `run()` cannot see them. This
    is the whole reason the reporting point is one frame further out.
    """
    with (
        patch(
            f"{_PUBLISHER_MODULE}.get_reflexio",
            side_effect=RuntimeError("storage pool dial timed out"),
        ),
        caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
        publish_timing.collect(),
        pytest.raises(RuntimeError, match="storage pool dial timed out"),
    ):
        publisher_api.add_user_interaction(
            org_id=_ORG_ID, request=_publish_request(), defer_learning=True
        )

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    assert "total_ms=" in lines[0]


def test_a_failed_publish_reports_its_phase_breakdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The phases measured before the raise are why the request failed.

    Without this, a storage timeout produced only `publish_request_failed`
    with an aggregate duration -- the breakdown was available for healthy slow
    requests and missing for broken ones.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        service = _service(temp_dir)
        assert service.storage is not None
        # Driven through the real worker entry point, which is where the
        # reporting lives; the failure is raised deep inside `run()`.
        stub = SimpleNamespace(
            publish_interaction=lambda **kwargs: service.run(
                kwargs["request"], defer_learning=True
            )
        )
        with (
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            patch.object(
                type(service.storage),
                "add_user_interactions_bulk",
                side_effect=RuntimeError("storage timed out"),
            ),
            patch(f"{_PUBLISHER_MODULE}.get_reflexio", return_value=stub),
            publish_timing.collect(),
            pytest.raises(RuntimeError, match="storage timed out"),
        ):
            publisher_api.add_user_interaction(
                org_id=_ORG_ID, request=_publish_request(), defer_learning=True
            )

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    line = lines[0]
    # The phases that closed before the raise, plus the failure metering event.
    for field_name in ("add_request_ms=", "embeddings_ms=", "metering_ms="):
        assert field_name in line, f"{field_name!r} missing from {line!r}"


def test_the_coverage_wait_is_not_counted_as_serving_time(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast publish stays silent however long the caller then waits.

    `total_ms` answers "how long did serving this take". The in-process
    coverage wait under `defer_learning=False` is blocking on a background
    worker; counting it reported a healthy publish as slow (measured at 609ms
    for one whose own phases summed to ~110ms).
    """
    monkeypatch.setenv(publish_timing.ENV_THRESHOLD_MS, "300")

    # `extraction_status` is called twice on this path and the two calls are
    # not alike: the FIRST is `run()`'s coverage wait, which is excluded, and
    # the second is `_safe_coverage`'s reporting read, which is counted.
    # Slowing both would prove nothing -- the line would be legitimately slow.
    calls = {"n": 0}

    def slow_first_poll(*_args: object, **_kwargs: object) -> dict[str, str]:
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.5)
        return {"status": "done", "reason": "complete"}

    with tempfile.TemporaryDirectory() as temp_dir:
        reflexio = _reflexio(temp_dir)
        storage = reflexio._get_storage()  # noqa: SLF001 -- patching its class
        with (
            patch(f"{_PUBLISHER_MODULE}.get_reflexio", return_value=reflexio),
            patch.object(type(storage), "extraction_status", slow_first_poll),
            caplog.at_level(logging.WARNING, logger=publish_timing.__name__),
            publish_timing.collect(),
        ):
            publisher_api.add_user_interaction(
                org_id=_ORG_ID, request=_publish_request(), defer_learning=False
            )
    assert calls["n"] >= 2, "the coverage wait and the reporting read both run"

    assert _timing_lines(caplog) == [], (
        "a fast publish was reported as slow because the coverage wait was "
        f"counted against it: {_timing_lines(caplog)}"
    )


def test_a_publish_cancelled_while_queued_still_reports(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dominant production shape: the client gives up before admission.

    Nearly every production publish ends as an ELB 460 (59/59 in the measured
    window above) -- the client's own ~8s
    timeout fires before the load balancer can respond. A request cancelled
    while still queued in `acquire_ingestion` never reaches the worker at all,
    so the route has to report it.

    A request that HAS reached the worker needs no such help: the work is
    shielded and runs to completion in a thread holding its own copy of this
    context, so it emits the full duration itself. That is why the route's
    emit is scoped to the admission exits rather than to the whole handler.
    """
    from reflexio.server.routes import interactions
    from reflexio.server.services.durable_learning import waiting

    async def never_admits(org_id: str, deadline: float) -> bool:
        await asyncio.sleep(30)  # queued behind other publishes for this org
        return True

    monkeypatch.setattr(waiting, "acquire_ingestion", never_admits)

    async def scenario() -> None:
        task = asyncio.create_task(
            interactions.publish_user_interaction(
                request=_starlette_request(),
                payload=_publish_request(),
                org_id=_ORG_ID,
            )
        )
        await asyncio.sleep(0.15)
        task.cancel()  # the client closed the connection
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.WARNING, logger=publish_timing.__name__):
        asyncio.run(scenario())

    lines = _timing_lines(caplog)
    assert len(lines) == 1, f"expected exactly one timing line, got {lines}"
    assert "admission_ms=" in lines[0], lines[0]


def _retention_entry_points() -> list[str]:
    """Every storage method by which a publish could pay retention cost.

    DERIVED from ``RetentionMixin``'s public surface rather than hand-listed, so
    a new retention entry point is watched the day it is added -- plus the two
    legacy interaction-cap methods, which predate the mixin and are a live
    bypass: ``count_all_interactions`` + ``delete_oldest_interactions`` reproduce
    the exact cost (an exact COUNT and an irreversible oldest-rows delete)
    without touching a single mixin method.
    """
    from reflexio.server.services.storage.retention_mixin import RetentionMixin

    derived = [
        name
        for name in vars(RetentionMixin)
        if not name.startswith("_") and callable(vars(RetentionMixin)[name])
    ]
    assert derived, "the RetentionMixin scan found nothing -- it moved or was renamed"
    return sorted({*derived, "count_all_interactions", "delete_oldest_interactions"})


def test_the_publish_path_issues_no_retention_round_trips() -> None:
    """INVERTED from a phase-attribution guard when its subject was deleted.

    This test used to assert that `retention_sweep_ms` was present and contained
    the sweep's elapsed time, because the sweep ran two lines above
    `publish_start` and was in no phase at all. The sweep has since moved to the
    lineage GC scheduler, so the subject of that assertion is gone. Where the
    deleted thing WAS the test's subject, invert the test: asserting
    `retention_sweep_ms == 0` would be a check that cannot fail, because a key
    that is never written reads as absent either way.

    BOTH halves are required. The missing-phase-key assertion alone still passes
    if someone calls a storage hook outside a `publish_timing.phase` wrapper,
    which is precisely the defect shape the instrument was built to find.

    The call half watches a DERIVED SET, not one method name. A review of the
    first version produced a working bypass in three lines -- `count_all_interactions`
    then `delete_oldest_interactions`, both live in production today, neither
    named `count_retention_target_rows` -- which reintroduced an exact count and
    an irreversible delete onto the publish path while the guard stayed green.

    Known limits, stated rather than left for the next reader to discover:
    it sees only the OSS SQLite storage class, so an enterprise-side publish
    hook is invisible to it; and `calls` is read synchronously after `run`
    returns, so a sweep moved onto a thread spawned by publish would race it.
    """
    watched = _retention_entry_points()
    calls: list[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        service = _service(temp_dir)
        storage = service.storage
        assert storage is not None
        storage_cls = type(storage)

        def _recorder(name: str):
            def _record(_self: object, *_args: object, **_kwargs: object) -> int:
                calls.append(name)
                return 0

            return _record

        patches = [
            patch.object(storage_cls, name, _recorder(name))
            for name in watched
            if hasattr(storage_cls, name)
        ]
        assert len(patches) >= 3, (
            f"only {len(patches)} of {len(watched)} entry points exist on "
            f"{storage_cls.__name__}; the scan is no longer watching the code it guards"
        )

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(publish_timing.collect())
            service.run(_publish_request(), defer_learning=True)
            snap = publish_timing.snapshot()

    assert snap is not None
    assert "retention_sweep_ms" not in snap, (
        "the publish path still reports a retention phase, so the sweep was not "
        f"removed from it: {sorted(snap)}"
    )
    assert calls == [], (
        "the publish path still reaches retention storage methods, so it still "
        f"pays for them whether or not a phase reports it: {sorted(set(calls))}"
    )
