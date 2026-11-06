# ruff: noqa: S101 -- Test-only assertions.
"""Shared behavioural contract for ``OpenWorldStoragePort`` implementations.

One definition, run against every backend. A backend's test module binds a
``port_harness`` fixture and subclasses :class:`OpenWorldPortContract`::

    class TestSQLiteOpenWorldPort(OpenWorldPortContract):
        pass

The class has no ``Test`` prefix, so pytest does not collect it here, only
through a binding.

Each test receives a FRESH harness: an empty store whose queue holds nothing
the test did not seed, because the queue read is store-wide and the ordering
assertions are exact.

Time. A backend that owns its clock (Postgres reads ``clock_timestamp()``)
cannot be moved, so the contract never moves it. Every seed is placed at an
absolute epoch derived from :meth:`PortHarness.now`, or from a value the store
returned, and "later" is reached by ageing stored rows backwards
(:meth:`PortHarness.age_progress`). A harness whose clock is injectable pins
it, which makes the same tests exact. Margins against a moving clock are
minutes wide; boundary cases are seeded relative to the stored
``attempted_at``, which the comparison under test reads directly.

All data is synthetic.

Invariants covered
------------------
I16 -- sweep progress (``OpenWorldSweepProgressStore``):

* one row per playbook per UTC day, its day and ``attempted_at`` from one
  clock reading; a re-record increments ``attempt_count``, replaces the
  outcome and instant, and keeps the earlier job when given none;
* an unusable identity raises and writes nothing;
* the queue holds current playbooks without a blocking row today, never
  attempted first, then oldest last attempt, ties by id;
* the activity window admits a playbook only through a published exposure no
  older than ``window_seconds`` before the start of today, on
  ``COALESCE(exposed_at, ingested_at)``; ``None`` admits every current
  playbook;
* ``rate_limited`` stops blocking after its cooldown, below the shared cap;
* ``ineligible_*`` stops blocking after its cooldown, below the shared cap,
  only when a session that exposed the playbook has a newer request (counted
  ``EVIDENCE_REATTEMPT_LOOKBACK_SECONDS`` late) or a newer outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NamedTuple, Protocol
from uuid import uuid4

import pytest

from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldSweepProgressStore,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    EVIDENCE_REATTEMPT_COOLDOWN_SECONDS,
    EVIDENCE_REATTEMPT_LOOKBACK_SECONDS,
    EVIDENCE_REATTEMPT_PREFIX,
    MAX_SAME_DAY_SWEEP_ATTEMPTS,
    NEVER_ATTEMPTED_EPOCH,
    RATE_LIMIT_RETRY_COOLDOWN_SECONDS,
    RETRY_ELIGIBLE_OUTCOME,
    SweepProgressRecord,
    SweepQueueEntry,
)

_DAY = 86_400
#: A synthetic activity window: two weeks, the shape of the production policy.
WINDOW_SECONDS = 14 * _DAY
_INELIGIBLE = f"{EVIDENCE_REATTEMPT_PREFIX}no_evidence"
#: Older than any aged attempt below, so activity this old is never "new".
_OLD = 3_000
#: Past the evidence cooldown and still inside the rate-limit cooldown.
_PAST_EVIDENCE_COOLDOWN = EVIDENCE_REATTEMPT_COOLDOWN_SECONDS + 100
_USER = "contract-user"


class ProgressRow(NamedTuple):
    """One stored ``open_world_sweep_progress`` row, as the harness reads it."""

    progress_day: str
    outcome: str
    job_id: int | None
    attempted_at: int
    attempt_count: int


class PortHarness(Protocol):
    """What a backend binding supplies: the store and direct-write seeders.

    Seeders write the backend's tables directly, bypassing the port, so a test
    can place rows the port has no method for (requests, outcomes, exposures)
    and move stored rows in time.
    """

    @property
    def storage(self) -> OpenWorldSweepProgressStore: ...

    def now(self) -> int:
        """The store's own clock, in whole epoch seconds."""
        ...

    def seed_playbook(self, *, current: bool = True) -> int:
        """A user playbook; ``current=False`` gives it a non-null status."""
        ...

    def seed_request(self, *, session_id: str, user_id: str, at_epoch: int) -> str:
        """A request created at ``at_epoch``; returns its id."""
        ...

    def seed_session_outcome(
        self, *, session_id: str, user_id: str, at_epoch: int
    ) -> None:
        """A session outcome whose ``created_at`` is ``at_epoch``."""
        ...

    def seed_exposure(
        self,
        *,
        user_playbook_id: int,
        request_id: str | None,
        session_id: str | None,
        user_id: str,
        exposed_at: int | None,
        ingested_at: int,
        published: bool = True,
    ) -> None:
        """An exposure event; ``published=False`` uses a non-publish id."""
        ...

    def age_progress(self, user_playbook_id: int, *, seconds: int) -> None:
        """Move every progress row's ``attempted_at`` back; the day stays."""
        ...

    def backdate_progress(self, user_playbook_id: int, *, days: int) -> None:
        """Move every progress row back by whole days, day and instant together."""
        ...

    def progress_rows(self, user_playbook_id: int) -> list[ProgressRow]:
        """The playbook's progress rows, oldest day first."""
        ...


def _utc_day(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()


def _record(
    harness: PortHarness, playbook_id: int, outcome: str, job_id: int | None = None
) -> SweepProgressRecord:
    return harness.storage.record_open_world_sweep_progress(
        user_playbook_id=playbook_id, outcome=outcome, job_id=job_id
    )


def _queue(
    harness: PortHarness, *, limit: int = 500, window_seconds: int | None = None
) -> tuple[SweepQueueEntry, ...]:
    return harness.storage.list_open_world_playbooks_without_progress(
        limit=limit, window_seconds=window_seconds
    )


def _queued(
    harness: PortHarness, *, window_seconds: int | None = WINDOW_SECONDS
) -> set[int]:
    return {
        entry.playbook_id for entry in _queue(harness, window_seconds=window_seconds)
    }


def _session() -> str:
    return f"session-{uuid4().hex[:10]}"


def _exposed_playbook(
    harness: PortHarness, *, session_only: bool = False
) -> tuple[int, str]:
    """A current playbook exposed ``_OLD`` seconds ago in its own session.

    Inside every window, and old enough that the exposure's own request is
    never "new" activity for an attempt aged less than ``_OLD``.
    """
    session = _session()
    at = harness.now() - _OLD
    playbook_id = harness.seed_playbook()
    request_id = harness.seed_request(session_id=session, user_id=_USER, at_epoch=at)
    harness.seed_exposure(
        user_playbook_id=playbook_id,
        request_id=None if session_only else request_id,
        session_id=session,
        user_id=_USER,
        exposed_at=at,
        ingested_at=at,
    )
    return playbook_id, session


def _spent(
    harness: PortHarness, *, outcome: str = _INELIGIBLE, session_only: bool = False
) -> tuple[int, str]:
    """An exposed playbook whose ``ineligible_*`` attempt is past its cooldown.

    Nothing has happened since the attempt, so it is still OUT of the queue;
    asserted before returning, so a case that later sees it IN knows its own
    seed did that.
    """
    playbook_id, session = _exposed_playbook(harness, session_only=session_only)
    _record(harness, playbook_id, outcome)
    harness.age_progress(playbook_id, seconds=_PAST_EVIDENCE_COOLDOWN)
    assert playbook_id not in _queued(harness)
    return playbook_id, session


def _attempted_at(harness: PortHarness, playbook_id: int) -> int:
    (row,) = harness.progress_rows(playbook_id)
    return row.attempted_at


class OpenWorldPortContract:
    """The port's behaviour, asserted identically on every backend."""

    # ----------------------------------------------------------------- I16
    # The record

    def test_record_returns_the_stored_row_from_one_clock_reading(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id = port_harness.seed_playbook()
        before = port_harness.now()

        record = _record(port_harness, playbook_id, "  attempt_error  ", 7)

        after = port_harness.now()
        assert record.user_playbook_id == playbook_id
        assert record.outcome == "attempt_error", "storage trims the label"
        assert record.job_id == 7
        assert before <= record.attempted_at <= after
        assert record.progress_day == _utc_day(record.attempted_at), (
            "the day and the instant come from one clock reading"
        )
        assert port_harness.progress_rows(playbook_id) == [
            ProgressRow(record.progress_day, "attempt_error", 7, record.attempted_at, 1)
        ]

    def test_one_row_per_day_counts_every_record(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id = port_harness.seed_playbook()
        first = _record(port_harness, playbook_id, "attempt_error", 11)
        port_harness.age_progress(playbook_id, seconds=120)

        second = _record(port_harness, playbook_id, "ineligible_no_evidence")

        assert second.progress_day == first.progress_day
        assert second.outcome == "ineligible_no_evidence"
        assert second.job_id == 11, "a re-record without a job keeps the earlier one"
        assert second.attempted_at > first.attempted_at - 120, (
            "the instant is replaced, not kept from the aged row"
        )
        third = _record(port_harness, playbook_id, "attempt_error", 12)
        assert third.job_id == 12, "a new job replaces the earlier one"
        (row,) = port_harness.progress_rows(playbook_id)
        assert row.attempt_count == 3
        assert row.attempted_at == third.attempted_at

    def test_a_new_day_starts_a_new_row(self, port_harness: PortHarness) -> None:
        playbook_id = port_harness.seed_playbook()
        _record(port_harness, playbook_id, "attempt_error", 7)
        port_harness.backdate_progress(playbook_id, days=1)

        today = _record(port_harness, playbook_id, "attempt_error")

        rows = port_harness.progress_rows(playbook_id)
        assert [row.attempt_count for row in rows] == [1, 1]
        assert rows[0].progress_day < rows[1].progress_day == today.progress_day
        assert today.job_id is None, "yesterday's job does not carry into today"

    @pytest.mark.parametrize(
        ("playbook_id", "outcome", "job_id"),
        [
            (0, "attempt_error", None),
            (-1, "attempt_error", None),
            (1, "   ", None),
            (1, "", None),
            (1, "attempt_error", 0),
        ],
    )
    def test_an_unusable_identity_raises_and_writes_nothing(
        self,
        port_harness: PortHarness,
        playbook_id: int,
        outcome: str,
        job_id: int | None,
    ) -> None:
        if playbook_id == 1:
            playbook_id = port_harness.seed_playbook()
        with pytest.raises(Exception, match="sweep progress identity is invalid"):
            _record(port_harness, playbook_id, outcome, job_id)
        assert port_harness.progress_rows(playbook_id) == []

    # ----------------------------------------------------------------- I16
    # The queue: membership and order

    def test_never_attempted_first_then_oldest_attempt(
        self, port_harness: PortHarness
    ) -> None:
        never, _ = _exposed_playbook(port_harness)
        two_days, _ = _exposed_playbook(port_harness)
        one_day, _ = _exposed_playbook(port_harness)
        today, _ = _exposed_playbook(port_harness)
        for playbook_id in (two_days, one_day, today):
            _record(port_harness, playbook_id, "attempt_error")
        port_harness.backdate_progress(two_days, days=2)
        port_harness.backdate_progress(one_day, days=1)

        queue = _queue(port_harness, window_seconds=WINDOW_SECONDS)

        assert queue == (
            SweepQueueEntry(never, NEVER_ATTEMPTED_EPOCH),
            SweepQueueEntry(two_days, _attempted_at(port_harness, two_days)),
            SweepQueueEntry(one_day, _attempted_at(port_harness, one_day)),
        ), "today's record spends the day; earlier days only order the queue"

    def test_ties_break_by_id_and_the_limit_reads_at_least_one(
        self, port_harness: PortHarness
    ) -> None:
        ids = sorted(port_harness.seed_playbook() for _ in range(3))

        assert [entry.playbook_id for entry in _queue(port_harness)] == ids
        assert _queue(port_harness, limit=2) == _queue(port_harness)[:2]
        assert _queue(port_harness, limit=0) == _queue(port_harness)[:1]
        assert _queue(port_harness, limit=-5) == _queue(port_harness)[:1]

    def test_only_current_playbooks_are_queued(self, port_harness: PortHarness) -> None:
        current = port_harness.seed_playbook()
        retired = port_harness.seed_playbook(current=False)
        at = port_harness.now() - _OLD
        for playbook_id in (current, retired):
            port_harness.seed_exposure(
                user_playbook_id=playbook_id,
                request_id=None,
                session_id=_session(),
                user_id=_USER,
                exposed_at=at,
                ingested_at=at,
            )

        assert _queued(port_harness) == {current}
        assert _queued(port_harness, window_seconds=None) == {current}

    def test_no_window_is_the_unfiltered_arm(self, port_harness: PortHarness) -> None:
        exposed, _ = _exposed_playbook(port_harness)
        unexposed = port_harness.seed_playbook()

        assert _queued(port_harness) == {exposed}
        assert _queued(port_harness, window_seconds=None) == {exposed, unexposed}

    def test_the_window_floors_at_the_start_of_today(
        self, port_harness: PortHarness
    ) -> None:
        now = port_harness.now()
        floor = now - now % _DAY - WINDOW_SECONDS
        inside = port_harness.seed_playbook()
        outside = port_harness.seed_playbook()
        for playbook_id, at in ((inside, floor + 3_600), (outside, floor - 3_600)):
            port_harness.seed_exposure(
                user_playbook_id=playbook_id,
                request_id=None,
                session_id=_session(),
                user_id=_USER,
                exposed_at=at,
                ingested_at=at,
            )

        assert _queued(port_harness) == {inside}

    def test_the_window_reads_exposed_at_then_ingested_at(
        self, port_harness: PortHarness
    ) -> None:
        now = port_harness.now()
        null_exposed = port_harness.seed_playbook()
        stale_exposed = port_harness.seed_playbook()
        port_harness.seed_exposure(
            user_playbook_id=null_exposed,
            request_id=None,
            session_id=_session(),
            user_id=_USER,
            exposed_at=None,
            ingested_at=now - 60,
        )
        port_harness.seed_exposure(
            user_playbook_id=stale_exposed,
            request_id=None,
            session_id=_session(),
            user_id=_USER,
            exposed_at=now - WINDOW_SECONDS - 2 * _DAY,
            ingested_at=now - 60,
        )

        assert _queued(port_harness) == {null_exposed}, (
            "a NULL exposed_at falls back to ingested_at; a stale exposed_at is "
            "not rescued by a recent ingested_at"
        )

    def test_only_published_exposures_count_as_activity(
        self, port_harness: PortHarness
    ) -> None:
        at = port_harness.now() - _OLD
        legacy = port_harness.seed_playbook()
        port_harness.seed_exposure(
            user_playbook_id=legacy,
            request_id=None,
            session_id=_session(),
            user_id=_USER,
            exposed_at=at,
            ingested_at=at,
            published=False,
        )

        assert _queued(port_harness) == set()
        assert _queued(port_harness, window_seconds=None) == {legacy}

    # ----------------------------------------------------------------- I16
    # The rate-limited re-attempt

    def test_rate_limited_returns_after_its_cooldown_until_the_cap(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, _ = _exposed_playbook(port_harness)

        _record(port_harness, playbook_id, RETRY_ELIGIBLE_OUTCOME)
        assert playbook_id not in _queued(port_harness), "inside the cooldown"
        port_harness.age_progress(
            playbook_id, seconds=RATE_LIMIT_RETRY_COOLDOWN_SECONDS - 60
        )
        assert playbook_id not in _queued(port_harness), "still inside the cooldown"
        port_harness.age_progress(playbook_id, seconds=120)
        assert playbook_id in _queued(port_harness), "the cooldown has elapsed"

        for _ in range(MAX_SAME_DAY_SWEEP_ATTEMPTS - 1):
            _record(port_harness, playbook_id, RETRY_ELIGIBLE_OUTCOME)
        port_harness.age_progress(
            playbook_id, seconds=RATE_LIMIT_RETRY_COOLDOWN_SECONDS + 60
        )
        assert playbook_id not in _queued(port_harness), (
            "the shared cap spends the day whatever the cooldown"
        )

    @pytest.mark.parametrize("outcome", ["attempt_error", "succeeded", "abstained_x"])
    def test_any_other_outcome_spends_the_day(
        self, port_harness: PortHarness, outcome: str
    ) -> None:
        playbook_id, session = _exposed_playbook(port_harness)
        _record(port_harness, playbook_id, outcome)
        port_harness.age_progress(
            playbook_id, seconds=RATE_LIMIT_RETRY_COOLDOWN_SECONDS + 60
        )
        port_harness.seed_request(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )

        assert playbook_id not in _queued(port_harness)

    # ----------------------------------------------------------------- I16
    # The evidence-gated re-attempt

    def test_a_new_request_in_an_exposed_session_readmits(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _spent(port_harness)
        port_harness.seed_request(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )
        assert playbook_id in _queued(port_harness)

    def test_a_new_outcome_on_an_exposed_session_readmits(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _spent(port_harness)
        port_harness.seed_session_outcome(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )
        assert playbook_id in _queued(port_harness)

    def test_a_session_only_exposure_is_placed_on_its_own_session(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _spent(port_harness, session_only=True)
        port_harness.seed_request(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )
        assert playbook_id in _queued(port_harness)

    def test_activity_elsewhere_does_not_readmit(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _spent(port_harness)
        now = port_harness.now()
        port_harness.seed_request(session_id=_session(), user_id=_USER, at_epoch=now)
        port_harness.seed_session_outcome(
            session_id=session, user_id="another-user", at_epoch=now
        )
        assert playbook_id not in _queued(port_harness), (
            "only the exposing session, for the exposing user, carries evidence"
        )

    def test_within_the_cooldown_new_activity_does_not_readmit(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _exposed_playbook(port_harness)
        _record(port_harness, playbook_id, _INELIGIBLE)
        port_harness.age_progress(
            playbook_id, seconds=EVIDENCE_REATTEMPT_COOLDOWN_SECONDS - 60
        )
        port_harness.seed_request(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )
        assert playbook_id not in _queued(port_harness)

    def test_the_shared_cap_stops_the_evidence_re_attempt(
        self, port_harness: PortHarness
    ) -> None:
        playbook_id, session = _exposed_playbook(port_harness)
        for _ in range(MAX_SAME_DAY_SWEEP_ATTEMPTS):
            _record(port_harness, playbook_id, _INELIGIBLE)
        port_harness.age_progress(playbook_id, seconds=_PAST_EVIDENCE_COOLDOWN)
        port_harness.seed_request(
            session_id=session, user_id=_USER, at_epoch=port_harness.now()
        )
        assert playbook_id not in _queued(port_harness)

    @pytest.mark.parametrize("session_only", [False, True])
    def test_the_request_lookback_is_exact(
        self, port_harness: PortHarness, session_only: bool
    ) -> None:
        """A request counts as of ``created_at + lookback``, strictly after."""
        late, late_session = _spent(port_harness, session_only=session_only)
        port_harness.seed_request(
            session_id=late_session,
            user_id=_USER,
            at_epoch=_attempted_at(port_harness, late)
            - EVIDENCE_REATTEMPT_LOOKBACK_SECONDS
            + 1,
        )
        early, early_session = _spent(port_harness, session_only=session_only)
        port_harness.seed_request(
            session_id=early_session,
            user_id=_USER,
            at_epoch=_attempted_at(port_harness, early)
            - EVIDENCE_REATTEMPT_LOOKBACK_SECONDS,
        )

        queued = _queued(port_harness)
        assert late in queued
        assert early not in queued

    @pytest.mark.parametrize("session_only", [False, True])
    def test_each_attempt_is_compared_with_its_own_instant(
        self, port_harness: PortHarness, session_only: bool
    ) -> None:
        """Activity after the day's EARLIEST attempt is not enough for a later one.

        The activity read is bounded by the earliest attempt of the day, so with
        a single spent playbook that bound alone decides every boundary case.
        An earlier attempt on another playbook lowers it, which leaves only the
        per-attempt comparison to refuse activity at this attempt's instant.
        """
        earlier, _ = _spent(port_harness)
        port_harness.age_progress(earlier, seconds=_OLD // 2)
        playbook_id, session = _spent(port_harness, session_only=session_only)
        attempted = _attempted_at(port_harness, playbook_id)
        port_harness.seed_request(
            session_id=session,
            user_id=_USER,
            at_epoch=attempted - EVIDENCE_REATTEMPT_LOOKBACK_SECONDS,
        )
        port_harness.seed_session_outcome(
            session_id=session, user_id=_USER, at_epoch=attempted
        )
        assert playbook_id not in _queued(port_harness)

        port_harness.seed_request(
            session_id=session,
            user_id=_USER,
            at_epoch=attempted - EVIDENCE_REATTEMPT_LOOKBACK_SECONDS + 1,
        )
        assert playbook_id in _queued(port_harness)
        assert earlier not in _queued(port_harness)

    def test_an_outcome_counts_only_strictly_after_the_attempt(
        self, port_harness: PortHarness
    ) -> None:
        after, after_session = _spent(port_harness)
        port_harness.seed_session_outcome(
            session_id=after_session,
            user_id=_USER,
            at_epoch=_attempted_at(port_harness, after) + 1,
        )
        at, at_session = _spent(port_harness)
        port_harness.seed_session_outcome(
            session_id=at_session,
            user_id=_USER,
            at_epoch=_attempted_at(port_harness, at),
        )

        queued = _queued(port_harness)
        assert after in queued
        assert at not in queued

    def test_evidence_needs_an_exposure_inside_the_window(
        self, port_harness: PortHarness
    ) -> None:
        """An out-of-window exposure on the active session carries nothing."""
        playbook_id, _ = _spent(port_harness)
        stale_session = _session()
        stale = port_harness.now() - WINDOW_SECONDS - 2 * _DAY
        port_harness.seed_exposure(
            user_playbook_id=playbook_id,
            request_id=None,
            session_id=stale_session,
            user_id=_USER,
            exposed_at=stale,
            ingested_at=stale,
        )
        port_harness.seed_request(
            session_id=stale_session, user_id=_USER, at_epoch=port_harness.now()
        )

        assert playbook_id not in _queued(port_harness)
        assert playbook_id in _queued(port_harness, window_seconds=None), (
            "the unfiltered arm counts every exposure"
        )
