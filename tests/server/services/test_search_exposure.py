"""Direct behavior tests for user-playbook search exposure identities."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reflexio.models.api_schema.domain import BlockingIssue, UserPlaybook
from reflexio.models.api_schema.domain.enums import BlockingIssueKind, Status
from reflexio.server.extensions import register_service
from reflexio.server.services import search_exposure as search_exposure_module
from reflexio.server.services.search_exposure import (
    SEARCH_EXPOSURE_RECORDER,
    SearchExposureBatch,
    SearchExposureOutcome,
    build_user_playbook_exposure_event,
    record_search_exposures,
    reset_uncorrelated_reporting_state,
    user_playbook_full_version_fingerprint,
)


def _playbook() -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=101,
        user_id="user-1",
        agent_version="agent-v1",
        request_id="source-request-1",
        playbook_name="Support policy",
        created_at=1_700_000_000,
        content="Escalate refund requests after verification.",
        trigger="refund escalation",
        rationale="Historical resolution pattern.",
        blocking_issue=BlockingIssue(
            kind=BlockingIssueKind.MISSING_TOOL, details="CRM access is absent."
        ),
        status=Status.ARCHIVED,
        source="support-import",
        source_interaction_ids=[11, 12],
        expanded_terms="refund return escalation",
        tags=["support", "refund"],
        embedding=[0.25] * 512,
        source_span="messages 4-6",
        notes="Reviewed by ops.",
        reader_angle="customer impact",
        merged_into=88,
        superseded_by=99,
        governance_subject_ref="subject:user-1",
        retired_at=1_700_000_050,
    )


def _batch(
    playbook: UserPlaybook,
    *,
    request_id: str | None = "request-1",
    session_id: str | None = "session-1",
    interaction_id: int | None = 41,
    invocation_id: str = "invocation-1",
) -> SearchExposureBatch:
    return SearchExposureBatch(
        org_id="org-1",
        request_id=request_id,
        session_id=session_id,
        interaction_id=interaction_id,
        user_id="user-1",
        user_playbooks=(playbook,),
        invocation_id=invocation_id,
    )


def _event(batch: SearchExposureBatch, playbook: UserPlaybook):
    return build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref="user:user-1",
        playbook_owner_governance_subject_ref="owner:user-1",
    )


def test_correlated_retries_keep_one_exposure_event_id_despite_invocation_id() -> None:
    playbook = _playbook()
    initial = _event(_batch(playbook, invocation_id="invocation-a"), playbook)
    retry = _event(_batch(playbook, invocation_id="invocation-b"), playbook)

    assert initial.exposure_event_id == retry.exposure_event_id


def test_correlation_free_invocations_get_distinct_exposure_event_ids() -> None:
    playbook = _playbook()
    first = _event(
        _batch(
            playbook,
            request_id=None,
            session_id=None,
            interaction_id=None,
            invocation_id="invocation-a",
        ),
        playbook,
    )
    second = _event(
        _batch(
            playbook,
            request_id=None,
            session_id=None,
            interaction_id=None,
            invocation_id="invocation-b",
        ),
        playbook,
    )

    assert first.exposure_event_id != second.exposure_event_id


def test_unscoped_exposure_keeps_unknown_subject_separate_from_playbook_owner() -> None:
    playbook = _playbook()
    batch = replace(_batch(playbook), user_id=None)

    event = build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref=None,
        playbook_owner_governance_subject_ref="owner:user-1",
    )

    assert event.user_id is None
    assert event.governance_subject_ref is None
    assert event.playbook_owner_user_id == "user-1"
    assert event.playbook_owner_governance_subject_ref == "owner:user-1"


@pytest.mark.parametrize("user_id", ["", " \t\n"], ids=["empty", "whitespace"])
def test_blank_retrieval_subject_normalizes_to_unscoped(user_id: str) -> None:
    playbook = _playbook()
    batch = replace(_batch(playbook), user_id=user_id)

    event = build_user_playbook_exposure_event(
        batch,
        playbook,
        exposed_at=1_700_000_100,
        ingested_at=1_700_000_101,
        governance_subject_ref=None,
        playbook_owner_governance_subject_ref="owner:user-1",
    )

    assert batch.user_id is None
    assert event.user_id is None
    assert event.governance_subject_ref is None
    assert event.playbook_owner_user_id == "user-1"


def test_scoped_exposure_rejects_a_playbook_owned_by_another_user() -> None:
    playbook = _playbook().model_copy(update={"user_id": "user-2"})
    batch = _batch(playbook)

    with pytest.raises(ValueError, match="does not match retrieval subject"):
        _event(batch, playbook)


def test_request_and_session_correlation_ids_normalize_whitespace_consistently() -> (
    None
):
    playbook = _playbook()
    whitespace = _batch(
        playbook,
        request_id=" \trequest-1\n",
        session_id="\tsession-1 ",
        interaction_id=None,
    )
    normalized = _batch(
        playbook,
        request_id="request-1",
        session_id="session-1",
        interaction_id=None,
    )
    blank = _batch(
        playbook,
        request_id=" \t",
        session_id="\n ",
        interaction_id=None,
        invocation_id="fallback-invocation",
    )
    absent = _batch(
        playbook,
        request_id=None,
        session_id=None,
        interaction_id=None,
        invocation_id="fallback-invocation",
    )

    assert (
        (whitespace.request_id, whitespace.session_id)
        == (
            normalized.request_id,
            normalized.session_id,
        )
        == ("request-1", "session-1")
    )
    assert (
        _event(whitespace, playbook).exposure_event_id
        == _event(normalized, playbook).exposure_event_id
    )
    assert (blank.request_id, blank.session_id) == (None, None)
    assert (
        _event(blank, playbook).exposure_event_id
        == _event(absent, playbook).exposure_event_id
    )


def test_embedding_changes_do_not_change_full_version_fingerprint() -> None:
    playbook = _playbook()
    reembedded = playbook.model_copy(update={"embedding": [0.5] * 512})

    assert user_playbook_full_version_fingerprint(playbook) == (
        user_playbook_full_version_fingerprint(reembedded)
    )


def test_internal_fields_are_excluded_from_user_playbook_serialization() -> None:
    serialized = _playbook().model_dump(mode="json")

    assert "governance_subject_ref" not in serialized
    assert "retired_at" not in serialized


# Every current UserPlaybook field is persisted except its derived embedding vector.
_PERSISTED_FIELD_CHANGES = [
    ("user_playbook_id", 102),
    ("user_id", "user-2"),
    ("agent_version", "agent-v2"),
    ("request_id", "source-request-2"),
    ("playbook_name", "Returns policy"),
    ("created_at", 1_700_000_001),
    ("content", "Verify returns before escalating."),
    ("trigger", "returns escalation"),
    ("rationale", "Updated resolution pattern."),
    (
        "blocking_issue",
        BlockingIssue(
            kind=BlockingIssueKind.PERMISSION_DENIED,
            details="CRM access was denied.",
        ),
    ),
    ("status", Status.PENDING),
    ("source", "returns-import"),
    ("source_interaction_ids", [11, 13]),
    ("expanded_terms", "return exchange escalation"),
    ("tags", ["support", "returns"]),
    ("source_span", "messages 7-9"),
    ("notes", "Needs legal review."),
    ("reader_angle", "policy compliance"),
    ("merged_into", 87),
    ("superseded_by", 100),
    ("governance_subject_ref", "subject:user-2"),
    ("retired_at", 1_700_000_051),
]


@pytest.mark.parametrize(("field", "value"), _PERSISTED_FIELD_CHANGES)
def test_full_version_fingerprint_changes_for_each_persisted_non_embedding_field(
    field: str, value: object
) -> None:
    playbook = _playbook()
    changed = playbook.model_copy(update={field: value})

    assert user_playbook_full_version_fingerprint(playbook) != (
        user_playbook_full_version_fingerprint(changed)
    )


def test_full_version_fingerprint_coverage_includes_each_persisted_model_field() -> (
    None
):
    assert {field for field, _value in _PERSISTED_FIELD_CHANGES} == (
        set(UserPlaybook.model_fields) - {"embedding"}
    )


def test_semantic_digest_and_fallback_identity_are_deterministic_and_domain_separated() -> (
    None
):
    playbook = _playbook()
    batch = _batch(
        playbook,
        request_id=None,
        session_id=None,
        interaction_id=None,
        invocation_id="invocation-a",
    )

    first = _event(batch, playbook)
    repeated = _event(replace(batch), playbook)

    assert (
        first.exposure_event_id
        == repeated.exposure_event_id
        == ("80b011b78df4a90e2238a7150d091c4d2f8c0e38d4343ccc7616e3536d40ca49")
    )
    assert (
        first.served_semantic_digest
        == repeated.served_semantic_digest
        == ("d321988fa077b43deb4df4c89753b98c757c7c3167a12ea26af9247e665cc942")
    )
    assert first.exposure_event_id != first.served_semantic_digest


class _CollectingRecorder:
    """Minimal recorder that keeps whatever it was handed."""

    def __init__(self) -> None:
        self.batches: list[SearchExposureBatch] = []

    def record(self, batch: SearchExposureBatch) -> None:
        self.batches.append(batch)


class _FailingRecorder:
    """Recorder that fails the write, to pin that the failure is CONTAINED."""

    def record(self, batch: SearchExposureBatch) -> None:
        raise RuntimeError("ledger unavailable")


def test_absent_recorder_reports_that_nothing_was_recorded() -> None:
    """No recorder registered is legal, but the caller must be able to see it.

    Exposure is append-only, so a caller reconstructing a corpus cannot learn
    later that its batches went nowhere. Returning ``None`` regardless made a
    silent no-op indistinguishable from a durable write.
    """
    outcome = record_search_exposures(_batch(_playbook()))

    assert outcome is SearchExposureOutcome.NO_RECORDER


def test_registered_recorder_reports_that_the_batch_was_recorded() -> None:
    """The positive direction: a real write must not report absence."""
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)
    batch = _batch(_playbook())

    outcome = record_search_exposures(batch)

    assert outcome is SearchExposureOutcome.RECORDED
    assert recorder.batches == [batch]


def test_recorder_failure_is_contained_and_named_not_propagated() -> None:
    """INVERTED: a failed audit write must not fail the search.

    This pinned the opposite -- the exception propagating -- and that was the
    defect. The recorder writes synchronously inside the request and neither
    search route guards the call, so the exception unwound into FastAPI as a
    bare 500 and discarded search results the route had already built. An
    append-only audit row with no live reader must not outrank the customer's
    answer; the adjacent ``enqueue_search_metering`` queues BILLING off the hot
    path.

    It is still not softened into ``RECORDED``. ``RECORDER_FAILED`` is a
    distinct outcome precisely so a caller that depends on the ledger can tell
    a lost write from a durable one -- exposure cannot be backfilled.
    """
    register_service(SEARCH_EXPOSURE_RECORDER, _FailingRecorder())

    outcome = record_search_exposures(_batch(_playbook()))

    assert outcome is SearchExposureOutcome.RECORDER_FAILED
    assert outcome is not SearchExposureOutcome.RECORDED, (
        "a lost write must never read as a durable one"
    )


def test_uncorrelated_batch_is_recorded_and_flagged_not_refused() -> None:
    """The row reaches the ledger, where the schema labels it.

    This INVERTS the previous contract deliberately. Refusing the write was
    intended to protect the org's reconstructability ratio, which a row with no
    correlation can only ever lower. Measured against production on 2026-09-27,
    that protected nothing: no org in the fleet carried a single ``request_id``,
    so the predicate matched 100% of live traffic and exposure intake went to
    zero fleet-wide for 16 days. A proportional dent in a ratio beats losing the
    whole audit trail -- which playbooks were served, under which fingerprints.

    The row is not passed off as healthy: ``integrity_state`` reads
    ``incomplete`` with ``missing_correlation``, and the anomaly is still
    reported.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    outcome = record_search_exposures(
        _batch(_playbook(), request_id=None, session_id=None, interaction_id=41)
    )

    assert outcome is SearchExposureOutcome.RECORDED_UNCORRELATED
    assert len(recorder.batches) == 1, "the audit trail must survive"
    assert recorder.batches[0].request_id is None
    assert recorder.batches[0].session_id is None


def test_blank_correlation_strings_are_flagged_like_absent_ones() -> None:
    """Normalization collapses blanks to ``None``; the classification follows it.

    A whitespace-only ``request_id`` must not read as correlation just because it
    is a non-empty string -- the schema's own ``btrim`` check would call it
    ``missing_correlation`` either way, so the two layers have to agree.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    outcome = record_search_exposures(
        _batch(_playbook(), request_id="   ", session_id="")
    )

    assert outcome is SearchExposureOutcome.RECORDED_UNCORRELATED
    assert len(recorder.batches) == 1


@pytest.mark.parametrize(
    ("request_id", "session_id"),
    [("request-1", "session-1"), ("request-1", None), (None, "session-1")],
)
def test_any_single_correlation_handle_is_still_recorded(
    request_id: str | None, session_id: str | None
) -> None:
    """The other direction: the guard must not swallow real evidence.

    A guard that refused everything would pass a one-sided "nothing recorded"
    assertion while emptying the ledger. Either stored correlation column on its
    own keeps the batch, matching the schema's ``missing_correlation`` reason,
    which fires only when both are blank.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)
    batch = _batch(_playbook(), request_id=request_id, session_id=session_id)

    outcome = record_search_exposures(batch)

    assert outcome is SearchExposureOutcome.RECORDED
    assert recorder.batches == [batch]


@pytest.fixture(autouse=True)
def _clean_uncorrelated_reporting_state():
    """The throttle is process-global; tests must not inherit each other's."""
    reset_uncorrelated_reporting_state()
    yield
    reset_uncorrelated_reporting_state()


class _AnomalySpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, message: str, **tags: object) -> None:
        self.calls.append((message, tags))


@pytest.fixture
def anomalies(monkeypatch) -> _AnomalySpy:
    spy = _AnomalySpy()
    monkeypatch.setattr(search_exposure_module, "capture_anomaly", spy)
    return spy


def _uncorrelated_batch() -> SearchExposureBatch:
    return _batch(_playbook(), request_id=None, session_id=None)


def test_an_uncorrelated_batch_is_reported_when_a_recorder_is_registered(
    anomalies: _AnomalySpy,
) -> None:
    """The row now lands, so this report is the ONLY signal of the gap.

    While the write was refused, an operator could in principle have noticed the
    absence of rows -- nobody did, for 16 days. Now the rows are present and
    merely flagged, so nothing about the table's size reveals the problem and
    this anomaly is the whole detector. It must not be removed on the grounds
    that the data is no longer being lost.
    """
    register_service(SEARCH_EXPOSURE_RECORDER, _CollectingRecorder())

    outcome = record_search_exposures(_uncorrelated_batch())

    assert outcome is SearchExposureOutcome.RECORDED_UNCORRELATED
    assert len(anomalies.calls) == 1
    message, tags = anomalies.calls[0]
    assert message == "search_exposure.uncorrelated"
    assert tags["uncorrelated_since_last_report"] == 1
    assert tags["first_report"] is True


def test_a_recorderless_deployment_is_not_told_about_a_non_problem(
    anomalies: _AnomalySpy,
) -> None:
    """No recorder is a SUPPORTED configuration, not a misconfiguration.

    Shared ``create_app()`` installs no default recorder, so local and no-auth
    OSS deployments persist nothing by design. Reporting there would fire on
    every search of every such deployment forever -- the signal has to be
    absent exactly where it would be noise.
    """
    outcome = record_search_exposures(_uncorrelated_batch())

    # NO_RECORDER, because there is genuinely nowhere to put the row. The
    # anomaly stays silent: a recorderless deployment persists nothing by
    # design, so telling it its exposures are uncorrelated is noise.
    assert outcome is SearchExposureOutcome.NO_RECORDER
    assert anomalies.calls == []


def test_a_correlated_batch_reports_nothing(anomalies: _AnomalySpy) -> None:
    register_service(SEARCH_EXPOSURE_RECORDER, _CollectingRecorder())

    outcome = record_search_exposures(_batch(_playbook()))

    assert outcome is SearchExposureOutcome.RECORDED
    assert anomalies.calls == []


def test_a_storm_of_uncorrelated_serves_reports_once_and_counts_the_rest(
    anomalies: _AnomalySpy,
) -> None:
    """Bounded by construction: the motivating deployment stored ~95k of these.

    Now that every one of them also WRITES a row, an unthrottled report would be
    one error-reporting event per search on top of the write.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    for _ in range(250):
        record_search_exposures(_uncorrelated_batch())

    assert len(anomalies.calls) == 1
    assert anomalies.calls[0][1]["uncorrelated_since_last_report"] == 1
    assert len(recorder.batches) == 250, (
        "throttling the SIGNAL must not throttle writes"
    )


def test_the_next_window_reports_the_serves_it_accumulated(
    anomalies: _AnomalySpy, monkeypatch
) -> None:
    """A throttle that only ever reports once would hide an ongoing outage."""
    register_service(SEARCH_EXPOSURE_RECORDER, _CollectingRecorder())
    record_search_exposures(_uncorrelated_batch())
    assert len(anomalies.calls) == 1

    for _ in range(4):
        record_search_exposures(_uncorrelated_batch())
    assert len(anomalies.calls) == 1, "still inside the first window"

    monkeypatch.setattr(
        search_exposure_module, "_UNCORRELATED_ANOMALY_THROTTLE_SECONDS", 0.0
    )
    record_search_exposures(_uncorrelated_batch())

    assert len(anomalies.calls) == 2
    second = anomalies.calls[1][1]
    # The four suppressed serves plus this one: a count, not a resampling.
    assert second["uncorrelated_since_last_report"] == 5
    assert second["first_report"] is False


def test_reporting_does_not_suppress_the_write(
    anomalies: _AnomalySpy,
) -> None:
    """Observability must not become a behaviour change -- in either direction.

    The anomaly report and the write are independent: reporting must not stop
    the row reaching the ledger, which is exactly the coupling that turned a
    diagnostic into 16 days of lost intake.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    outcome = record_search_exposures(_uncorrelated_batch())

    assert outcome is SearchExposureOutcome.RECORDED_UNCORRELATED
    assert len(recorder.batches) == 1
    assert len(anomalies.calls) == 1, "the signal must survive alongside the write"


def test_a_failed_write_is_neither_counted_nor_reported(
    anomalies: _AnomalySpy,
) -> None:
    """The report asserts the row LANDED, so it must follow the write.

    Two things break if the anomaly is emitted before ``recorder.record``. The
    message says the exposure "is recorded but flagged", which is false for a
    batch the recorder then refused. Worse, the emission burns the hour-long
    throttle window: the next 3600s of genuinely-stored uncorrelated serves go
    unreported, which is precisely the blindness this signal exists to remove.
    """
    register_service(SEARCH_EXPOSURE_RECORDER, _FailingRecorder())

    assert record_search_exposures(_uncorrelated_batch()) is (
        SearchExposureOutcome.RECORDER_FAILED
    )

    uncorrelated_reports = [
        c for c in anomalies.calls if c[0] == "search_exposure.uncorrelated"
    ]
    assert uncorrelated_reports == [], (
        "a write that never landed must not be counted as an uncorrelated row"
    )

    # And the throttle is untouched, so the next real miss still reports.
    register_service(SEARCH_EXPOSURE_RECORDER, _CollectingRecorder(), override=True)
    assert record_search_exposures(_uncorrelated_batch()) is (
        SearchExposureOutcome.RECORDED_UNCORRELATED
    )
    uncorrelated_reports = [
        c for c in anomalies.calls if c[0] == "search_exposure.uncorrelated"
    ]
    assert len(uncorrelated_reports) == 1
    assert uncorrelated_reports[0][1]["first_report"] is True
    assert uncorrelated_reports[0][1]["uncorrelated_since_last_report"] == 1, (
        "the failed write must not be counted either"
    )


def test_an_empty_uncorrelated_batch_persists_no_row_so_reports_nothing(
    anomalies: _AnomalySpy,
) -> None:
    """The unified ``/api/search`` hands over an empty batch on purpose.

    When a production agent's search surfaces no user playbooks,
    ``unified_search_endpoint`` still records one empty batch (pinned by
    ``test_no_user_playbook_results_record_one_empty_synchronous_batch``). An
    empty batch writes no exposure event, so there is no row for
    ``missing_correlation`` to be true of, and an anomaly claiming one landed
    would be false.

    The throttle is what makes this matter rather than merely untidy. No org in
    the fleet sends a correlation id, so every empty search would qualify --
    these would be the COMMON case, and each one would consume the hour-long
    window, crowding out the report for a serve that really did store an
    uncorrelated row. This anomaly is the only detector there is.
    """
    recorder = _CollectingRecorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    empty = SearchExposureBatch(
        org_id="org-1",
        request_id=None,
        session_id=None,
        interaction_id=None,
        user_id="user-1",
        user_playbooks=(),
    )

    outcome = record_search_exposures(empty)

    # Still RECORDED_UNCORRELATED: the recorder did accept the batch, and the
    # caller must not be told a write was skipped when it was not.
    assert outcome is SearchExposureOutcome.RECORDED_UNCORRELATED
    assert len(recorder.batches) == 1
    assert recorder.batches[0].user_playbooks == ()
    assert anomalies.calls == [], "no row landed, so there is nothing to report"

    # The window was never opened, so a real uncorrelated serve still reports.
    assert record_search_exposures(_uncorrelated_batch()) is (
        SearchExposureOutcome.RECORDED_UNCORRELATED
    )
    assert len(anomalies.calls) == 1
    assert anomalies.calls[0][1]["first_report"] is True
    assert anomalies.calls[0][1]["uncorrelated_since_last_report"] == 1, (
        "the empty batch must not be counted either"
    )


def test_a_contained_recorder_failure_is_still_reported(
    anomalies: _AnomalySpy,
) -> None:
    """Containing the failure must not make it silent.

    Swallowing a lost append-only write with no signal would be strictly worse
    than the 500 it replaces: the 500 at least told somebody. Unlike the
    uncorrelated report this one is NOT throttled -- that condition is normal
    traffic and would fire forever, whereas a recorder failure is a fault,
    bounded by the outage causing it and worth paging on.
    """
    register_service(SEARCH_EXPOSURE_RECORDER, _FailingRecorder())

    outcome = record_search_exposures(_batch(_playbook()))

    assert outcome is SearchExposureOutcome.RECORDER_FAILED
    assert len(anomalies.calls) == 1
    message, tags = anomalies.calls[0]
    assert message == "search_exposure.recorder_failed"
    assert tags["playbooks"] == 1
    assert tags["uncorrelated"] is False
    assert tags["level"] == "error"


def test_a_recorder_raising_a_base_exception_still_unwinds() -> None:
    """``Exception`` only. Interpreter-level control flow is not ours to eat.

    Catching ``BaseException`` here would swallow ``KeyboardInterrupt``,
    ``SystemExit`` and ``GeneratorExit``, turning a shutdown signal into a
    logged anomaly and a served response.
    """

    class _Exiting:
        def record(self, _batch: SearchExposureBatch) -> None:
            raise KeyboardInterrupt

    register_service(SEARCH_EXPOSURE_RECORDER, _Exiting())

    with pytest.raises(KeyboardInterrupt):
        record_search_exposures(_batch(_playbook()))


def test_a_failing_reporter_does_not_break_the_search_path(monkeypatch) -> None:
    """capture_anomaly is best-effort; a broken reporter must not 500 a search."""

    def _boom(message: str, **tags: object) -> None:
        raise RuntimeError("reporter down")

    monkeypatch.setattr(search_exposure_module, "capture_anomaly", _boom)
    register_service(SEARCH_EXPOSURE_RECORDER, _CollectingRecorder())

    with pytest.raises(RuntimeError, match="reporter down"):
        record_search_exposures(_uncorrelated_batch())


def test_invalid_owner_is_rejected_even_without_recorder(monkeypatch) -> None:
    monkeypatch.setattr(search_exposure_module, "get_service", lambda *_: None)
    batch = replace(_batch(_playbook()), user_id="another-user")
    with pytest.raises(ValueError):
        record_search_exposures(batch)


def test_oversized_batch_is_rejected_before_recorder_lookup(monkeypatch) -> None:
    def unexpected_lookup(*_):
        pytest.fail("Invalid batches must not reach recorder lookup")

    monkeypatch.setattr(search_exposure_module, "get_service", unexpected_lookup)
    batch = replace(_batch(_playbook()), user_playbooks=(_playbook(),) * 101)
    with pytest.raises(ValueError, match="at most"):
        record_search_exposures(batch)


def test_unscoped_mixed_owners_remain_valid() -> None:
    first = _playbook()
    second = first.model_copy(
        update={"user_id": "another-user", "user_playbook_id": 102}
    )
    batch = replace(_batch(first), user_id=None, user_playbooks=(first, second))
    search_exposure_module.validate_search_exposure_batch(batch)
