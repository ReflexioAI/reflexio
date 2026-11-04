"""Source-backed session compaction: evidence fidelity, budgets and resume cursors."""

from __future__ import annotations

import pytest

from reflexio.server.services.offline_tuner.evidence import compaction
from reflexio.server.services.offline_tuner.evidence.compaction import (
    MAX_COMPACTION_BATCHES,
    CompactionError,
    CompactionPoint,
    CompactionSource,
    SessionCompaction,
    compact_session,
    compaction_batch_count,
    compaction_source,
    source_batches,
    validate_compaction,
)


def _src(
    text: str, *, interaction_id: int = 1, field_path: str = "content"
) -> CompactionSource:
    return compaction_source(
        request_id="request",
        interaction_id=interaction_id,
        field_path=field_path,
        speaker="user",
        text=text,
    )


def _letters(count: int) -> str:
    return "".join(chr(ord("a") + i % 26) for i in range(count))


def _point(
    source: CompactionSource,
    quote: str,
    *,
    detail: str = "A fact",
    start_offset: int | None = None,
) -> CompactionPoint:
    return CompactionPoint(
        category="constraint",
        detail=detail,
        source_id=source.source_id,
        exact_quote=quote,
        start_offset=start_offset,
    )


def _single(point: CompactionPoint) -> SessionCompaction:
    return SessionCompaction(points=[point])


def test_source_identity_is_deterministic_and_binds_every_member() -> None:
    base = _src("hello")
    assert _src("hello") == base
    assert len(base.source_id) == 12
    assert _src("hello!").source_id != base.source_id
    assert _src("hello", interaction_id=2).source_id != base.source_id
    assert _src("hello", field_path="other").source_id != base.source_id


@pytest.mark.parametrize(
    ("quote", "offset", "message"),
    [
        ("not present", None, "does not resolve to original evidence"),
        ("UTC", 0, "offset does not resolve to source"),
        ("UTC", 999, "offset does not resolve to source"),
    ],
    ids=("unknown-quote", "wrong-offset", "offset-past-end"),
)
def test_validate_rejects_quotes_that_do_not_resolve(
    quote: str, offset: int | None, message: str
) -> None:
    source = _src("Do not use UTC.")
    with pytest.raises(CompactionError, match=message):
        validate_compaction(
            _single(_point(source, quote, start_offset=offset)),
            {source.source_id: source},
        )


def test_validate_rejects_a_point_citing_an_unknown_source() -> None:
    source = _src("Do not use UTC.")
    other = _src("Different text")
    with pytest.raises(CompactionError, match="does not resolve to original"):
        validate_compaction(
            _single(_point(other, "Different")), {source.source_id: source}
        )


def test_validate_accepts_an_exact_offset() -> None:
    source = _src("Do not use UTC.")
    validate_compaction(
        _single(_point(source, "UTC", start_offset=11)), {source.source_id: source}
    )


@pytest.mark.parametrize(
    ("detail", "quote"),
    [("   ", "UTC"), ("A fact", " ")],
    ids=("blank-detail", "blank-quote"),
)
def test_validate_rejects_whitespace_only_evidence(detail: str, quote: str) -> None:
    source = _src("Do not use UTC. ")
    with pytest.raises(CompactionError, match="empty evidence"):
        validate_compaction(
            _single(_point(source, quote, detail=detail)), {source.source_id: source}
        )


def test_validate_bounds_the_serialized_output_in_bytes() -> None:
    # Multi-byte detail text: well inside the character limits, over the byte cap.
    source = _src("é" * 240)
    points = [_point(source, "é" * 240, detail="é" * 512) for _ in range(24)]
    with pytest.raises(CompactionError, match="bounded output budget"):
        validate_compaction(
            SessionCompaction(points=points), {source.source_id: source}
        )


def test_batch_budget_floor_is_enforced() -> None:
    with pytest.raises(ValueError, match="too small"):
        list(source_batches([_src("x")], max_batch_bytes=2047))


def test_oversized_record_is_split_without_losing_text_and_keeps_offsets() -> None:
    text = _letters(1000)
    batches = list(source_batches([_src(text)], max_batch_bytes=2048))
    records = [record for batch in batches for record in batch]
    assert len(records) == 5
    assert "".join(str(r["text"]) for r in records) == text
    for record in records:
        start, end = int(str(record["start"])), int(str(record["end"]))
        assert text[start:end] == record["text"]
        assert end - start <= 2048 // 8


def test_source_identity_larger_than_the_batch_is_refused() -> None:
    with pytest.raises(CompactionError, match="source identity exceeds batch budget"):
        list(source_batches([_src("x", field_path="f" * 4000)], max_batch_bytes=2048))


def test_empty_source_text_is_not_a_compactable_session() -> None:
    with pytest.raises(CompactionError, match="no compactable source records"):
        compaction_batch_count([_src("")])


def test_read_budget_is_checked_before_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compaction, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(CompactionError, match="exceeds the read budget"):
        compaction_batch_count([_src("x" * 11)])


def test_call_budget_rejects_more_batches_than_allowed() -> None:
    # ~256 chars per record at the floor budget forces more than 128 batches.
    sources = [
        _src("y" * 240, interaction_id=i) for i in range(MAX_COMPACTION_BATCHES * 8)
    ]
    with pytest.raises(CompactionError, match="compaction call budget"):
        compaction_batch_count(sources, max_batch_bytes=2048)


def _fixed_complete(point_for):
    calls: list[str] = []

    def complete(payload: str) -> SessionCompaction:
        calls.append(payload)
        return point_for(len(calls))

    return complete, calls


def test_compact_session_rejects_duplicate_source_identities() -> None:
    source = _src("abc")
    with pytest.raises(CompactionError, match="not unique"):
        compact_session([source, source], complete=lambda _p: pytest.fail("called"))


@pytest.mark.parametrize(
    ("batches", "value"),
    [
        (-1, None),
        (MAX_COMPACTION_BATCHES + 1, "value"),
        (True, "value"),
        (1, None),
        (0, "value"),
    ],
    ids=("negative", "too-large", "bool", "count-without-value", "value-without-count"),
)
def test_invalid_resume_cursor_is_refused_before_any_call(
    batches: object, value: object
) -> None:
    source = _src("Do not use UTC.")
    resume = _single(_point(source, "UTC")) if value else None
    with pytest.raises(CompactionError, match="invalid compaction resume cursor"):
        compact_session(
            [source],
            complete=lambda _p: pytest.fail("provider called"),
            resume_batches=batches,  # type: ignore[arg-type]
            resume_value=resume,
        )


def test_resume_cursor_beyond_the_source_batches_is_refused() -> None:
    source = _src("Do not use UTC.")
    with pytest.raises(CompactionError, match="exceeds source batches"):
        compact_session(
            [source],
            complete=lambda _p: pytest.fail("provider called"),
            resume_batches=2,
            resume_value=_single(_point(source, "UTC")),
        )


def test_candidate_citing_a_later_unread_fragment_is_refused() -> None:
    first = _src(_letters(1200), interaction_id=1)
    later = _src("Later secret detail", interaction_id=2)
    complete, calls = _fixed_complete(lambda _n: _single(_point(later, "secret")))
    with pytest.raises(CompactionError, match="outside its input"):
        compact_session([first, later], complete=complete, max_batch_bytes=2048)
    assert len(calls) == 1


def test_offset_point_outside_the_current_fragment_window_is_refused() -> None:
    text = _letters(3000)
    source = _src(text)
    # Batch 1 holds [0:1200), batch 2 [1200:2400). The first answer cites an
    # offset from the second, still unread, window.
    late_start = 1300
    early = _point(source, text[10:14], start_offset=10)
    late = _point(source, text[late_start : late_start + 4], start_offset=late_start)

    responses = {1: _single(late)}
    complete, _calls = _fixed_complete(lambda n: responses.get(n, _single(early)))
    with pytest.raises(CompactionError, match="outside its input"):
        compact_session([source], complete=complete, max_batch_bytes=2048)


def test_rolling_memory_admits_previous_points_and_checkpoints_each_batch() -> None:
    first = _src("goal: ship " + _letters(1189), interaction_id=1)
    second = _src("b" * 240, interaction_id=2)
    keep = _point(first, "goal: ship")
    complete, calls = _fixed_complete(lambda _n: _single(keep))
    checkpoints: list[tuple[int, SessionCompaction]] = []
    before: list[int] = []

    result = compact_session(
        [first, second],
        complete=complete,
        max_batch_bytes=2048,
        checkpoint=lambda n, v: checkpoints.append((n, v)),
        before_call=lambda: before.append(len(calls)),
    )

    assert result == _single(keep)
    assert len(calls) == compaction_batch_count([first, second], max_batch_bytes=2048)
    assert [n for n, _ in checkpoints] == list(range(1, len(calls) + 1))
    assert before == list(range(len(calls)))
    assert '"previous":null' in calls[0]
    assert '"previous":{' in calls[-1]


def _three_batch_sources() -> list[CompactionSource]:
    return [_src(ch * 1200, interaction_id=i) for i, ch in enumerate("xyz")]


def test_resume_skips_completed_batches_and_validates_the_receipt() -> None:
    sources = _three_batch_sources()
    total = compaction_batch_count(sources, max_batch_bytes=2048)
    assert total == 3
    receipt = _single(_point(sources[0], "xxxx"))
    complete, calls = _fixed_complete(lambda _n: receipt)

    result = compact_session(
        sources,
        complete=complete,
        max_batch_bytes=2048,
        resume_batches=2,
        resume_value=receipt,
    )

    assert result == receipt
    assert len(calls) == total - 2


def test_resume_receipt_citing_evidence_beyond_the_cursor_is_refused() -> None:
    sources = _three_batch_sources()
    unread = _single(_point(sources[2], "zzzz"))
    with pytest.raises(CompactionError, match="cites unread evidence"):
        compact_session(
            sources,
            complete=lambda _p: pytest.fail("provider called"),
            max_batch_bytes=2048,
            resume_batches=1,
            resume_value=unread,
        )


def test_resume_receipt_with_unresolvable_quote_is_refused() -> None:
    sources = _three_batch_sources()
    forged = _single(_point(sources[0], "not in source"))
    with pytest.raises(CompactionError, match="does not resolve"):
        compact_session(
            sources,
            complete=lambda _p: pytest.fail("provider called"),
            max_batch_bytes=2048,
            resume_batches=1,
            resume_value=forged,
        )


def test_resume_receipt_offset_before_the_read_window_is_refused() -> None:
    text = _letters(3000)
    source = _src(text)
    # The receipt claims batch one was read but cites a later offset.
    forged = _single(_point(source, text[1300:1304], start_offset=1300))
    with pytest.raises(CompactionError, match="cites unread evidence"):
        compact_session(
            [source],
            complete=lambda _p: pytest.fail("provider called"),
            max_batch_bytes=2048,
            resume_batches=1,
            resume_value=forged,
        )


def test_fully_resumed_session_returns_the_receipt_without_a_call() -> None:
    source = _src("Do not use UTC.")
    receipt = _single(_point(source, "UTC", start_offset=11))
    assert (
        compact_session(
            [source],
            complete=lambda _p: pytest.fail("provider called"),
            resume_batches=1,
            resume_value=receipt,
        )
        == receipt
    )


def test_offset_citation_into_an_earlier_window_is_refused_even_if_quoted_before() -> (
    None
):
    text = _letters(3000)
    source = _src(text)
    retained = _point(source, text[10:20])  # no offset: retained by quote only
    pinned = _point(source, text[10:20], start_offset=10)
    complete, calls = _fixed_complete(
        lambda n: _single(retained) if n == 1 else _single(pinned)
    )
    # Batch two's window starts at 1200 and the retained memory carries no
    # offset, so an offset-pinned citation to [10:20) has no admitted window.
    with pytest.raises(CompactionError, match="outside its input"):
        compact_session([source], complete=complete, max_batch_bytes=2048)
    assert len(calls) == 2
