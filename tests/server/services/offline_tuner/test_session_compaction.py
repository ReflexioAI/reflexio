"""Source fidelity, bounded batches, and eligibility invariants for compaction."""

import json
from dataclasses import replace

import pytest

from reflexio.server.services.offline_tuner.evidence.compaction import (
    CompactionError,
    CompactionIdentity,
    CompactionPoint,
    CompactionSource,
    SessionCompaction,
    compact_session,
    compaction_batch_count,
    source_batches,
)


def source(text: str, source_id: str = "source-1") -> CompactionSource:
    return CompactionSource(source_id, "request-1", 1, "content", "user", text)


def point(source_id: str, quote: str) -> CompactionPoint:
    return CompactionPoint(
        category="constraint",
        detail="User supplied this constraint",
        source_id=source_id,
        exact_quote=quote,
    )


def test_single_large_unicode_record_is_read_without_truncation() -> None:
    original = source("汉字 🦊\n" * 3000)
    batches = list(source_batches([original], max_batch_bytes=4096))
    fragments = [record for batch in batches for record in batch]
    assert len(batches) > 1
    assert "".join(str(r["text"]) for r in fragments) == original.text
    assert fragments[0]["start"] == 0
    assert fragments[-1]["end"] == len(original.text)
    for batch in batches:
        assert (
            len(json.dumps(batch, ensure_ascii=False, separators=(",", ":")).encode())
            <= 4096
        )


def test_rolling_memory_can_retain_an_early_constraint_after_later_turns() -> None:
    sources = [
        source("Do not use UTC. " + "x" * 5000),
        source("The timezone bug appeared later.", "source-2"),
    ]
    calls = []
    checkpoints = []

    def complete(payload: str) -> SessionCompaction:
        data = json.loads(payload)
        calls.append(data)
        previous = data["previous"]
        if previous:
            return SessionCompaction.model_validate(previous)
        return SessionCompaction(points=[point("source-1", "Do not use UTC.")])

    result = compact_session(
        sources,
        complete=complete,
        max_batch_bytes=2048,
        checkpoint=lambda index, value: checkpoints.append((index, value)),
    )
    assert len(calls) > 1
    assert result.points[0].exact_quote == "Do not use UTC."
    assert len(checkpoints) == len(calls)
    assert "The timezone bug appeared later." in json.dumps(calls[-1])


@pytest.mark.parametrize(
    "bad", [point("missing", "fiction"), point("source-1", "rewritten text")]
)
def test_model_cannot_invent_or_rewrite_original_evidence(bad: CompactionPoint) -> None:
    with pytest.raises(CompactionError, match="original evidence"):
        compact_session(
            [source("Original text")],
            complete=lambda _: SessionCompaction(points=[bad]),
        )


def test_model_cannot_cite_a_later_fragment_it_has_not_read() -> None:
    original = source("x" * 6000 + "critical late correction")
    with pytest.raises(CompactionError, match="outside its input"):
        compact_session(
            [original],
            max_batch_bytes=2048,
            complete=lambda _: SessionCompaction(
                points=[point("source-1", "critical late correction")]
            ),
        )


def test_authorization_runs_before_each_paid_call() -> None:
    calls = []

    def authorize() -> None:
        calls.append("authorize")
        if calls.count("authorize") == 2:
            raise CompactionError("source erased")

    def complete(_: str) -> SessionCompaction:
        calls.append("model")
        return SessionCompaction(points=[point("source-1", "x")])

    with pytest.raises(CompactionError, match="erased"):
        compact_session(
            [source("x" * 6000)],
            max_batch_bytes=2048,
            complete=complete,
            before_call=authorize,
        )
    assert calls == ["authorize", "model", "authorize"]


def test_reuse_identity_changes_for_every_authorization_and_model_boundary() -> None:
    original = CompactionIdentity(
        "org",
        "project",
        "user",
        "session",
        "a" * 64,
        "cutoff",
        (("subject", 1),),
        "model-1",
    )
    for field, value in [
        ("tenant_id", "other"),
        ("project_id", "other"),
        ("user_id", "other"),
        ("session_id", "other"),
        ("trajectory_digest", "b" * 64),
        ("through_request_id", "later"),
        ("governance_epochs", (("subject", 2),)),
        ("model", "model-2"),
    ]:
        assert replace(original, **{field: value}).digest != original.digest


def test_empty_and_duplicate_sources_are_explicit_failures() -> None:
    with pytest.raises(CompactionError, match="no compactable"):
        compact_session([], complete=lambda _: pytest.fail("no model call"))
    with pytest.raises(CompactionError, match="not unique"):
        compact_session(
            [source("a"), source("b")], complete=lambda _: pytest.fail("no model call")
        )


def test_partial_checkpoint_resumes_without_repeating_completed_calls() -> None:
    sources = [source("Keep this constraint. " + "x" * 3000)]
    receipt = SessionCompaction(points=[point("source-1", "Keep this constraint.")])
    batches = list(source_batches(sources, max_batch_bytes=2048))
    calls = []

    def complete(payload: str) -> SessionCompaction:
        data = json.loads(payload)
        calls.append(data)
        assert data["previous"] == receipt.model_dump(mode="json")
        return receipt

    result = compact_session(
        sources,
        complete=complete,
        max_batch_bytes=2048,
        resume_batches=1,
        resume_value=receipt,
    )
    assert result == receipt
    assert len(calls) == len(batches) - 1


def test_resume_receipt_cannot_cite_unread_later_evidence() -> None:
    sources = [source("x" * 3000 + "Unseen future text.")]
    receipt = SessionCompaction(points=[point("source-1", "Unseen future text.")])
    with pytest.raises(CompactionError, match="unread evidence"):
        compact_session(
            sources,
            complete=lambda _: receipt,
            max_batch_bytes=2048,
            resume_batches=1,
            resume_value=receipt,
        )


@pytest.mark.parametrize("resume_batches", [0, 1, 128])
def test_impossible_call_budget_fails_before_authorization_model_or_checkpoint(
    resume_batches: int,
) -> None:
    # Metadata makes each small source require its own batch; raw bytes alone
    # cannot detect this overrun. Even a resumed job must reject before spending.
    sources = [
        replace(source("x", f"source-{i}"), request_id="r" * 1800) for i in range(129)
    ]
    assert len(list(source_batches(sources, max_batch_bytes=2048))) == 129
    effects = []
    receipt = SessionCompaction(points=[point("source-0", "x")])
    with pytest.raises(CompactionError, match="compaction call budget"):
        compact_session(
            sources,
            max_batch_bytes=2048,
            before_call=lambda: effects.append("authorize"),
            complete=lambda _: effects.append("model") or receipt,
            checkpoint=lambda *_: effects.append("checkpoint"),
            resume_batches=resume_batches,
            resume_value=receipt if resume_batches else None,
        )
    assert effects == []


def test_exact_call_budget_is_admitted_and_every_batch_is_processed() -> None:
    sources = [
        replace(source("x", f"source-{i}"), request_id="r" * 1800) for i in range(128)
    ]
    assert compaction_batch_count(sources, max_batch_bytes=2048) == 128
    checkpoints = []

    def complete(payload: str) -> SessionCompaction:
        record = json.loads(payload)["source_records"][0]
        return SessionCompaction(points=[point(record["source_id"], record["text"])])

    result = compact_session(
        sources,
        max_batch_bytes=2048,
        complete=complete,
        checkpoint=lambda index, _: checkpoints.append(index),
    )
    assert checkpoints == list(range(1, 129))
    assert result.points[0].source_id == "source-127"


def test_single_record_budget_includes_both_json_array_brackets() -> None:
    original = source("x")
    baseline = list(source_batches([original], max_batch_bytes=2048))[0]
    size = len(json.dumps(baseline, ensure_ascii=False, separators=(",", ":")).encode())
    fitting = replace(original, request_id=original.request_id + "r" * (2048 - size))
    assert (
        len(
            json.dumps(
                list(source_batches([fitting], max_batch_bytes=2048))[0],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        == 2048
    )
    oversized = replace(fitting, request_id=fitting.request_id + "r")
    with pytest.raises(CompactionError, match="source identity exceeds"):
        compaction_batch_count([oversized], max_batch_bytes=2048)
