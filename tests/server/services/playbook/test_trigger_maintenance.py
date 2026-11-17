"""Portable boundary tests: real requests/parsing, recorded fake inference only."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    FinalPlaybookRevision,
    TriggerBatch,
    TriggerGate,
    TriggerSourceFrame,
    run_trigger_maintenance,
)


@pytest.fixture
def frame():
    return TriggerSourceFrame(
        incumbent_id=9,
        owner_id="owner",
        incumbent_version="a" * 64,
        evidence_bundle_digest="b" * 64,
        final_content_artifact_digest="c" * 64,
        original_content="Old wording.",
        original_trigger="After a successful charge",
        final_content="After a timeout the charge result is unknown. Check the ledger before retrying.",
    )


def gate(frame, *, gap="opposed_consultation_state", missing=None):
    return {
        "contrast": {
            "gap": gap,
            "missed_task": "Determine whether a timed-out charge completed",
            "original_excludes_because": "Requires known success",
            "content_actually_provides": "Ledger inspection when the result is unknown",
            "revision_admits_because": "Consult with unknown result",
            "support": [
                {"field": "trigger", "quote": frame.original_trigger},
                {"field": "content", "quote": "the charge result is unknown"},
            ],
            "missing_facts": missing or [],
        },
        "revision": {
            "action": "rewrite",
            "trigger": "gate proposal MUST NOT reach builder",
            "reason": "Contrast",
            "support": [{"field": "content", "quote": "Check the ledger"}],
        },
    }


def built(frame, *, action="rewrite", index=0):
    return {
        "revisions": [
            {
                "index": index,
                "revision": {
                    "action": action,
                    "trigger": frame.original_trigger
                    if action == "keep"
                    else "When a charge result is unknown after a timeout",
                    "reason": "Consult the actual ledger guidance",
                    "support": [{"field": "content", "quote": "Check the ledger"}],
                },
            }
        ]
    }


def transport(*outputs):
    replies = iter(outputs)
    calls = []

    def invoke(system, payload, schema):
        calls.append((system, payload, schema))
        # Exercise provider JSON parsing, not unchecked model_copy data.
        return schema.model_validate_json(json.dumps(next(replies)))

    return invoke, calls


def test_single_rule_requests_bind_final_content_without_gate_contamination(frame):
    invoke, calls = transport(gate(frame), built(frame))
    result = run_trigger_maintenance(frame, invoke)
    assert result.status == "repaired"
    assert result.revision.revised_content == frame.final_content
    assert (
        result.revision.revised_trigger
        == "When a charge result is unknown after a timeout"
    )
    assert result.source_digest == frame.digest
    assert result.revision.changes(frame)
    assert [c[2] for c in calls] == [TriggerGate, TriggerBatch]
    assert calls[0][1] == {
        "source": {"trigger": frame.original_trigger, "content": frame.final_content}
    }
    assert calls[1][1] == {
        "sources": [
            {
                "index": 0,
                "original_trigger": frame.original_trigger,
                "content": frame.final_content,
                "gap": "opposed_consultation_state",
            }
        ]
    }
    assert "gate proposal MUST NOT reach builder" not in json.dumps(calls[1][1])
    assert result.gate is not None and result.builder is not None
    assert result.gate.revision.trigger == "gate proposal MUST NOT reach builder"
    assert (
        result.builder.revisions[0].revision.trigger == result.revision.revised_trigger
    )


@pytest.mark.parametrize("original", [None, "", "  \t", "长" * 321])
def test_preserves_null_blank_and_long_trigger_without_inference(frame, original):
    frame = frame.model_copy(update={"original_trigger": original})
    invoke = Mock(side_effect=AssertionError("No inference permitted"))
    result = run_trigger_maintenance(frame, invoke)
    invoke.assert_not_called()
    assert result.status == "preserved_without_inference"
    assert result.revision.revised_trigger == original
    assert result.revision.revised_content == frame.final_content


@pytest.mark.parametrize(
    "gap,missing",
    [("none", []), ("uncertain", []), ("promised_help_absent", ["Authority missing"])],
)
def test_unresolved_gate_preserves_original_and_raw_output(frame, gap, missing):
    raw = gate(frame, gap=gap, missing=missing)
    invoke, calls = transport(raw)
    result = run_trigger_maintenance(frame, invoke)
    assert len(calls) == 1
    assert result.status == "gate_keep"
    assert result.revision.revised_trigger == frame.original_trigger
    assert result.gate is not None
    assert result.gate.model_dump(mode="json") == raw
    assert result.builder is None


@pytest.mark.parametrize(
    "gap,missing",
    [("none", []), ("uncertain", []), ("promised_help_absent", ["Missing authority"])],
)
@pytest.mark.parametrize("damage", ["changed_keep", "fake_quote"])
def test_unresolved_gate_still_refuses_malformed_revision(frame, gap, missing, damage):
    raw = gate(frame, gap=gap, missing=missing)
    if damage == "changed_keep":
        raw["revision"]["action"] = "keep"
    else:
        raw["revision"]["support"][0]["quote"] = "not supplied by the source"
    invoke, calls = transport(raw)
    with pytest.raises(ValueError):
        run_trigger_maintenance(frame, invoke)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "damage", ["fake_quote", "missing_trigger_support", "empty_contrast", "false_keep"]
)
def test_invalid_resolved_gate_is_failure_not_successful_keep(frame, damage):
    raw = gate(frame)
    if damage == "fake_quote":
        raw["contrast"]["support"][0]["quote"] = "not in source"
    elif damage == "missing_trigger_support":
        raw["contrast"]["support"] = raw["contrast"]["support"][1:]
    elif damage == "empty_contrast":
        raw["contrast"]["missed_task"] = ""
    else:
        raw["revision"]["action"] = "keep"
    invoke, calls = transport(raw)
    with pytest.raises(ValueError):
        run_trigger_maintenance(frame, invoke)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "wrong_index",
        "duplicate_index",
        "fake_quote",
        "long",
        "unchanged_rewrite",
        "changed_keep",
    ],
)
def test_builder_refuses_bad_inventory_quotes_or_revision(frame, damage):
    raw = built(frame)
    r = raw["revisions"][0]["revision"]
    if damage == "wrong_index":
        raw["revisions"][0]["index"] = 1
    elif damage == "duplicate_index":
        raw["revisions"] *= 2
    elif damage == "fake_quote":
        r["support"][0]["quote"] = frame.original_content
    elif damage == "long":
        r["trigger"] = "x" * 321
    elif damage == "unchanged_rewrite":
        r["trigger"] = frame.original_trigger
    else:
        r["action"] = "keep"
    invoke, calls = transport(gate(frame), raw)
    with pytest.raises(ValueError):
        run_trigger_maintenance(frame, invoke)
    assert len(calls) == 2


def test_builder_can_reject_gate_with_exact_keep(frame):
    invoke, calls = transport(gate(frame), built(frame, action="keep"))
    result = run_trigger_maintenance(frame, invoke)
    assert len(calls) == 2 and result.status == "builder_keep"
    assert result.revision.revised_trigger == frame.original_trigger


def test_exact_ceiling_is_eligible_and_keep_preserves_unicode(frame):
    frame = frame.model_copy(update={"original_trigger": "字" * 320})
    invoke, calls = transport(gate(frame, gap="none"))
    result = run_trigger_maintenance(frame, invoke)
    assert len(calls) == 1
    assert result.revision.revised_trigger == frame.original_trigger


def test_source_only_builder_cannot_override_the_gate_restriction(frame):
    from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
        trigger_builder_request,
    )

    with pytest.raises(ValueError, match="resolved supported gap"):
        trigger_builder_request(frame, "none")  # type: ignore[arg-type]


@pytest.mark.parametrize("stage", ["gate", "builder"])
def test_transport_failure_is_not_retried_or_relabelled(frame, stage):
    first = Mock(side_effect=TimeoutError("fake provider failure"))

    def invoke(system, payload, schema):
        if stage == "builder" and schema is TriggerGate:
            return TriggerGate.model_validate_json(json.dumps(gate(frame)))
        return first(system, payload, schema)

    with pytest.raises(TimeoutError):
        run_trigger_maintenance(frame, invoke)
    assert first.call_count == 1


def test_oversized_final_content_refused_before_inference(frame):
    frame = frame.model_copy(update={"final_content": "word " * 70_000})
    invoke = Mock(side_effect=AssertionError("No inference permitted"))
    with pytest.raises(ValueError, match="input budget"):
        run_trigger_maintenance(frame, invoke)
    invoke.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "owner_id",
        "incumbent_version",
        "evidence_bundle_digest",
        "final_content_artifact_digest",
        "original_content",
        "original_trigger",
        "final_content",
    ],
)
def test_frame_binding_covers_source_and_final_content(frame, field):
    replacement = (
        "d" * 64
        if field.endswith("digest") or field == "incumbent_version"
        else "other text"
    )
    other = TriggerSourceFrame.model_validate_json(
        frame.model_copy(update={field: replacement}).model_dump_json()
    )
    assert other.digest != frame.digest


def test_final_revision_requires_explicit_nullable_trigger_and_detects_both_changes(
    frame,
):
    with pytest.raises(ValidationError):
        FinalPlaybookRevision.model_validate(
            {"revised_content": frame.original_content}
        )
    noop = FinalPlaybookRevision(
        revised_content=frame.original_content, revised_trigger=frame.original_trigger
    )
    assert not noop.changes(frame)
    assert noop.model_copy(update={"revised_trigger": "other"}).changes(frame)
    assert noop.model_copy(update={"revised_content": frame.final_content}).changes(
        frame
    )
    assert FinalPlaybookRevision(
        revised_content=frame.original_content, revised_trigger=None
    ).changes(frame)
