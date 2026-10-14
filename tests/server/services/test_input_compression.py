"""Exercise the same public compression interface as both runtime callers."""

import json

import pytest

from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionChoice,
    CompressionError,
    CompressionPoint,
    EvidenceSource,
    OmissionCounts,
    compress_input,
    compression_tokens,
)

SCOPE = {"org_id": "org-a", "project_id": "project-a", "user_id": "alice"}


def source(index=1, text="Keep incident summaries concise."):
    return EvidenceSource(
        source_id=f"i{index}:content",
        request_id=f"request-{index}",
        interaction_id=index,
        field="content",
        speaker="User",
        source_order=index,
        text=text,
        request_source="coding_agent",
    )


def packets_for(call):
    packets = json.loads(call.messages[1]["content"])["input_packets"]
    text_by_ref = {}
    origins = packets[0].get("source_origins", {}) if packets else {}
    for packet in packets:
        for record in packet["evidence"]:
            record.update(origins.get(record.get("origin_ref"), {}))
            if "text_ref" in record:
                record["text"] = text_by_ref[record["text_ref"]]
            text_by_ref[record["source_id"]] = record["text"]
    return packets


def selected(call):
    packets = packets_for(call)
    evidence = [s for packet in packets for s in packet["evidence"]]
    # Deterministic fixture selects the original constraint wherever it appears.
    refs = [s["source_id"] for s in evidence if "approval" in s["text"]]
    choice = CompressionChoice(
        overview="Approval constraints." if refs else "",
        unresolved=[],
        omissions=[],
        points=[
            CompressionPoint(
                category="constraint",
                detail="Keep approval conditions.",
                evidence_refs=refs[:16],
            )
        ]
        if refs
        else [],
    )
    while call.validate(choice) and choice.points[0].evidence_refs:
        choice.points[0].evidence_refs.pop()
        choice.omissions = ["Some evidence was omitted to fit the test budget."]
        if not choice.points[0].evidence_refs:
            choice.points = []
            break
    return choice


def run(sources, complete=selected, **kwargs):
    options = {
        "scope": SCOPE,
        "policy": "extraction",
        "model_contract": "fixture-model-v1",
        "budget": CompressionBudget(output_tokens=1024, section_tokens=1200),
        "complete": complete,
    }
    options.update(kwargs)
    return compress_input(sources, **options)


def test_small_input_is_unchanged_without_model_or_cache_access():
    original = source()

    def forbidden(*_args):
        pytest.fail("No model/checkpoint access for small input")

    result = run([original], complete=forbidden, load=forbidden, save=forbidden)
    assert result.passthrough
    assert result.packet.evidence == [original]
    assert result.calls == result.cache_hits == 0
    assert result.covered_source_ids == (original.source_id,)


def test_host_omission_totals_survive_paid_reductions_and_cache_replay():
    originals = [
        source(index, f"Record {index}: " + "diagnostic " * 650)
        for index in range(1, 5)
    ]
    cache = {}
    leaf_calls = []

    def complete(call):
        packets = packets_for(call)
        leaf = len(packets) == 1
        leaf_calls.append(leaf)
        refs = [s["source_id"] for packet in packets for s in packet["evidence"]]
        return CompressionChoice(
            points=[
                CompressionPoint(category="selected_evidence", evidence_refs=[ref])
                for ref in (refs if leaf else refs[:1])
            ],
            # Simulate the native adapter's host-counted semantic recovery.
            omission_counts=OmissionCounts(unseen_reference_groups=int(leaf)),
        )

    options = {
        "budget": CompressionBudget(output_tokens=1024, section_tokens=2000),
        "load": cache.get,
        "save": cache.__setitem__,
    }
    result = run(originals, complete=complete, **options)
    assert result.sections >= 2 and result.calls > result.sections
    assert result.packet.omission_counts.unseen_reference_groups == sum(leaf_calls)
    assert result.packet.omission_counts.capacity_points > 0
    assert result.output_tokens == compression_tokens(result.packet.render()) <= 1024

    replay = run(
        originals,
        complete=lambda _call: pytest.fail("Replay must use saved selections"),
        **options,
    )
    assert replay.packet == result.packet
    assert replay.calls == 0 and replay.cache_hits == result.calls


def test_fitting_union_keeps_both_children_host_omission_totals():
    from reflexio.server.services.input_compression import EvidencePacket, _combine

    packets = [
        EvidencePacket(omission_counts=OmissionCounts(capacity_points=3)),
        EvidencePacket(omission_counts=OmissionCounts(unseen_reference_groups=2)),
    ]
    packet, _levels = _combine(
        packets,
        output_tokens=1024,
        measure_output=lambda packet: compression_tokens(packet.render()),
        parent_cost=lambda packet: compression_tokens(packet.render()),
        intermediate_limit=2048,
        select=lambda *_args, **_kwargs: pytest.fail("Fitting union needs no call"),
    )
    assert packet.omission_counts == OmissionCounts(
        capacity_points=3, unseen_reference_groups=2
    )


@pytest.mark.parametrize("policy", ["extraction", "offline_tuning"])
def test_all_source_characters_seen_and_early_condition_survives(policy):
    text = (
        "Staging needs approval; production requires separate approval.\n"
        + "汉字 🦊\n" * 1200
    )
    original = source(text=text)
    seen = []

    def complete(call):
        for packet in packets_for(call):
            seen.extend(packet["evidence"])
        return selected(call)

    result = run([original], complete=complete, policy=policy)
    assert not result.passthrough
    assert result.sections > 1
    fragments = {(s["start"], s["text"]): s for s in seen}
    ordered = sorted(fragments.values(), key=lambda s: s["start"])
    assert "".join(s["text"] for s in ordered) == text
    assert original.text == text
    assert "production requires separate approval" in result.packet.render()
    assert result.output_tokens <= 1024
    assert result.covered_source_ids == (original.source_id,)


def test_cached_sections_are_revalidated_and_scope_model_policy_bound():
    sources = [source(text="Require approval.\n" + "status " * 3000)]
    cache = {}

    def save(key, value):
        cache[key] = value

    first = run(sources, load=cache.get, save=save)
    assert first.calls > 0

    def forbidden(_call):
        pytest.fail("Cached completion must not be repeated")

    replay = run(sources, complete=forbidden, load=cache.get)
    assert replay.packet == first.packet
    assert replay.calls == 0
    assert replay.cache_hits == first.calls
    for override in (
        {"scope": {**SCOPE, "org_id": "org-b"}},
        {"scope": {**SCOPE, "project_id": "project-b"}},
        {"scope": {**SCOPE, "user_id": "bob"}},
        {"model_contract": "another-model"},
        {"policy": "offline_tuning"},
    ):
        assert run(sources, load=cache.get, **override).calls > 0
    entry = next(iter(cache))
    cache[entry]["points"] = [
        {"category": "constraint", "detail": "Fabricated", "evidence_refs": ["unseen"]}
    ]
    with pytest.raises(CompressionError, match="unseen"):
        run(sources, complete=forbidden, load=cache.get)


def test_unseen_evidence_and_forged_quotations_cannot_be_published():
    def forge(call):
        choice = CompressionChoice(
            overview="",
            points=[
                CompressionPoint(
                    category="constraint",
                    detail="Invented",
                    evidence_refs=["future-source"],
                )
            ],
            unresolved=[],
            omissions=[],
        )
        errors = call.validate(choice)
        assert len(errors) == 1
        assert "points[0].evidence_refs[0]" in errors[0]
        assert "future-source" not in errors[0]
        return choice

    with pytest.raises(CompressionError, match="unseen"):
        run([source(text="x " * 3000)], complete=forge)


def test_section_and_complete_call_budget_fail_before_spending():
    def forbidden(_call):
        pytest.fail("Preflight must reject before model dispatch")

    sources = [source(text="large tool result " * 2000)]
    for budget in (
        CompressionBudget(output_tokens=500, section_tokens=1000, max_sections=1),
        CompressionBudget(output_tokens=500, section_tokens=1000, max_calls=1),
        CompressionBudget(output_tokens=500, max_source_bytes=1),
        CompressionBudget(output_tokens=500, input_tokens=100),
    ):
        with pytest.raises(CompressionError):
            run(sources, complete=forbidden, budget=budget)


def test_output_limit_includes_the_copied_evidence_not_only_selection_json():
    def choose_everything(call):
        packets = packets_for(call)
        refs = [s["source_id"] for p in packets for s in p["evidence"]]
        choice = CompressionChoice(
            overview="",
            points=[
                CompressionPoint(
                    category="action",
                    detail="All",
                    evidence_refs=refs[:16],
                )
            ],
            unresolved=[],
            omissions=[],
        )
        assert not call.validate(choice)
        return choice

    result = run(
        [source(text="distinct payload " * 2000)],
        complete=choose_everything,
        budget=CompressionBudget(output_tokens=128, section_tokens=1500),
    )
    assert result.output_tokens <= 128
    assert not result.packet.evidence and not result.packet.points
    assert result.omitted_evidence_count > 0
    assert result.packet.omissions


def test_independent_sections_do_not_erase_earlier_evidence_before_combination():
    sources = [
        source(i, f"Constraint {i}: approval is required. " + "log " * 700)
        for i in range(1, 9)
    ]
    leaf_inputs = []

    def complete(call):
        packets = packets_for(call)
        if len(packets) == 1:
            assert packets[0]["points"] == []
            leaf_inputs.append(packets[0])
        return selected(call)

    result = run(sources, complete=complete)
    assert len(leaf_inputs) == result.sections
    assert result.reduction_levels >= 1
    assert result.calls <= 2 * result.sections - 1
    assert result.output_tokens == compression_tokens(result.packet.render())


def test_empty_sources_and_empty_content_do_not_crash_or_claim_model_work():
    assert run([]).passthrough
    with pytest.raises(CompressionError, match="metadata"):
        run([source(text="")], budget=CompressionBudget(output_tokens=1))


def test_duplicate_source_identity_and_missing_tenant_scope_rejected():
    with pytest.raises(CompressionError, match="not unique"):
        run([source(), source()])
    with pytest.raises(CompressionError, match="scoped"):
        run([source()], scope={"user_id": "alice"})


def test_interrupted_work_reuses_only_completed_nodes():
    sources = [source(text="Require approval. " + "trace " * 2500)]
    cache = {}
    count = 0

    def fail_second(call):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("worker interrupted")
        return selected(call)

    with pytest.raises(RuntimeError, match="interrupted"):
        run(
            sources,
            complete=fail_second,
            load=cache.get,
            save=lambda key, value: cache.update({key: value}),
        )
    assert len(cache) == 1
    resumed = run(sources, load=cache.get)
    assert resumed.cache_hits == 1
    assert "Require approval" in resumed.packet.render()


def test_appending_sources_reuses_stable_completed_sections():
    # A completed full section has the same key when later sources are appended.
    sources = [source(i, "Repeated log " * 400) for i in range(1, 5)]
    cache = {}
    first = run(sources, load=cache.get, save=lambda k, v: cache.update({k: v}))
    appended = run(
        [*sources, source(5, "A late approval correction. " * 100)], load=cache.get
    )
    assert appended.cache_hits > 0
    assert first.source_digest != appended.source_digest
    assert appended.covered_source_ids[-1] == "i5:content"


def test_consumer_rendering_controls_passthrough_and_output_measurement():
    from reflexio.server.services.input_compression import packet_tokens

    original = source(text="unimportant output " * 100)
    calls = []

    def complete(call):
        calls.append(call)
        return CompressionChoice(
            overview="", points=[], unresolved=[], omissions=["Repeated output."]
        )

    result = run(
        [original],
        complete=complete,
        budget=CompressionBudget(output_tokens=1024),
        measure_output=lambda p: packet_tokens(p) * 8,
        output_contract="consumer-framing-v1",
    )
    assert calls and not result.passthrough
    assert result.output_tokens == packet_tokens(result.packet) * 8 <= 1024


def test_distinct_intermediate_products_fit_requests_and_root_keeps_full_budget():
    import hashlib

    from reflexio.server.services.input_compression import packet_tokens

    originals = [
        source(
            i,
            "".join(hashlib.sha256(f"{i}-{j}".encode()).hexdigest() for j in range(45))
            + " routine output " * 1500,
        )
        for i in range(1, 5)
    ]
    observed = []

    def complete(call):
        raw = json.loads(call.messages[1]["content"])
        packets = packets_for(call)
        refs = [
            s["source_id"] for p in packets for s in p["evidence"] if s["start"] < 2880
        ]
        choice = CompressionChoice(
            overview="",
            points=[
                CompressionPoint(
                    category="constraint",
                    detail="Distinct source facts.",
                    evidence_refs=[ref],
                )
                for ref in refs
            ],
            unresolved=[],
            omissions=[],
        )
        while call.validate(choice) and choice.points:
            choice.points[0].evidence_refs.pop()
            if not choice.points[0].evidence_refs:
                choice.points = []
        observed.append(raw["next_request_token_budget"])
        return choice

    result = run(
        originals,
        complete=complete,
        budget=CompressionBudget(
            output_tokens=8000, input_tokens=12000, section_tokens=2500
        ),
    )
    assert result.sections > 2
    assert result.output_tokens <= 8000
    assert result.output_tokens == packet_tokens(result.packet)
    assert result.output_tokens > min(value for value in observed if value is not None)


def test_one_point_can_retain_more_than_sixteen_distinct_conditions():
    from reflexio.server.services.input_compression import _materialize, packet_tokens

    originals = [source(i, f"service {i} needs its own approval") for i in range(1, 36)]
    choice = CompressionChoice(
        overview="",
        points=[
            CompressionPoint(
                category="constraint",
                detail="Distinct per-service approval requirements.",
                evidence_refs=[f"e{i}" for i in range(1, 36)],
            )
        ],
        unresolved=[],
        omissions=[],
    )
    packet = _materialize(
        choice, {f"e{i}": s for i, s in enumerate(originals, 1)}, 16384, packet_tokens
    )
    assert packet.evidence == originals
    assert len(packet.points[0].evidence_refs) == 35


def test_bounded_packing_retains_whole_qualified_points_and_reports_loss():
    from reflexio.server.services.input_compression import _pack_choice, packet_tokens

    available = {
        "e1": source(1, "Production is allowed only after a separate approval."),
        "e2": source(2, "A later reply withdraws that approval."),
        "e3": source(3, "routine log " * 1000),
    }
    choice = CompressionChoice(
        overview="",
        points=[
            CompressionPoint(
                category="constraint",
                detail="Approval was required then withdrawn.",
                evidence_refs=["e1", "e2"],
            ),
            CompressionPoint(
                category="action", detail="Routine tooling.", evidence_refs=["e3"]
            ),
        ],
        unresolved=[],
        omissions=[],
    )
    packet = _pack_choice(choice, available, 1024, packet_tokens, lambda _p: True)
    assert {s.source_id for s in packet.evidence} == {"i1:content", "i2:content"}
    assert packet.points[0].evidence_refs == ["i1:content", "i2:content"]
    assert len(packet.points) == 1
    assert "omitted 1 ranked evidence points" in packet.omissions[-1]
    assert packet_tokens(packet) <= 1024


def test_priced_wire_overhead_of_many_distinct_short_origins_fits_all_parents():
    import math

    from reflexio.server.services.input_compression import packet_tokens

    sources = [
        source(i, f"Approval constraint {i}: owner {i} must sign.").model_copy(
            update={"source_id": str(i), "request_id": str(i), "request_source": ""}
        )
        for i in range(1, 91)
    ]
    calls = []

    def choose_all(call):
        payload = json.loads(call.messages[1]["content"])
        refs = [s["source_id"] for p in payload["input_packets"] for s in p["evidence"]]
        assert all(
            "retained_token_estimate" in s
            for p in payload["input_packets"]
            for s in p["evidence"]
        )
        rendered = json.dumps(
            {
                "messages": call.messages,
                "schema": CompressionChoice.model_json_schema(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert math.ceil(compression_tokens(rendered) * 1.2) <= 6000
        calls.append(len(payload["input_packets"]))
        return CompressionChoice(
            overview="",
            points=[
                CompressionPoint(
                    category="constraint",
                    detail="Distinct conditions",
                    evidence_refs=refs,
                )
            ],
            unresolved=[],
            omissions=[],
        )

    result = run(
        sources,
        complete=choose_all,
        budget=CompressionBudget(
            output_tokens=10000, input_tokens=6000, section_tokens=800
        ),
        measure_output=lambda p: packet_tokens(p) * 4,
        output_contract="expanded-test-v1",
    )
    assert result.sections > 2 and 2 in calls
    assert result.output_tokens <= 10000


def test_measurement_integrity_errors_are_not_converted_to_budget_omissions():
    from reflexio.server.services.input_compression import _pack_choice

    def reject_source(_packet):
        raise CompressionError("Original source identity changed")

    choice = CompressionChoice(
        overview="",
        points=[
            CompressionPoint(
                category="constraint",
                detail="Original constraint",
                evidence_refs=["e1"],
            )
        ],
        unresolved=[],
        omissions=[],
    )
    with pytest.raises(CompressionError, match="identity changed"):
        _pack_choice(choice, {"e1": source()}, 1024, reject_source, lambda _p: True)


def test_observed_single_point_shape_is_source_checked_partial_output():
    from pydantic import ValidationError

    from reflexio.server.services.input_compression import _pack_choice, packet_tokens

    choice = CompressionChoice.model_validate(
        {
            "category": "constraint",
            "detail": "Approval is required.",
            "evidence_refs": ["e1"],
        }
    )
    packet = _pack_choice(
        choice, {"e1": source()}, 1024, packet_tokens, lambda _p: True
    )
    assert len(packet.points) == len(packet.evidence) == 1
    assert "completeness is unverified" in packet.omissions[0]
    with pytest.raises(CompressionError, match="unseen"):
        _pack_choice(choice, {}, 1024, packet_tokens, lambda _p: True)
    with pytest.raises(ValidationError):
        CompressionChoice.model_validate(
            {"category": "constraint", "detail": "Missing citations"}
        )


def test_navigation_labels_do_not_relax_original_evidence_validation():
    from reflexio.server.services.input_compression import _pack_choice, packet_tokens

    choice = CompressionChoice(
        overview="",
        points=[
            CompressionPoint(
                category="implementation_constraint",
                detail="A navigation label is not testimony.",
                evidence_refs=["e1"],
            )
        ],
        unresolved=[],
        omissions=[],
    )
    packet = _pack_choice(
        choice, {"e1": source()}, 4096, packet_tokens, lambda _p: True
    )
    assert packet.evidence[0].text == source().text
    choice.points[0] = choice.points[0].model_copy(
        update={"evidence_refs": ["not-an-original-source"]}
    )
    with pytest.raises(CompressionError, match="unseen source evidence"):
        _pack_choice(choice, {"e1": source()}, 4096, packet_tokens, lambda _p: True)


def test_selected_fields_keep_original_tool_order_and_contiguous_fragments():
    from reflexio.server.services.input_compression import _materialize, packet_tokens

    sources = [
        source().model_copy(
            update={
                "source_id": f"field-{field}-{start}",
                "field": field,
                "source_order": 1,
                "start": start,
                "text": "original",
            }
        )
        for field, start in [
            ("content", 0),
            ("tools_used[2]", 0),
            ("tools_used[2]", 512),
            ("tools_used[10]", 0),
        ]
    ]
    available = {f"e{i}": s for i, s in enumerate(sources, 1)}
    choice = CompressionChoice(
        overview="",
        points=[
            CompressionPoint(
                category="action",
                detail="Ordered actions",
                evidence_refs=list(reversed(available)),
            )
        ],
        unresolved=[],
        omissions=[],
    )
    packet = _materialize(choice, available, 4096, packet_tokens)
    assert packet.evidence == sources
