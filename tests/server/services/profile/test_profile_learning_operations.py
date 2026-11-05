"""Coherent units and exact geometry through the real unchanged semantic kernel."""

import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reflexio.server.services.profile.learning import compiler as candidate
from reflexio.server.services.profile.learning.operation_author import (
    OperationBook,
    current_book,
    operation_scope,
)


def payload(text="User paints Saturdays at Cedar Studio."):
    return {
        "original_window": [],
        "NEW": [{"id": "NEW-0", "content": "Requested removal of venue"}],
        "EXISTING": [{"id": "EXISTING-0", "content": text, "expires": 999}],
    }


def target(start=0, end=3, row="EXISTING-0"):
    return {"row_id": row, "start_token": start, "end_token": end}


def forget(book, eid="e"):
    return {
        "snapshot": book.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [{"clauses": [{"action": "carry", "target": target()}]}],
                "forget": [
                    {
                        "source_new_id": "NEW-0",
                        "targets": [target(3, 6)],
                        "evidence": [
                            {"excerpt_id": eid, "stance": "forget_instruction"}
                        ],
                    }
                ],
            }
        ],
    }


def test_exact_unicode_whitespace_and_input_immutability():
    raw = payload("用户 paints\n Saturdays at Cedar Studio.")
    before = deepcopy(raw)
    book = OperationBook(raw)
    group = book.compile(forget(book))["groups"][0]
    assert group["writes"][0]["clauses"][0]["content"] == "用户 paints\n Saturdays"
    assert [d["literal"] for d in group["clause_dispositions"]] == [
        "用户 paints\n Saturdays",
        "at Cedar Studio.",
    ]
    assert group["clause_dispositions"][0]["write_indexes"] == [0]
    assert group["retire_ids"] == ["EXISTING-0"]
    assert raw == before
    assert book.rendered()["EXISTING"][0]["token_positions"] == {
        "0": "用户",
        "1": "paints",
        "2": "Saturdays",
        "3": "at",
        "4": "Cedar",
        "5": "Studio.",
    }


@pytest.mark.parametrize(
    "change",
    [
        "snapshot",
        "foreign",
        "negative",
        "reverse",
        "overflow",
        "overlap",
        "missing",
        "extra",
    ],
)
def test_invalid_wire_is_inert_and_feedback_has_locations(change):
    book = OperationBook(payload())
    wire = forget(book)
    clause = wire["groups"][0]["writes"][0]["clauses"][0]
    if change == "snapshot":
        wire["snapshot"] = "wrong"
    elif change == "foreign":
        clause["target"]["row_id"] = "EXISTING-1"
    elif change == "negative":
        clause["target"]["start_token"] = -1
    elif change == "reverse":
        clause["target"].update(start_token=3, end_token=2)
    elif change == "overflow":
        clause["target"]["end_token"] = 999
    elif change == "overlap":
        wire["groups"][0]["forget"][0]["targets"][0] = target(2, 6)
    elif change == "missing":
        wire["groups"] = []
    elif change == "extra":
        clause["content"] = "invented carry text"
    before = deepcopy(book.payload)
    with pytest.raises(ValidationError) as failure:
        book.compile(wire)
    errors = failure.value.errors(
        include_url=False, include_input=False, include_context=False
    )
    assert errors and all(e["loc"] for e in errors)
    assert book.payload == before


def test_replacement_evidence_and_destinations_are_derived_once():
    book = OperationBook(payload("Old workplace."))
    evidence = [{"excerpt_id": "e", "stance": "assertion"}]
    wire = {
        "snapshot": book.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [
                    {
                        "clauses": [
                            {
                                "action": "assert",
                                "source_new_id": "NEW-0",
                                "content": "New workplace.",
                                "evidence": evidence,
                                "replaces": [target(0, 2)],
                            }
                        ]
                    }
                ],
            }
        ],
    }
    result = book.compile(wire)["groups"][0]
    assert result["clause_dispositions"] == [
        {
            "existing_id": "EXISTING-0",
            "literal": "Old workplace.",
            "disposition": "replaced",
            "write_indexes": [0],
            "evidence": evidence,
        }
    ]
    assert result["writes"][0]["clauses"][0]["origin"]["evidence"] == evidence


def test_repeated_identical_literals_keep_occurrence_order_and_flattened_slots():
    book = OperationBook(payload("Same fact. Same fact."))
    wire = {
        "snapshot": book.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [
                    {
                        "clauses": [
                            {"action": "carry", "target": target(2, 4)},
                            {"action": "carry", "target": target(0, 2)},
                        ]
                    }
                ],
            }
        ],
    }
    result = book.compile(wire)["groups"][0]
    assert [d["write_indexes"] for d in result["clause_dispositions"]] == [[1], [0]]


def test_structured_carry_splits_provenance_and_retains_output_order():
    raw = payload("First fact. Second fact.")
    raw["EXISTING"][0]["clauses"] = [
        {"content": "First fact."},
        {"content": "Second fact."},
    ]
    b = OperationBook(raw)
    wire = {
        "snapshot": b.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [{"clauses": [{"action": "carry", "target": target(0, 4)}]}],
            }
        ],
    }
    group = b.compile(wire)["groups"][0]
    assert [c["content"] for c in group["writes"][0]["clauses"]] == [
        "First fact.",
        "Second fact.",
    ]
    assert [d["write_indexes"] for d in group["clause_dispositions"]] == [[0], [1]]


def test_same_incumbent_cannot_be_changed_in_separate_groups():
    raw = payload()
    raw["NEW"].append({"id": "NEW-1"})
    b = OperationBook(raw)
    wire = forget(b)
    wire["groups"].append(
        {
            "new_ids": ["NEW-1"],
            "writes": [{"clauses": [{"action": "carry", "target": target()}]}],
        }
    )
    with pytest.raises(ValidationError) as failure:
        b.compile(wire)
    assert all(
        e["type"] == "operation_incumbent_multiple_groups"
        for e in failure.value.errors()
    )


def test_metadata_and_scope_are_request_bound():
    a = OperationBook(payload())
    raw = payload()
    raw["EXISTING"][0]["expires"] += 1
    b = OperationBook(raw)
    with pytest.raises(ValidationError):
        b.compile(forget(a))
    with pytest.raises(ValueError):
        current_book()
    with operation_scope() as scope:
        scope["book"] = a
        assert current_book() is a
    with pytest.raises(ValueError):
        current_book()


@pytest.fixture
def native():
    cm = importlib.import_module(
        "reflexio.server.services.profile.components.consolidator"
    )
    return SimpleNamespace(
        output=cm.ProfileConsolidator.OUTPUT_SCHEMA,
        consolidator=cm.ProfileConsolidator,
        cm=cm,
    )


def window(role="user", text="Delivery entrance is north.", action_description=None):
    return [
        {
            "session_id": "session",
            "request_id": "request",
            "source": "chat",
            "interactions": [
                {
                    "interaction_id": 1,
                    "created_at": 200,
                    "role": role,
                    "content": text,
                    "user_action_description": action_description,
                }
            ],
        }
    ]


def book(source=None):
    from reflexio.server.services.profile.learning.source_evidence import EvidenceBook

    return EvidenceBook(
        source or window(), user_id="owner", request_id="request", org_id="org"
    )


def excerpts(evidence):
    return evidence.rendered_window[0]["interactions"][0]["content"]["excerpts"]


def compiler(native, source):
    con = object.__new__(native.consolidator)
    con.request_context = SimpleNamespace(org_id="provider-free")
    con.lineage_sources_by_profile_id = {}
    con.consolidated_output_indices = set()
    con.enrichment_audit = []
    con.review_window = source
    return con


def profile(native, identity, text, timestamp=100, ttl="one_week", owner="owner"):
    ttl_value = native.cm.ProfileTimeToLive(ttl)
    return native.cm.UserProfile(
        profile_id=identity,
        user_id=owner,
        content=text,
        last_modified_timestamp=timestamp,
        expiration_timestamp=native.cm.calculate_expiration_timestamp(
            timestamp, ttl_value
        ),
        profile_time_to_live=ttl_value,
        generated_from_request_id="source-" + identity,
        source_span="Original quotation: " + text,
        source_interaction_ids=[timestamp],
    )


def actual(
    native,
    text="Forget Cedar Studio but keep Saturday painting.",
    old_content="User paints Saturdays at Cedar Studio.",
    fresh="Requested removal of Cedar Studio",
    action_description=None,
):

    source = window(text=text, action_description=action_description)
    evidence = book(source)
    con = compiler(native, source)
    old = [profile(native, "old", old_content, ttl="one_month", timestamp=100)]
    new = [profile(native, "new", fresh, timestamp=300)]
    with operation_scope() as scope, evidence.bound():
        con._operation_scope = scope
        con._render_dedup_prompt(new, old)
        b = scope["book"]
    return con, evidence, b, new, old, excerpts(evidence)[0]["excerpt_id"]


def hydrate(native, evidence, b, wire):
    with operation_scope() as scope, evidence.bound():
        scope["book"] = b
        return native.output.model_validate(wire)


def test_native_selective_forget_preserves_retention_and_attribution(native):
    con, e, b, new, old, eid = actual(native)
    out = hydrate(native, e, b, forget(b, eid))
    assert con._plan_errors(out, new, old) == []
    rows, retired, _ = con._build_deduplicated_results(
        new, old, out, "owner", "request"
    )
    assert [p.content for p in rows] == ["User paints Saturdays"] and retired == ["old"]
    assert rows[0].expiration_timestamp == old[0].expiration_timestamp
    assert rows[0].source_interaction_ids == old[0].source_interaction_ids


def test_native_missing_residual_stays_rejected_without_auto_publication(native):
    con, e, b, new, old, eid = actual(native)
    wire = forget(b, eid)
    wire["groups"][0]["writes"] = []
    with pytest.raises(ValidationError) as failure:
        hydrate(native, e, b, wire)
    assert failure.value.errors()[0]["type"] == "operation_incomplete_row_coverage"
    assert old[0].content == "User paints Saturdays at Cedar Studio."


def test_native_replacement_with_one_evidence_decision_and_bad_terms_nonfatal(native):
    con, e, b, new, old, eid = actual(
        native,
        text="My books are at West Branch now.",
        old_content="My books are at East Branch.",
        fresh="My books are at West Branch.",
    )
    wire = {
        "snapshot": b.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [
                    {
                        "clauses": [
                            {
                                "action": "assert",
                                "source_new_id": "NEW-0",
                                "content": new[0].content,
                                "evidence": [
                                    {"excerpt_id": eid, "stance": "assertion"}
                                ],
                                "replaces": [target(0, 6)],
                            }
                        ],
                        "search_terms": {"bad": "metadata"},
                    }
                ],
            }
        ],
    }
    out = hydrate(native, e, b, wire)
    assert con._plan_errors(out, new, old) == []
    rows, retired, _ = con._build_deduplicated_results(
        new, old, out, "owner", "request"
    )
    assert [p.content for p in rows] == [new[0].content] and retired == ["old"]
    wrong = deepcopy(wire)
    wrong["groups"][0]["writes"][0]["clauses"][0]["evidence"][0]["stance"] = (
        "historical_quote"
    )
    out = hydrate(native, e, b, wrong)
    assert con._plan_errors(out, new, old)


@pytest.mark.parametrize("retention_field", ["content", "user_action_description"])
@pytest.mark.parametrize("rule", ["one_week", "2026-11-05T00:00:00Z"])
def test_native_explicit_renewal_changes_only_retention(native, retention_field, rule):
    instruction = (
        "Please remember my painting fact for one week starting now."
        if rule == "one_week"
        else "Please keep my painting fact through 2026-11-05T00:00:00Z."
    )
    con, e, b, new, old, eid = actual(
        native,
        text=instruction if retention_field == "content" else "Remember this forever.",
        fresh="Renew painting memory",
        action_description=instruction if retention_field != "content" else None,
    )
    if retention_field != "content":
        eid = e.rendered_window[0]["interactions"][0][retention_field]["excerpts"][0][
            "excerpt_id"
        ]
    wire = {
        "snapshot": b.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0"],
                "writes": [
                    {
                        "clauses": [
                            {
                                "action": "carry",
                                "target": target(0, 6),
                                "explicit_retention": {
                                    "source_new_id": "NEW-0",
                                    "rule": rule,
                                    "boundary": "through",
                                    "evidence": [
                                        {
                                            "excerpt_id": eid,
                                            "stance": "retention_instruction",
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ],
            }
        ],
    }
    out = hydrate(native, e, b, wire)
    assert con._plan_errors(out, new, old) == []
    assert set(native.output.model_json_schema()["properties"]) == {
        "snapshot",
        "groups",
    }
    rows, retired, _ = con._build_deduplicated_results(
        new, old, out, "owner", "request"
    )
    assert retired == ["old"]
    assert rows[0].last_modified_timestamp == old[0].last_modified_timestamp
    assert rows[0].expiration_timestamp == (
        200 + 7 * 86400 if rule == "one_week" else 1793836800
    )


def test_native_author_renderer_preserves_canonical_semantic_policy(native):
    con, evidence, _, new, old, _ = actual(native)
    with operation_scope() as scope, evidence.bound():
        con._operation_scope = scope
        prompt = con._render_dedup_prompt(new, old)
    assert prompt.count(candidate.CONTENT_GUIDANCE) == 1
    assert "Forgetting instructions are actions, never new biographical facts" in prompt


def test_prompt_action_example_compiles_to_carry_and_scoped_forgetting():
    import json

    from reflexio.server.services.profile.learning.operation_author import GUIDANCE

    example = GUIDANCE.split("the group shape is:\n", 1)[1].split("\nCode carries", 1)[
        0
    ]
    group = json.loads(example)
    book = OperationBook(payload("Cycles on Sundays along River Road."))
    compiled = book.compile({"snapshot": book.snapshot, "groups": [group]})["groups"][0]
    assert len(compiled["writes"]) == 1
    clause = compiled["writes"][0]["clauses"][0]
    assert clause["content"] == "Cycles on Sundays"
    assert clause["origin"] == {"kind": "carry", "existing_id": "EXISTING-0"}
    assert [item["literal"] for item in compiled["clause_dispositions"]] == [
        "Cycles on Sundays",
        "along River Road.",
    ]
    assert [item["disposition"] for item in compiled["clause_dispositions"]] == [
        "retained",
        "forgotten",
    ]


@pytest.mark.parametrize("stance", ["forget_instruction", "historical_quote", "draft"])
def test_native_whole_forget_with_redundant_fresh_preference_keeps_authority(
    native, stance
):

    source = window(
        text="Forget my contact number, but keep my preference for concise replies."
    )
    evidence = book(source)
    con = compiler(native, source)
    old = [
        profile(native, "phone", "The user contact number is 555-0102.", timestamp=100),
        profile(
            native, "preference", "The user prefers concise replies.", timestamp=100
        ),
    ]
    new = [
        profile(
            native, "request", "Requested removal of contact number", timestamp=300
        ),
        profile(native, "redundant", "prefers concise replies", timestamp=300),
    ]
    with operation_scope() as scope, evidence.bound():
        con._operation_scope = scope
        con._render_dedup_prompt(new, old)
        b = scope["book"]
    wire = {
        "snapshot": b.snapshot,
        "groups": [
            {
                "new_ids": ["NEW-0", "NEW-1"],
                "writes": [
                    {
                        "clauses": [
                            {"action": "carry", "target": target(0, 5, "EXISTING-1")}
                        ]
                    }
                ],
                "forget": [
                    {
                        "source_new_id": "NEW-0",
                        "targets": [target(0, 6)],
                        "evidence": [
                            {
                                "excerpt_id": excerpts(evidence)[0]["excerpt_id"],
                                "stance": stance,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    out = hydrate(native, evidence, b, wire)
    errors = con._plan_errors(out, new, old)
    if stance != "forget_instruction":
        assert any("explicit forget evidence" in error for error in errors)
        rows, retired, _ = con._build_deduplicated_results(
            new, old, out, "owner", "request"
        )
        assert rows == [] and retired == []
    else:
        assert errors == []
        rows, retired, _ = con._build_deduplicated_results(
            new, old, out, "owner", "request"
        )
        assert rows == [] and retired == ["phone"]
        assert out.duplicate_groups[0].writes == []
        assert "EXISTING-1" not in out.duplicate_groups[0].item_ids
        assert old[1].content == "The user prefers concise replies."
        assert old[1].last_modified_timestamp == 100


def retention_write():
    return SimpleNamespace(
        evidence=[
            SimpleNamespace(
                session_id="session",
                interaction_id=1,
                role="user",
                stance="retention_instruction",
            )
        ],
        time_to_live="one_day",
        explicit_deadline=None,
    )


def test_relative_retention_uses_instruction_clock_not_extraction_clock(native):
    source = window(text="Please remember my tea preference for one day.")
    source[0]["interactions"][0]["created_at"] = 100
    con = compiler(native, source)
    # Delayed extraction and carried facts must not restart the instruction.
    assert con._deadline(retention_write()) == 86500


@pytest.mark.parametrize("timestamp", [None, True, "100", 100.5])
def test_relative_retention_refuses_missing_or_untyped_instruction_clock(
    native, timestamp
):
    source = window(text="Please remember my tea preference for one day.")
    source[0]["interactions"][0]["created_at"] = timestamp
    with pytest.raises(ValueError, match="instruction timestamp"):
        compiler(native, source)._deadline(retention_write())


def test_relative_retention_refuses_conflicting_instruction_clocks(native):
    source = window(text="Please remember my tea preference for one day.")
    source[0]["interactions"][0]["created_at"] = 100
    second = deepcopy(source[0]["interactions"][0])
    second.update(interaction_id=2, created_at=200)
    source[0]["interactions"].append(second)
    write = retention_write()
    write.evidence.append(
        SimpleNamespace(
            session_id="session",
            interaction_id=2,
            role="user",
            stance="retention_instruction",
        )
    )
    with pytest.raises(ValueError, match="instruction timestamp"):
        compiler(native, source)._deadline(write)
