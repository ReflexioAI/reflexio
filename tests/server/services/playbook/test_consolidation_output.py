"""Schema tests for PlaybookConsolidationOutput as a 4-kind discriminated union.

These tests pin down the shape of the consolidation LLM output schema after the
4-kind redesign: ``unify`` (subsumes the legacy ``duplicate`` and
``prefer_new``), ``reject_new`` (renamed from ``prefer_existing``),
``differentiate``, and ``independent``. The legacy 5-kind names must no
longer parse — that structural guarantee is what the new schema buys.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.playbook.components.consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    RejectNewDecision,
    UnifyDecision,
    validate_consolidation_output,
)


def _playbook(idx: int) -> UserPlaybook:
    return UserPlaybook(
        content=f"content {idx}",
        trigger=f"trigger {idx}",
        request_id=f"req-{idx}",
        agent_version="v1",
        source_interaction_ids=[idx],
    )


def test_active_prompt_documents_partition_contract():
    prompt_path = (
        Path(__file__).resolve().parents[4]
        / "reflexio"
        / "server"
        / "prompt"
        / "prompt_bank"
        / "playbook_consolidation"
        / "v2.4.0.prompt.md"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "Every NEW label MUST appear exactly once" in prompt
    assert '"new_id": ["NEW-0", "NEW-1"]' in prompt
    assert "`archive_existing_ids` only refers to rows from the EXISTING list" in prompt


def test_output_json_schema_is_provider_safe_by_construction():
    """``model_json_schema()`` emits no ``oneOf``/``discriminator`` natively.

    Option C (``ProviderSafeUnionMixin``): the discriminated union is folded to
    ``anyOf`` at the model boundary, so the emitted JSON schema is accepted by
    strict structured-output providers WITHOUT ``make_strict_json_schema`` and
    WITHOUT any provider-detection gate (the gap that caused Sentry
    ``PYTHON-FASTAPI-9J``). Asserting on the raw schema proves the property is
    intrinsic to the model, not bolted on by a downstream normalizer.
    """
    raw = json.dumps(PlaybookConsolidationOutput.model_json_schema())
    # Bare-word (not just the ``"oneOf":`` JSON-key form) so a docstring/
    # description leak of these tokens into the wire schema is also caught —
    # the whole point is to keep them out of what ships to the provider.
    assert "oneOf" not in raw
    assert "discriminator" not in raw
    # The four variants survive as an ``anyOf`` branch (generation stays
    # constrained to exactly those shapes).
    assert '"anyOf"' in raw


def test_provider_safe_schema_keeps_keyed_validation_errors():
    """Folding the wire schema must NOT weaken validation.

    The discriminator stays in the core (validation) schema, so a malformed
    ``unify`` payload yields a single, pinpointed error at the ``unify``
    variant's missing field — not the noisy multi-error spray a plain
    (non-discriminated) union produces. This precise-error property is Option
    C's advantage over flattening the union (Option B).
    """
    with pytest.raises(ValidationError) as exc_info:
        PlaybookConsolidationOutput.model_validate(
            {
                "decisions": [{"kind": "unify", "new_id": "NEW-0"}]
            }  # missing content/trigger/rationale
        )
    errors = exc_info.value.errors()
    # Keyed dispatch routed validation to the ``unify`` variant only.
    assert all("unify" in err["loc"] for err in errors)
    # And the reported failure is the missing required field, not a union mismatch.
    assert any(err["type"] == "missing" and "content" in err["loc"] for err in errors)


def test_decision_kinds_are_discriminated_union():
    """A bare ``UnifyDecision`` round-trips through the output schema."""
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[1],
        content="Recommend X.",
        trigger="when Y",
        rationale="r",
    )
    out = PlaybookConsolidationOutput(decisions=[unify])
    assert out.decisions[0].kind == "unify"


def test_differentiate_requires_both_refined_triggers():
    """Empty refined triggers must fail validation."""
    with pytest.raises(ValidationError):
        DifferentiateDecision(
            new_id="NEW-0",
            existing_id=42,
            refined_new_trigger="",  # empty
            refined_existing_trigger="when narrow",
        )


def test_unify_has_no_polarity_field():
    """``UnifyDecision`` no longer declares a polarity field (wording-derived)."""
    # Parses fine without any polarity; orientation is inferred from wording at
    # apply time, not declared on the decision.
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[1],
        content="X",
        trigger="t",
        rationale="r",
    )
    assert "polarity" not in UnifyDecision.model_fields
    assert "polarity" not in unify.model_dump()


def test_unify_accepts_empty_archive_existing_ids():
    """``unify`` with an empty archive list is a valid insert-without-archive."""
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[],
        content="X",
        trigger="t",
        rationale="r",
    )
    assert unify.archive_existing_ids == []


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("NEW-0", "NEW-0"),
        ("[NEW-0]", "NEW-0"),
        (["NEW-0"], "NEW-0"),
        (["[NEW-0]"], "NEW-0"),
    ],
)
def test_single_new_id_accepts_prompt_label_variants(raw_value, expected):
    """Single-NEW decision fields tolerate common LLM label echoes."""
    d = DifferentiateDecision(
        new_id=raw_value,  # type: ignore[arg-type]
        existing_id=0,
        refined_new_trigger="narrow new",
        refined_existing_trigger="narrow existing",
    )
    assert d.new_id == expected


@pytest.mark.parametrize(
    "decision",
    [
        UnifyDecision(
            new_id=["[NEW-0]", "NEW-1"],  # type: ignore[arg-type]
            archive_existing_ids=[],
            content="X",
            trigger="t",
            rationale="r",
        ),
        RejectNewDecision(
            new_id=["[NEW-0]", "NEW-1"],  # type: ignore[arg-type]
            superseded_by_existing_id=0,
        ),
        IndependentDecision(new_id=["[NEW-0]", "NEW-1"]),  # type: ignore[arg-type]
    ],
)
def test_multi_new_decisions_accept_new_id_lists(decision):
    """Decision kinds with clear multi-NEW semantics keep canonical NEW ids."""
    assert decision.new_ids == ["NEW-0", "NEW-1"]


def test_differentiate_rejects_multi_new_id_list():
    """``differentiate`` has one pair of refined triggers, so multi-NEW is ambiguous."""
    with pytest.raises(ValidationError):
        DifferentiateDecision(
            new_id=["NEW-0", "NEW-1"],  # type: ignore[arg-type]
            existing_id=0,
            refined_new_trigger="narrow new",
            refined_existing_trigger="narrow existing",
        )


def test_reject_new_requires_superseded_existing_id():
    """``RejectNewDecision`` must name the superseding existing id."""
    with pytest.raises(ValidationError):
        RejectNewDecision(new_id="NEW-0")  # type: ignore[call-arg]


def test_all_four_kinds_round_trip_through_output():
    """All four kinds parse via the discriminated union in one batch."""
    decisions: list[ConsolidationDecision] = [
        UnifyDecision(
            new_id="NEW-0",
            archive_existing_ids=[1],
            content="X",
            trigger="t",
            rationale="r",
        ),
        RejectNewDecision(new_id="NEW-1", superseded_by_existing_id=2),
        DifferentiateDecision(
            new_id="NEW-2",
            existing_id=4,
            refined_new_trigger="when A and B",
            refined_existing_trigger="when A and not B",
        ),
        IndependentDecision(new_id="NEW-3"),
    ]
    out = PlaybookConsolidationOutput(decisions=decisions)
    kinds = [d.kind for d in out.decisions]
    assert kinds == [
        "unify",
        "reject_new",
        "differentiate",
        "independent",
    ]


def test_validate_consolidation_output_accepts_exactly_once_coverage():
    output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id=["NEW-0", "NEW-1"],
                archive_existing_ids=[],
                content="X",
                trigger="t",
                rationale="r",
            ),
            RejectNewDecision(new_id="NEW-2", superseded_by_existing_id=0),
            DifferentiateDecision(
                new_id="NEW-3",
                existing_id=0,
                refined_new_trigger="narrow new",
                refined_existing_trigger="narrow existing",
            ),
            IndependentDecision(new_id="NEW-4"),
        ]
    )

    assert validate_consolidation_output([_playbook(i) for i in range(5)], output) == []


def test_validate_consolidation_output_reports_missing_new_id():
    output = PlaybookConsolidationOutput(
        decisions=[IndependentDecision(new_id="NEW-0")]
    )

    errors = validate_consolidation_output([_playbook(0), _playbook(1)], output)

    assert any("missing NEW ids: NEW-1" in error for error in errors)


def test_validate_consolidation_output_reports_duplicate_new_id():
    output = PlaybookConsolidationOutput(
        decisions=[
            IndependentDecision(new_id="NEW-0"),
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[],
                content="X",
                trigger="t",
                rationale="r",
            ),
        ]
    )

    errors = validate_consolidation_output([_playbook(0)], output)

    assert any("NEW-0 appears in decision[0]" in error for error in errors)


def test_validate_consolidation_output_reports_unknown_new_id():
    output = PlaybookConsolidationOutput(
        decisions=[IndependentDecision(new_id="NEW-99")]
    )

    errors = validate_consolidation_output([_playbook(0)], output)

    assert any("unknown NEW id NEW-99" in error for error in errors)


@pytest.mark.parametrize(
    "legacy_payload",
    [
        # Legacy ``duplicate`` shape — subsumed by ``unify``.
        {
            "kind": "duplicate",
            "item_ids": ["NEW-0", "EXISTING-1"],
            "merged_content": "X",
            "merged_trigger": "t",
            "merged_rationale": "r",
            "merged_polarity": "positive",
        },
        # Legacy ``prefer_new`` shape — subsumed by ``unify``.
        {"kind": "prefer_new", "new_id": "NEW-0", "existing_id": 1},
        # Legacy ``prefer_existing`` shape — renamed to ``reject_new``.
        {"kind": "prefer_existing", "new_id": "NEW-0", "existing_id": 1},
    ],
)
def test_legacy_kind_literals_no_longer_parse(legacy_payload):
    """Old 5-kind discriminator values must fail to parse under the 4-kind union.

    This is the structural guarantee that the 4-kind redesign buys: callers
    that still produce the old shapes will fail loudly at decode time rather
    than silently misroute decisions.
    """
    with pytest.raises(ValidationError):
        PlaybookConsolidationOutput.model_validate({"decisions": [legacy_payload]})


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ([0, 1], [0, 1]),
        (["EXISTING-0", "EXISTING-3"], [0, 3]),
        (["[EXISTING-0]", "[EXISTING-3]"], [0, 3]),
        (["existing-2"], [2]),
        (["5", "EXISTING-6"], [5, 6]),
        ([0, "EXISTING-7"], [0, 7]),
        ("[EXISTING-8]", [8]),
        (None, []),
    ],
)
def test_unify_archive_existing_ids_accepts_position_labels(raw_value, expected):
    """``archive_existing_ids`` tolerates either bare ints or ``EXISTING-N`` strings.

    Strong structured-output models (GPT-4o, Claude) honor ``list[int]`` and return
    bare integers, but weaker models (MiniMax-M3) ignore the int constraint and
    return the literal ``"EXISTING-0"`` label from the prompt. The validator strips
    the prefix so both shapes round-trip; the apply path keeps using ints
    downstream.
    """
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=raw_value,  # type: ignore[arg-type]
        content="c",
        trigger="t",
        rationale="r",
    )
    assert unify.archive_existing_ids == expected


def test_reject_new_superseded_id_coerces_position_label():
    """``superseded_by_existing_id`` strips an ``EXISTING-N`` label to ``N``.

    The prompt instructs the model to emit the list position as a bare integer,
    but weaker models (MiniMax-M3) return the literal ``"EXISTING-4"`` label.
    Coercing here keeps this field consistent with ``archive_existing_ids`` so
    one stray label does not fail the whole batch.
    """
    r = RejectNewDecision(new_id="NEW-0", superseded_by_existing_id="EXISTING-4")  # type: ignore[arg-type]
    assert r.superseded_by_existing_id == 4


def test_reject_new_superseded_id_accepts_position_label_list():
    """``reject_new`` can name one-or-more EXISTING refs in list form."""
    r = RejectNewDecision(
        new_id="NEW-0",
        superseded_by_existing_id=["[EXISTING-1]", "EXISTING-4"],  # type: ignore[arg-type]
    )
    assert r.superseded_by_existing_ids == [1, 4]


def test_differentiate_existing_id_coerces_position_label():
    """``existing_id`` strips an ``EXISTING-N`` label to ``N`` (see
    ``test_reject_new_superseded_id_coerces_position_label``).
    """
    d = DifferentiateDecision(
        new_id="NEW-0",
        existing_id="EXISTING-9",  # type: ignore[arg-type]
        refined_new_trigger="narrow new",
        refined_existing_trigger="narrow existing",
    )
    assert d.existing_id == 9


def test_differentiate_existing_id_accepts_one_item_list():
    """``differentiate`` tolerates a one-item EXISTING list, but stays singular."""
    d = DifferentiateDecision(
        new_id="NEW-0",
        existing_id=["[EXISTING-2]"],  # type: ignore[arg-type]
        refined_new_trigger="narrow new",
        refined_existing_trigger="narrow existing",
    )
    assert d.existing_id == 2


def test_differentiate_existing_id_rejects_multi_item_list():
    """A single pair of refined triggers cannot safely cover multiple existing rows."""
    with pytest.raises(ValidationError):
        DifferentiateDecision(
            new_id="NEW-0",
            existing_id=["EXISTING-1", "EXISTING-2"],  # type: ignore[arg-type]
            refined_new_trigger="narrow new",
            refined_existing_trigger="narrow existing",
        )


def test_reject_new_superseded_id_rejects_non_numeric_label():
    """A label that is not a position integer still fails loudly."""
    with pytest.raises(ValidationError):
        RejectNewDecision(new_id="NEW-0", superseded_by_existing_id="EXISTING-foo")  # type: ignore[arg-type]


def test_reject_new_superseded_id_accepts_bare_int():
    """Sanity check: bare ints still pass."""
    r = RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=12345)
    assert r.superseded_by_existing_id == 12345


def test_differentiate_existing_id_accepts_bare_int():
    """Sanity check: bare ints still pass."""
    d = DifferentiateDecision(
        new_id="NEW-0",
        existing_id=12345,
        refined_new_trigger="narrow new",
        refined_existing_trigger="narrow existing",
    )
    assert d.existing_id == 12345


def test_existing_position_rejects_garbage():
    """Garbage strings that aren't ``EXISTING-N`` or numeric still fail loudly."""
    with pytest.raises(ValidationError):
        UnifyDecision(
            new_id="NEW-0",
            archive_existing_ids=["not-a-real-id"],  # type: ignore[list-item]
            content="c",
            trigger="t",
            rationale="r",
        )


@pytest.mark.parametrize("bad", [-1, "-1", "EXISTING--1", "existing_-2"])
def test_existing_position_rejects_negative(bad):
    """List positions are always ``>= 0`` — negatives are rejected.

    A negative position (or any post-strip negative parse like ``EXISTING--1``)
    can only fail later in apply. Fail fast here so the validator behaves as
    a sharp domain check, not a passthrough that defers the error.
    """
    with pytest.raises(ValidationError):
        UnifyDecision(
            new_id="NEW-0",
            archive_existing_ids=[bad],  # type: ignore[list-item]
            content="c",
            trigger="t",
            rationale="r",
        )


def test_unify_coerced_position_resolves_via_existing_by_position():
    """Apply-path regression: a coerced ``"EXISTING-N"`` resolves to the
    expected row through ``existing_by_position`` (the lookup
    ``_apply_unify`` actually uses).

    The unit tests above prove decode-time coercion (``"EXISTING-3"`` → ``3``).
    This test closes the loop: index 3 in the position-keyed lookup must
    point at the third existing playbook, so the coercion is end-to-end
    correct for the one field where it's wired.
    """
    from reflexio.models.api_schema.domain.entities import UserPlaybook

    existing = [
        UserPlaybook(
            content=f"existing-{i}",
            trigger="t",
            user_playbook_id=1000 + i,
            agent_version="v0",
            request_id="r0",
        )
        for i in range(5)
    ]
    # This is the same construction _build_deduplicated_results uses.
    existing_by_position = {f"EXISTING-{idx}": pb for idx, pb in enumerate(existing)}

    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=["EXISTING-3"],  # type: ignore[list-item]
        content="c",
        trigger="t",
        rationale="r",
    )
    # Coerced to int position
    assert unify.archive_existing_ids == [3]
    # And that position resolves through the apply-path lookup
    resolved = existing_by_position.get(f"EXISTING-{unify.archive_existing_ids[0]}")
    assert resolved is not None
    assert resolved.user_playbook_id == 1003
    assert resolved.content == "existing-3"
