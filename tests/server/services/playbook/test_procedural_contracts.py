"""Native transfer regressions for selected V8 source/bundle/group contracts."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.procedural import codec
from reflexio.server.services.playbook.procedural.catalog import catalog
from reflexio.server.services.playbook.procedural.compilation import (
    compile_changes,
    proposals,
    witnesses,
)
from reflexio.server.services.playbook.procedural.models import (
    Citation,
    ContractError,
    Create,
    Document,
    Input,
    Maintenance,
    Page,
    Quotation,
    RemoveResource,
    Resource,
    ResultVerdict,
    Retire,
    Revise,
    Source,
)
from reflexio.server.services.playbook.procedural.review import (
    accepted_groups,
    prepare,
    reduce,
)


def _document(*pages: Page) -> Document:
    return Document(
        name="deploy-service",
        pages=list(pages) or [Page(path="SKILL.md", content="Deploy after approval.")],
    )


def _incumbent(identity: int = 31, *, content: str | None = None) -> dict:
    return {
        "user_playbook_id": identity,
        "content": content or codec.encode(_document()),
        "trigger": "Deploying a service",
        "source_interaction_ids": [1],
        "source_span": "Ask for approval before deployment.",
    }


def _quotes() -> dict[str, Quotation]:
    return {
        "EXCERPT_1_1_1": Quotation(
            interaction_id=2, text="Use staging before production."
        ),
        "EXCERPT_2_1_1": Quotation(interaction_id=3, text="print('ready')"),
    }


def _create(group: str = "A", name: str = "deploy-service") -> Create:
    return Create(
        group=group,
        name=name,
        trigger="Preparing a release",
        entry="Use staging before production.",
        evidence="EXCERPT_1_1_1",
    )


def _verdict(unit: str, **updates: str) -> ResultVerdict:
    value = {
        "unit": unit,
        "decision": "accept",
        "result_support": "pass",
        "scope_conditions": "pass",
        "useful_nonduplicate": "pass",
        "preservation_and_correction": "pass",
        "resource_consistency": "pass",
        "reason": "Original source supports this complete result.",
    }
    return ResultVerdict.model_validate({**value, **updates})


def test_bundle_canonical_bytes_preserve_delimiter_like_text_and_unicode() -> None:
    text = "Release café.\n## scripts/fake.py\n\nThis is still entry text."
    doc = _document(
        Page(path="SKILL.md", content=text),
        Page(path="assets/template.txt", content="✓"),
    )
    encoded = codec.encode(doc)
    expected_header = {
        "name": "deploy-service",
        "pages": [
            {"path": "SKILL.md", "chars": len(text)},
            {"path": "assets/template.txt", "chars": 1},
        ],
    }
    assert (
        encoded
        == codec.PREFIX
        + json.dumps(expected_header, separators=(",", ":"))
        + " -->\n\n## SKILL.md\n\n"
        + text
        + "\n## assets/template.txt\n\n✓"
    )
    assert codec.decode(encoded) == doc
    assert codec.decode("Ordinary existing playbook") is None
    with pytest.raises(ContractError, match="Invalid persisted"):
        codec.decode(encoded + "unframed tail")


@pytest.mark.parametrize(
    "text",
    [
        "Consult EXCERPT_1_1_1",
        "An embedded\x00NUL",
        "[missing](references/missing.md)",
        "[unsafe](file:///tmp/private)",
        "[missing][label]",
    ],
)
def test_bundle_rejects_invalid_saved_resources(text: str) -> None:
    with pytest.raises(ContractError):
        codec.encode(_document(Page(path="SKILL.md", content=text)))


def test_bundle_requires_unique_safe_paths_and_resolving_anchors() -> None:
    with pytest.raises(ValidationError):
        Page(path="../outside", content="bad")
    with pytest.raises(ContractError, match="unique resource"):
        codec.encode(
            _document(
                Page(path="SKILL.md", content="a"), Page(path="SKILL.md", content="b")
            )
        )
    doc = _document(
        Page(path="SKILL.md", content="[Checklist](references/check.md#verify)"),
        Page(path="references/check.md", content="# Verify\nCheck staging."),
    )
    assert codec.decode(codec.encode(doc)) == doc


def test_catalog_addresses_visible_pieces_without_joining_omitted_middle() -> None:
    head, tail = "a" * 601, "last visible piece"
    data = Input(
        owner="owner",
        sources=[
            Source(
                interaction_id=2,
                request_id="r",
                owner="owner",
                role="User",
                request_source="app",
                text=head + "\n\n" + tail,
            )
        ],
        existing=[],
        agent_context="Deployment agent",
        definition="Learn recurring deployment procedures",
        tool_context="",
    )
    body, quotes, segments = catalog(data, {2: [head, tail]})
    assert list(quotes) == ["EXCERPT_1_1_1", "EXCERPT_1_1_2", "EXCERPT_1_2_1"]
    assert [q.text for q in quotes.values()] == ["a" * 600, "a", tail]
    assert segments == {"SEGMENT_1_1": (2, head), "SEGMENT_1_2": (2, tail)}
    assert "text" not in body["sources"][0]
    with pytest.raises(ContractError, match="Visible pieces differ"):
        catalog(data, {2: [head + tail]})
    with pytest.raises(ValidationError, match="one owner"):
        data.model_copy(update={"owner": "other"}).__class__.model_validate(
            {**data.model_dump(), "owner": "other"}
        )


def test_flat_revision_inherits_resources_and_keeps_untouched_incumbents() -> None:
    original = _incumbent(
        content=codec.encode(
            _document(
                Page(path="SKILL.md", content="Deploy after approval."),
                Page(path="references/check.md", content="Check logs."),
            )
        )
    )
    incumbents = {
        "P1": original,
        "P2": _incumbent(32, content="Unrelated useful instruction"),
    }
    frozen = deepcopy(incumbents)
    revision = Revise(**_create().model_dump(), target="P1")
    value = proposals([("revise_procedure", revision)], incumbents, _quotes())
    result = compile_changes(
        value, incumbents, _quotes(), {2: ["Use staging before production."]}
    )
    record = result.records["W0"]
    decoded = codec.decode(record["content"])
    assert decoded is not None
    assert decoded.pages[1].content == "Check logs."
    assert record["replaces"] == [31]
    assert record["source_interaction_ids"] == [1, 2]
    assert (
        record["source_span"]
        == "Ask for approval before deployment.\n\nUse staging before production."
    )
    assert result.keep_existing == [32]
    assert result.retired_ids == []
    assert incumbents == frozen


def test_flat_complete_resource_edit_citation_and_exact_script_copy() -> None:
    quote = "EXCERPT_2_1_1"
    items = [
        ("create_procedure", _create()),
        (
            "attach_resource",
            Resource(
                group="A",
                document_name="deploy-service",
                path="scripts/check.py",
                content="print('ready')",
                evidence=quote,
            ),
        ),
        ("cite_evidence", Citation(group="A", evidence=quote)),
    ]
    value = proposals(items, {}, _quotes())
    result = compile_changes(
        value,
        {},
        _quotes(),
        {2: ["Use staging before production."], 3: ["print('ready')"]},
    )
    assert value.changes[0].support == ["EXCERPT_1_1_1", quote]
    decoded = codec.decode(result.records["W0"]["content"])
    assert decoded is not None
    assert decoded.pages[1].content == "print('ready')"
    with pytest.raises(ContractError, match="New script bytes"):
        compile_changes(
            value,
            {},
            _quotes(),
            {2: ["Use staging before production."], 3: ["a different program"]},
        )


def test_resources_require_same_complete_entry_and_allow_explicit_removal() -> None:
    existing = {
        "P1": _incumbent(
            content=codec.encode(
                _document(
                    Page(path="SKILL.md", content="Deploy."),
                    Page(path="assets/old.txt", content="Old."),
                )
            )
        )
    }
    revision = Revise(**_create().model_dump(), target="P1")
    removal = RemoveResource(
        group="A",
        document_name="deploy-service",
        path="assets/old.txt",
        evidence="EXCERPT_1_1_1",
    )
    value = proposals(
        [("revise_procedure", revision), ("remove_resource", removal)],
        existing,
        _quotes(),
    )
    document = value.changes[0].document
    assert document is not None
    assert [page.path for page in document.pages] == ["SKILL.md"]
    with pytest.raises(ContractError, match="complete entry"):
        proposals([("remove_resource", removal)], existing, _quotes())
    with pytest.raises(ContractError, match="only once"):
        proposals(
            [
                ("revise_procedure", revision),
                ("remove_resource", removal),
                ("remove_resource", removal),
            ],
            existing,
            _quotes(),
        )


def test_retirement_only_plan_is_not_an_empty_noop() -> None:
    incumbents = {
        "P1": _incumbent(),
        "P2": _incumbent(32, content="Keep this valid rule"),
    }
    retire = Retire(group="obsolete", target="P1", evidence="EXCERPT_1_1_1")
    value = proposals([("retire_procedure", retire)], incumbents, _quotes())
    result = compile_changes(value, incumbents, _quotes(), {})
    assert result.records == {} and result.selected == []
    assert result.retired_ids == [31] and result.keep_existing == [32]
    noop = compile_changes(Maintenance(changes=[]), incumbents, _quotes(), {})
    assert noop.retired_ids == [] and noop.keep_existing == [31, 32]


def test_complete_batch_rejects_double_consumption_or_duplicate_names() -> None:
    revision = Revise(**_create().model_dump(), target="P1")
    existing = {"P1": _incumbent()}
    retire = Retire(group="B", target="P1", evidence="EXCERPT_1_1_1")
    value = proposals(
        [("revise_procedure", revision), ("retire_procedure", retire)],
        existing,
        _quotes(),
    )
    with pytest.raises(ContractError, match="consumed only once"):
        compile_changes(value, existing, _quotes(), {})
    with pytest.raises(ContractError, match="one complete entry"):
        proposals(
            [("create_procedure", _create()), ("create_procedure", _create())],
            {},
            _quotes(),
        )
    with pytest.raises(ContractError, match="duplicate procedure"):
        compile_changes(
            proposals([("create_procedure", _create())], existing, _quotes()),
            existing,
            _quotes(),
            {},
        )


@pytest.mark.parametrize("bad", ["", "EXCERPT_1_1_1 EXCERPT_1_1_1", "EXCERPT_missing"])
def test_unavailable_or_duplicate_witness_ids_are_rejected(bad: str) -> None:
    with pytest.raises(ContractError, match="unique exact"):
        witnesses(bad, _quotes())


def _review_payload() -> dict:
    return {
        "changes": [
            {"group": "coupled", "operation": "upsert"},
            {"group": "coupled", "operation": "retire"},
            {"group": "independent", "operation": "upsert"},
        ],
        "proposed_records": {
            "W0": {"content": "Complete revised rule"},
            "W2": {"content": "Independent rule"},
        },
    }


def test_full_result_review_preserves_coupling_and_rejects_unknown_group() -> None:
    payload = _review_payload()
    prepared = prepare(payload)
    assert prepared["result_review_units"][0]["resulting_record"] == {
        "content": "Complete revised rule"
    }
    assert prepared["result_review_units"][0]["coupled_units"] == ["U0", "U1"]
    assert "result_review_units" not in payload
    review = reduce(
        payload,
        [_verdict("U2"), _verdict("U1", scope_conditions="unknown"), _verdict("U0")],
    )
    assert accepted_groups(review) == {"independent"}
    assert review.groups[0].decision == "unknown"
    failed = reduce(
        payload, [_verdict("U0"), _verdict("U1", result_support="fail"), _verdict("U2")]
    )
    assert failed.groups[0].decision == "reject" and accepted_groups(failed) == {
        "independent"
    }


@pytest.mark.parametrize(
    "units", [["U0", "U2"], ["U0", "U0", "U2"], ["U0", "U1", "U9"]]
)
def test_full_result_reviewer_must_account_every_unit_once(units: list[str]) -> None:
    with pytest.raises(ContractError, match="exactly once"):
        reduce(_review_payload(), [_verdict(unit) for unit in units])


def test_full_result_review_requires_actual_result_for_upsert() -> None:
    payload = _review_payload()
    payload["proposed_records"].pop("W2")
    with pytest.raises(ContractError, match="complete compiled result"):
        prepare(payload)
    with pytest.raises(ContractError, match="Reserved"):
        prepare({**_review_payload(), "result_review_units": []})


def test_existing_read_view_preserves_all_resources_after_plain_body_edit() -> None:
    original = codec.encode(
        _document(
            Page(path="SKILL.md", content="Deploy after approval."),
            Page(path="references/check.md", content="# Verify\nCheck staging."),
            Page(path="scripts/check.py", content="print('ready')"),
        )
    )
    edited = original.replace(
        "Deploy after approval.", "Deploy only after explicit approval."
    )
    frozen = edited
    with pytest.raises(ContractError, match="Invalid persisted"):
        codec.decode(edited)
    document = codec.read_existing(edited)
    assert document is not None
    assert document.pages == [
        Page(path="SKILL.md", content="Deploy only after explicit approval."),
        Page(path="references/check.md", content="# Verify\nCheck staging."),
        Page(path="scripts/check.py", content="print('ready')"),
    ]
    assert edited == frozen
    assert codec.decode(codec.encode(document)) == document
    assert codec.encode(document) != edited


def test_valid_existing_framing_keeps_literal_markers_but_stale_ambiguous_input_fails() -> (
    None
):
    original = codec.encode(
        _document(
            Page(
                path="SKILL.md",
                content="Example: \n## assets/check.txt\n\nThis is a literal example.",
            ),
            Page(path="assets/check.txt", content="A real resource."),
        )
    )
    assert codec.read_existing(original) == codec.decode(original)
    stale = original.replace("literal example.", "longer literal example.")
    with pytest.raises(ContractError, match="cannot be read unambiguously"):
        codec.read_existing(stale)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "reordered",
        "unsafe",
        "extra_manifest",
        "invalid_count",
        "leading_bytes",
        "broken_reference",
    ],
)
def test_existing_read_view_never_guesses_a_malformed_resource_partition(
    change: str,
) -> None:
    original = codec.encode(
        _document(
            Page(path="SKILL.md", content="Deploy after approval."),
            Page(path="references/check.md", content="# Verify\nCheck staging."),
        )
    )
    edited = original.replace(
        "Deploy after approval.", "Deploy only after explicit approval."
    )
    if change == "missing":
        edited = edited.replace(
            "\n## references/check.md\n\n", "\nMissing section marker\n"
        )
    elif change == "reordered":
        first, last = "\n## SKILL.md\n\n", "\n## references/check.md\n\n"
        edited = (
            edited.replace(first, "TEMP").replace(last, first).replace("TEMP", last)
        )
    elif change == "unsafe":
        edited = edited.replace("references/check.md", "../outside.md")
    elif change == "extra_manifest":
        edited = edited.replace('{"name":', '{"unexpected":true,"name":', 1)
    elif change == "invalid_count":
        edited = edited.replace('"chars":22', '"chars":true', 1)
    elif change == "leading_bytes":
        edited = edited.replace(" -->\n", " -->\nUnframed bytes", 1)
    else:
        edited = edited.replace(
            "explicit approval.", "explicit approval. [Missing](references/missing.md)"
        )
    with pytest.raises(ContractError, match="cannot be read unambiguously"):
        codec.read_existing(edited)


def test_native_tuner_exact_clause_edit_remains_consumable_without_rewriting_its_bytes() -> (
    None
):
    from reflexio.server.services.playbook.rule_evidence import models as rule_models
    from reflexio.server.services.playbook.rule_evidence import wire

    original = codec.encode(
        _document(
            Page(path="SKILL.md", content="Deploy after approval."),
            Page(path="assets/template.txt", content="Keep this template unchanged."),
        )
    )
    data = rule_models.RuleEvidenceInput.model_validate(
        {
            "incumbent": {
                "id": 31,
                "user_id": "owner",
                "version": "frozen",
                "trigger": "Deploying a service",
                "content": original,
            },
            "sessions": [
                {
                    "session_id": "source",
                    "user_id": "owner",
                    "exposed_version": "frozen",
                    "source_limitations": "",
                    "messages": [
                        {
                            "turn": 1,
                            "role": "User",
                            "content": "Use explicit approval before deployment.",
                        }
                    ],
                }
            ],
            "agent_context": "Deployment assistant",
            "success_definition": "Follow explicit deployment requirements",
            "available_tools": [],
            "tool_context_status": "missing",
        }
    )
    clause = next(
        item["id"]
        for item in wire.catalog(original)
        if item["text"].strip() == "Deploy after approval."
    )
    value = wire.tune_schema(data).model_validate(
        {
            "hypothesis": "Keep approval explicit.",
            "edits": [
                {
                    "clause_start": clause,
                    "clause_end": clause,
                    "replacement": "Deploy only after explicit approval.",
                    "defect": "Approval was underspecified.",
                    "evidence_refs": {"r1": "REF_1"},
                }
            ],
            "abstention_reason": "",
        }
    )
    result = wire.rebuild(data, [], value)
    assert result.revised_content is not None
    raw_tuner_result = result.revised_content
    assert raw_tuner_result == original.replace(
        "Deploy after approval.", "Deploy only after explicit approval."
    )
    document = codec.read_existing(raw_tuner_result)
    assert document is not None
    assert document.pages[1].content == "Keep this template unchanged."
    incumbents = {"P1": _incumbent(content=raw_tuner_result)}
    produced = proposals(
        [("revise_procedure", Revise(**_create().model_dump(), target="P1"))],
        incumbents,
        _quotes(),
    )
    compiled = compile_changes(
        produced, incumbents, _quotes(), {2: ["Use staging before production."]}
    )
    canonical = codec.decode(compiled.records["W0"]["content"])
    assert canonical is not None
    assert canonical.pages[1].content == "Keep this template unchanged."
    assert (
        result.revised_content == raw_tuner_result
        and incumbents["P1"]["content"] == raw_tuner_result
    )
