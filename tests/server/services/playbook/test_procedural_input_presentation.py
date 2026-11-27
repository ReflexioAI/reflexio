"""Lossless late wire views preserve source, full-result review and journal inputs."""

import json
from copy import deepcopy

import pytest
import yaml

from reflexio.server.services.playbook.procedural import review
from reflexio.server.services.playbook.procedural.models import (
    ContractError,
    Create,
    Revise,
)
from reflexio.server.services.playbook.procedural.transport import (
    _input_content,
    _single_call_schema,
)


def payload(spans, replacements=None, suffix="\n\nNew source witness."):
    replacements = (
        list(range(1, len(spans) + 1)) if replacements is None else replacements
    )
    record = {
        "content": "Complete supported instruction.",
        "trigger": "When preparing a release.",
        "source_span": "\n\n".join(dict.fromkeys(span for span in spans if span))
        + suffix,
        "source_interaction_ids": [1, 2],
        "replaces": replacements,
    }
    return review.prepare(
        {
            "source": {
                "existing": {
                    f"P{i}": {"user_playbook_id": i, "source_span": span}
                    for i, span in enumerate(spans, 1)
                }
            },
            "changes": [{"operation": "upsert", "group": "g"}],
            "proposed_records": {"W0": record},
            "keep_existing": [],
            "retired_ids": [],
        }
    )


@pytest.mark.parametrize(
    "text",
    [
        "a\x85b",
        "a\nb\n\n",
        "a\r\nb\r",
        "a\u2028b\u2029c",
        "a\n\x85b\u2028c\r\n",
        "\x00\x01\x1f\x7f\x9f",
        "yes: no\non: 2026-11-27\nnull: 12",
        '你好 👋 "quotes" \\n',
        "---\n&anchor *alias !tag {'verbatim_concatenation': []}\n",
        "\n",
        "trailing\n",
        "trailing\n\n",
        "leading spaces\n  exact  ",
    ],
)
def test_nonreader_wire_preserves_every_source_codepoint(text):
    body = {"source_span": text, "yes": "yes", "numeric": "012", "none": None}
    original = deepcopy(body)
    assert yaml.safe_load(_input_content("maintain", body)) == body == original
    assert _input_content("source", body) == json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize(
    "spans", [["old"], ["old", "other"], ["same", "same"], ["", "old"]]
)
@pytest.mark.parametrize("suffix", ["", "\n\n", "\n\nNew\r\n\x85source\n"])
def test_review_view_expands_to_the_entire_original_body(spans, suffix):
    original = payload(spans, suffix=suffix)
    snapshot = deepcopy(original)
    view = review.input_view(original)
    assert original == snapshot
    assert review._expand_provenance(view) == original
    assert view["source"]["existing"] == original["source"]["existing"]
    projected = view["proposed_records"]["W0"]
    assert projected is view["result_review_units"][0]["resulting_record"]
    assert isinstance(projected["source_span"], dict)
    assert projected["content"] == original["proposed_records"]["W0"]["content"]
    assert (
        review._expand_provenance(yaml.safe_load(_input_content("review", original)))
        == original
    )


@pytest.mark.parametrize(
    "replacements,span",
    [
        ([], "independent ADD"),
        ([99], "old\n\nnew"),
        ([1, 1], "old\n\nnew"),
        ([1], "unrelated"),
        ([1], "older prefix is different"),
    ],
)
def test_unproven_provenance_remains_raw(replacements, span):
    original = payload(["old"], replacements=replacements)
    original["proposed_records"]["W0"]["source_span"] = span
    assert review.input_view(original) == original


def test_review_view_checks_distinct_result_copies_and_rejects_foreign_references():
    original = payload(["old"])
    original["result_review_units"][0]["resulting_record"] = deepcopy(
        original["proposed_records"]["W0"]
    )
    view = review.input_view(original)
    assert review._expand_provenance(view) == original
    view["proposed_records"]["W0"]["source_span"]["verbatim_concatenation"][0][
        "incumbent_source_span"
    ] = "https://untrusted.invalid"
    with pytest.raises(ContractError, match="exact incumbent"):
        review._expand_provenance(view)


def test_batch_schema_shares_exact_witness_enum_and_still_rejects_unknown_witness():
    choices = [f"EXCERPT_1_1_{i}" for i in range(85)]
    schema = _single_call_schema(
        {"create_procedure": Create, "revise_procedure": Revise}, choices
    )
    encoded = json.dumps(schema.model_json_schema())
    assert encoded.count(json.dumps(choices)) == 1
    assert encoded.count('"$ref": "#/$defs/OriginalSourceWitness"') == 2
    with pytest.raises(ValueError):
        schema.model_validate(
            {
                "calls": [
                    {
                        "id": "one",
                        "name": "create_procedure",
                        "arguments": {
                            "group": "g",
                            "name": "release",
                            "trigger": "Release",
                            "entry": "Do the supported action.",
                            "evidence": "EXCERPT_unknown",
                        },
                    }
                ]
            }
        )


def test_encoder_mismatch_refuses_before_supplied_evidence_can_be_dispatched(
    monkeypatch,
):
    from reflexio.server.services.playbook.procedural import transport

    monkeypatch.setattr(
        transport.yaml, "dump", lambda *_args, **_kwargs: "source_span: changed\n"
    )
    with pytest.raises(ContractError, match="encoding changed"):
        _input_content("maintain", {"source_span": "Original exact source."})
