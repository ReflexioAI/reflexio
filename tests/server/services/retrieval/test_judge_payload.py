"""Exact wire fixture and input-isolation contracts; no provider or Lab imports."""

import hashlib
import json
from copy import deepcopy

import pytest

from reflexio.server.services.retrieval.judge_payload import CRITERIA, MODEL, payload

pytestmark = pytest.mark.unit


def pools():
    return {
        "profiles": [{"id": "p1", "text": "Synthetic profile evidence"}],
        "agent_playbooks": [{"id": 7, "trigger": "Synthetic agent trigger"}],
        "user_playbooks": [{"id": 11, "trigger": "Synthetic user trigger"}],
    }


def test_frozen_all_type_wire_digest_dictionary_order_questions_and_strings():
    wire = payload("Synthetic request", pools())
    # Frozen from the accepted J1 payload over these synthetic inputs, including
    # all exact question/criterion strings, key order, model and field names.
    digest = hashlib.sha256(
        json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == "467adbaae5eb43cadb33ed3a91aed402c5335ad43af1e2f9fd33a9a4c399e3df"
    assert list(wire) == ["model", "state", "questions"]
    assert wire["model"] == MODEL == "jev-1.14.0"
    assert list(wire["questions"]) == [
        f"item_{i}_{kind}" for i in range(3) for kind in ("usefulness", "conflict")
    ]


def test_candidate_evidence_isolated_ids_never_sent_no_input_mutation():
    source = pools()
    saved = deepcopy(source)
    wire = payload("Synthetic request", source)
    for index, (kind, textfield) in enumerate(
        (("profile", "text"), ("agent", "trigger"), ("user", "trigger"))
    ):
        for category in ("usefulness", "conflict"):
            candidate = wire["questions"][f"item_{index}_{category}"]["instructions"][
                "candidate"
            ]
            assert set(candidate) == {"type", textfield} and candidate["type"] == kind
    assert wire["state"] == {"request": "Synthetic request"}
    assert source == saved
    wire["questions"]["item_0_usefulness"]["criteria"][0] = "caller mutation"
    assert wire["questions"]["item_1_usefulness"]["criteria"] == CRITERIA
    assert (
        payload("Synthetic request", source)["questions"]["item_0_usefulness"][
            "criteria"
        ]
        == CRITERIA
    )


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "bool", "empty", "extra", "wrong_field", "pool21", "missing_type"],
)
def test_strict_source_projection_rejection(invalid):
    source = pools()
    if invalid == "duplicate":
        source["user_playbooks"].append(dict(source["user_playbooks"][0]))
    elif invalid == "bool":
        source["agent_playbooks"][0]["id"] = True
    elif invalid == "empty":
        source["profiles"][0]["text"] = "   "
    elif invalid == "extra":
        source["user_playbooks"][0]["content"] = "Forbidden Content"
    elif invalid == "wrong_field":
        source["profiles"][0] = {"id": "p1", "trigger": "wrong"}
    elif invalid == "pool21":
        source["profiles"] = [{"id": str(i), "text": "E"} for i in range(21)]
    else:
        source.pop("profiles")
    with pytest.raises(ValueError):
        payload("Task", source)


def test_empty_wire_is_pure_and_does_not_read_configuration():
    wire = payload("Task", {k: [] for k in pools()})
    assert wire == {"model": MODEL, "state": {"request": "Task"}, "questions": {}}
