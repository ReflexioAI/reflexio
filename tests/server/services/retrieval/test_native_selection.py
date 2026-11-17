"""Provider-free contracts for the transferred J1 decision, not relevance grades."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from operator import setitem
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from reflexio.server.services.retrieval.judge_payload import MODEL, payload
from reflexio.server.services.retrieval.native_selection import (
    KINDS,
    NativePlan,
    select,
)

pytestmark = pytest.mark.unit


def inputs(count: int = 6) -> tuple[dict[str, Any], dict[str, Any]]:
    pools = {
        k: [
            {"id": i + 1, "text" if k == "profiles" else "trigger": f"Evidence{i}"}
            for i in range(count)
        ]
        for k in KINDS
    }
    return pools, {k: list(range(1, min(count, 5) + 1)) for k in KINDS}


def reply(
    pools: dict[str, Any], probs: list[float] | None = None, clash: float = 0.1
) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for number, _ in enumerate(item for k in KINDS for item in pools[k]):
        answers[f"item_{number}_usefulness"] = {
            "type": "score",
            "probabilities": dict(
                zip("0123", probs or [0.3, 0.24, 0.46, 0], strict=True)
            ),
        }
        answers[f"item_{number}_conflict"] = {"type": "noul", "noul": clash}
    return {"model": MODEL, "answers": answers}


def test_native_off_immutable_snapshot_no_payload_or_key_requirement(monkeypatch):
    pools, anchors = inputs()
    plan = NativePlan("Original task", pools=pools, anchors=anchors)
    pools["profiles"][0]["text"] = "callerbuffer changed"
    anchors["profiles"].clear()
    for key_present in (False, True):
        if key_present:
            monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-does-not-enable")
        else:
            monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with patch(
            "reflexio.server.services.retrieval.judge_payload.payload",
            side_effect=AssertionError,
        ):
            selection = select(plan)
        assert selection.mode == "native" and selection.ids["profiles"] == (
            1,
            2,
            3,
            4,
            5,
        )
    assert plan.projection()["profiles"][0]["text"] == "Evidence0"
    with pytest.raises(TypeError):
        setitem(cast(Any, plan.anchors), "profiles", ())
    with pytest.raises(FrozenInstanceError):
        cast(Any, plan).request = "changed"


def test_empty_enabled_path_has_zero_calls():
    pools, anchors = inputs(0)
    judge = Mock(side_effect=AssertionError("emptydispatch"))
    selection = select(NativePlan("Task", pools=pools, anchors=anchors), judge=judge)
    assert selection.mode == "native_empty" and not any(selection.ids.values())
    judge.assert_not_called()


def test_exact_one_payload_call_preserves_all_inputs_and_native_anchor_order():
    pools, anchors = inputs()
    for kind in KINDS:
        anchors[kind] = [5, 1, 4, 2, 3]
    response = reply(pools)
    saved = deepcopy((pools, anchors, response))
    judge = Mock(return_value=response)
    selection = select(
        NativePlan("Original task", pools=pools, anchors=anchors), judge=judge
    )
    assert selection.mode == "enhanced"
    assert selection.ids == {k: tuple(anchors[k]) for k in KINDS}
    judge.assert_called_once_with(payload("Original task", pools))
    assert (pools, anchors, response) == saved
    with pytest.raises(TypeError):
        setitem(cast(Any, selection.ids), "profiles", ())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "probabilities,clash,kept",
    [
        ([0, 0.5, 0.5, 0], 0.4999, True),
        ([0, 0.5001, 0.4999, 0], 0.1, False),
        ([0.5, 0, 0.3, 0.2], 0.1, True),
        ([0, 0.5, 0.5, 0], 0.5, False),
    ],
)
def test_positive_p50_gate_and_conflict_strictness(kind, probabilities, clash, kept):
    pools, anchors = inputs(0)
    pools[kind] = [{"id": 1, "text" if kind == "profiles" else "trigger": "Evidence"}]
    response = reply(pools, probabilities, clash)
    selection = select(
        NativePlan("Task", pools=pools, anchors=anchors), judge=lambda _: response
    )
    assert selection.ids[kind] == ((1,) if kept else ())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "probabilities,clash,pruned",
    [
        ([0.46, 0.49, 0.05, 0], 0.1, True),
        ([0.46, 0.4899, 0.0501, 0], 0.1, False),
        ([0.3, 0.24, 0.46, 0], 0.9, True),
        ([0.3, 0.24, 0.46, 0], 0.8999, False),
    ],
)
def test_negative_p95_or_conflict_p90_prunes_only_at_frozen_threshold(
    kind, probabilities, clash, pruned
):
    pools, anchors = inputs(0)
    pools[kind] = [{"id": 1, "text" if kind == "profiles" else "trigger": "Evidence"}]
    anchors[kind] = [1]
    response = reply(pools, probabilities, clash)
    selection = select(
        NativePlan("Task", pools=pools, anchors=anchors), judge=lambda _: response
    )
    assert selection.ids[kind] == (() if pruned else (1,))


@pytest.mark.parametrize(
    "novel_probs,expected",
    [([0, 0, 1, 0], (5, 1, 4, 2, 3)), ([0, 0, 0.99999, 0.00001], (5, 1, 4, 2, 6))],
)
def test_equal_novel_gain_preserves_native_anchors_higher_gain_replaces_last_weakest(
    novel_probs, expected
):
    pools, anchors = inputs()
    for kind in KINDS:
        anchors[kind] = [5, 1, 4, 2, 3]
    response = reply(pools, [0, 0, 1, 0])
    for i in (5, 11, 17):
        response["answers"][f"item_{i}_usefulness"]["probabilities"] = dict(
            zip("0123", novel_probs, strict=True)
        )
    selection = select(
        NativePlan("Task", pools=pools, anchors=anchors), judge=lambda _: response
    )
    assert all(ids == expected for ids in selection.ids.values())


def test_strong_negative_prune_then_stable_refill_obeys_per_type_cap():
    pools, anchors = inputs(8)
    response = reply(pools, [0, 0, 0, 1])
    for i in (0, 8, 16):
        response["answers"][f"item_{i}_usefulness"]["probabilities"] = dict(
            zip("0123", [1, 0, 0, 0], strict=True)
        )
    selection = select(
        NativePlan("Task", pools=pools, anchors=anchors), judge=lambda _: response
    )
    assert all(ids == (2, 3, 4, 5, 6) for ids in selection.ids.values())


def test_all_three_type_projections_preserve_typed_identities():
    pools, anchors = inputs(0)
    for kind, identity in zip(KINDS, ["profile-a", 7, "user-a"], strict=True):
        pools[kind] = [
            {"id": identity, "text" if kind == "profiles" else "trigger": "Evidence"}
        ]
        anchors[kind] = [identity]
    plan = NativePlan("Task", pools=pools, anchors=anchors)
    assert plan.projection() == pools
    assert select(plan).ids == {k: tuple(v) for k, v in anchors.items()}


@pytest.mark.parametrize("bad", [[1, 1], [99], [True], [1, 2, 3, 4, 5, 6]])
def test_invalid_anchor_universe_rejected_before_judge(bad):
    pools, anchors = inputs()
    anchors["profiles"] = bad
    with pytest.raises(ValueError):
        NativePlan("Task", pools=pools, anchors=anchors)


@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
def test_enabled_failure_propagates_without_retry_or_native_fallback(failure):
    pools, anchors = inputs()
    judge = Mock(side_effect=failure("synthetic unavailable"))
    with pytest.raises(failure):
        select(NativePlan("Task", pools=pools, anchors=anchors), judge=judge)
    judge.assert_called_once()


@pytest.mark.parametrize(
    "change",
    ["model", "missing", "extra", "nan", "inf", "mass", "bool", "type", "grades"],
)
def test_complete_reply_validation_precedes_any_pruning_or_partial_result(change):
    pools, anchors = inputs()
    response = reply(pools, [1, 0, 0, 0])
    answer = response["answers"]["item_17_conflict"]
    if change == "model":
        response["model"] = "unapproved"
    elif change == "missing":
        del response["answers"]["item_17_conflict"]
    elif change == "extra":
        response["answers"]["unasked"] = {}
    elif change in ("nan", "inf"):
        answer["noul"] = float(change)
    elif change == "mass":
        response["answers"]["item_17_usefulness"]["probabilities"]["1"] = 0.03
    elif change == "bool":
        answer["noul"] = True
    elif change == "type":
        answer["type"] = "score"
    else:
        response["answers"]["item_17_usefulness"]["probabilities"].pop("3")
    plan = NativePlan("Task", pools=pools, anchors=anchors)
    with pytest.raises(ValueError):
        select(plan, judge=lambda _: response)
    assert plan.anchors == {k: tuple(v) for k, v in anchors.items()}


@pytest.mark.parametrize("mutation", ["model", "questions"])
def test_transport_cannot_redefine_expected_wire_identity(mutation):
    pools, anchors = inputs()
    response = reply(pools)

    def judge(wire):
        if mutation == "model":
            wire["model"] = response["model"] = "unapproved"
        else:
            wire["questions"]["unsolicited"] = response["answers"]["unsolicited"] = {}
        return response

    with pytest.raises(ValueError):
        select(NativePlan("Task", pools=pools, anchors=anchors), judge=judge)


@pytest.mark.parametrize(
    "probabilities,valid", [([0.52, 0, 0.5, 0], True), ([0.5201, 0, 0.5, 0], False)]
)
def test_rounded_probability_mass_tolerance_is_inclusive_without_normalizing(
    probabilities, valid
):
    pools, anchors = inputs(1)
    response = reply(pools, probabilities)
    plan = NativePlan("Task", pools=pools, anchors=anchors)
    if valid:
        assert all(
            ids == (1,) for ids in select(plan, judge=lambda _: response).ids.values()
        )
    else:
        with pytest.raises(ValueError):
            select(plan, judge=lambda _: response)


@pytest.mark.parametrize("response", [None, [], {"model": MODEL, "answers": []}])
def test_nonobject_or_wrong_answer_container_refuses(response):
    pools, anchors = inputs(1)
    with pytest.raises(ValueError):
        select(
            NativePlan("Task", pools=pools, anchors=anchors), judge=lambda _: response
        )
