"""Information retention and reference invariants in bounded historical checks."""

from types import SimpleNamespace
from typing import cast

import pytest
import tiktoken

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
)
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.server.services.playbook.components.overflow_consolidation import (
    HistoricalQuote,
    HistoricalScan,
    _merge_quotes,
    _neighbor_bounds,
    _partition,
    _remap_proposal,
    _selected_rows,
)


def _old(identity):
    return UserPlaybook(
        user_playbook_id=identity, user_id="u", agent_version="v1", request_id="old"
    )


def test_historical_neighbor_context_preserves_condition_across_unicode_boundary():
    text = "工具 🙂 result " * 6000
    ranges = _partition(text)
    assert len(ranges) > 1
    boundary = ranges[0][1]
    left_start, left_end = _neighbor_bounds(text, *ranges[0])
    right_start, right_end = _neighbor_bounds(text, *ranges[1])
    assert left_start == 0 and right_start < boundary < left_end
    quote = HistoricalQuote(
        start=boundary - 5, end=boundary + 8, text=text[boundary - 5 : boundary + 8]
    )
    update = HistoricalScan(coverage="complete", quotes=[quote])
    assert _merge_quotes([], update, text, left_start, left_end) == [quote]
    assert _merge_quotes([], update, text, right_start, right_end) == [quote]
    encoding = tiktoken.get_encoding("cl100k_base")
    assert (
        len(encoding.encode(text[right_start:boundary], disallowed_special=())) <= 256
    )
    assert len(encoding.encode(text[boundary:left_end], disallowed_special=())) <= 256


@pytest.mark.parametrize("failure", ["invented", "outside", "undetermined", "capacity"])
def test_historical_carriers_never_invent_drop_or_accept_incomplete_evidence(failure):
    text = "Separate approval is needed to deploy. " + "evidence " * 5000
    prior = [HistoricalQuote(start=0, end=37, text=text[:37])]
    if failure == "invented":
        update = HistoricalScan(
            coverage="complete", quotes=[HistoricalQuote(start=0, end=6, text="Deploy")]
        )
        expected = ValueError
    elif failure == "outside":
        update = HistoricalScan(
            coverage="complete",
            quotes=[HistoricalQuote(start=35, end=45, text=text[35:45])],
        )
        expected = ValueError
    elif failure == "undetermined":
        update = HistoricalScan(coverage="undetermined")
        expected = InputBudgetConfigurationError
    else:
        update = HistoricalScan(
            coverage="complete",
            quotes=[HistoricalQuote(start=38, end=len(text), text=text[38:])],
        )
        expected = InputBudgetExceededError
    with pytest.raises(expected):
        _merge_quotes(prior, update, text, 0, 38 if failure == "outside" else len(text))
    assert len(prior) == 1 and prior[0].text == text[:37]


def test_historical_positions_stay_stable_when_database_ids_overlap_positions():
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    all_old = [_old(900), _old(1), _old(500)]
    subset = [all_old[2], all_old[0]]
    output = PlaybookConsolidationOutput(
        decisions=[RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=1)]
    )
    mapped = _remap_proposal(owner, output, subset, all_old, 7)
    assert isinstance(mapped.decisions[0], RejectNewDecision)
    assert mapped.decisions[0].new_ids == ["NEW-7"]
    assert mapped.decisions[0].superseded_by_existing_ids == [0]
    assert _selected_rows(owner, mapped, all_old) == [all_old[0]]


@pytest.mark.parametrize("conflict", ["two_survivors", "reject_into_archive"])
def test_one_historical_rule_cannot_be_archived_into_conflicting_survivors(conflict):
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    first = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[0],
        content="rule",
        trigger="scope",
        rationale="reason",
    )
    second = (
        UnifyDecision(
            new_id="NEW-1",
            archive_existing_ids=[0],
            content="other",
            trigger="scope",
            rationale="reason",
        )
        if conflict == "two_survivors"
        else RejectNewDecision(new_id="NEW-1", superseded_by_existing_id=0)
    )
    output = PlaybookConsolidationOutput(decisions=[first, second])
    assert _selected_rows(owner, output, [_old(51)], enforce_conflicts=False) == [
        _old(51)
    ]
    with pytest.raises(InputBudgetConfigurationError):
        _selected_rows(owner, output, [_old(51)])


def test_candidate_shortlists_reconcile_shared_history_using_original_positions(
    monkeypatch,
):
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    old = [_old(100 + index) for index in range(40)]
    new = [
        row.model_copy(update={"user_playbook_id": 0, "content": f"candidate-{index}"})
        for index, row in enumerate(old[:2])
    ]
    coordinator = SimpleNamespace(
        playbook_key=lambda row: row.content,
        candidate_historical_matches={
            "candidate-0": (101, 106, 110, 115, 117),
            "candidate-1": (103, 106, 113, 118, 120),
        },
    )
    seen = []

    def propose(owner, coordinator, candidates, existing):
        seen.append([row.user_playbook_id for row in existing])
        if len(candidates) == 1:
            assert len(existing) == 5
            position = next(
                index
                for index, row in enumerate(existing)
                if row.user_playbook_id == 106
            )
            ids = "NEW-0"
        else:
            assert [row.user_playbook_id for row in existing] == [106]
            position, ids = 0, ["NEW-0", "NEW-1"]
        return PlaybookConsolidationOutput(
            decisions=[
                UnifyDecision(
                    new_id=ids,
                    archive_existing_ids=[position],
                    content="combined constraints",
                    trigger="same task",
                    rationale="preserve both lessons",
                )
            ]
        )

    monkeypatch.setattr(module, "_cached_proposal", propose)
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    output = module._bounded_proposals(
        owner, cast(OverflowCoordinator, coordinator), new, old
    )
    assert len(seen) == 3
    assert isinstance(output.decisions[0], UnifyDecision)
    assert output.decisions[0].new_ids == ["NEW-0", "NEW-1"]
    assert output.decisions[0].archive_existing_ids == [6]
    assert _selected_rows(owner, output, old) == [old[6]]


def test_identical_historical_source_geometry_does_not_count_twice_or_merge_provenance():
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
    )
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        _admit_history,
    )

    checkpoint = {}
    prepared = PreparedExtractionInput(
        "window", tuple(PreparedChunk(index, ()) for index in range(41))
    )
    coordinator = SimpleNamespace(
        checkpoint=checkpoint, require_prepared=lambda: prepared
    )
    coordinator.save_checkpoint = lambda value: checkpoint.update(value)
    first = _old(51).model_copy(
        update={
            "source_span": "Original evidence. " * 12000,
            "source_interaction_ids": [1],
        }
    )
    second = first.model_copy(
        update={
            "user_playbook_id": 52,
            "content": "Different historical rule",
            "source_interaction_ids": [900],
        }
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    plans = _admit_history(cast(OverflowCoordinator, coordinator), [first, second])
    assert len(plans) == len(checkpoint["historical_chunks"]) == 1
    assert 41 + sum(map(len, plans.values())) <= 48
    assert first.source_interaction_ids == [1] and second.source_interaction_ids == [
        900
    ]
    assert first.user_playbook_id == 51 and second.user_playbook_id == 52
