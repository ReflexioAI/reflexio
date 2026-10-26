"""Information retention and reference invariants in bounded historical checks."""

from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

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


def _history_fixture(text):
    from dataclasses import asdict
    from unittest.mock import MagicMock

    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    receipts = {}
    coordinator = SimpleNamespace(
        source_digest="fixture-source",
        completed_step=receipts.get,
        require_remaining_work=MagicMock(),
    )
    coordinator.step_key = lambda stage, inputs: OverflowCoordinator.step_key(
        cast(OverflowCoordinator, coordinator), stage, inputs
    )
    coordinator.complete_step = lambda key, output, provenance: receipts.update(
        {key: {"output": output, "model_provenance": asdict(provenance)}}
    )
    owner = SimpleNamespace(model_name="fixture-model", client=MagicMock())
    row = _old(51).model_copy(update={"source_span": text, "created_at": 1000})
    return (
        cast(PlaybookConsolidator, owner),
        cast(OverflowCoordinator, coordinator),
        row,
        receipts,
    )


def test_historical_selection_display_is_lossless_and_server_copies_unicode_json():
    import json

    from reflexio.server.services.playbook.components.overflow_consolidation import (
        HistoricalScanSelection,
        _historical_scan_request,
        _selection_scan,
        _validate_selection_scan,
    )

    text = (
        "Unshown prefix. " * 1000
        + 'Exact "quote" 🙂\nΩ' * 100
        + " Unshown tail." * 1000
    )
    owner, coordinator, row, _ = _history_fixture(text)
    start = len("Unshown prefix. " * 1000)
    _, messages, low, high, snippets = _historical_scan_request(
        owner, coordinator, row, [], [], start, start + 1700
    )
    payload = json.loads(messages[1]["content"])
    assert payload["author_role"] == "unknown"
    assert payload["chunk"]["primary_start"] == start
    assert low > 0 and high < len(text)
    assert (
        "".join(item["text"] for item in payload["chunk"]["snippets"]) == text[low:high]
    )
    assert all(len(quote.text) <= 1024 for quote in snippets.values())
    identity = next(key for key, quote in snippets.items() if "🙂" in quote.text)
    update = _selection_scan(
        HistoricalScanSelection(coverage="complete", snippet_ids=[identity]), snippets
    )
    quote = update.quotes[0]
    assert quote.text == text[quote.start : quote.end]
    assert '"quote"' in quote.text and "🙂" in quote.text
    unknown = HistoricalScanSelection(coverage="complete", snippet_ids=["h:0:10"])
    feedback = _validate_selection_scan(
        unknown, prior=[], snippets=snippets, text=text, start=low, end=high
    )
    assert feedback and "Unshown prefix" not in str(feedback)
    with pytest.raises(ValueError, match="unshown"):
        _selection_scan(unknown, snippets)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_historical_both_receipt_versions_are_reused_without_dispatch(version):
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        SCAN_POLICY,
        _historical_scan_request,
        _remaining_history_calls,
        _scan_history,
        _scan_request,
    )

    text = 'Keep exact "quoted" Unicode 🙂 evidence.'
    owner, coordinator, row, receipts = _history_fixture(text)
    v1_key, v1_messages, _, _ = _scan_request(
        owner, coordinator, row, [], [], 0, len(text)
    )
    assert v1_messages[0]["content"] == SCAN_POLICY
    key = (
        v1_key
        if version == "v1"
        else _historical_scan_request(owner, coordinator, row, [], [], 0, len(text))[0]
    )
    assert (key == v1_key) == (version == "v1")
    quote = HistoricalQuote(start=0, end=len(text), text=text)
    receipts[key] = {
        "output": HistoricalScan(coverage="complete", quotes=[quote]).model_dump(),
        "model_provenance": {"model_name": "fixture-model", "provider": None},
    }
    assert _remaining_history_calls(owner, coordinator, row, [(0, len(text))], []) == 0
    assert _scan_history(owner, coordinator, row, [(0, len(text))], []) == [quote]
    cast(
        MagicMock, owner.client.generate_chat_response_with_provenance
    ).assert_not_called()
    cast(MagicMock, coordinator.require_remaining_work).assert_not_called()


def test_historical_new_receipt_is_server_hydrated_and_retains_prior_quote():
    import json

    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        HistoricalScanSelection,
        _remaining_history_calls,
        _scan_history,
        _scan_request,
    )

    text = "Prior condition. " + 'New "condition" 🙂' * 80
    owner, coordinator, row, receipts = _history_fixture(text)
    boundary = len("Prior condition. ")
    prior = HistoricalQuote(start=0, end=boundary, text=text[:boundary])
    key = _scan_request(owner, coordinator, row, [], [], 0, boundary)[0]
    receipts[key] = {
        "output": HistoricalScan(coverage="complete", quotes=[prior]).model_dump()
    }

    def select(**kwargs):
        assert kwargs["response_format"] is HistoricalScanSelection
        payload = json.loads(kwargs["messages"][1]["content"])
        assert payload["selected_quotes"] == [prior.model_dump()]
        assert payload["remaining_carrier_tokens"] < 2048
        identity = payload["chunk"]["snippets"][-1]["snippet_id"]
        selected = HistoricalScanSelection(coverage="complete", snippet_ids=[identity])
        assert kwargs["structured_output_validator"](selected) == []
        return CompletionResult(selected, ModelProvenance(model_name="fixture-model"))

    cast(
        MagicMock, owner.client.generate_chat_response_with_provenance
    ).side_effect = select
    ranges = [(0, boundary), (boundary, len(text))]
    assert _remaining_history_calls(owner, coordinator, row, ranges, []) == 1
    quotes = _scan_history(owner, coordinator, row, ranges, [])
    assert quotes[0] == prior and len(quotes) == 2
    assert all(quote.text == text[quote.start : quote.end] for quote in quotes)
    assert len(receipts) == 2
    assert all(
        "quotes" in receipt["output"] and "snippet_ids" not in receipt["output"]
        for receipt in receipts.values()
    )
    assert _remaining_history_calls(owner, coordinator, row, ranges, []) == 0
    assert _scan_history(owner, coordinator, row, ranges, []) == quotes
    assert (
        cast(MagicMock, owner.client.generate_chat_response_with_provenance).call_count
        == 1
    )


def test_historical_selection_capacity_hold_never_saves_or_removes_prior_evidence():
    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        HistoricalScanSelection,
        _historical_scan_request,
        _scan_history,
        _scan_request,
        _validate_selection_scan,
    )

    prefix = "word " * 1700
    text = prefix + "🙂Ω" * 512
    prior = [HistoricalQuote(start=0, end=len(prefix), text=prefix)]
    snippets = {
        "shown": HistoricalQuote(
            start=len(prefix), end=len(text), text=text[len(prefix) :]
        )
    }
    with pytest.raises(InputBudgetExceededError, match="carrier capacity"):
        _validate_selection_scan(
            HistoricalScanSelection(coverage="complete", snippet_ids=["shown"]),
            prior=prior,
            snippets=snippets,
            text=text,
            start=len(prefix),
            end=len(text),
        )
    assert prior == [HistoricalQuote(start=0, end=len(prefix), text=prefix)]
    owner, coordinator, row, receipts = _history_fixture(text)
    old_key = _scan_request(owner, coordinator, row, [], [], 0, len(prefix))[0]
    saved = {"output": HistoricalScan(coverage="complete", quotes=prior).model_dump()}
    receipts[old_key] = saved
    new_key, _, _, _, shown = _historical_scan_request(
        owner, coordinator, row, [], prior, len(prefix), len(text)
    )
    cast(
        MagicMock, owner.client.generate_chat_response_with_provenance
    ).return_value = CompletionResult(
        HistoricalScanSelection(coverage="complete", snippet_ids=list(shown)),
        ModelProvenance(model_name="fixture-model"),
    )
    with pytest.raises(InputBudgetExceededError, match="carrier capacity"):
        _scan_history(
            owner, coordinator, row, [(0, len(prefix)), (len(prefix), len(text))], []
        )
    assert receipts == {old_key: saved}
    assert new_key not in receipts


@pytest.mark.parametrize("corrected", [True, False])
def test_historical_capacity_correction_preserves_prior_and_bounds_dispatch(corrected):
    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        HistoricalScanSelection,
        _scan_history,
        _scan_request,
    )

    prefix = "word " * 1700
    text = prefix + "🙂Ω" * 512
    prior = HistoricalQuote(start=0, end=len(prefix), text=prefix)
    owner, coordinator, row, receipts = _history_fixture(text)
    key = _scan_request(owner, coordinator, row, [], [], 0, len(prefix))[0]
    saved = {"output": HistoricalScan(coverage="complete", quotes=[prior]).model_dump()}
    receipts[key] = saved
    upstream_guard = MagicMock()

    def completion(**kwargs):
        import json

        body = json.loads(kwargs["messages"][1]["content"])
        oversized = HistoricalScanSelection(
            coverage="complete",
            snippet_ids=[item["snippet_id"] for item in body["chunk"]["snippets"]],
        )
        guard = extraction_provider_guard.get()
        assert guard is not None
        guard({}, 10, ("fixture-model",), True)
        feedback = kwargs["structured_output_validator"](oversized)
        assert feedback and "remaining capacity" in feedback[0]
        assert "🙂Ω" not in feedback[0]
        guard({}, 10, ("fixture-model",), True)
        output = (
            HistoricalScanSelection(coverage="complete", snippet_ids=[])
            if corrected
            else oversized
        )
        assert kwargs["structured_output_validator"](output) == []
        return CompletionResult(output, ModelProvenance(model_name="fixture-model"))

    cast(
        MagicMock, owner.client.generate_chat_response_with_provenance
    ).side_effect = completion
    token = extraction_provider_guard.set(upstream_guard)
    try:
        if corrected:
            assert _scan_history(
                owner,
                coordinator,
                row,
                [(0, len(prefix)), (len(prefix), len(text))],
                [],
            ) == [prior]
            assert len(receipts) == 2
        else:
            with pytest.raises(InputBudgetExceededError, match="carrier capacity"):
                _scan_history(
                    owner,
                    coordinator,
                    row,
                    [(0, len(prefix)), (len(prefix), len(text))],
                    [],
                )
            assert receipts == {key: saved}
        assert receipts[key] == saved
        assert upstream_guard.call_count == 2
        assert extraction_provider_guard.get() is upstream_guard
    finally:
        extraction_provider_guard.reset(token)


def test_historical_neighbor_reselection_uses_physical_quote_coverage():
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        HistoricalScanSelection,
        _selection_scan,
    )

    text = "Preserve approval. Keep the deadline."
    prior = [
        HistoricalQuote(start=0, end=18, text=text[:18]),
        HistoricalQuote(start=18, end=len(text), text=text[18:]),
    ]
    snippets = {"neighbor": HistoricalQuote(start=2, end=25, text=text[2:25])}
    selected = HistoricalScanSelection(coverage="complete", snippet_ids=["neighbor"])
    assert _selection_scan(selected, snippets, prior).quotes == []
    # A one-character gap must remain new evidence, even with equal text nearby.
    gap = [prior[0].model_copy(update={"end": 17, "text": text[:17]}), prior[1]]
    assert _selection_scan(selected, snippets, gap).quotes == [snippets["neighbor"]]
    assert len(prior) == 2


def test_rebuilt_proposal_reuses_receipt_across_clock_and_unrelated_witness_changes(
    monkeypatch,
):
    from dataclasses import asdict
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from reflexio.models.api_schema.domain import entities
    from reflexio.models.config_schema import PlaybookConfig
    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        current_overflow_coordinator,
    )
    from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        _cached_proposal,
    )
    from reflexio.server.services.playbook.components.overflow_evidence import (
        historical_playbook_digest,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookContent,
    )
    from reflexio.server.services.playbook.service import (
        PlaybookGenerationServiceConfig,
    )

    old = _old(51).model_copy(
        update={"created_at": 1000, "content": "Require approval before deploying."}
    )
    checkpoint = {"historical_playbooks": {"51": historical_playbook_digest(old)}}
    receipts = {}
    keys = []
    coordinator = SimpleNamespace(
        source_digest="frozen-source",
        checkpoint=checkpoint,
        saved_run=SimpleNamespace(
            created_at=datetime(2026, 10, 1, 12, tzinfo=UTC), committed_output=None
        ),
        completed_step=receipts.get,
        require_remaining_work=MagicMock(),
    )

    def step_key(stage, inputs):
        key = OverflowCoordinator.step_key(
            cast(OverflowCoordinator, coordinator), stage, inputs
        )
        keys.append(key)
        return key

    def complete_step(key, output, provenance):
        receipts[key] = {"output": output, "model_provenance": asdict(provenance)}

    coordinator.step_key = step_key
    coordinator.complete_step = complete_step
    extractor = PlaybookExtractor.__new__(PlaybookExtractor)
    extractor.config = PlaybookConfig(extraction_definition_prompt="Fixture extraction")
    extractor.service_config = PlaybookGenerationServiceConfig(
        user_id="u", agent_version="v1", request_id="new", source="api"
    )
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    owner.model_name = "fixture-model"
    owner.model_provenance = ModelProvenance(
        model_name="fixture-model", provider="fixture"
    )
    owner.request_context = MagicMock()
    # Exercise the real formatter: NEW Last Modified is part of the real key.
    owner.request_context.prompt_manager.render_prompt.side_effect = (
        lambda _prompt_id, values: str(values)
    )
    output = PlaybookConsolidationOutput(
        decisions=[RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=0)]
    )
    provider = MagicMock(return_value=output)
    monkeypatch.setattr(owner, "_consolidation_decisions", provider)
    token = current_overflow_coordinator.set(cast(OverflowCoordinator, coordinator))
    try:
        prompts = []
        for minute in (1, 20):
            ambient = datetime(2026, 10, 1, 13, minute, tzinfo=UTC)
            monkeypatch.setattr(
                entities,
                "datetime",
                SimpleNamespace(now=lambda _, frozen=ambient: frozen),
            )
            candidate = extractor._build_user_playbook(
                StructuredPlaybookContent(content="Require approval before deploying."),
                source_interaction_ids=[7],
            )
            assert candidate is not None
            prompts.append(owner._render_consolidation_prompt([candidate], [old]))
            assert (
                _cached_proposal(
                    owner, cast(OverflowCoordinator, coordinator), [candidate], [old]
                )
                == output
            )
            checkpoint["historical_playbooks"]["999"] = "unrelated-history-witness"
        assert prompts[0] == prompts[1]
        assert keys[0] == keys[1]
        assert provider.call_count == 1
        assert coordinator.require_remaining_work.call_count == 1
        assert len(receipts) == 1

        # A selected OLD change is load-bearing, so it must never reuse the
        # previous proposal even though unrelated witness additions are ignored.
        changed = old.model_copy(
            update={"content": "Require separate approval and tests."}
        )
        checkpoint["historical_playbooks"]["51"] = historical_playbook_digest(changed)
        assert (
            _cached_proposal(
                owner, cast(OverflowCoordinator, coordinator), [candidate], [changed]
            )
            == output
        )
        assert keys[2] != keys[0]
        assert provider.call_count == 2
        assert coordinator.require_remaining_work.call_count == 2
        assert len(receipts) == 2
    finally:
        current_overflow_coordinator.reset(token)


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


def _rule(identity, content, trigger="after preparation"):
    return _old(identity).model_copy(
        update={"content": content, "trigger": trigger, "source_interaction_ids": []}
    )


_APPROVAL = (
    "Prepare the release branch only; production deployment needs a separate "
    "explicit approval from the owner each time."
)
_PREFLIGHT = (
    "Run the migration registration preflight and the deploy guards locally "
    "before opening any pull request that adds a migration."
)
# Restates the OLD rule and adds a constraint; a lexical "same lesson" check
# matches it, so superseding it by OLD would silently drop the new constraint.
_APPROVAL_PLUS = _APPROVAL + " Never deploy on Fridays."


def _uncleared_inputs():
    old = [
        _rule(51, "Unrelated lesson about formatting tables in reports."),
        _rule(52, _APPROVAL),
    ]
    new = [
        _rule(0, _APPROVAL_PLUS).model_copy(update={"request_id": "current"}),
        # Two siblings describing the same lesson, neither matching history.
        _rule(0, _PREFLIGHT, "before opening a pull request"),
        _rule(0, _PREFLIGHT + " Always.", "before opening a pull request"),
    ]
    return new, old


def test_uncleared_proposal_never_changes_history_or_drops_a_lesson():
    from reflexio.server.services.playbook.components.consolidator import (
        IndependentDecision,
    )
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        unconsolidated_decisions,
    )

    new, old = _uncleared_inputs()
    plan = unconsolidated_decisions(new)
    assert plan.decisions == [IndependentDecision(new_id=f"NEW-{i}") for i in range(3)]
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    owner.consolidated_output_indices = set()
    owner.request_context = cast(Any, SimpleNamespace(org_id="org"))
    build = partial(
        owner._build_deduplicated_results,
        new_playbooks=new,
        existing_playbooks=old,
        dedup_output=plan,
        generation_request_id="current",
        agent_version="v1",
    )
    # The overlap guard would reject this plan (NEW-0 against OLD, and the two
    # siblings against each other); the no-merge plan admits it.
    same = partial(
        PlaybookConsolidator._describes_same_lesson,
        content_threshold=0.7,
        min_content_length=80,
    )
    assert same(new[0], old[1]) and same(new[1], new[2])
    with pytest.raises(ValueError, match="Overlapping consolidation survivors"):
        build()
    rows, archived, groups = build(admit_overlapping_survivors=True)
    assert archived == [] and groups == []
    assert [row.content for row in rows] == [row.content for row in new]
    assert [row.content for row in old] == [old[0].content, _APPROVAL]


def test_deduplicate_completes_an_uncleared_window_without_touching_history(
    monkeypatch,
):
    from reflexio.models.config_schema import DeduplicationConfig
    from reflexio.server.services.extraction import overflow_coordinator
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    def refuse(*_):
        raise module.ConsolidationNotClearedError(
            "Historical consolidation is unsafe or undetermined"
        )

    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setattr(module, "_verified_proposal", refuse)
    coordinator = SimpleNamespace(remember_historical_playbooks=lambda _rows: None)
    token = overflow_coordinator.current_overflow_coordinator.set(
        cast(Any, coordinator)
    )
    try:
        owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
        owner._dedup_config = DeduplicationConfig()
        owner.request_context = cast(Any, SimpleNamespace(org_id="org"))
        new, old = _uncleared_inputs()
        for _ in range(2):  # every retry of the window reaches the same result
            rows, archived, groups = owner.deduplicate(
                [new], "current", "v1", user_id="u", existing_playbooks=old
            )
            assert archived == [] and groups == []
            assert [row.content for row in rows] == [row.content for row in new]
            assert owner.consolidation_not_cleared is True
        # A later VERIFIED plan with the same overlapping survivors is still
        # rejected: admission is scoped to the uncleared call that set it.
        monkeypatch.setattr(
            module,
            "_verified_proposal",
            lambda *_: module.unconsolidated_decisions(new),
        )
        with pytest.raises(ValueError, match="Overlapping consolidation survivors"):
            owner.deduplicate(
                [new], "current", "v1", user_id="u", existing_playbooks=old
            )
        assert owner.consolidation_not_cleared is False
    finally:
        overflow_coordinator.current_overflow_coordinator.reset(token)


@pytest.mark.parametrize(
    "cause",
    [
        "Historical consolidation is unsafe or undetermined",
        "Historical evidence coverage is undetermined",
        "Historical rule lacks verified quotations",
    ],
)
def test_uncleared_history_completes_with_no_merge_instead_of_retrying(
    monkeypatch, cause
):
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    def refuse(*_):
        raise module.ConsolidationNotClearedError(cause)

    monkeypatch.setattr(module, "_verified_proposal", refuse)
    new = [_rule(0, "A fresh, unrelated lesson about release notes.")]
    old = [_rule(51, "Prepare only; deployment needs separate approval each time.")]
    owner = cast(PlaybookConsolidator, SimpleNamespace())
    coordinator = cast(OverflowCoordinator, SimpleNamespace())
    first = module.verified_consolidation_decisions(owner, coordinator, new, old)
    # Replays are identical: no saved receipt can turn into a retry loop.
    assert (
        module.verified_consolidation_decisions(owner, coordinator, new, old) == first
    )
    assert [decision.kind for decision in first.decisions] == ["independent"]


@pytest.mark.parametrize(
    "error",
    [
        InputBudgetConfigurationError("Historical source partition changed"),
        InputBudgetExceededError("Original and historical coverage exceeds 48 chunks"),
    ],
)
def test_capacity_and_integrity_failures_still_stop_publication(monkeypatch, error):
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    def fail(*_):
        raise error

    monkeypatch.setattr(module, "_verified_proposal", fail)
    with pytest.raises(type(error)):
        module.verified_consolidation_decisions(
            cast(PlaybookConsolidator, SimpleNamespace()),
            cast(OverflowCoordinator, SimpleNamespace()),
            [_rule(0, "lesson")],
            [],
        )


def _dedup_owner(monkeypatch, error):
    from reflexio.models.config_schema import DeduplicationConfig

    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    owner._dedup_config = DeduplicationConfig()
    owner.request_context = cast(Any, SimpleNamespace(org_id="org"))

    def decide(*_):
        raise error

    monkeypatch.setattr(owner, "_window_consolidation_decisions", decide)
    return owner


def _deduplicate(owner):
    return owner.deduplicate(
        [[_rule(0, "new lesson")]],
        "request",
        "v1",
        user_id="u",
        existing_playbooks=[_rule(51, "old lesson")],
    )


@pytest.mark.parametrize("kind", ["suspended", "checkpoint", "lease"])
def test_durable_control_flow_is_not_reported_as_consolidation_failure(
    monkeypatch, caplog, kind
):
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowSuspendedError,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        OverflowCheckpointConflictError,
    )

    error = {
        "suspended": OverflowSuspendedError("Extraction turn needs continuation"),
        "checkpoint": OverflowCheckpointConflictError("revision moved"),
        "lease": LeaseLostError("lease lost"),
    }[kind]
    owner = _dedup_owner(monkeypatch, error)
    caplog.set_level("DEBUG")
    with pytest.raises(type(error)):
        _deduplicate(owner)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad decision"),
        InputBudgetConfigurationError("Historical source partition changed"),
    ],
)
def test_consolidation_failures_are_still_reported(monkeypatch, caplog, error):
    owner = _dedup_owner(monkeypatch, error)
    with pytest.raises(type(error)):
        _deduplicate(owner)
    assert any(
        r.levelname == "ERROR" and r.getMessage() == "Failed to identify duplicates"
        for r in caplog.records
    )


def _retried_llm_failures() -> list[Exception]:
    from reflexio.server.llm._litellm_types import (
        LiteLLMClientError,
        StructuredOutputRepairError,
    )

    return [
        TimeoutError(),
        # The subprocess-isolated provider call: the concrete litellm class is
        # carried as ``upstream_error_type`` (PYTHON-FASTAPI-15G / 15Z).
        LiteLLMClientError("API call failed", upstream_error_type="APIConnectionError"),
        LiteLLMClientError("API call failed", upstream_status_code=529),
        # An exhausted repair ladder (PYTHON-FASTAPI-MN).
        StructuredOutputRepairError(
            "Structured output repair exhausted", failure_kind="parse", model="m"
        ),
    ]


@pytest.mark.parametrize(
    "error", _retried_llm_failures(), ids=lambda e: type(e).__name__
)
def test_retried_llm_failures_are_warnings_not_errors(monkeypatch, caplog, error):
    """The caller retries these and owns escalation; ERROR here pages per attempt."""
    owner = _dedup_owner(monkeypatch, error)
    caplog.set_level("DEBUG")
    with pytest.raises(type(error)):
        _deduplicate(owner)
    failures = [
        r for r in caplog.records if r.getMessage() == "Failed to identify duplicates"
    ]
    assert [r.levelname for r in failures] == ["WARNING"]
    assert failures[0].exc_info is not None
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


def test_non_transient_provider_failure_is_still_an_error(monkeypatch, caplog):
    from reflexio.server.llm._litellm_types import LiteLLMClientError

    owner = _dedup_owner(
        monkeypatch,
        LiteLLMClientError("bad request", upstream_error_type="BadRequestError"),
    )
    with pytest.raises(LiteLLMClientError):
        _deduplicate(owner)
    assert [
        r.levelname
        for r in caplog.records
        if r.getMessage() == "Failed to identify duplicates"
    ] == ["ERROR"]


# --- Legacy history without a stored source_span ---


def _legacy(identity=51, **update):
    values = {
        "content": "Prepare only. Deployment needs separate approval.",
        "trigger": "after preparation",
        "rationale": "Preserve approval scope",
        "source_span": None,
        "created_at": 1000,
    }
    values.update(update)
    return _old(identity).model_copy(update=values)


def _admit_fixture(chunks=1):
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    checkpoint: dict[str, Any] = {}
    prepared = PreparedExtractionInput(
        "window", tuple(PreparedChunk(index, ()) for index in range(chunks))
    )
    coordinator = SimpleNamespace(
        checkpoint=checkpoint, require_prepared=lambda: prepared
    )
    coordinator.save_checkpoint = lambda value: checkpoint.update(value)
    return cast(OverflowCoordinator, coordinator), checkpoint


def test_span_rows_keep_saved_receipt_keys_payloads_and_digests():
    """Pinned from origin/main d11e2c43: saved v1/v2 receipts must keep replaying."""
    import hashlib
    import json

    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    text = 'Keep exact "quoted" Unicode 🙂 evidence.'
    owner, coordinator, row, _ = _history_fixture(text)
    row = row.model_copy(
        update={
            "content": "Require approval.",
            "trigger": "before deploy",
            "rationale": "Owner said so",
        }
    )
    claims = [{"content": "new claim"}]

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    v1 = module._scan_request(owner, coordinator, row, claims, [], 0, len(text))
    v2 = module._historical_scan_request(
        owner, coordinator, row, claims, [], 0, len(text)
    )
    assert v1[0] == "fbe0a3552afb77eecef383c7dafdfa7c5ae9650bf2c232a6b33b7613d89a6075"
    assert digest(v1[1]) == (
        "bb0a34e4a1a0d647a1545d9315d74233f6a5a5a1f7e26ca6f44144bb28fc3ac7"
    )
    assert v2[0] == "cb3d21c4dfdd99368be2abe47da3cc547ff8f9a994c2b8f83c084ecc770e95c4"
    assert digest(v2[1]) == (
        "a97af025b1b720d147fbd9fdcd793e8435eccef2c86f0ac579135b0b13138550"
    )
    assert (
        module._historical_source_digest(row)
        == hashlib.sha256(text.encode()).hexdigest()
    )
    policies = {
        "SCAN_POLICY": "f82dc251467c5be90cdbe4d14ca0438094cf32e0cbe68a68996a25cc1515cf98",
        "SCAN_SELECTION_POLICY": "f0665772e44d0ecbc93992c17c4b7e999402922cd042a4a977fd1b2ce7aa0cf4",
        "VERIFY_POLICY": "ab2d80a8f279a5f19c71d1e82e013e5a7b22776bb31717bd4b63e5f3c57da196",
    }
    for name, expected in policies.items():
        assert hashlib.sha256(getattr(module, name).encode()).hexdigest() == expected
    quote = HistoricalQuote(start=0, end=len(text), text=text)
    from reflexio.server.services.playbook.components.overflow_evidence import (
        historical_playbook_digest,
    )

    # The verification record for a span row is the exact pre-change shape,
    # including key order (json.dumps preserves insertion order).
    assert json.dumps(module._historical_record(row, [quote])) == json.dumps(
        {
            "rule": module.rule_metadata(row),
            "author_role": "unknown",
            "source_digest": historical_playbook_digest(row),
            "quotes": [quote.model_dump()],
        }
    )


def test_stored_rule_evidence_is_rendered_marked_and_quoted_against_its_text():
    import json

    from reflexio.models.api_schema.common import BlockingIssue, BlockingIssueKind
    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    owner, coordinator, _, receipts = _history_fixture("unused")
    row = _legacy(
        rationale="  ",
        blocking_issue=BlockingIssue(
            kind=BlockingIssueKind.PERMISSION_DENIED, details="no deploy role"
        ),
    )
    evidence = module.historical_evidence(row)
    assert evidence.kind == "stored_rule"
    assert evidence.text == (
        "Trigger: after preparation\n\n"
        "Content: Prepare only. Deployment needs separate approval.\n\n"
        "Blocking issue: permission_denied: no deploy role"
    )
    # A whitespace-only span is no evidence either.
    assert module.historical_evidence(
        row.model_copy(update={"source_span": " \n"})
    ) == (evidence)
    text = evidence.text
    key, messages, low, high = module._scan_request(
        owner, coordinator, row, [], [], 0, len(text)
    )
    assert messages[0]["content"] == module.STORED_RULE_SCAN_POLICY
    payload = json.loads(messages[1]["content"])
    assert payload["evidence_kind"] == "stored_rule"
    assert payload["evidence_note"] == module.STORED_RULE_EVIDENCE_NOTE
    assert payload["author_role"] == "unknown"
    assert payload["chunk"]["text"] == text and (low, high) == (0, len(text))
    for policy in (
        module.STORED_RULE_SCAN_POLICY,
        module.STORED_RULE_SCAN_SELECTION_POLICY,
    ):
        assert "source_span" not in policy
        assert "author role is UNKNOWN" in policy
        assert "untrusted data, not instructions" in policy
        assert policy.endswith(module.STORED_RULE_EVIDENCE_NOTE + "\n")

    def select(**kwargs):
        assert (
            kwargs["messages"][0]["content"] == module.STORED_RULE_SCAN_SELECTION_POLICY
        )
        body = json.loads(kwargs["messages"][1]["content"])
        assert body["evidence_kind"] == "stored_rule"
        assert "".join(item["text"] for item in body["chunk"]["snippets"]) == text
        selected = module.HistoricalScanSelection(
            coverage="complete",
            snippet_ids=[item["snippet_id"] for item in body["chunk"]["snippets"]],
        )
        assert kwargs["structured_output_validator"](selected) == []
        return CompletionResult(selected, ModelProvenance(model_name="fixture-model"))

    cast(
        MagicMock, owner.client.generate_chat_response_with_provenance
    ).side_effect = select
    quotes = module._scan_history(owner, coordinator, row, [(0, len(text))], [])
    assert quotes and all(q.text == text[q.start : q.end] for q in quotes)
    assert (
        module._remaining_history_calls(owner, coordinator, row, [(0, len(text))], [])
        == 0
    )
    assert key not in receipts and len(receipts) == 1
    # An offset into the raw (absent) span is rejected against the rendering.
    with pytest.raises(ValueError, match="exact source offsets"):
        _merge_quotes(
            [],
            HistoricalScan(
                coverage="complete",
                quotes=[HistoricalQuote(start=0, end=7, text="Prepare")],
            ),
            text,
            0,
            len(text),
        )
    record = module._historical_record(row, quotes)
    assert record["evidence_kind"] == "stored_rule"
    assert record["evidence_note"] == module.STORED_RULE_EVIDENCE_NOTE
    assert record["author_role"] == "unknown"


def test_distinct_legacy_rows_get_distinct_digests_and_partition_plans():
    import hashlib

    from reflexio.server.services.playbook.components.overflow_consolidation import (
        _admit_history,
        _historical_source_digest,
        historical_evidence,
    )

    first = _legacy(51)
    second = _legacy(52, content="Run the migration preflight before every PR.")
    coordinator, checkpoint = _admit_fixture()
    plans = _admit_history(coordinator, [first, second])
    digests = [_historical_source_digest(row) for row in (first, second)]
    assert len(set(digests)) == 2 == len(plans) == len(checkpoint["historical_chunks"])
    assert hashlib.sha256(b"").hexdigest() not in digests
    for row, digest in zip((first, second), digests, strict=True):
        assert plans[digest] == [(0, len(historical_evidence(row).text))]
    # Re-admitting the same rows (a retry) reproduces the same plans.
    assert _admit_history(coordinator, [first, second]) == plans
    # Domain separation: a span equal to a rendering never shares its digest.
    spanned = first.model_copy(update={"source_span": historical_evidence(first).text})
    assert _historical_source_digest(spanned) != digests[0]
    assert digests[0].startswith("stored_rule:")


def test_legacy_row_gaining_a_span_does_not_trip_partition_changed():
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        _admit_history,
        _historical_source_digest,
    )

    legacy = _legacy(51)
    coordinator, checkpoint = _admit_fixture()
    _admit_history(coordinator, [legacy])
    saved = dict(checkpoint["historical_chunks"])
    spanned = legacy.model_copy(update={"source_span": "Original quote. " * 40})
    plans = _admit_history(coordinator, [spanned])
    assert list(plans) == [_historical_source_digest(spanned)]
    # The earlier stored-rule entry is kept (append-only), never overwritten.
    assert checkpoint["historical_chunks"] == {
        **saved,
        _historical_source_digest(spanned): [[0, len(spanned.source_span or "")]],
    }


def test_legacy_row_without_any_rule_text_takes_the_no_merge_plan(monkeypatch):
    from reflexio.server.services.playbook.components import (
        overflow_consolidation as module,
    )

    blank = _legacy(51, content="", trigger=None, rationale=" ")
    new = [_rule(0, "A fresh lesson about release notes.")]
    output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[0],
                content="merged",
                trigger="scope",
                rationale="reason",
            )
        ]
    )
    monkeypatch.setattr(module, "_bounded_proposals", lambda *_: output)
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    owner.consolidation_not_cleared = False
    coordinator, checkpoint = _admit_fixture()
    for _ in range(2):  # every retry reaches the same completed plan
        plan = module.verified_consolidation_decisions(owner, coordinator, new, [blank])
        assert plan == module.unconsolidated_decisions(new)
        assert owner.consolidation_not_cleared is True
    assert not checkpoint.get("historical_chunks")
