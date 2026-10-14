"""Information retention and reference invariants in bounded historical checks."""

from types import SimpleNamespace
from typing import cast
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
