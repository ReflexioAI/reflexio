"""Extraction-only presentation preserves late evidence and global source labels."""

import json
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.extraction.input_preparation import (
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
    prepare_extraction_input,
    resolve_evidence_span,
    resolve_visible_quote,
)
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    present_extraction_chunk,
    protected_quote_view,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedEvidence,
    ProtectedRecord,
    ProtectedRecordUpdate,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    construct_profile_extraction_messages_from_sessions,
)


def test_profile_and_playbook_prepared_prompts_keep_late_correction_and_source_labels():
    original = (
        "tool progress " * 2000
        + "Do not deploy before the production check."
        + " more tool progress" * 2000
    )
    sessions = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[
                Interaction(
                    interaction_id=7,
                    request_id="r",
                    user_id="u",
                    content=original,
                    role="user",
                )
            ],
            arrival_order=True,
        )
    ]
    before = sessions[0].model_dump(mode="json")
    prepared = prepare_extraction_input(sessions, user_id="u", chunk_tokens=1500)
    assert len(prepared.chunks) > 1
    manager = Mock()
    manager.render_prompt.side_effect = lambda _prompt, variables: json.dumps(variables)
    chunk = next(
        item
        for item in prepared.chunks
        if "Do not deploy before the production check." in item.text
    )
    with present_extraction_chunk(prepared, chunk, ProtectedRecord()):
        profile = construct_profile_extraction_messages_from_sessions(
            manager, sessions, [], "agent", "context", "definition"
        )
        assert "Do not deploy before the production check." in json.dumps(profile)
        playbook = build_playbook_prompt_context(sessions, label_turns=True)
        assert "Do not deploy before the production check." in playbook.text
        assert set(playbook.evidence_units) == {
            span.evidence_ref for span in (*chunk.neighbors, *chunk.spans)
        }
        assert {unit.turn_ref for unit in playbook.evidence_units.values()} == {"T1"}
        assert all(
            unit.interaction_id == 7 for unit in playbook.evidence_units.values()
        )
    assert current_overflow_presentation.get() is None
    assert sessions[0].model_dump(mode="json") == before
    # The regular formatter keeps its existing bounded presentation.
    assert (
        "Do not deploy before the production check."
        not in build_playbook_prompt_context(sessions).text
    )


def test_protected_neighbor_quotation_remains_resolvable_in_later_chunk():
    sessions = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[
                Interaction(
                    interaction_id=7,
                    request_id="r",
                    user_id="u",
                    role="user",
                    content="原文 🙂 scoped approval " * 1500,
                )
            ],
        )
    ]
    prepared = prepare_extraction_input(
        sessions, user_id="u", chunk_tokens=1800, require_overlap=True
    )
    assert len(prepared.chunks) > 2
    neighbor = prepared.chunks[1].neighbors[0]
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(neighbor.evidence_ref,),
                    quote=neighbor.text,
                )
            ]
        ),
        visible_spans=prepared.chunks[1].neighbors,
    )
    with present_extraction_chunk(prepared, prepared.chunks[-1], record):
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        refs = {span.evidence_ref: span for span in presentation.visible_spans}
        assert refs[neighbor.evidence_ref] == neighbor
        context = build_playbook_prompt_context(sessions, label_turns=True)
        assert context.evidence_units[neighbor.evidence_ref].role == "user"
        assert context.evidence_units[neighbor.evidence_ref].interaction_id == 7


def test_protected_quotation_exposes_only_shown_text_across_profile_and_playbook():
    from reflexio.server.services.playbook.components.extractor import (
        _fresh_candidate_record,
        _fresh_output_validation_errors,
    )
    from reflexio.server.services.playbook.playbook_evidence import as_playbook_content
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredExtractedPlaybookContent,
        StructuredExtractedPlaybookList,
        StructuredPlaybookEvidence,
        StructuredReferencedExtractedPlaybookContent,
        StructuredReferencedExtractedPlaybookList,
    )
    from reflexio.server.services.profile.components.extractor import (
        _profile_evidence_validation_errors,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    quote = "Keep preparation scoped."
    hidden = "Never infer deployment approval."
    prefix = f"Earlier context. {quote} {hidden}"
    current = "Current independent message."
    duplicate = quote
    original = prefix + current + duplicate
    spans = (
        SourceSpan(7, "r", "user", "content", 0, len(prefix), prefix, 1),
        SourceSpan(
            7,
            "r",
            "user",
            "content",
            len(prefix),
            len(prefix) + len(current),
            current,
            1,
        ),
        SourceSpan(
            7,
            "r",
            "user",
            "content",
            len(prefix) + len(current),
            len(original),
            duplicate,
            1,
        ),
    )
    chunks = tuple(PreparedChunk(index, (span,)) for index, span in enumerate(spans))
    prepared = PreparedExtractionInput("fixture-source", chunks)
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(spans[0].evidence_ref,),
                    quote=quote,
                )
            ]
        ),
        visible_spans=spans,
    )
    snapshot = record.model_dump(mode="json")
    shown, protected_spans = protected_quote_view(prepared, record)
    quoted = protected_spans[0]
    assert quoted.start == prefix.index(quote)
    assert quoted.end == quoted.start + len(quote)
    assert quoted.text == quote
    assert resolve_evidence_span(quoted.evidence_ref, spans) == quoted
    assert shown.evidence[0].evidence_refs == (quoted.evidence_ref,)
    sessions = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[
                Interaction(
                    interaction_id=7,
                    request_id="r",
                    user_id="u",
                    role="user",
                    content=original,
                )
            ],
        )
    ]
    with present_extraction_chunk(prepared, chunks[1], record):
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        assert hidden not in presentation.text
        assert spans[0].evidence_ref not in presentation.text
        visible = presentation.visible_spans
        context = build_playbook_prompt_context(sessions, label_turns=True)
        assert set(context.evidence_units) == {
            quoted.evidence_ref,
            spans[1].evidence_ref,
        }

        def profile_output(text):
            return StructuredProfilesOutput.model_validate(
                {
                    "profiles": [
                        {
                            "content": "Keep scope narrow.",
                            "time_to_live": "infinity",
                            "source_span": text,
                        }
                    ]
                }
            )

        assert _profile_evidence_validation_errors(profile_output(quote), visible) == []
        assert _profile_evidence_validation_errors(profile_output(hidden), visible)

        def typed_output(ref):
            return StructuredReferencedExtractedPlaybookList(
                playbooks=[
                    StructuredReferencedExtractedPlaybookContent(
                        rationale="The instruction limits preparation scope.",
                        evidence_kind="correction",
                        trigger="When preparing a change",
                        content="Keep preparation scoped.",
                        evidence_refs=[ref],
                    )
                ]
            )

        accepted = typed_output(quoted.evidence_ref)
        assert _fresh_output_validation_errors(accepted, context, spans) == []
        assert _fresh_output_validation_errors(
            typed_output(spans[0].evidence_ref), context, spans
        )
        carrier = _fresh_candidate_record(
            as_playbook_content(accepted.playbooks[0]), context, spans
        )
        assert carrier.evidence[0].quote == quote
        assert carrier.evidence[0].evidence_refs == (quoted.evidence_ref,)
        # The unshown duplicate elsewhere in the window cannot make the shown
        # quotation ambiguous at this prompt boundary.
        explicit = StructuredExtractedPlaybookList(
            playbooks=[
                StructuredExtractedPlaybookContent(
                    rationale="The instruction limits preparation scope.",
                    evidence_kind="correction",
                    trigger="When preparing a change",
                    content="Keep preparation scoped.",
                    evidence=[
                        StructuredPlaybookEvidence(turn_ref="T1", source_span=quote)
                    ],
                )
            ]
        )
        assert _fresh_output_validation_errors(explicit, context, spans) == []
    assert record.model_dump(mode="json") == snapshot
    assert prepared.chunks == chunks


def test_protected_quote_subranges_preserve_contiguous_offsets_and_reject_ambiguity():
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    text = "prefix. uniquely grounded quotation. suffix."
    boundary = text.index("grounded")
    spans = (
        SourceSpan(7, "r", "user", "content", 0, boundary, text[:boundary], 1),
        SourceSpan(7, "r", "user", "content", boundary, len(text), text[boundary:], 1),
    )
    prepared = PreparedExtractionInput("fixture-source", (PreparedChunk(0, spans),))
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=tuple(span.evidence_ref for span in spans),
                    quote="uniquely grounded quotation.",
                )
            ]
        ),
        visible_spans=spans,
    )
    shown, derived = protected_quote_view(prepared, record)
    assert "".join(span.text for span in derived) == record.evidence[0].quote
    assert derived[0].end == derived[1].start
    assert (
        ProtectedRecord().merge(
            ProtectedRecordUpdate(evidence=list(shown.evidence)), visible_spans=spans
        )
        == shown
    )
    for ref in ("i7:content:0:9999", "i8:content:0:1", "i7:expert_content:0:1"):
        with pytest.raises(ValueError):
            resolve_evidence_span(ref, spans)
    repeated = SourceSpan(7, "r", "user", "content", 0, 9, "same same", 1)
    repeated_plan = PreparedExtractionInput(
        "fixture-source", (PreparedChunk(0, (repeated,)),)
    )
    ambiguous = ProtectedRecord(
        evidence=(
            ProtectedEvidence(
                kind="instruction",
                evidence_refs=(repeated.evidence_ref,),
                quote="same",
            ),
        )
    )
    with pytest.raises(InputBudgetConfigurationError, match="unique original"):
        protected_quote_view(repeated_plan, ambiguous)


def test_visible_quote_resolver_preserves_physical_offsets_and_profile_repeated_support():
    from dataclasses import replace

    first = SourceSpan(7, "r", "user", "content", 0, 9, "same tail", 1)
    overlap = replace(first, start=2, text=first.text[2:])
    assert resolve_visible_quote("same", (first, overlap)) == (
        replace(first, end=4, text="same"),
    )
    later = replace(first, start=20, end=29)
    with pytest.raises(ValueError, match="unique"):
        resolve_visible_quote("same", (first, later))
    shown = resolve_visible_quote("same", (first, later), require_unique=False)
    assert [(span.start, span.end, span.text) for span in shown] == [
        (0, 4, "same"),
        (20, 24, "same"),
    ]
    foreign = replace(later, interaction_id=8, request_id="other", source_order=2)
    assert resolve_visible_quote("same", (first, foreign), turn_ref="T1") == shown[:1]
    left = replace(first, end=4, text="same")
    right = replace(first, start=4, text=" tail")
    assert (
        "".join(span.text for span in resolve_visible_quote("same tail", (left, right)))
        == "same tail"
    )
    gap = replace(right, start=5, end=10)
    with pytest.raises(ValueError):
        resolve_visible_quote("same tail", (left, gap))


def test_quote_only_protected_rendering_retains_the_record_capacity_guard():
    import tiktoken

    from reflexio.server.services.extraction.input_preparation import (
        ExtractionInputBudgetExceededError,
    )

    quotes = [f"<specific-quotation-{index:03}>" for index in range(128)]
    text = "x" * 10000 + " ".join(quotes)
    span = SourceSpan(7, "r", "user", "content", 0, len(text), text, 1)
    plan = PreparedExtractionInput("fixture-source", (PreparedChunk(0, (span,)),))
    encoding = tiktoken.get_encoding("cl100k_base")
    for count in range(1, 129):
        record = ProtectedRecord(
            evidence=tuple(
                ProtectedEvidence(
                    kind="instruction", evidence_refs=(span.evidence_ref,), quote=quote
                )
                for quote in quotes[:count]
            )
        )
        if len(encoding.encode(record.render(), disallowed_special=())) > 2048:
            break
        try:
            protected_quote_view(plan, record)
        except ExtractionInputBudgetExceededError:
            assert record.evidence[-1].quote == quotes[count - 1]
            return
    pytest.fail(
        "Fixture must cross the rendered-ref capacity boundary while the stored record still fits"
    )


@pytest.mark.parametrize("producer", ["system", "internal", ""])
def test_v3_provider_context_keeps_internal_producer_and_original_quote_role(
    monkeypatch, producer
):
    from dataclasses import replace
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.extraction.overflow_presentation import (
        protected_source_origins,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    text = "This generated user-role message reports approval."
    earlier = RequestInteractionDataModel(
        session_id="s",
        request=Request(request_id="r", user_id="u", session_id="s", source=producer),
        interactions=[
            Interaction(
                interaction_id=7, request_id="r", user_id="u", role="User", content=text
            )
        ],
    )
    plan = prepare_extraction_input(
        [earlier], user_id="u", require_overlap=True, include_request_source=True
    )
    span = plan.chunks[0].spans[0]
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(span.evidence_ref,),
                    quote="reports approval",
                    role="Assistant",
                )
            ]
        ),
        visible_spans=(span,),
    )
    assert (
        record.evidence[0].role == "User" and record.evidence[0].kind == "instruction"
    )
    original_record = record.model_dump(mode="json")
    # Protected schema and durable serialization remain unchanged across versions.
    assert '"request_source"' not in json.dumps(
        ProtectedRecordUpdate.model_json_schema()
    )
    assert all("request_source" not in item for item in original_record["evidence"])
    metadata = protected_source_origins(plan, record)
    origin = metadata["protected_source_origins"][span.evidence_ref]
    assert origin["request_source"] == producer and origin["role"] == "User"
    assert "not user approval" in metadata["source_attribution_policy"]
    # Raw carried proof receives the same producer context even with no record.
    carried = protected_source_origins(plan, source_spans=(span,))
    assert (
        carried["protected_source_origins"][span.evidence_ref]["request_source"]
        == producer
    )
    with pytest.raises(InputBudgetConfigurationError, match="origin changed"):
        protected_source_origins(
            plan, source_spans=(replace(span, request_source="forged-human"),)
        )
    manager = Mock()
    manager.render_prompt.side_effect = lambda _prompt, variables: json.dumps(variables)
    with present_extraction_chunk(plan, plan.chunks[0], record):
        messages = construct_profile_extraction_messages_from_sessions(
            manager, [earlier], [], "agent", "context", "definition"
        )
    seen = []

    def respond(params, hard_timeout):
        seen.append(params["messages"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"profiles":[]}', tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20, total_tokens=120
            ),
            model="gpt-4o",
            _hidden_params={"custom_llm_provider": "openai"},
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-provider")
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", respond)
    result = client.generate_chat_response_with_provenance(
        messages=messages, response_format=StructuredProfilesOutput
    )
    assert result.value == StructuredProfilesOutput(profiles=[])
    assert len(seen) == 1
    wire = json.dumps(seen[0]).replace("\\", "").replace(" ", "")
    assert f'"request_source":"{producer}"' in wire
    assert "doesnotprovedirecthumanauthorship" in wire
    assert record.model_dump(mode="json") == original_record
    for dated in (False, True):
        old = prepare_extraction_input([earlier], user_id="u", require_overlap=dated)
        assert protected_source_origins(old, record) == {}
        with present_extraction_chunk(old, old.chunks[0], record):
            presentation = current_overflow_presentation.get()
            assert presentation is not None
            shown = json.loads(presentation.text)
        assert set(shown) == {"protected_record", "original_chunk"}
        assert "request_source" not in json.dumps(shown)
