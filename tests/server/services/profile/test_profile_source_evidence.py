"""Permission compilation uses the same visible originals as profile validation."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.services.extraction.input_preparation import SourceSpan
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extraction.overflow_pipeline import (
    _candidate_source_spans,
)
from reflexio.server.services.extraction.resumable_agent import encode_committed_output
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.profile.components.extractor import (
    ProfileExtractor,
    _profile_evidence_validation_errors,
    profile_forget_authority_texts,
    profile_source_evidence_texts,
)
from reflexio.server.services.profile.lifecycle import metadata
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.service import ProfileGenerationServiceConfig
from tests.server.services.extraction import test_resume_worker as resume

storage = resume.storage
request_context = resume.request_context


def _permission(kind):
    quote = (
        "Delete this memory on 2033-05-18T03:33:20Z."
        if kind == "deadline"
        else "Forget my address."
    )
    fields = (
        {
            "explicit_storage_deadline": {
                "deadline_utc": "2033-05-18T03:33:20Z",
                "source_quote": quote,
            }
        }
        if kind == "deadline"
        else {"explicit_forget_quote": quote}
    )
    return quote, fields


def _raw(kind):
    _, fields = _permission(kind)
    return {"content": "An extracted fact", "source_span": "A visible fact", **fields}


def _convert(raw, texts, user_texts=()):
    return ProfileExtractor._convert_raw_to_user_profiles(
        [raw], "u", "r", source_texts=texts, user_source_texts=user_texts
    )[0]


@pytest.mark.parametrize("kind", ["deadline", "forget"])
def test_ordinary_view_authorizes_visible_action_but_not_hidden_fields(kind):
    quote, _ = _permission(kind)
    interaction = Interaction(
        interaction_id=1,
        user_id="u",
        request_id="r",
        content="A visible fact",
        user_action=UserActionType.CLICK,
        user_action_description=quote,
        expert_content="Hidden expert permission",
        shadow_content="Hidden shadow permission",
    )
    models = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[interaction],
        )
    ]
    texts = profile_source_evidence_texts(models)
    assert quote in "\n".join(texts)
    assert _convert(_raw(kind), texts, profile_forget_authority_texts(models))
    hidden = interaction.model_copy(update={"user_action": UserActionType.NONE})
    hidden_models = [models[0].model_copy(update={"interactions": [hidden]})]
    with pytest.raises(ValueError, match="absent from original source"):
        _convert(_raw(kind), profile_source_evidence_texts(hidden_models))
    assert "Hidden expert permission" not in texts
    assert "Hidden shadow permission" not in texts


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["deadline", "forget"])
@pytest.mark.parametrize("role", ["User", "Assistant", "Tool"])
def test_fresh_run_converts_permission_in_actual_visible_action_view(
    request_context, kind, monkeypatch, role
):
    quote, _ = _permission(kind)
    interaction = Interaction(
        role=role,
        interaction_id=1,
        user_id="u",
        request_id="r",
        content="A visible fact",
        user_action=UserActionType.CLICK,
        user_action_description=quote,
    )
    models = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[interaction],
        )
    ]
    extractor = ProfileExtractor(
        request_context,
        MagicMock(),
        ProfileExtractorConfig(extraction_definition_prompt="Extract preferences"),
        ProfileGenerationServiceConfig(
            user_id="u", request_id="r", window_interactions=models
        ),
        "Test context",
    )
    generated = MagicMock(return_value=[_raw(kind)])
    monkeypatch.setattr(extractor, "_generate_raw_updates_from_sessions", generated)
    if kind == "forget" and role != "User":
        with pytest.raises(ValueError, match="User instruction"):
            extractor.run()
        return
    outcome = extractor.run()
    assert isinstance(outcome, ExtractionOutcome)
    assert len(outcome.items) == 1
    authority = metadata(outcome.items[0])
    assert (
        authority["explicit_storage_deadline"]["source_quote"]
        if kind == "deadline"
        else authority["explicit_forget_quote"]
    ) == quote
    generated.assert_called_once()


@pytest.mark.parametrize("kind", ["deadline", "forget"])
@pytest.mark.parametrize(
    "field",
    [
        "content",
        "expert_content",
        "shadow_content",
        "user_action_description",
        "tools_used[0]",
    ],
)
def test_prepared_permission_uses_every_visible_original_field(kind, field):
    quote, _ = _permission(kind)
    spans = (
        SourceSpan(1, "r", "User", "content", 0, 14, "A visible fact"),
        SourceSpan(2, "r", "User", field, 0, len(quote), quote),
    )
    raw = _raw(kind)
    output = StructuredProfilesOutput.model_validate({"profiles": [raw]})
    errors = _profile_evidence_validation_errors(output, spans)
    texts = profile_source_evidence_texts([], visible_spans=spans)
    users = profile_forget_authority_texts([], visible_spans=spans)
    if kind == "forget" and field not in {"content", "user_action_description"}:
        assert errors
        with pytest.raises(ValueError, match="User instruction"):
            _convert(raw, texts, users)
    else:
        assert errors == []
        assert _convert(raw, texts, users)


@pytest.mark.parametrize("kind", ["deadline", "forget"])
@pytest.mark.parametrize("defect", ["gap", "other-field", "hidden-range"])
def test_permission_cannot_join_missing_or_hidden_original_ranges(kind, defect):
    quote, _ = _permission(kind)
    middle = len(quote) // 2
    if defect == "hidden-range":
        permission = (SourceSpan(2, "r", "User", "content", 0, middle, quote[:middle]),)
    else:
        start = middle + 1 if defect == "gap" else middle
        permission = (
            SourceSpan(2, "r", "User", "content", 0, middle, quote[:middle]),
            SourceSpan(
                2,
                "r",
                "User",
                "content" if defect == "gap" else "expert_content",
                start,
                len(quote),
                quote[start:],
            ),
        )
    spans = (
        SourceSpan(1, "r", "User", "content", 0, 14, "A visible fact"),
        *permission,
    )
    assert _profile_evidence_validation_errors(
        StructuredProfilesOutput.model_validate({"profiles": [_raw(kind)]}), spans
    )
    with pytest.raises(ValueError, match="absent from original source"):
        _convert(_raw(kind), profile_source_evidence_texts([], visible_spans=spans))


@pytest.mark.parametrize("kind", ["deadline", "forget"])
def test_overflow_carries_permission_separate_from_factual_source_quote(kind):
    quote, _ = _permission(kind)
    spans = (
        SourceSpan(1, "r", "User", "content", 0, 14, "A visible fact"),
        SourceSpan(1, "r", "User", "user_action_description", 0, len(quote), quote),
        SourceSpan(1, "r", "User", "expert_content", 0, 14, "Hidden surplus"),
    )
    carried = _candidate_source_spans([_raw(kind)], "profiles", spans)
    assert {span.field for span in carried} == {"content", "user_action_description"}
    assert (
        _profile_evidence_validation_errors(
            StructuredProfilesOutput.model_validate({"profiles": [_raw(kind)]}), carried
        )
        == []
    )
    assert _convert(
        _raw(kind),
        profile_source_evidence_texts([], visible_spans=carried),
        profile_forget_authority_texts([], visible_spans=carried),
    )


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["deadline", "forget"])
@pytest.mark.parametrize(
    "field", ["content", "user_action_description", "expert_content", "shadow_content"]
)
@pytest.mark.parametrize("role", ["User", "Assistant", "Tool"])
def test_resumed_conversion_reconstructs_only_ordinary_visible_evidence(
    storage, request_context, kind, field, role
):
    quote, permission = _permission(kind)
    resume._seed_interactions(storage)
    resume._seed_ready_run(storage)
    statements = {
        "content": "UPDATE interactions SET content=? WHERE interaction_id=1",
        "user_action_description": "UPDATE interactions SET user_action='click',user_action_description=? WHERE interaction_id=1",
        "expert_content": "UPDATE interactions SET expert_content=? WHERE interaction_id=1",
        "shadow_content": "UPDATE interactions SET shadow_content=? WHERE interaction_id=1",
    }
    with storage.commit_scope():
        storage.conn.execute(statements[field], (quote,))
        storage.conn.execute(
            "UPDATE interactions SET role=? WHERE interaction_id=1", (role,)
        )
    run = storage.get_agent_run("run_1")
    run = replace(
        run,
        committed_output=encode_committed_output(
            StructuredProfilesOutput.model_validate(
                {"profiles": [{"content": "Fact", "source_span": quote, **permission}]}
            ),
            None,
        ),
    )
    client = MagicMock()
    worker = ExtractionResumeWorker(request_context=request_context, llm_client=client)
    client.reset_mock()  # Constructor's client-presence check is not inference.
    if field in {"expert_content", "shadow_content"} or (
        kind == "forget" and role != "User"
    ):
        with pytest.raises(ValueError, match="absent from original source"):
            worker._items_from_committed_output(run)
    else:
        items, _, _ = worker._items_from_committed_output(run)
        assert len(items) == 1
        assert items[0].source_interaction_ids == run.binding.source_interaction_ids
        if kind == "deadline":
            assert (
                metadata(items[0])["explicit_storage_deadline"]["source_quote"] == quote
            )
        else:
            assert metadata(items[0])["explicit_forget_quote"] == quote
    client.assert_not_called()
    assert client.mock_calls == []


@pytest.mark.parametrize("role", ["User", "Assistant", "Tool"])
@pytest.mark.parametrize(
    "field",
    [
        "content",
        "user_action_description",
        "tools_used[0]",
        "expert_content",
        "shadow_content",
    ],
)
def test_overflow_forget_requires_user_instruction_not_attached_context(role, field):
    quote, _ = _permission("forget")
    spans = (
        SourceSpan(1, "r", "User", "content", 0, 14, "A visible fact"),
        SourceSpan(2, "r", role, field, 0, len(quote), quote),
    )
    carried = _candidate_source_spans([_raw("forget")], "profiles", spans)
    facts = profile_source_evidence_texts([], visible_spans=carried)
    users = profile_forget_authority_texts([], visible_spans=carried)
    assert quote in facts
    output = StructuredProfilesOutput.model_validate({"profiles": [_raw("forget")]})
    if role == "User" and field in {"content", "user_action_description"}:
        assert not _profile_evidence_validation_errors(output, carried)
        assert _convert(_raw("forget"), facts, users)
    else:
        assert _profile_evidence_validation_errors(output, carried)
        with pytest.raises(ValueError, match="User instruction"):
            _convert(_raw("forget"), facts, users)


def test_assistant_span_cannot_complete_user_instruction_gap():
    quote, _ = _permission("forget")
    middle = len(quote) // 2
    spans = (
        SourceSpan(1, "r", "User", "content", 0, middle, quote[:middle]),
        SourceSpan(1, "r", "Assistant", "content", middle, len(quote), quote[middle:]),
    )
    assert quote in profile_source_evidence_texts([], visible_spans=spans)
    assert quote not in profile_forget_authority_texts([], visible_spans=spans)
