"""Upgrade compatibility consumes saved receipts without new authoring authority."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.service_schemas import ProfileTimeToLive
from reflexio.server.services.extraction.recovery import _validate_saved_output
from reflexio.server.services.profile import lifecycle
from reflexio.server.services.profile.committed_output import (
    read_saved_structured_output,
)
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
    StructuredProfilesOutput,
    calculate_expiration_timestamp,
)
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)

T0 = 1700000000


def saved_run(payload, **kwargs):
    return AgentRunRecord(
        id="saved",
        binding=AgentBinding("org", "profile", "user", "request", None, None, [1]),
        status=AgentRunStatus.AGENT_COMPLETED,
        generation_request_snapshot={},
        committed_output=payload,
        agent_completed_at=datetime.fromtimestamp(T0, UTC),
        **kwargs,
    )


def convert(payload, run=None, **kwargs):
    output = read_saved_structured_output(StructuredProfilesOutput, payload)
    return ProfileExtractor._convert_raw_to_user_profiles(
        [item.model_dump() for item in output.profiles or []],
        "user",
        "request",
        [1],
        committed_run=run,
        **kwargs,
    )


@pytest.mark.parametrize("ttl", list(ProfileTimeToLive))
def test_saved_ttl_preserves_receipt_and_visibility_without_authority(ttl):
    payload = {
        "profiles": [
            {
                "content": "likes tea",
                "time_to_live": ttl.value,
                "source_span": "tea",
                "notes": "source note",
                "reader_angle": "preference",
            }
        ]
    }
    before = json.dumps(payload)
    decoded = read_saved_structured_output(StructuredProfilesOutput, payload)
    assert isinstance(decoded, StructuredProfilesOutput)
    assert decoded.model_dump(mode="json") == payload
    _validate_saved_output(StructuredProfilesOutput, payload)
    run = saved_run(payload)
    (first,) = convert(payload, run)
    (second,) = convert(payload, replace(run, updated_at=datetime.now(UTC)))
    expected = calculate_expiration_timestamp(T0, ttl)
    assert first.expiration_timestamp == second.expiration_timestamp == expected
    assert first.last_modified_timestamp == second.last_modified_timestamp == T0
    assert first.source_interaction_ids == [1]
    assert (first.source_span, first.notes, first.reader_angle) == (
        "tea",
        "source note",
        "preference",
    )
    metadata = lifecycle.metadata(first)
    assert metadata["created_at"] is None
    assert metadata["legacy_origin"] is True
    assert metadata["explicit_storage_deadline"] is None
    assert metadata["explicit_forget_quote"] is None
    assert lifecycle.attention_epoch(first, now=T0 + 900) == T0 + 900
    assert json.dumps(payload) == before
    with pytest.raises(ValidationError, match="estimated lifetime"):
        StructuredProfilesOutput.model_validate(payload)
    assert "time_to_live" not in ProfileAddItem.model_json_schema()["properties"]


@pytest.mark.parametrize("ttl", [None, "invalid", 42, {}, []])
def test_corrupt_saved_ttl_is_invalid(ttl):
    with pytest.raises(ValidationError):
        read_saved_structured_output(
            StructuredProfilesOutput,
            {"profiles": [{"content": "tea", "time_to_live": ttl}]},
        )


def test_missing_collection_and_unknown_fields_remain_invalid():
    for payload in (
        {},
        {"profiles": None},
        {"profiles": [], "unknown": 1},
        {"profiles": [{"content": "tea", "time_to_live": "one_day", "unknown": 1}]},
    ):
        with pytest.raises(ValueError):
            _validate_saved_output(StructuredProfilesOutput, payload)


def test_custom_data_preserved_without_dropping_fields():
    payload = {
        "profiles": [{"content": "tea", "time_to_live": "one_day", "category": "food"}]
    }
    before = deepcopy(payload)
    (profile,) = convert(payload, saved_run(payload))
    assert profile.custom_features is not None
    assert profile.custom_features["category"] == "food"
    assert payload == before


def test_legacy_clock_requires_recorded_completion_or_preparation():
    payload = {"profiles": [{"content": "tea", "time_to_live": "one_day"}]}
    run = replace(
        saved_run(payload),
        agent_completed_at=None,
        created_at=datetime.fromtimestamp(T0 - 500, UTC),
    )
    (profile,) = convert(payload, run)
    assert profile.last_modified_timestamp == T0 - 500
    with pytest.raises(ValueError, match="recorded clock"):
        convert(payload, replace(run, created_at=None))
    with pytest.raises(ValueError, match="guessed TTL"):
        convert(payload)


@pytest.mark.parametrize(
    "authority",
    [
        {"explicit_forget_quote": "forget tea"},
        {
            "explicit_storage_deadline": {
                "deadline_utc": "2024-01-01T00:00:00Z",
                "source_quote": "forget tea",
            }
        },
    ],
)
def test_legacy_extra_fields_cannot_acquire_new_storage_authority(authority):
    payload = {"profiles": [{"content": "tea", "time_to_live": "one_day", **authority}]}
    with pytest.raises(ValidationError, match="cannot acquire storage authority"):
        convert(payload, saved_run(payload), source_texts=["forget tea"])


def test_modern_saved_permissions_retain_strict_evidence_validation():
    payload = {"profiles": [{"content": "tea", "explicit_forget_quote": "forget tea"}]}
    with pytest.raises(ValueError, match="absent from original source"):
        convert(payload, saved_run(payload))
    (profile,) = convert(payload, saved_run(payload), source_texts=["forget tea"])
    assert lifecycle.metadata(profile)["explicit_forget_quote"] == "forget tea"


def test_legacy_payload_cannot_author_lifecycle_clocks_or_change_scope():
    payload = {
        "profiles": [
            {
                "content": "tea",
                "time_to_live": "one_day",
                lifecycle.KEY: {"created_at": T0},
            }
        ]
    }
    with pytest.raises(ValueError, match="cannot acquire storage authority"):
        convert(payload, saved_run(payload))
    clean = {"profiles": [{"content": "tea", "time_to_live": "one_day"}]}
    run = saved_run(clean)
    with pytest.raises(ValueError, match="scope changed"):
        convert(clean, replace(run, binding=replace(run.binding, user_id="other")))
    with pytest.raises(ValueError, match="absent from committed"):
        convert(clean, saved_run({"profiles": []}))


def test_extractor_run_carries_saved_execution_clock_through_actual_raw_generation(
    monkeypatch,
):
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import MagicMock

    from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
    from reflexio.models.api_schema.service_schemas import Interaction, Request
    from reflexio.models.config_schema import ProfileExtractorConfig
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.tools import ToolLoopTrace
    from reflexio.server.services.extraction.outcome import ExtractionOutcome
    from reflexio.server.services.extraction.resumable_agent import AgentRunResult
    from reflexio.server.services.profile.components import extractor as module
    from reflexio.server.services.profile.service import ProfileGenerationServiceConfig

    payload = {"profiles": [{"content": "tea", "time_to_live": "one_day"}]}
    run = saved_run(payload)
    extractor = object.__new__(ProfileExtractor)
    extractor.request_context = cast(
        RequestContext, SimpleNamespace(prompt_manager=MagicMock())
    )
    extractor.client = MagicMock()
    extractor.config = ProfileExtractorConfig(
        extraction_definition_prompt="Preferences"
    )
    extractor.service_config = cast(
        ProfileGenerationServiceConfig,
        SimpleNamespace(
            existing_data=[], user_id="user", request_id="request", source=None
        ),
    )
    extractor.agent_context = "test"
    extractor.default_generation_model_name = "mock"
    models = [
        RequestInteractionDataModel(
            session_id="session",
            request=Request(request_id="request", user_id="user", session_id="session"),
            interactions=[
                Interaction(
                    interaction_id=1,
                    request_id="request",
                    user_id="user",
                    content="tea",
                )
            ],
        )
    ]
    monkeypatch.setattr(extractor, "_get_interactions", lambda: models)
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setattr(
        module,
        "prepare_profile_extraction_prompt",
        lambda **_: SimpleNamespace(messages=[], history_inputs=None),
    )
    cached = AgentRunResult(
        run_id=run.id,
        output=read_saved_structured_output(StructuredProfilesOutput, payload),
        pending_tool_call_ids=[],
        messages=[],
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        execution_run=run,
    )
    monkeypatch.setattr(module, "run_resumable_extraction_agent", lambda **_: cached)
    outcome = extractor.run()
    assert isinstance(outcome, ExtractionOutcome)
    assert extractor._last_resumable_run is run
    assert outcome.execution_run is run
    assert outcome.items[0].last_modified_timestamp == T0
    assert outcome.items[0].expiration_timestamp == T0 + 86400
    extractor.client.generate_chat_response_with_provenance.assert_not_called()
