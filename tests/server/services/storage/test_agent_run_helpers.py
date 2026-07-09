from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.storage.storage_base import (
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    canonical_json,
    human_feedback_scope,
    normalize_dedup_text,
)


def test_scope_hash_uses_canonical_json_ordering():
    left = {"scope_kind": "org", "org_id": "org_1"}
    right = {"org_id": "org_1", "scope_kind": "org"}

    assert canonical_json(left) == canonical_json(right)
    assert build_scope_hash(left) == build_scope_hash(right)


def test_human_feedback_scope_never_includes_user_id():
    scope = human_feedback_scope("org_1")

    assert scope == {"org_id": "org_1", "scope_kind": "org"}
    assert "user_id" not in scope


def test_dedup_key_normalizes_case_unicode_and_whitespace():
    key_a = build_pending_tool_call_dedup_key(
        tool_name="Ask_Human",
        question_text="  What\u00a0is   the user's deployment target? ",
        answer_format=" Plain Text ",
    )
    key_b = build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="what is the user's deployment target?",
        answer_format="plain text",
    )

    assert key_a == key_b


def test_missing_answer_format_normalizes_to_empty_string():
    assert normalize_dedup_text(None) == ""
    assert build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="Need deployment target?",
        answer_format=None,
    ) == build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="Need deployment target?",
        answer_format="",
    )


class _ExtractorConfig(BaseModel):
    extraction_definition_prompt: str = "Extract deployment preferences."


def test_build_extractor_agent_run_record_maps_generation_request_id_to_legacy_binding_field():
    request = Request(
        request_id="req_source_1",
        user_id="user_1",
        source="api",
        agent_version="v1",
        session_id="session_1",
    )
    interaction = Interaction(
        interaction_id=42,
        user_id="user_1",
        request_id="req_source_1",
        content="Remember that I deploy to ECS.",
    )
    request_interaction_data_models = [
        RequestInteractionDataModel(
            session_id="session_1",
            request=request,
            interactions=[interaction],
        )
    ]

    run = build_extractor_agent_run_record(
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        generation_request_id="rerun_ab12cd34",
        agent_version="v1",
        source="api",
        request_interaction_data_models=request_interaction_data_models,
        extractor_config=_ExtractorConfig(),
        service_config={"request_id": "rerun_ab12cd34"},
        agent_context="context",
    )

    assert run.binding.request_id == "rerun_ab12cd34"
    assert run.generation_request_snapshot["request_id"] == "rerun_ab12cd34"
    assert run.id.startswith("ar_")


def test_build_extractor_agent_run_record_accepts_legacy_request_id_keyword():
    request = Request(
        request_id="req_source_1",
        user_id="user_1",
        source="api",
        agent_version="v1",
        session_id="session_1",
    )
    interaction = Interaction(
        interaction_id=42,
        user_id="user_1",
        request_id="req_source_1",
        content="Remember that I deploy to ECS.",
    )
    request_interaction_data_models = [
        RequestInteractionDataModel(
            session_id="session_1",
            request=request,
            interactions=[interaction],
        )
    ]

    run = build_extractor_agent_run_record(
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        request_id="legacy_req",
        agent_version="v1",
        source="api",
        request_interaction_data_models=request_interaction_data_models,
        extractor_config=_ExtractorConfig(),
        service_config={"request_id": "legacy_req"},
        agent_context="context",
    )

    assert run.binding.request_id == "legacy_req"
    assert run.generation_request_snapshot["request_id"] == "legacy_req"
