"""Integration tests for PlaybookGenerationService."""

import contextlib
import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import litellm
import pytest

pytestmark = pytest.mark.integration


# Disable global mock mode for the message construction test so LLM client mock is used
@pytest.fixture
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so tests use their own mocks."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    LLMConfig,
    PendingToolCallConfig,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.playbook.components.reviewer import (
    CandidateReviewDecision,
    PlaybookCandidateReviewOutput,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
    StructuredPlaybookContent,
    StructuredPlaybookEvidence,
    StructuredPlaybookList,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
)
from tests.server.test_utils import skip_in_precommit


def create_request_interaction_data_model(
    user_id: str,
    request_id: str,
    interactions: list[Interaction],
    agent_version: str = "test_agent_1",
) -> RequestInteractionDataModel:
    """Helper function to create a RequestInteractionDataModel for testing."""
    request = Request(
        request_id=request_id,
        user_id=user_id,
        source="test",
        agent_version=agent_version,
        session_id="session_1",
        created_at=1000,
    )
    return RequestInteractionDataModel(
        session_id="session_1",
        request=request,
        interactions=interactions,
    )


@pytest.fixture
def mock_request_context(tmp_path, monkeypatch):
    """Create a mock request context with necessary components."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    context = MagicMock(spec=RequestContext)
    context.org_id = "test_org_123"
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_a, **_k: [0.0] * 512)
    real_storage = SQLiteStorage(
        org_id=context.org_id, db_path=str(tmp_path / "runs.db")
    )
    # One actual storage identity owns every lease and writer. Spy on individual
    # domain methods; wrapping the whole storage creates a different target when
    # shared SQL methods call their actual self internally.
    context.storage = real_storage
    for name in (
        "get_operation_state",
        "claim_due_playbook_aggregation",
        "try_acquire_in_progress_lock",
        "get_user_playbooks",
        "search_user_playbooks",
        "governance_subject_ref_for_user",
        "get_last_k_interactions_grouped",
        "get_interactions_by_ids",
        "get_request",
        "save_user_playbooks",
    ):
        monkeypatch.setattr(
            real_storage, name, MagicMock(wraps=getattr(real_storage, name))
        )
    # Mock get_operation_state to return None by default (no in-progress state)
    context.storage.get_operation_state.return_value = None
    context.storage.claim_due_playbook_aggregation.return_value = None
    # Mock try_acquire_in_progress_lock to return success
    context.storage.try_acquire_in_progress_lock.return_value = {"acquired": True}
    # Mock get_user_playbooks to return empty list (for existing playbooks check)
    context.storage.get_user_playbooks.return_value = []
    context.storage.search_user_playbooks.return_value = []
    context.storage.governance_subject_ref_for_user.return_value = "subject-test-user"
    # Mock get_last_k_interactions_grouped to return empty by default
    context.storage.get_last_k_interactions_grouped.return_value = ([], [])
    context.configurator = MagicMock()
    context.configurator.get_agent_context.return_value = "Synthetic agent context"
    context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "runs.db")),
        profile_extractor_config=None,
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Test playbook definition",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
        llm_config=LLMConfig(generation_model_name="gpt-4o-mini"),
        pending_tool_call_config=PendingToolCallConfig(),
        playbook_window_size=100,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    # Mock window_size for extractor
    context.configurator.get_config.return_value.playbook_window_size = 100
    # Pin the pending-tool-call config to the real default (disabled). Without
    # this, the auto-generated MagicMock makes the resumable-agent path think
    # pending tools are enabled and feeds a MagicMock similarity_threshold into
    # a numeric comparison (prior_answer_search), raising TypeError.
    context.configurator.get_config.return_value.pending_tool_call_config = (
        PendingToolCallConfig()
    )
    context.prompt_manager = PromptManager()
    try:
        yield context
    finally:
        from reflexio.server.services.playbook import aggregation_scheduler

        # The session autouse cleanup runs after fixture teardown. This fixture
        # owns the DB, so join its scheduler before the driver can be closed.
        with aggregation_scheduler._LOCAL_SCHEDULERS_LOCK:
            scheduler = aggregation_scheduler._LOCAL_SCHEDULERS.pop(
                context.org_id, None
            )
        if scheduler is not None:
            scheduler.stop()
            assert not scheduler.is_running(), "test-owned scheduler did not stop"
        with real_storage._lock:
            real_storage.conn.close()


@pytest.fixture
def playbook_generation_service(mock_request_context):
    """Create a PlaybookGenerationService instance with mocked dependencies."""
    mock_client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini", fallback_models=[]))
    service = PlaybookGenerationService(
        llm_client=mock_client, request_context=mock_request_context
    )
    return service  # noqa: RET504


@pytest.fixture
def test_interactions(mock_request_context):
    """Create test interactions for playbook generation."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user_123",
            request_id="test_request_1",
            content="I need help with my account",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="Clicked help button",
            interacted_image_url="https://example.com/help",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user_123",
            request_id="test_request_1",
            content="Thank you for your help!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="Clicked thank you button",
            interacted_image_url="https://example.com/thank-you",
        ),
    ]

    source = create_request_interaction_data_model(
        "test_user_123", "test_request_1", interactions
    )
    mock_request_context.storage.add_request(source.request)
    for row in interactions:
        mock_request_context.storage._insert_interaction(row)
    return interactions


def _setup_mock_chat_completion(
    service,
    should_generate=True,
    content="The agent was helpful and provided accurate information",
):
    """Helper function to set up mock chat completion responses.

    Playbook extraction now delivers its result as a native structured
    response: the loop calls ``generate_chat_response`` with ``response_format=``
    and reads the parsed output model from a plain (no-tool) turn. So the
    extraction turn returns the parsed ``StructuredPlaybookList`` directly. The
    boolean should-generate gate still returns a plain string.
    """

    def mock_generate_chat_response(messages, **kwargs):
        """Route on prompt content / extraction mode to the right mock response."""
        response_format = kwargs.get("response_format")
        if len(messages) >= 2:
            try:
                payload = json.loads(messages[1]["content"])
            except (ValueError, TypeError, KeyError):
                payload = None
            if (
                isinstance(payload, dict)
                and {"new", "existing", "sessions"} <= payload.keys()
            ):
                if "proposed_memory" in payload:
                    return json.dumps({"findings": []})
                return json.dumps(
                    {
                        "decisions": [
                            {
                                "kind": "create",
                                "new_ids": [row["id"]],
                                "reason": "Preserve the supported candidate",
                                "outputs": [
                                    {
                                        key: row[key]
                                        for key in (
                                            "trigger",
                                            "content",
                                            "rationale",
                                            "evidence_refs",
                                        )
                                    }
                                ],
                            }
                            for row in payload["new"]
                        ]
                    }
                )
        if response_format is PlaybookCandidateReviewOutput:
            return PlaybookCandidateReviewOutput(
                decisions=[
                    CandidateReviewDecision(
                        id="C1",
                        decision="accept",
                        reason_code="grounded_useful",
                        evidence_ids=["C1-E1"],
                    )
                ]
            )
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_generate_playbook call
        # Support both old and new prompt formats
        if (
            "Output just a boolean value" in prompt_content
            or "Return only true or false" in prompt_content
        ):
            return "true" if should_generate else "false"
        # Otherwise this is a playbook extraction turn: the loop requests a
        # response_format, so return the parsed structured output directly
        # (the structured-response terminus).
        return StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    trigger="interacting with users",
                    content=content,
                    rationale="The user's explicit acceptance applies to future support interactions and supports reusing the successful response behavior.",
                    evidence_kind="verified-success",
                    future_task_class="support interactions",
                    improvement_mechanism="reuses behavior the user explicitly accepted",
                    reader_angle="verified-success",
                    evidence=[
                        StructuredPlaybookEvidence(
                            turn_ref="T2",
                            source_span="Thank you for your help!",
                        )
                    ],
                )
            ]
        )

    service.client.generate_chat_response = MagicMock(
        side_effect=mock_generate_chat_response
    )
    service.client.generate_chat_response_with_provenance = MagicMock(
        side_effect=lambda *args, **kwargs: CompletionResult(
            mock_generate_chat_response(*args, **kwargs), ModelProvenance()
        )
    )


def _setup_procedural_wire(service, *, noop=False):
    """Script the current source/compiler/full-result model return contracts."""
    calls = []

    def wire(params, _timeout):
        calls.append(params)
        body = json.loads(params["messages"][1]["content"])
        names = {tool["function"]["name"] for tool in params["tools"]}
        if "record_observation" in names:
            if noop:
                name, value = (
                    "finish_batch",
                    {"status": "complete", "reason": "No reusable change selected."},
                )
            else:
                evidence = body["sources"][-1]["segments"][0]["excerpts"][0]["id"]
                name, value = (
                    "record_observation",
                    {
                        "topic": "Account support",
                        "evidence": evidence,
                        "observation": "The user accepted the assistance.",
                    },
                )
        elif "create_procedure" in names:
            name, value = (
                "create_procedure",
                {
                    "group": "support",
                    "name": "account-support",
                    "trigger": "interacting with users",
                    "entry": "Use accurate information when assisting with account issues.",
                    "evidence": body["source_observations"][0]["evidence"],
                },
            )
        elif "review_result" in names:
            name, value = (
                "review_result",
                {
                    "unit": "U0",
                    "decision": "accept",
                    "result_support": "pass",
                    "scope_conditions": "pass",
                    "useful_nonduplicate": "pass",
                    "preservation_and_correction": "pass",
                    "resource_consistency": "pass",
                    "reason": "The complete output is supported.",
                },
            )
        else:
            raise AssertionError(f"Unexpected retired semantic call: {names}")
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"return-{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(value),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    service.client._completion_with_hard_timeout = MagicMock(side_effect=wire)
    return calls


def _ordinary_source(context, interactions):
    # Keep textual action metadata; only the independent visual input is absent.
    rows = [row.model_copy(update={"interacted_image_url": ""}) for row in interactions]
    context.storage._execute(
        "UPDATE interactions SET interacted_image_url='' WHERE user_id=?",
        ("test_user_123",),
    )
    group = create_request_interaction_data_model(
        "test_user_123", "test_request_1", rows
    )
    context.storage.get_last_k_interactions_grouped.return_value = ([group], rows)
    context.storage.get_interactions_by_ids.side_effect = lambda ids: [
        row for row in rows if row.interaction_id in ids
    ]
    context.storage.get_request.return_value = group.request
    return group


def test_playbook_generation_with_storage(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    from reflexio.server.services.playbook.procedural import codec

    _ordinary_source(mock_request_context, test_interactions)
    calls = _setup_procedural_wire(playbook_generation_service)
    playbook_generation_service.run(
        PlaybookGenerationRequest(
            request_id="test_request_1",
            agent_version="test_agent_1",
            user_id="test_user_123",
            auto_run=False,
        )
    )
    mock_request_context.storage.save_user_playbooks.assert_called_once()
    row = mock_request_context.storage.save_user_playbooks.call_args[0][0][0]
    assert isinstance(row, UserPlaybook)
    assert row.agent_version == "test_agent_1" and row.request_id == "test_request_1"
    assert row.user_id == "test_user_123"
    assert row.trigger == "interacting with users"
    document = codec.decode(row.content)
    assert document is not None
    assert (
        document.pages[0].content
        == "Use accurate information when assisting with account issues."
    )
    assert row.source_interaction_ids == [2] and row.source_span
    assert len(calls) == 3


def test_playbook_generation_with_should_not_generate(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    _ordinary_source(mock_request_context, test_interactions)
    mock_request_context.configurator.get_config.return_value.playbook_stride_size = 1
    calls = _setup_procedural_wire(playbook_generation_service, noop=True)
    playbook_generation_service.run(
        PlaybookGenerationRequest(
            request_id="test_request_1",
            agent_version="test_agent_1",
            user_id="test_user_123",
            auto_run=True,
        )
    )
    assert len(calls) == 1
    assert "record_observation" in {
        tool["function"]["name"] for tool in calls[0]["tools"]
    }
    mock_request_context.storage.save_user_playbooks.assert_not_called()


def test_playbook_message_construction_with_interactions(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    _ordinary_source(mock_request_context, test_interactions)
    calls = _setup_procedural_wire(playbook_generation_service)
    playbook_generation_service.run(
        PlaybookGenerationRequest(
            request_id="test_request_1",
            agent_version="test_agent_1",
            user_id="test_user_123",
            auto_run=False,
        )
    )
    assert len(calls) == 3
    source = json.loads(calls[0]["messages"][1]["content"])
    assert source["owner"] == "test_user_123"
    assert "source_field_context" in source
    assert {row["interaction_id"] for row in source["sources"]} == {1, 2}
    text_by_id = {
        row["interaction_id"]: "\n\n".join(
            excerpt["text"]
            for segment in row["segments"]
            for excerpt in segment["excerpts"]
        )
        for row in source["sources"]
    }
    assert "I need help with my account" in text_by_id[1]
    assert "Thank you for your help!" in text_by_id[2]
    assert any(
        segment.get("field") == "native_user_action_display"
        for row in source["sources"]
        for segment in row["segments"]
    )
    assert all(
        row["request_id"] == "test_request_1" and row["owner"] == "test_user_123"
        for row in source["sources"]
    )


@skip_in_precommit
def test_multimodal_playbook_generation_with_storage(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    """Test playbook generation with mocked storage."""
    _setup_mock_chat_completion(playbook_generation_service)

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions when extractor calls get_last_k_interactions_grouped
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )
    mock_request_context.storage.get_interactions_by_ids.side_effect = (
        lambda interaction_ids: [
            interaction
            for interaction in test_interactions
            if interaction.interaction_id in interaction_ids
        ]
    )
    mock_request_context.storage.get_request.return_value = (
        request_interaction_data_model.request
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    playbook_generation_service.run(request)

    # Verify storage was called with correct playbook
    mock_request_context.storage.save_user_playbooks.assert_called_once()
    saved_playbooks = mock_request_context.storage.save_user_playbooks.call_args[0][0]

    assert len(saved_playbooks) == 1
    playbook = saved_playbooks[0]
    assert isinstance(playbook, UserPlaybook)
    assert playbook.agent_version == "test_agent_1"
    assert playbook.request_id == "test_request_1"
    # Verify top-level fields are populated
    assert playbook.trigger == "interacting with users"
    assert playbook.content == "The agent was helpful and provided accurate information"


@skip_in_precommit
def test_playbook_generation_with_empty_interactions(
    playbook_generation_service, mock_request_context
):
    """Test playbook generation with empty interactions."""
    # No need to set up mock since service should return early with empty interactions

    # Storage returns empty interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = ([], [])

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    playbook_generation_service.run(request)

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


@skip_in_precommit
def test_playbook_generation_with_no_playbook_config(
    playbook_generation_service, mock_request_context, test_interactions
):
    """Test playbook generation with no playbook config."""
    _setup_mock_chat_completion(playbook_generation_service)

    # Set empty playbook config
    mock_request_context.configurator.get_config.return_value.user_playbook_extractor_config = None

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    playbook_generation_service.run(request)

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


@skip_in_precommit
def test_multimodal_playbook_generation_with_should_not_generate(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    """Test playbook generation when should_generate_playbook returns false."""
    _setup_mock_chat_completion(playbook_generation_service, should_generate=False)

    # Exercise the automatic judge; manual runs deliberately bypass it.
    root_config = mock_request_context.configurator.get_config.return_value
    root_config.skip_should_run_check = False
    root_config.playbook_stride_size = 1

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=True,
    )

    playbook_generation_service.run(request)

    playbook_generation_service.client.generate_chat_response.assert_called_once()
    playbook_generation_service.client.generate_chat_response_with_provenance.assert_not_called()

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


def test_multimodal_playbook_message_construction_with_interactions(
    mock_request_context,
    disable_mock_llm_response,
):
    """Test that interactions are formatted correctly in rendered playbook prompts."""
    from unittest.mock import patch

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager

    # Create test interactions
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user_123",
            request_id="test_request_1",
            content="I need help with my account",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="help button",
            interacted_image_url="https://example.com/help",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user_123",
            request_id="test_request_1",
            content="Thank you for your help!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
    ]

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=interactions,
    )

    # Add mock prompt_manager to request context
    mock_request_context.prompt_manager = PromptManager()

    # Mock storage to return interactions when extractor calls get_last_k_interactions_grouped
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        interactions,
    )

    # Use a real LiteLLMClient
    llm_config = LiteLLMConfig(model="gpt-4o-mini")
    llm_client = LiteLLMClient(llm_config)
    service = PlaybookGenerationService(
        llm_client=llm_client, request_context=mock_request_context
    )

    # Capture the messages sent to generate_chat_response
    captured_messages = []

    def mock_generate_chat_response(messages, **kwargs):
        captured_messages.append(messages)
        # Return appropriate responses based on content
        # Check if this is should_generate or playbook extraction
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        if "Output just a boolean value" in prompt_content:
            return "true"
        # Return list-shaped output expected by the new extractor schema
        return StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    trigger="assisting users",
                    content="The agent was helpful",
                )
            ]
        )

    def mock_generate_with_provenance(*args, **kwargs):
        return CompletionResult(
            mock_generate_chat_response(*args, **kwargs), ModelProvenance()
        )

    with (
        patch(
            "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
            side_effect=mock_generate_chat_response,
        ),
        patch(
            "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response_with_provenance",
            side_effect=mock_generate_with_provenance,
        ),
    ):
        # Create playbook generation request with new API
        request = PlaybookGenerationRequest(
            request_id="test_request_1",
            agent_version="test_agent_1",
            user_id="test_user_123",
            auto_run=False,  # Skip stride check for testing
        )

        # Run playbook generation
        with contextlib.suppress(Exception):
            # We're just validating message construction, errors are ok
            service.run(request)

    # Validate that messages were captured
    assert len(captured_messages) > 0, "No messages were captured"

    # Find the message that contains the playbook_extraction_main prompt
    found_interactions_in_prompt = False
    for messages in captured_messages:
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                # Message content might be a list of dicts (multimodal) or a string
                content_str = ""
                if isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict) and "text" in item:
                            content_str += item["text"]
                else:
                    content_str = str(message["content"])

                # Check if this message contains interactions
                if any(
                    pattern in content_str
                    for pattern in [
                        "User: ```I need help",
                        "user: ```I need help",
                        "[Interaction",
                        "Session:",
                    ]
                ):
                    # Validate the interactions are formatted correctly in the rendered prompt
                    # Note: format might be "User:" (capital) or "user:" depending on the prompt
                    # Content is wrapped in backticks in the prompt template
                    has_interaction1 = (
                        "User: ```I need help with my account```" in content_str
                        or "user: ```I need help with my account```" in content_str
                    )
                    has_interaction2 = (
                        "User: ```Thank you for your help!```" in content_str
                        or "user: ```Thank you for your help!```" in content_str
                    )

                    if has_interaction1 and has_interaction2:
                        # Found both content interactions
                        found_interactions_in_prompt = True
                        break
        if found_interactions_in_prompt:
            break

    assert found_interactions_in_prompt, (
        "Did not find interactions in any rendered prompt"
    )
