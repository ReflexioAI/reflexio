"""Tests for View converter functions in api_schema.ui.converters.

Verifies that each converter correctly strips internal fields (embedding,
image_encoding, embedding_text) while preserving all user-facing fields.
"""

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    BlockingIssue,
    Interaction,
    ProfileChangeLog,
    StructuredData,
    ToolUsed,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import (
    BlockingIssueKind,
    PlaybookStatus,
    ProfileTimeToLive,
    RegularVsShadow,
    Status,
    UserActionType,
)
from reflexio.models.api_schema.ui.converters import (
    to_agent_playbook_view,
    to_evaluation_result_view,
    to_interaction_view,
    to_profile_change_log_view,
    to_profile_view,
    to_structured_data_view,
    to_user_playbook_view,
)
from reflexio.models.api_schema.ui.entities import (
    AgentPlaybookView,
    EvaluationResultView,
    InteractionView,
    ProfileChangeLogView,
    ProfileView,
    StructuredDataView,
    UserPlaybookView,
)

_FAKE_EMBEDDING = [0.1] * 512


def _make_structured_data() -> StructuredData:
    return StructuredData(
        rationale="test rationale",
        trigger="test trigger",
        instruction="test instruction",
        pitfall="test pitfall",
        blocking_issue=BlockingIssue(
            kind=BlockingIssueKind.MISSING_TOOL,
            details="missing foobar tool",
        ),
        embedding_text="internal embedding text",
    )


class TestToStructuredDataView:
    """to_structured_data_view: strips embedding_text, preserves all 5 visible fields."""

    def test_strips_embedding_text(self) -> None:
        sd = _make_structured_data()
        view = to_structured_data_view(sd)
        assert isinstance(view, StructuredDataView)
        assert "embedding_text" not in StructuredDataView.model_fields

    def test_preserves_all_visible_fields(self) -> None:
        sd = _make_structured_data()
        view = to_structured_data_view(sd)
        assert view.rationale == "test rationale"
        assert view.trigger == "test trigger"
        assert view.instruction == "test instruction"
        assert view.pitfall == "test pitfall"
        assert view.blocking_issue is not None
        assert view.blocking_issue.kind == BlockingIssueKind.MISSING_TOOL
        assert view.blocking_issue.details == "missing foobar tool"

    def test_none_fields_preserved(self) -> None:
        sd = StructuredData()
        view = to_structured_data_view(sd)
        assert view.rationale is None
        assert view.trigger is None
        assert view.instruction is None
        assert view.pitfall is None
        assert view.blocking_issue is None


class TestToInteractionView:
    """to_interaction_view: strips embedding and image_encoding, preserves all other fields."""

    def test_strips_embedding_and_image_encoding(self) -> None:
        interaction = Interaction(
            interaction_id=42,
            user_id="user1",
            request_id="req1",
            created_at=1000000,
            role="User",
            content="hello",
            user_action=UserActionType.CLICK,
            user_action_description="clicked button",
            interacted_image_url="",
            image_encoding="base64data",
            shadow_content="shadow",
            tools_used=[
                ToolUsed(tool_name="search", tool_data={"input": {"q": "test"}})
            ],
            embedding=_FAKE_EMBEDDING,
        )
        view = to_interaction_view(interaction)
        assert isinstance(view, InteractionView)
        assert "embedding" not in InteractionView.model_fields
        assert "image_encoding" not in InteractionView.model_fields

    def test_preserves_all_other_fields(self) -> None:
        tools = [ToolUsed(tool_name="search", tool_data={"input": {"q": "test"}})]
        interaction = Interaction(
            interaction_id=42,
            user_id="user1",
            request_id="req1",
            created_at=1000000,
            role="Assistant",
            content="hello world",
            user_action=UserActionType.CLICK,
            user_action_description="clicked button",
            interacted_image_url="",
            image_encoding="base64data",
            shadow_content="shadow text",
            tools_used=tools,
            embedding=_FAKE_EMBEDDING,
        )
        view = to_interaction_view(interaction)
        assert view.interaction_id == 42
        assert view.user_id == "user1"
        assert view.request_id == "req1"
        assert view.created_at == 1000000
        assert view.role == "Assistant"
        assert view.content == "hello world"
        assert view.user_action == UserActionType.CLICK
        assert view.user_action_description == "clicked button"
        assert view.interacted_image_url == ""
        assert view.shadow_content == "shadow text"
        assert len(view.tools_used) == 1
        assert view.tools_used[0].tool_name == "search"


class TestToProfileView:
    """to_profile_view: strips embedding, preserves all other fields."""

    def test_strips_embedding(self) -> None:
        profile = UserProfile(
            profile_id="p1",
            user_id="user1",
            content="content",
            last_modified_timestamp=1000000,
            generated_from_request_id="req1",
            profile_time_to_live=ProfileTimeToLive.ONE_WEEK,
            custom_features={"key": "value"},
            source="test",
            status=Status.CURRENT,
            extractor_names=["ext1"],
            embedding=_FAKE_EMBEDDING,
        )
        view = to_profile_view(profile)
        assert isinstance(view, ProfileView)
        assert "embedding" not in ProfileView.model_fields

    def test_preserves_all_other_fields(self) -> None:
        profile = UserProfile(
            profile_id="p1",
            user_id="user1",
            content="likes cats",
            last_modified_timestamp=1000000,
            generated_from_request_id="req1",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
            expiration_timestamp=2000000,
            custom_features={"pref": "dark"},
            source="chat",
            status=Status.PENDING,
            extractor_names=["ext1", "ext2"],
            embedding=_FAKE_EMBEDDING,
        )
        view = to_profile_view(profile)
        assert view.profile_id == "p1"
        assert view.user_id == "user1"
        assert view.content == "likes cats"
        assert view.last_modified_timestamp == 1000000
        assert view.generated_from_request_id == "req1"
        assert view.profile_time_to_live == ProfileTimeToLive.ONE_MONTH
        assert view.expiration_timestamp == 2000000
        assert view.custom_features == {"pref": "dark"}
        assert view.source == "chat"
        assert view.status == Status.PENDING
        assert view.extractor_names == ["ext1", "ext2"]


class TestToUserPlaybookView:
    """to_user_playbook_view: strips embedding, converts structured_data to StructuredDataView."""

    def test_strips_embedding(self) -> None:
        rf = UserPlaybook(
            user_playbook_id=10,
            user_id="user1",
            agent_version="v1",
            request_id="req1",
            playbook_name="fb",
            created_at=1000000,
            content="content",
            structured_data=_make_structured_data(),
            status=Status.CURRENT,
            source="test",
            source_interaction_ids=[1, 2],
            embedding=_FAKE_EMBEDDING,
        )
        view = to_user_playbook_view(rf)
        assert isinstance(view, UserPlaybookView)
        assert "embedding" not in UserPlaybookView.model_fields

    def test_converts_structured_data_to_view(self) -> None:
        rf = UserPlaybook(
            user_playbook_id=10,
            user_id="user1",
            agent_version="v1",
            request_id="req1",
            structured_data=_make_structured_data(),
            embedding=_FAKE_EMBEDDING,
        )
        view = to_user_playbook_view(rf)
        assert isinstance(view.structured_data, StructuredDataView)
        assert view.structured_data.rationale == "test rationale"
        assert view.structured_data.trigger == "test trigger"
        assert view.structured_data.instruction == "test instruction"
        assert view.structured_data.pitfall == "test pitfall"
        assert "embedding_text" not in StructuredDataView.model_fields

    def test_preserves_all_other_fields(self) -> None:
        rf = UserPlaybook(
            user_playbook_id=10,
            user_id="user1",
            agent_version="v1",
            request_id="req1",
            playbook_name="my_playbook",
            created_at=1000000,
            content="some content",
            structured_data=_make_structured_data(),
            status=Status.ARCHIVED,
            source="web",
            source_interaction_ids=[3, 4, 5],
            embedding=_FAKE_EMBEDDING,
        )
        view = to_user_playbook_view(rf)
        assert view.user_playbook_id == 10
        assert view.user_id == "user1"
        assert view.agent_version == "v1"
        assert view.request_id == "req1"
        assert view.playbook_name == "my_playbook"
        assert view.created_at == 1000000
        assert view.content == "some content"
        assert view.status == Status.ARCHIVED
        assert view.source == "web"
        assert view.source_interaction_ids == [3, 4, 5]


class TestToAgentPlaybookView:
    """to_agent_playbook_view: strips embedding, converts structured_data to StructuredDataView."""

    def test_strips_embedding(self) -> None:
        fb = AgentPlaybook(
            agent_playbook_id=20,
            playbook_name="fb",
            agent_version="v1",
            created_at=1000000,
            content="content",
            structured_data=_make_structured_data(),
            playbook_status=PlaybookStatus.APPROVED,
            playbook_metadata="meta",
            embedding=_FAKE_EMBEDDING,
            status=Status.CURRENT,
        )
        view = to_agent_playbook_view(fb)
        assert isinstance(view, AgentPlaybookView)
        assert "embedding" not in AgentPlaybookView.model_fields

    def test_converts_structured_data_to_view(self) -> None:
        fb = AgentPlaybook(
            agent_playbook_id=20,
            agent_version="v1",
            content="content",
            structured_data=_make_structured_data(),
            embedding=_FAKE_EMBEDDING,
        )
        view = to_agent_playbook_view(fb)
        assert isinstance(view.structured_data, StructuredDataView)
        assert view.structured_data.instruction == "test instruction"
        assert "embedding_text" not in StructuredDataView.model_fields

    def test_preserves_all_other_fields(self) -> None:
        fb = AgentPlaybook(
            agent_playbook_id=20,
            playbook_name="my_fb",
            agent_version="v2",
            created_at=1000000,
            content="important playbook",
            structured_data=_make_structured_data(),
            playbook_status=PlaybookStatus.REJECTED,
            playbook_metadata="some meta",
            embedding=_FAKE_EMBEDDING,
            status=Status.ARCHIVED,
        )
        view = to_agent_playbook_view(fb)
        assert view.agent_playbook_id == 20
        assert view.playbook_name == "my_fb"
        assert view.agent_version == "v2"
        assert view.created_at == 1000000
        assert view.content == "important playbook"
        assert view.playbook_status == PlaybookStatus.REJECTED
        assert view.playbook_metadata == "some meta"
        assert view.status == Status.ARCHIVED


class TestToEvaluationResultView:
    """to_evaluation_result_view: strips embedding, preserves all other fields."""

    def test_strips_embedding(self) -> None:
        result = AgentSuccessEvaluationResult(
            result_id=5,
            agent_version="v1",
            session_id="sess1",
            is_success=True,
            embedding=_FAKE_EMBEDDING,
        )
        view = to_evaluation_result_view(result)
        assert isinstance(view, EvaluationResultView)
        assert "embedding" not in EvaluationResultView.model_fields

    def test_preserves_all_other_fields(self) -> None:
        result = AgentSuccessEvaluationResult(
            result_id=5,
            agent_version="v1",
            session_id="sess1",
            is_success=False,
            failure_type="timeout",
            failure_reason="took too long",
            evaluation_name="speed_eval",
            created_at=1000000,
            regular_vs_shadow=RegularVsShadow.SHADOW_IS_BETTER,
            number_of_correction_per_session=3,
            user_turns_to_resolution=7,
            is_escalated=True,
            embedding=_FAKE_EMBEDDING,
        )
        view = to_evaluation_result_view(result)
        assert view.result_id == 5
        assert view.agent_version == "v1"
        assert view.session_id == "sess1"
        assert view.is_success is False
        assert view.failure_type == "timeout"
        assert view.failure_reason == "took too long"
        assert view.evaluation_name == "speed_eval"
        assert view.created_at == 1000000
        assert view.regular_vs_shadow == RegularVsShadow.SHADOW_IS_BETTER
        assert view.number_of_correction_per_session == 3
        assert view.user_turns_to_resolution == 7
        assert view.is_escalated is True


class TestToProfileChangeLogView:
    """to_profile_change_log_view: converts nested profiles to ProfileView lists."""

    def _make_profile(self, profile_id: str) -> UserProfile:
        return UserProfile(
            profile_id=profile_id,
            user_id="user1",
            content=f"content for {profile_id}",
            last_modified_timestamp=1000000,
            generated_from_request_id="req1",
            embedding=_FAKE_EMBEDDING,
        )

    def test_nested_profiles_converted_to_profile_view(self) -> None:
        log = ProfileChangeLog(
            id=1,
            user_id="user1",
            request_id="req1",
            created_at=1000000,
            added_profiles=[self._make_profile("p1")],
            removed_profiles=[self._make_profile("p2")],
            mentioned_profiles=[self._make_profile("p3")],
        )
        view = to_profile_change_log_view(log)
        assert isinstance(view, ProfileChangeLogView)
        assert len(view.added_profiles) == 1
        assert len(view.removed_profiles) == 1
        assert len(view.mentioned_profiles) == 1
        for pv in [
            *view.added_profiles,
            *view.removed_profiles,
            *view.mentioned_profiles,
        ]:
            assert isinstance(pv, ProfileView)
            assert "embedding" not in ProfileView.model_fields

    def test_preserves_top_level_fields(self) -> None:
        log = ProfileChangeLog(
            id=42,
            user_id="user1",
            request_id="req99",
            created_at=1000000,
            added_profiles=[self._make_profile("p1"), self._make_profile("p2")],
            removed_profiles=[],
            mentioned_profiles=[self._make_profile("p3")],
        )
        view = to_profile_change_log_view(log)
        assert view.id == 42
        assert view.user_id == "user1"
        assert view.request_id == "req99"
        assert view.created_at == 1000000
        assert len(view.added_profiles) == 2
        assert len(view.removed_profiles) == 0
        assert len(view.mentioned_profiles) == 1

    def test_profile_content_preserved(self) -> None:
        log = ProfileChangeLog(
            id=1,
            user_id="user1",
            request_id="req1",
            created_at=1000000,
            added_profiles=[self._make_profile("p1")],
            removed_profiles=[],
            mentioned_profiles=[],
        )
        view = to_profile_change_log_view(log)
        assert view.added_profiles[0].profile_id == "p1"
        assert view.added_profiles[0].content == "content for p1"
