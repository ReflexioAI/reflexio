from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    InteractionData,
    PublishUserInteractionRequest,
    UserActionType,
)
from reflexio.server.api_endpoints.precondition_checks import (
    validate_delete_user_profile_request,
    validate_publish_user_interaction_request,
)


def _make_publish_request(
    interactions: list[InteractionData] | None = None,
) -> PublishUserInteractionRequest:
    # Always ``model_construct``: these tests exercise the precondition guard
    # in isolation, and that guard exists specifically to cover callers that
    # bypass Pydantic. Building a validated request here would instead trip
    # ``PublishUserInteractionRequest``'s own validators (min_length=1, and the
    # contentless-interaction rule), so the guard itself would never be reached.
    return PublishUserInteractionRequest.model_construct(
        user_id="test-user",
        session_id="test-session",
        interaction_data_list=interactions if interactions is not None else [],
    )


class TestValidatePublishUserInteractionRequest:
    def test_empty_interaction_data_list(self):
        request = _make_publish_request(interactions=None)
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert msg == "No interaction data provided"

    def test_user_action_without_description(self):
        interaction = InteractionData(
            content="hello",
            user_action=UserActionType.CLICK,
            user_action_description="",
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert "user_action_description" in msg

    def test_empty_session_id(self):
        request = PublishUserInteractionRequest.model_construct(
            user_id="test-user",
            session_id="",
            interaction_data_list=[InteractionData(content="hello")],
        )
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert msg == "session_id is required and cannot be empty"

    def test_both_image_url_and_image_encoding(self):
        interaction = InteractionData(
            content="hello",
            interacted_image_url="https://example.com/image.png",
            image_encoding="base64data",
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert "cannot both be set" in msg

    def test_all_fields_empty_with_none_action_is_rejected(self):
        # Regression: this branch MUST be reachable. It previously could not
        # fire, because ``UserActionType`` is a ``StrEnum`` with NONE = "none"
        # — a truthy string — so the old ``not interaction_data.user_action``
        # test was always False and the whole "all empty" chain was dead code.
        # A publish of 50 such interactions was accepted with 200 and stored 50
        # rows with content='', which silently produced no learnings at all.
        # The comparison must be ``!= UserActionType.NONE``, never truthiness.
        interaction = InteractionData(
            content="",
            interacted_image_url="",
            image_encoding="",
            user_action=UserActionType.NONE,
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert "is empty" in msg

    def test_tools_used_only_is_accepted(self):
        # A tool-call-only agent turn carries real information. Making the
        # emptiness guard live against its original narrow field list (content
        # / image_url / image_encoding / user_action) would have started
        # rejecting these, so the predicate must consider tools_used too.
        interaction = InteractionData(
            content="",
            tools_used=[ToolUsed(tool_name="search", tool_data={"query": "x"})],
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_shadow_content_only_is_accepted(self):
        interaction = InteractionData(content="", shadow_content="shadow variant")
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_expert_content_only_is_accepted(self):
        interaction = InteractionData(content="", expert_content="expert answer")
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_contentless_interaction_among_real_turns_is_allowed(self):
        # An empty row beside real turns is skipped by the request model, not
        # rejected -- plugins append empty placeholder turns unconditionally,
        # and failing the batch wedged them into a permanent retry loop.
        request = _make_publish_request(
            [
                InteractionData(content="real turn"),
                InteractionData(),
                InteractionData(content="another real turn"),
            ]
        )
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_wholly_empty_batch_is_rejected(self):
        request = _make_publish_request([InteractionData(), InteractionData()])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert "every interaction is empty" in msg

    def test_valid_with_content(self):
        interaction = InteractionData(content="hello world")
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_valid_with_user_action_and_description(self):
        interaction = InteractionData(
            content="hello",
            user_action=UserActionType.CLICK,
            user_action_description="Clicked the submit button",
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_valid_with_image_url_only(self):
        interaction = InteractionData(
            interacted_image_url="https://example.com/image.png",
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_valid_with_image_encoding_only(self):
        interaction = InteractionData(image_encoding="base64data")
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

    def test_multiple_interactions_second_fails_action_check(self):
        good = InteractionData(content="hello")
        bad = InteractionData(
            content="hello",
            user_action=UserActionType.CLICK,
            user_action_description="",
        )
        request = _make_publish_request([good, bad])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert "user_action_description" in msg


class TestValidateDeleteUserProfileRequest:
    def test_no_profile_id_and_no_search_query(self):
        request = DeleteUserProfileRequest(
            user_id="test-user", profile_id="", search_query=""
        )
        valid, msg = validate_delete_user_profile_request(request)
        assert valid is False
        assert msg == "Profile id or search query is required"

    def test_with_profile_id(self):
        request = DeleteUserProfileRequest(user_id="test-user", profile_id="prof-123")
        valid, msg = validate_delete_user_profile_request(request)
        assert valid is True
        assert msg == ""

    def test_with_search_query(self):
        request = DeleteUserProfileRequest(
            user_id="test-user", search_query="some query"
        )
        valid, msg = validate_delete_user_profile_request(request)
        assert valid is True
        assert msg == ""

    def test_with_both_profile_id_and_search_query(self):
        request = DeleteUserProfileRequest(
            user_id="test-user", profile_id="prof-123", search_query="some query"
        )
        valid, msg = validate_delete_user_profile_request(request)
        assert valid is True
        assert msg == ""
