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
    if interactions is not None:
        return PublishUserInteractionRequest(
            user_id="test-user",
            interaction_data_list=interactions,
        )
    # Bypass Pydantic min_length=1 validation to test the precondition check
    return PublishUserInteractionRequest.model_construct(
        user_id="test-user",
        interaction_data_list=[],
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
        assert msg == "User action description is required for user action"

    def test_both_image_url_and_image_encoding(self):
        interaction = InteractionData(
            content="hello",
            interacted_image_url="https://example.com/image.png",
            image_encoding="base64data",
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is False
        assert (
            msg == "Image encoding and interacted image url cannot be provided together"
        )

    def test_all_fields_empty_with_none_action_passes(self):
        # UserActionType.NONE is "none" (truthy), so the "all empty" branch
        # in the source is unreachable via normal model construction.
        # With NONE, the validator considers user_action as present and passes.
        interaction = InteractionData(
            content="",
            interacted_image_url="",
            image_encoding="",
            user_action=UserActionType.NONE,
        )
        request = _make_publish_request([interaction])
        valid, msg = validate_publish_user_interaction_request(request)
        assert valid is True
        assert msg == ""

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
        assert msg == "User action description is required for user action"


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
