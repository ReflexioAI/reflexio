from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    PublishUserInteractionRequest,
)


def validate_publish_user_interaction_request(
    request: PublishUserInteractionRequest,
) -> tuple[bool, str]:
    """
    Validate the publish user interaction request

    Args:
        request (PublishUserInteractionRequest): The request to validate

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating if the request is valid and a message
    """
    if not request.interaction_data_list:
        return False, "No interaction data provided"

    # Defense-in-depth: session_id is a required NonEmptyStr on
    # PublishUserInteractionRequest, so the normal validated API path already
    # rejects empty/missing values with a 422. This guard additionally covers
    # paths that bypass Pydantic validation (e.g. ``model_construct``).
    if not request.session_id or not request.session_id.strip():
        return False, "session_id is required and cannot be empty"

    # Delegates to InteractionData.shape_error(), the single definition of
    # the per-interaction rules, so this guard and PublishUserInteractionRequest's
    # validator cannot diverge. The emptiness rule in particular used to be dead
    # code here: it ended in `not interaction_data.user_action`, and UserActionType
    # is a StrEnum whose NONE member is the truthy string "none", so the branch
    # could never fire and a publish of entirely empty interactions was accepted.
    for index, interaction_data in enumerate(request.interaction_data_list):
        if reason := interaction_data.shape_error():
            return False, f"interaction_data_list[{index}] {reason}"

    # An individual empty interaction is skipped by the request model, not
    # rejected (plugins emit empty placeholder turns). Only a wholly empty
    # batch -- the original incident -- is fatal.
    if not any(
        interaction_data.carries_content()
        for interaction_data in request.interaction_data_list
    ):
        return False, "every interaction is empty"

    return True, ""


def validate_delete_user_profile_request(
    request: DeleteUserProfileRequest,
) -> tuple[bool, str]:
    """
    Validate the delete user profile request

    Args:
        request (DeleteUserProfileRequest): The request to validate

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating if the request is valid and a message
    """

    if not request.profile_id and not request.search_query:
        return False, "Profile id or search query is required"

    return True, ""
