from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    PublishUserInteractionRequest,
    UserActionType,
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

    for interaction_data in request.interaction_data_list:
        if (
            interaction_data.user_action != UserActionType.NONE
            and not interaction_data.user_action_description
        ):
            return False, "User action description is required for user action"

        if interaction_data.interacted_image_url and interaction_data.image_encoding:
            return (
                False,
                "Image encoding and interacted image url cannot be provided together",
            )

        if (
            not interaction_data.content
            and not interaction_data.interacted_image_url
            and not interaction_data.image_encoding
            and not interaction_data.user_action
        ):
            return (
                False,
                "Text interaction, interacted image url, image encoding, and user action cannot be all empty",
            )

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
