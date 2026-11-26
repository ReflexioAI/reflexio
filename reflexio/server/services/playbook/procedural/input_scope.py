"""Select the actual source modality, never an old/new method preference."""

from collections.abc import Sequence

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.services.playbook.playbook_service_utils import has_expert_content


def is_ordinary_text(groups: Sequence[RequestInteractionDataModel]) -> bool:
    """Expert comparisons and visual inputs retain independent capabilities.

    Ordinary textual tool/action/shadow fields remain source evidence. Retrieval
    attribution and citations are metadata, not a reason to switch engines.
    """
    interactions = [item for group in groups for item in group.interactions]
    return (
        bool(interactions)
        and not has_expert_content(interactions)
        and not any(
            item.interacted_image_url or item.image_encoding for item in interactions
        )
    )


def native_display_fields(interaction: Interaction) -> list[str]:
    """Follow render_interaction's actual visible-piece order, not hidden fields."""
    from reflexio.models.api_schema.domain.enums import UserActionType

    fields = []
    if interaction.content:
        fields.extend("native_tool_display" for _ in interaction.tools_used or [])
        fields.append("content")
    if interaction.user_action != UserActionType.NONE:
        fields.append("native_user_action_display")
    return fields
