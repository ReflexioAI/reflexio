"""Converters from domain models to UI View models.

This module bridges the domain layer and UI layer. It imports from both
domain/entities (input types) and ui/entities (output types).
"""

from ..domain.entities import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    ProfileChangeLog,
    UserPlaybook,
    UserProfile,
    to_visible_structured_data,
)
from .entities import (
    AgentPlaybookView,
    EvaluationResultView,
    InteractionView,
    ProfileChangeLogView,
    ProfileView,
    UserPlaybookView,
)

__all__ = [
    "to_structured_data_view",
    "to_interaction_view",
    "to_profile_view",
    "to_user_playbook_view",
    "to_agent_playbook_view",
    "to_evaluation_result_view",
    "to_profile_change_log_view",
]

to_structured_data_view = to_visible_structured_data


def to_interaction_view(interaction: Interaction) -> InteractionView:
    """Convert internal Interaction to user-facing InteractionView.

    Args:
        interaction (Interaction): Full internal interaction

    Returns:
        InteractionView: View without embedding and image_encoding
    """
    return InteractionView(
        interaction_id=interaction.interaction_id,
        user_id=interaction.user_id,
        request_id=interaction.request_id,
        created_at=interaction.created_at,
        role=interaction.role,
        content=interaction.content,
        user_action=interaction.user_action,
        user_action_description=interaction.user_action_description,
        interacted_image_url=interaction.interacted_image_url,
        shadow_content=interaction.shadow_content,
        expert_content=interaction.expert_content,
        tools_used=interaction.tools_used,
    )


def to_profile_view(profile: UserProfile) -> ProfileView:
    """Convert internal UserProfile to user-facing ProfileView.

    Args:
        profile (UserProfile): Full internal user profile

    Returns:
        ProfileView: View without embedding
    """
    return ProfileView(
        profile_id=profile.profile_id,
        user_id=profile.user_id,
        content=profile.content,
        last_modified_timestamp=profile.last_modified_timestamp,
        generated_from_request_id=profile.generated_from_request_id,
        profile_time_to_live=profile.profile_time_to_live,
        expiration_timestamp=profile.expiration_timestamp,
        custom_features=profile.custom_features,
        source=profile.source,
        status=profile.status,
        extractor_names=profile.extractor_names,
    )


def to_user_playbook_view(rf: UserPlaybook) -> UserPlaybookView:
    """Convert internal UserPlaybook to user-facing UserPlaybookView.

    Args:
        rf (UserPlaybook): Full internal user playbook

    Returns:
        UserPlaybookView: View without embedding, with StructuredDataView
    """
    return UserPlaybookView(
        user_playbook_id=rf.user_playbook_id,
        user_id=rf.user_id,
        agent_version=rf.agent_version,
        request_id=rf.request_id,
        playbook_name=rf.playbook_name,
        created_at=rf.created_at,
        content=rf.content,
        structured_data=to_structured_data_view(rf.structured_data),
        status=rf.status,
        source=rf.source,
        source_interaction_ids=rf.source_interaction_ids,
    )


def to_agent_playbook_view(fb: AgentPlaybook) -> AgentPlaybookView:
    """Convert internal AgentPlaybook to user-facing AgentPlaybookView.

    Args:
        fb (AgentPlaybook): Full internal agent playbook

    Returns:
        AgentPlaybookView: View without embedding, with StructuredDataView
    """
    return AgentPlaybookView(
        agent_playbook_id=fb.agent_playbook_id,
        playbook_name=fb.playbook_name,
        agent_version=fb.agent_version,
        created_at=fb.created_at,
        content=fb.content,
        structured_data=to_structured_data_view(fb.structured_data),
        playbook_status=fb.playbook_status,
        playbook_metadata=fb.playbook_metadata,
        status=fb.status,
    )


def to_evaluation_result_view(
    result: AgentSuccessEvaluationResult,
) -> EvaluationResultView:
    """Convert internal AgentSuccessEvaluationResult to user-facing EvaluationResultView.

    Args:
        result (AgentSuccessEvaluationResult): Full internal evaluation result

    Returns:
        EvaluationResultView: View without embedding
    """
    return EvaluationResultView(
        result_id=result.result_id,
        agent_version=result.agent_version,
        session_id=result.session_id,
        is_success=result.is_success,
        failure_type=result.failure_type,
        failure_reason=result.failure_reason,
        evaluation_name=result.evaluation_name,
        created_at=result.created_at,
        regular_vs_shadow=result.regular_vs_shadow,
        number_of_correction_per_session=result.number_of_correction_per_session,
        user_turns_to_resolution=result.user_turns_to_resolution,
        is_escalated=result.is_escalated,
    )


def to_profile_change_log_view(log: ProfileChangeLog) -> ProfileChangeLogView:
    """Convert internal ProfileChangeLog to user-facing ProfileChangeLogView.

    Args:
        log (ProfileChangeLog): Full internal profile change log

    Returns:
        ProfileChangeLogView: View with ProfileView lists instead of UserProfile lists
    """
    return ProfileChangeLogView(
        id=log.id,
        user_id=log.user_id,
        request_id=log.request_id,
        created_at=log.created_at,
        added_profiles=[to_profile_view(p) for p in log.added_profiles],
        removed_profiles=[to_profile_view(p) for p in log.removed_profiles],
        mentioned_profiles=[to_profile_view(p) for p in log.mentioned_profiles],
    )
