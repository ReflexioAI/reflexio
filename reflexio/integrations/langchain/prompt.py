"""Prompt helpers for injecting Reflexio insights into LangChain agent prompts.

These utilities fetch playbooks and user profiles from Reflexio
and format them as structured text for inclusion in system prompts.

No LangChain dependency required — works with plain strings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.client.client import ReflexioClient
    from reflexio.models.api_schema.service_schemas import (
        AgentPlaybook,
        UserPlaybook,
        UserProfile,
    )

logger = logging.getLogger(__name__)


def _format_playbook_item(item: AgentPlaybook | UserPlaybook) -> str:
    """Format a single playbook item. content is the primary behavioral guideline."""
    if item.content:
        return f"- {item.content}"
    return ""


def _format_profile(profile: UserProfile) -> str:
    """Format a single user profile for context."""
    return f"- {profile.content}"


def get_reflexio_context(
    client: ReflexioClient,
    query: str,
    *,
    agent_version: str = "",
    user_id: str = "",
    top_k: int = 5,
) -> str:
    """Search Reflexio for context relevant to a specific query.

    Performs a unified search across playbooks and profiles,
    returning formatted text suitable for injection into a prompt.

    Args:
        client (ReflexioClient): Reflexio client instance
        query (str): Search query
        agent_version (str): Filter by agent version
        user_id (str): Filter profiles by user ID
        top_k (int): Maximum results per entity type

    Returns:
        str: Formatted context string, or empty string if nothing found
    """
    try:
        resp = client.search(
            query=query,
            agent_version=agent_version or None,
            user_id=user_id or None,
            top_k=top_k,
        )
    except Exception:
        logger.debug("Failed to search Reflexio", exc_info=True)
        return ""

    sections: list[str] = []

    if resp.agent_playbooks:
        agent_playbook_text = "\n".join(
            line for f in resp.agent_playbooks if (line := _format_playbook_item(f))
        )
        if agent_playbook_text:
            sections.append(f"**Behavioral Guidelines:**\n{agent_playbook_text}")

    if resp.user_playbooks:
        user_playbook_text = "\n".join(
            line for rf in resp.user_playbooks if (line := _format_playbook_item(rf))
        )
        if user_playbook_text:
            sections.append(f"**User Playbooks:**\n{user_playbook_text}")

    if resp.profiles:
        profile_text = "\n".join(_format_profile(p) for p in resp.profiles)
        sections.append(f"**User Context:**\n{profile_text}")

    return "\n\n".join(sections)
