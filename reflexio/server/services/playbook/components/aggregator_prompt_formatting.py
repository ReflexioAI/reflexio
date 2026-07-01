from __future__ import annotations

from reflexio.models.api_schema.service_schemas import UserPlaybook


def get_direction_key(fb: UserPlaybook) -> str:
    """
    Extract a similarity key from a user playbook for grouping.

    Returns the raw content used for token-overlap comparison. Grouping is
    purely content-similarity based: under Option B a skill may legitimately
    hold mixed-orientation rules (do-rules and avoid-rules for different
    sub-aspects of one task), so whole-content polarity is NOT derived or
    gated here. Preserving distinct do/avoid rules when similar items are
    merged is the aggregation prompt's responsibility.

    Args:
        fb: A user playbook item

    Returns:
        str: Content used as the similarity key for grouping
    """
    return fb.content or ""


def token_overlap(str1: str, str2: str, threshold: float = 0.6) -> bool:
    """
    Check if two strings have significant token overlap using asymmetric containment.

    Computes the ratio of shared tokens to the smaller set, so a short string
    contained in a longer one still counts as a match.

    Args:
        str1: First string
        str2: Second string
        threshold: Minimum overlap ratio

    Returns:
        bool: True if overlap ratio >= threshold
    """
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    if not tokens1 or not tokens2:
        return False
    intersection = len(tokens1 & tokens2)
    overlap_ratio = max(intersection / len(tokens1), intersection / len(tokens2))
    return overlap_ratio >= threshold


def group_playbooks_by_direction(
    cluster_playbooks: list[UserPlaybook],
    threshold: float = 0.6,
) -> list[list[UserPlaybook]]:
    """
    Group playbooks by similarity of their content.

    Uses greedy single-linkage: each playbook is assigned to the first
    existing group that has any member with sufficient token overlap.
    Groups are returned sorted by size descending (largest first).

    Grouping is purely content-similarity based and does NOT gate on a
    derived whole-content polarity. Under Option B a skill may hold
    mixed-orientation rules (do-rules and avoid-rules for different
    sub-aspects), and whole-content polarity is undefined for such a skill.
    Keeping a do-rule and an avoid-rule as distinct rules when similar items
    are merged into one skill is the aggregation prompt's responsibility,
    not a mechanical split here.

    Args:
        cluster_playbooks: List of raw playbooks to group
        threshold: Token overlap threshold for grouping

    Returns:
        list[list[UserPlaybook]]: Groups sorted by size descending
    """
    groups: list[list[UserPlaybook]] = []

    for fb in cluster_playbooks:
        key = get_direction_key(fb)
        matched = False
        for group in groups:
            if any(
                token_overlap(
                    key,
                    get_direction_key(group_fb),
                    threshold,
                )
                for group_fb in group
            ):
                group.append(fb)
                matched = True
                break
        if not matched:
            groups.append([fb])

    # Sort by group size descending (largest first)
    groups.sort(key=len, reverse=True)
    return groups


def format_structured_cluster_input(
    cluster_playbooks: list[UserPlaybook],
    direction_overlap_threshold: float = 0.6,
) -> str:
    """
    Format a cluster of playbooks for structured aggregation prompt.

    When the cluster forms a single similarity group, uses the flat-list
    format. When distinct similarity groups are detected (multiple groups),
    uses a grouped format so the LLM can see which items are similar and
    preserve distinct rules (e.g. a do-rule and an avoid-rule) as separate
    rules in the merged skill rather than collapsing them.

    Args:
        cluster_playbooks: List of raw playbooks in this cluster
        direction_overlap_threshold: Token overlap threshold for grouping by direction

    Returns:
        str: Formatted input for the aggregation prompt
    """
    groups = group_playbooks_by_direction(
        cluster_playbooks, threshold=direction_overlap_threshold
    )

    if len(groups) <= 1:
        return format_flat(cluster_playbooks)
    return format_grouped(groups)


def format_flat(cluster_playbooks: list[UserPlaybook]) -> str:
    """
    Format playbooks as flat bullet lists (original format, used when no conflict).

    Args:
        cluster_playbooks: List of raw playbooks in this cluster

    Returns:
        str: Formatted input with separate field lists
    """
    triggers = []
    rationales = []

    for fb in cluster_playbooks:
        if fb.trigger:
            triggers.append(fb.trigger)
        if fb.rationale:
            rationales.append(fb.rationale)

    lines: list[str] = []

    if triggers:
        lines.append("TRIGGER conditions (to be consolidated):")
        lines.extend(f"- {trigger}" for trigger in triggers)
    else:
        lines.append("TRIGGER conditions: (none specified)")

    if rationales:
        lines.append("RATIONALE summaries:")
        lines.extend(f"- {r}" for r in rationales)

    append_freeform_observations(lines, cluster_playbooks)

    return "\n".join(lines)


def format_grouped(
    groups: list[list[UserPlaybook]],
) -> str:
    """
    Format playbooks in grouped layout (used when conflicting directions are detected).

    Args:
        groups: AgentPlaybook groups sorted by size descending

    Returns:
        str: Formatted input with group headers and per-playbook fields
    """
    lines: list[str] = [
        "The following playbook items are grouped by similarity. "
        "Groups are ordered by size (largest first).",
        "",
    ]

    for idx, group in enumerate(groups, start=1):
        count_label = "playbook" if len(group) == 1 else "playbooks"
        lines.append(f"Group {idx} ({len(group)} {count_label}):")
        for fb in group:
            parts: list[str] = []
            if fb.trigger:
                parts.append(f'Trigger: "{fb.trigger}"')
            if fb.rationale:
                parts.append(f'Rationale: "{fb.rationale}"')
            if not parts and fb.content:
                parts.append(f'AgentPlaybook: "{fb.content}"')
            if parts:
                lines.append(f"  - {parts[0]}")
                lines.extend(f"    {p}" for p in parts[1:])
        lines.append("")

    return "\n".join(lines)


def append_freeform_observations(
    lines: list[str], cluster_playbooks: list[UserPlaybook]
) -> None:
    """Append freeform observations from cluster playbooks to output lines."""
    freeform_observations = [
        fb.content for fb in cluster_playbooks if not fb.trigger and fb.content
    ]
    if freeform_observations:
        lines.append("Freeform observations (from freeform cluster members):")
        lines.extend(f"- {obs}" for obs in freeform_observations)
