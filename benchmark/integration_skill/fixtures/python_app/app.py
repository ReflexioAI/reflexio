"""Existing synchronous customer application; search and publish already work."""

from reflexio import InteractionData


def handle_turn(
    client,
    model,
    *,
    user_message,
    user_id,
    session_id,
    source="support",
    agent_version="support@1",
):
    session = client.for_session(session_id)
    try:
        results = session.search(
            query=user_message,
            user_id=user_id,
            source=source,
            agent_version=agent_version,
            entity_types=["profiles", "user_playbooks", "agent_playbooks"],
            agent_playbook_status_filter=["approved"],
            top_k=3,
        )
        context = []
        if results.success:
            for label, items in (
                ("Profile", results.profiles),
                ("User playbook", results.user_playbooks),
                ("Approved agent playbook", results.agent_playbooks),
            ):
                context.extend(
                    f"{label}: {item.content}"
                    for item in items
                    if not item.content.startswith("[discard]")
                )
    except Exception:
        context = []
    response = model.generate(user_message, "\n".join(context))
    result = session.publish_interaction(
        user_id=user_id,
        source=source,
        agent_version=agent_version,
        interactions=[
            InteractionData(role="User", content=user_message),
            InteractionData(role="Agent", content=response),
        ],
    )
    if not result.success:
        model.diagnostic("publish failed")
    for warning in result.warnings:
        model.diagnostic(warning)
    return response
