"""Existing Python 3.11 HTTP integration, with an injected HTTP adapter."""


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
    identities = {
        "user_id": user_id,
        "session_id": session_id,
        "source": source,
        "agent_version": agent_version,
    }
    try:
        results = client.post(
            "/api/search",
            json={
                **identities,
                "query": user_message,
                "entity_types": ["profiles", "user_playbooks", "agent_playbooks"],
                "agent_playbook_status_filter": ["approved"],
                "top_k": 3,
            },
            timeout=5,
        )
        context = []
        if results["success"]:
            for label, key in (
                ("Profile", "profiles"),
                ("User playbook", "user_playbooks"),
                ("Approved agent playbook", "agent_playbooks"),
            ):
                context.extend(
                    f"{label}: {item['content']}"
                    for item in results[key]
                    if not item["content"].startswith("[discard]")
                )
    except Exception:
        context = []
    response = model.generate(user_message, "\n".join(context))
    result = client.post(
        "/api/publish_interaction",
        json={
            **identities,
            "interaction_data_list": [
                {"role": "User", "content": user_message},
                {"role": "Agent", "content": response},
            ],
        },
        timeout=5,
    )
    if not result["success"]:
        model.diagnostic("publish failed")
    for warning in result.get("warnings", []):
        model.diagnostic(warning)
    return response
