"""E2E: session-scoped search dedup through real SQLite storage.

Drives ``Reflexio.unified_search`` (the same path the ``/api/search`` route
calls) against a real per-test SQLite database: a search carrying a
``session_id`` never re-returns rows already served to that session and
backfills next-best matches; other sessions and session-less searches are
unaffected.
"""

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.api_schema.service_schemas import UserPlaybook


def _seed_playbooks(reflexio_instance, count: int = 5) -> None:
    storage = reflexio_instance.request_context.storage
    assert storage is not None
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id="dedup-user",
                agent_version="dedup-agent",
                request_id=f"dedup-req-{i}",
                content=f"When refunding an order always verify the receipt id variant {i}",
                trigger="refund order receipt",
            )
            for i in range(1, count + 1)
        ]
    )


def _search_ids(reflexio_instance, org_id: str, session_id: str | None) -> set[int]:
    response = reflexio_instance.unified_search(
        UnifiedSearchRequest(
            query="how to refund an order",
            user_id="dedup-user",
            agent_version="dedup-agent",
            entity_types=["user_playbooks"],
            top_k=2,
            session_id=session_id,
        ),
        org_id=org_id,
    )
    assert response.success
    return {p.user_playbook_id for p in response.user_playbooks or []}


def test_session_dedup_end_to_end(
    reflexio_instance_playbook_only, test_org_id, cleanup_playbook_only
):
    instance = reflexio_instance_playbook_only
    _seed_playbooks(instance)
    session_a = f"{test_org_id}-dedup-a"
    session_b = f"{test_org_id}-dedup-b"

    first = _search_ids(instance, test_org_id, session_a)
    assert len(first) == 2

    # Same session: previously served rows skipped, next-best backfilled.
    second = _search_ids(instance, test_org_id, session_a)
    assert len(second) == 2
    assert first.isdisjoint(second)

    # A concurrent session sees the full result set.
    other = _search_ids(instance, test_org_id, session_b)
    assert other == first

    # Pool exhausts: the last remaining row, then nothing.
    third = _search_ids(instance, test_org_id, session_a)
    assert len(third) == 1
    assert third.isdisjoint(first | second)
    assert _search_ids(instance, test_org_id, session_a) == set()

    # Session-less searches are unaffected and never consume any session.
    assert _search_ids(instance, test_org_id, None) == first
    assert _search_ids(instance, test_org_id, None) == first
    assert _search_ids(instance, test_org_id, session_b) == second
