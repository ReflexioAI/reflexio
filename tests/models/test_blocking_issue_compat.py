"""Read-compatibility for the deprecated ``blocking_issue`` field.

``blocking_issue`` is no longer produced or used by the active extraction /
aggregation / display code path, but the ``BlockingIssue`` type and the
domain-model field are intentionally retained so existing stored rows that
still carry a non-empty ``blocking_issue`` keep deserializing without error.
Storage was not migrated; these tests guard that read-compat contract.
"""

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    BlockingIssue,
    BlockingIssueKind,
)


def test_agent_playbook_hydrates_nonempty_blocking_issue():
    """A stored playbook dict with a non-empty blocking_issue still loads."""
    pb = AgentPlaybook.model_validate(
        {
            "agent_version": "v1",
            "content": "suggest using the API endpoint instead",
            "blocking_issue": {
                "kind": "missing_tool",
                "details": "No direct database query tool available",
            },
        }
    )

    assert pb.blocking_issue == BlockingIssue(
        kind=BlockingIssueKind.MISSING_TOOL,
        details="No direct database query tool available",
    )


def test_agent_playbook_defaults_blocking_issue_to_none():
    """New playbooks (no blocking_issue) default to None — never populated."""
    pb = AgentPlaybook(agent_version="v1", content="do the thing")

    assert pb.blocking_issue is None
