"""
Unit tests for blocking_issue handling in PlaybookAggregator.

Tests the formatting and processing methods that carry blocking_issue
through the aggregation pipeline.
"""

from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import (
    BlockingIssue,
    BlockingIssueKind,
    StructuredData,
    UserPlaybook,
)
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
    StructuredPlaybookContent,
    format_structured_fields_for_display,
)


@pytest.fixture
def aggregator():
    """Create a PlaybookAggregator with mocked dependencies."""
    mock_llm_client = MagicMock()
    mock_request_context = MagicMock()
    mock_request_context.storage = MagicMock()
    mock_request_context.configurator = MagicMock()

    return PlaybookAggregator(
        llm_client=mock_llm_client,
        request_context=mock_request_context,
        agent_version="1.0",
    )


class TestFormatClusterInput:
    """Tests for _format_cluster_input with blocking_issue."""

    def test_includes_blocking_issues_in_output(self, aggregator):
        """Test that blocking issues from cluster playbooks appear in formatted output."""
        playbooks = [
            UserPlaybook(
                agent_version="1.0",
                request_id="req1",
                playbook_name="test",
                content="content1",
                structured_data=StructuredData(
                    trigger="user asks to delete files",
                    instruction="inform user about permission requirements",
                    blocking_issue=BlockingIssue(
                        kind=BlockingIssueKind.PERMISSION_DENIED,
                        details="No admin file deletion access",
                    ),
                ),
            ),
            UserPlaybook(
                agent_version="1.0",
                request_id="req2",
                playbook_name="test",
                content="content2",
                structured_data=StructuredData(
                    trigger="user requests file removal",
                    instruction="suggest contacting admin",
                    blocking_issue=BlockingIssue(
                        kind=BlockingIssueKind.PERMISSION_DENIED,
                        details="Lacks write permissions on shared drive",
                    ),
                ),
            ),
        ]

        result = aggregator._format_cluster_input(playbooks)

        # New per-item format: each playbook is a numbered block with Blocked by: line
        assert "[permission_denied] No admin file deletion access" in result
        assert "[permission_denied] Lacks write permissions on shared drive" in result

    def test_omits_blocked_by_when_no_blocking_issues(self, aggregator):
        """Test that Blocked by: is absent when no playbooks have blocking_issue."""
        playbooks = [
            UserPlaybook(
                agent_version="1.0",
                request_id="req1",
                playbook_name="test",
                content="content1",
                structured_data=StructuredData(
                    trigger="user asks a question",
                    instruction="provide a detailed answer",
                ),
            ),
        ]

        result = aggregator._format_cluster_input(playbooks)

        assert "Blocked by:" not in result

    def test_includes_only_non_none_blocking_issues(self, aggregator):
        """Test that only playbooks with blocking_issue have a Blocked by: line."""
        playbooks = [
            UserPlaybook(
                agent_version="1.0",
                request_id="req1",
                playbook_name="test",
                content="content1",
                structured_data=StructuredData(
                    trigger="user asks to query DB",
                    instruction="use API instead",
                    blocking_issue=BlockingIssue(
                        kind=BlockingIssueKind.MISSING_TOOL,
                        details="No DB query tool",
                    ),
                ),
            ),
            UserPlaybook(
                agent_version="1.0",
                request_id="req2",
                playbook_name="test",
                content="content2",
                structured_data=StructuredData(
                    trigger="user asks to query DB",
                    instruction="suggest API endpoint",
                ),
                # No blocking_issue
            ),
        ]

        result = aggregator._format_cluster_input(playbooks)

        assert "[missing_tool] No DB query tool" in result
        # Only one blocking issue line
        assert result.count("[missing_tool]") == 1


class TestFormatStructuredFieldsForDisplay:
    """Tests for format_structured_fields_for_display with blocking_issue."""

    def test_includes_blocked_by_line(self, aggregator):
        """Test that blocking_issue is formatted as 'Blocked by:' line."""
        structured = StructuredPlaybookContent(
            instruction="use API endpoint",
            trigger="user asks for DB access",
            blocking_issue=BlockingIssue(
                kind=BlockingIssueKind.EXTERNAL_DEPENDENCY,
                details="Database service is unavailable",
            ),
        )

        result = format_structured_fields_for_display(structured)

        assert (
            "Blocked by: [external_dependency] Database service is unavailable"
            in result
        )

    def test_omits_blocked_by_when_none(self, aggregator):
        """Test that no 'Blocked by:' line when blocking_issue is None."""
        structured = StructuredPlaybookContent(
            instruction="validate inputs",
            trigger="processing data",
        )

        result = format_structured_fields_for_display(structured)

        assert "Blocked by:" not in result


class TestProcessAggregationResponse:
    """Tests for _process_aggregation_response with blocking_issue."""

    def test_carries_blocking_issue_to_playbook(self, aggregator):
        """Test that blocking_issue from LLM response is set on the resulting AgentPlaybook."""
        response = PlaybookAggregationOutput(
            playbook=StructuredPlaybookContent(
                instruction="inform user about limitation",
                trigger="user requests restricted action",
                blocking_issue=BlockingIssue(
                    kind=BlockingIssueKind.POLICY_RESTRICTION,
                    details="Corporate policy blocks external API calls",
                ),
            )
        )
        cluster_playbooks = [
            UserPlaybook(
                agent_version="1.0",
                request_id="req1",
                playbook_name="test",
                content="content",
            ),
        ]

        result = aggregator._process_aggregation_response(response, cluster_playbooks)

        assert result is not None
        assert result.structured_data.blocking_issue is not None
        assert (
            result.structured_data.blocking_issue.kind
            == BlockingIssueKind.POLICY_RESTRICTION
        )
        assert "Corporate policy" in result.structured_data.blocking_issue.details
        assert "Blocked by: [policy_restriction]" in result.content

    def test_playbook_without_blocking_issue(self, aggregator):
        """Test that AgentPlaybook has no blocking_issue when LLM doesn't return one."""
        response = PlaybookAggregationOutput(
            playbook=StructuredPlaybookContent(
                instruction="provide clear instructions",
                trigger="user is confused",
            )
        )
        cluster_playbooks = [
            UserPlaybook(
                agent_version="1.0",
                request_id="req1",
                playbook_name="test",
                content="content",
            ),
        ]

        result = aggregator._process_aggregation_response(response, cluster_playbooks)

        assert result is not None
        assert result.structured_data.blocking_issue is None
        assert "Blocked by:" not in result.content
