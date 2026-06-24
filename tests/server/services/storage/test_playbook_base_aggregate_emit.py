"""Unit tests for PlaybookMixin.save_agent_playbook_with_aggregate_event base default.

Tests the base-class default directly via an unbound-method call with a mock
self, so the SQLite override (which has its own tests) does not interfere.

Two tests:
  1. Retry + loud: append_lineage_event always fails → retried
     _AGGREGATE_EVENT_EMIT_ATTEMPTS times, capture_anomaly called with
     level="error", method RETURNS the saved playbook (does not raise).
  2. Happy path: append succeeds on first call → called exactly once,
     capture_anomaly NOT called, emitted event has correct op and reason.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import AgentPlaybook
from reflexio.server.services.storage.storage_base._playbook import (
    _AGGREGATE_EVENT_EMIT_ATTEMPTS,
    PlaybookMixin,
)


def _make_saved_playbook() -> AgentPlaybook:
    pb = AgentPlaybook(
        playbook_name="test-pb",
        agent_version="v2",
        content="Do the thing.",
    )
    pb.agent_playbook_id = 42
    return pb


def _make_mock_self(saved_pb: AgentPlaybook, append_side_effect=None) -> MagicMock:
    """Build a minimal mock self that satisfies PlaybookMixin's attribute accesses."""
    mock_self = MagicMock()
    mock_self.save_agent_playbooks.return_value = [saved_pb]
    mock_self.org_id = "org-x"
    if append_side_effect is not None:
        mock_self.append_lineage_event.side_effect = append_side_effect
    return mock_self


class TestPlaybookBaseAggregateEmit:
    def test_retry_and_loud_on_persistent_failure(self):
        """append fails every time → retried N times, capture_anomaly(level='error'), no raise."""
        saved_pb = _make_saved_playbook()
        mock_self = _make_mock_self(
            saved_pb, append_side_effect=RuntimeError("transient db error")
        )

        with patch(
            "reflexio.server.services.storage.storage_base._playbook.capture_anomaly"
        ) as mock_capture:
            result = PlaybookMixin.save_agent_playbook_with_aggregate_event(
                mock_self,
                AgentPlaybook(playbook_name="test-pb", agent_version="v2", content="x"),
                source_ids=["1", "2"],
                request_id="r-fail",
                run_mode="full_archive",
            )

        # Method must return the saved playbook — never raise
        assert result is saved_pb

        # append_lineage_event retried exactly _AGGREGATE_EVENT_EMIT_ATTEMPTS times
        assert (
            mock_self.append_lineage_event.call_count == _AGGREGATE_EVENT_EMIT_ATTEMPTS
        )

        # capture_anomaly called once with level="error"
        mock_capture.assert_called_once()
        _, kwargs = mock_capture.call_args
        assert kwargs.get("level") == "error"

    def test_happy_path_first_attempt_succeeds(self):
        """append succeeds on first try → called once, capture_anomaly NOT called."""
        saved_pb = _make_saved_playbook()
        mock_self = _make_mock_self(saved_pb)

        with patch(
            "reflexio.server.services.storage.storage_base._playbook.capture_anomaly"
        ) as mock_capture:
            result = PlaybookMixin.save_agent_playbook_with_aggregate_event(
                mock_self,
                AgentPlaybook(playbook_name="test-pb", agent_version="v2", content="x"),
                source_ids=["10", "11"],
                request_id="r-ok",
                run_mode="full_archive",
            )

        assert result is saved_pb

        # append called exactly once — no retry on success
        assert mock_self.append_lineage_event.call_count == 1

        # Verify the emitted event has correct op and reason
        (event,) = mock_self.append_lineage_event.call_args.args
        assert event.op == "aggregate"
        assert event.reason == "aggregate:full_archive"
        assert event.prov_relation == "wasDerivedFrom"
        assert event.actor == "aggregator"
        assert event.source_ids == ["10", "11"]
        assert event.request_id == "r-ok"

        # capture_anomaly must NOT be called on success
        mock_capture.assert_not_called()

    def test_empty_request_id_raises_before_save(self):
        """Empty request_id raises ValueError before any storage write (no orphan row)."""
        saved_pb = _make_saved_playbook()
        mock_self = _make_mock_self(saved_pb)

        with pytest.raises(ValueError, match="non-empty request_id"):
            PlaybookMixin.save_agent_playbook_with_aggregate_event(
                mock_self,
                AgentPlaybook(playbook_name="test-pb", agent_version="v2", content="x"),
                source_ids=["1"],
                request_id="",
                run_mode="full_archive",
            )

        # No storage call must have been made
        mock_self.save_agent_playbooks.assert_not_called()
        mock_self.append_lineage_event.assert_not_called()
