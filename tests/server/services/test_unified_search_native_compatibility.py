"""Native fanout and source-lookup contracts independent of optional judging."""

from typing import Any, cast
from unittest.mock import Mock

import pytest

from reflexio.server import search_runtime
from reflexio.server.services.retrieval.session_dedup import session_seen_cache
from reflexio.server.services.unified_search_service import _run_phase_b as run_phase_b
from tests.server.services.test_unified_search_agent_selection import (
    native as native_fixture,
)
from tests.server.services.test_unified_search_agent_selection import run

pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    yield from cast(Any, native_fixture).__wrapped__(monkeypatch)


@pytest.mark.parametrize("has_window", [False, True])
def test_widened_fanout_preserves_each_native_scope(native, monkeypatch, has_window):
    from datetime import UTC, datetime

    from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
    from reflexio.models.config_schema import SearchMode
    from reflexio.server.services import unified_search_service as service

    storage = native.storage
    storage.supports_unified_hybrid_search = True
    combined = Mock(
        side_effect=AssertionError("Different per-type limits require fanout")
    )
    monkeypatch.setattr(service, "_run_phase_b_single_rpc", combined)
    monkeypatch.setattr(service, "_unified_single_rpc_enabled", lambda: True)
    profiles, agents, users = (
        Mock(return_value=[]),
        Mock(return_value=[]),
        Mock(return_value=[]),
    )
    monkeypatch.setattr(service, "_search_profiles_via_storage", profiles)
    monkeypatch.setattr(service, "_search_agent_playbooks_via_storage", agents)
    monkeypatch.setattr(service, "_search_user_playbooks_via_storage", users)
    request = UnifiedSearchRequest(
        query="q",
        user_id="u",
        agent_version="v1",
        tags=["billing"],
        source="unit",
        threshold=0.7,
        search_mode=SearchMode.FTS,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC) if has_window else None
    result = run_phase_b(
        request=request,
        org_id="org",
        storage=storage,
        embedding=None,
        query="q",
        top_k=5,
        playbook_top_k=20,
        threshold=0.7,
        search_mode=SearchMode.FTS,
        start_time=start,
    )
    assert result == ([], [], [])
    assert profiles.call_args.args[2] == 5
    assert agents.call_args.args[2:4] == (20, 0.7)
    assert agents.call_args.args[7] is not None  # Native SearchOptions.
    user_request = users.call_args.args[1]
    assert user_request.top_k == 20 and user_request.threshold == 0.7
    assert user_request.user_id == "u" and user_request.agent_version == "v1"
    assert user_request.tags == ["billing"] and user_request.source == "unit"
    assert (
        user_request.start_time == start and user_request.search_mode == SearchMode.FTS
    )
    combined.assert_not_called()


def test_native_lineage_failure_retains_existing_behavior(native):
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.side_effect = (
        RuntimeError("offline")
    )
    result = run(native)
    assert result.success and result.user_playbooks == native.users


def test_native_lineage_deadline_never_accepts_session(native):
    import time

    deadline = search_runtime.SearchDeadlineError("source lookup expired")
    native.storage.get_source_user_playbook_ids_for_agent_playbooks.side_effect = (
        deadline
    )
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 1)
    token = search_runtime._scope.set(scope)
    try:
        with pytest.raises(search_runtime.SearchDeadlineError) as raised:
            run(native, session_id="source-deadline")
        assert raised.value is deadline
        assert scope.success_callbacks == []
    finally:
        search_runtime._scope.reset(token)
    assert not session_seen_cache.seen("org", "source-deadline")
