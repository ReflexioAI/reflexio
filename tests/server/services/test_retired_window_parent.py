"""Old windows can create a current parent without mixing extraction contracts."""

import copy
import json
from dataclasses import replace

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.services.extraction import resumable_agent
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("fail_commit", [False, True])
@pytest.mark.parametrize("pending_tools", [False, True])
def test_first_parent_of_old_profile_window_completes_and_replays_once(
    pipeline, monkeypatch, fail_commit, pending_tools
):
    engine, _, _ = pipeline
    storage = engine.get_storage()
    if pending_tools:
        monkeypatch.setattr(
            resumable_agent, "pending_tool_calls_enabled", lambda _: True
        )
        monkeypatch.setattr(
            resumable_agent,
            "_pending_tool_call_config",
            lambda _: PendingToolCallConfig(enabled=True),
        )
    prepare = storage.prepare_extraction
    legacy_policy = None

    def admit_old_window(*args, **kwargs):
        nonlocal legacy_policy
        window = prepare(*args, **kwargs)
        if window is None or legacy_policy is not None:
            return window
        assert storage.get_agent_run(f"window:{window.window_id}") is None
        policy = copy.deepcopy(window.policy)
        policy["extractor"]["extractor_name"] = "historical-profile-label"
        with storage.conn:
            storage.conn.execute(
                "UPDATE extraction_windows SET policy=? WHERE window_id=?",
                (json.dumps(policy), window.window_id),
            )
        legacy_policy = policy
        return replace(window, policy=policy)

    monkeypatch.setattr(storage, "prepare_extraction", admit_old_window)
    # Existing real HTTP/storage/executor test drives both a normal completion
    # and an injected commit rollback. External model calls stay controlled;
    # it asserts exact provider usage, retry, publication and replay counts.
    durable.test_enabled_overflow_profile_uses_real_window_commit_once(
        pipeline,
        monkeypatch,
        gate_negative=False,
        fail_commit=fail_commit,
        short_session=False,
    )
    assert legacy_policy is not None
    row = storage.conn.execute(
        "SELECT policy FROM extraction_windows WHERE user_id=? LIMIT 1", ("u",)
    ).fetchone()
    assert json.loads(row[0]) == legacy_policy
    row = storage.conn.execute(
        "SELECT generation_request_snapshot FROM _agent_runs WHERE user_id=? LIMIT 1",
        ("u",),
    ).fetchone()
    snapshot = json.loads(row[0])
    assert "extractor_name" not in snapshot["extractor_config"]
    assert snapshot["overflow"]["attempts"]
