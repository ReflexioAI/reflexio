"""Attention uses exact final accepted profiles and immutable scoped snapshots."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server import search_runtime
from reflexio.server.services.profile.attention import (
    access_recording,
    candidate_attention,
    record_accepted_profiles,
    storage_snapshot,
)
from reflexio.server.services.profile.lifecycle import KEY, attention_context
from reflexio.server.services.retrieval.recency import decay_for_item

DAY = 86400


def profile(identity="p", ancestors=None):
    return UserProfile(
        profile_id=identity,
        user_id="u",
        content="tea",
        generated_from_request_id="r",
        last_modified_timestamp=999,
        custom_features={
            KEY: {
                "created_at": 100,
                "ancestor_ids": ancestors or [],
                "explicit_storage_deadline": None,
            }
        },
    )


def test_snapshot_reads_bounded_candidates_and_ancestors_without_recording():
    storage = MagicMock()
    storage.get_profile_access_epochs.return_value = {"old": 100 + 20 * DAY}
    item = profile(ancestors=["old"])
    with storage_snapshot(storage):
        with candidate_attention([item]):
            assert decay_for_item(
                item, entity_type="profiles", now=100 + 30 * DAY
            ) == pytest.approx(2 ** (-1 / 3))
        storage.get_profile_access_epochs.return_value = {}
        with candidate_attention([item]):
            assert decay_for_item(
                item, entity_type="profiles", now=100 + 30 * DAY
            ) == pytest.approx(2 ** (-1 / 3))
    storage.get_profile_access_epochs.assert_called_once_with(
        user_id="u", profile_ids=["old", "p"]
    )
    storage.record_served_profile_access.assert_not_called()


def test_explicit_assessment_snapshot_is_readonly_and_preserved():
    storage = MagicMock()
    item = profile()
    with attention_context({"p": 100 + 30 * DAY}), storage_snapshot(storage):
        with candidate_attention([item]):
            assert decay_for_item(item, entity_type="profiles", now=100 + 30 * DAY) == 1
        record_accepted_profiles(storage, [item])
    storage.get_profile_access_epochs.assert_not_called()
    storage.record_served_profile_access.assert_not_called()


def test_no_session_embedded_acceptance_records_only_final_ids():
    storage = MagicMock()
    record_accepted_profiles(storage, [profile("final")])
    assert storage.record_served_profile_access.call_count == 1
    assert storage.record_served_profile_access.call_args.kwargs["profile_ids"] == [
        "final"
    ]
    assert storage.record_served_profile_access.call_args.kwargs["user_id"] == "u"


def test_dashboard_and_empty_responses_do_not_record():
    storage = MagicMock()
    with access_recording(False):
        record_accepted_profiles(storage, [profile()])
    record_accepted_profiles(storage, [])
    storage.record_served_profile_access.assert_not_called()


def test_unknown_origin_legacy_is_neutral_and_not_inferred_from_mtime():
    item = profile().model_copy(update={"custom_features": None})
    assert decay_for_item(item, entity_type="profiles", now=100 + 300 * DAY) == 1


def test_snapshot_rejects_cross_owner_candidates():
    storage = MagicMock()
    another = profile("x").model_copy(update={"user_id": "another"})
    with (
        storage_snapshot(storage),
        pytest.raises(ValueError, match="crossed owners"),
        candidate_attention([profile(), another]),
    ):
        pass
    storage.get_profile_access_epochs.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("abandoned", [False, True])
async def test_http_acceptance_records_but_timeout_does_not(abandoned):
    storage = MagicMock()
    delivered = False
    messages = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return await asyncio.Future()

    async def send(message):
        messages.append(message)

    async def app(scope, receive, send):
        record_accepted_profiles(storage, [profile("final")])
        if abandoned:
            await asyncio.sleep(0.1)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    runtime = search_runtime.SearchRuntimeMiddleware(
        app, timeout=0.02 if abandoned else 1
    )
    await runtime(
        {
            "type": "http",
            "path": "/api/search",
            "method": "POST",
            "headers": [],
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1),
        },
        receive,
        send,
    )
    await asyncio.gather(*runtime.tasks, return_exceptions=True)
    assert storage.record_served_profile_access.call_count == (0 if abandoned else 1)
    assert messages[0]["status"] == (504 if abandoned else 200)


def test_full_large_ancestry_is_read_in_bounded_batches():
    ancestors = [f"ancestor-{index:05}" for index in range(3000)]
    storage = MagicMock()
    storage.get_profile_access_epochs.side_effect = lambda **kw: {
        identity: 100 + 30 * DAY
        for identity in kw["profile_ids"]
        if identity == ancestors[-1]
    }
    item = profile(ancestors=ancestors)
    with storage_snapshot(storage), candidate_attention([item]):
        assert decay_for_item(item, entity_type="profiles", now=100 + 30 * DAY) == 1
    batches = [
        call.kwargs["profile_ids"]
        for call in storage.get_profile_access_epochs.call_args_list
    ]
    assert all(0 < len(batch) <= 256 for batch in batches)
    assert {identity for batch in batches for identity in batch} == {"p", *ancestors}
    storage.record_served_profile_access.assert_not_called()


def test_large_ancestry_respects_cancellation_between_batches(monkeypatch):
    storage = MagicMock()
    storage.get_profile_access_epochs.return_value = {}
    item = profile(ancestors=[f"ancestor-{index:05}" for index in range(3000)])
    checkpoints = 0

    def cancelled():
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 3:
            raise search_runtime.SearchDeadlineError("expired between attention reads")

    monkeypatch.setattr(search_runtime, "checkpoint", cancelled)
    with (
        storage_snapshot(storage),
        pytest.raises(search_runtime.SearchDeadlineError),
        candidate_attention([item]),
    ):
        pass
    assert storage.get_profile_access_epochs.call_count == 2
    assert all(
        len(call.kwargs["profile_ids"]) <= 256
        for call in storage.get_profile_access_epochs.call_args_list
    )
    storage.record_served_profile_access.assert_not_called()
