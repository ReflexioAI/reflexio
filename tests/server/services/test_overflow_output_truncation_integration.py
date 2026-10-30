"""A request cut off by its output limit is not resent unchanged until quota runs out."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.overflow_coordinator import (
    MAX_TRUNCATED_REQUEST_DISPATCHES,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_overflow_policy_revalidation import (
    _make_due,
    _respond,
    _state,
)

_TURNS = 8


def _extraction_request(params) -> bool:
    """Profile extraction, as opposed to the deduplication call ``_respond`` serves."""
    schema = getattr(params.get("response_format"), "__name__", "") or json.dumps(
        params.get("response_format"), default=str
    )
    return "ProfileDeduplicationOutput" not in schema and not any(
        "unique_ids" in str(message.get("content", ""))
        for message in params["messages"]
    )


def _failing_extraction(finish_reason: str, content: str):
    """Answer every extraction call with a body that cannot parse."""

    def respond(params, hard_timeout):
        if not _extraction_request(params):
            return _respond(params, hard_timeout)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content, tool_calls=None, refusal=None
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=8192,
                total_tokens=8292,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    return respond


def _run(pipeline, monkeypatch, respond):
    engine, worker, client = pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    assert client.post(
        "/api/publish_interaction", json=payload("truncated", force=True)
    ).json()["success"]
    worker.drain_org(storage.org_id, 1, 300)
    for _ in range(_TURNS):
        _make_due(storage)
        worker.drain_org(storage.org_id, 1, 300)
    with storage._stream_sql(read_only=True) as db:
        window_id = db.query(
            "SELECT window_id FROM extraction_cursors WHERE user_id='u' AND kind='profile'"
        )[0]["window_id"]
    return storage, window_id, dispatch


def _request_counts(dispatch) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in dispatch.call_args_list:
        params = call.args[0]
        if _extraction_request(params):
            key = json.dumps(
                [params.get("model"), params.get("messages"), params.get("max_tokens")],
                sort_keys=True,
                default=str,
            )
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_truncated_request_is_held_after_bounded_dispatches(
    pipeline, monkeypatch, caplog
):
    storage, window_id, dispatch = _run(
        pipeline, monkeypatch, _failing_extraction("length", "")
    )
    cursor, run, checkpoint = _state(storage, window_id)
    counts = _request_counts(dispatch)
    assert counts
    assert max(counts.values()) == MAX_TRUNCATED_REQUEST_DISPATCHES
    manifest = run.generation_request_snapshot["overflow"]
    # Each distinct truncated request spent at most its bounded dispatches,
    # not the remaining 128-attempt budget.
    assert len(manifest["attempts"]) == sum(counts.values())
    assert len(manifest["attempts"]) <= MAX_TRUNCATED_REQUEST_DISPATCHES * len(counts)
    assert set(checkpoint["truncated_requests"].values()) == {
        MAX_TRUNCATED_REQUEST_DISPATCHES
    }
    assert cursor["retry_at"] == float("inf")
    assert cursor["last_error"] == "input_budget_exceeded"
    assert storage.extraction_status("u", "truncated") == {
        "status": "failed",
        "reason": "input_budget_exceeded",
    }
    assert storage.count_all_profiles() == 0
    assert "error=OutputTruncatedRequestError" in caplog.text


@pytest.mark.parametrize(
    ("finish_reason", "content"), [("stop", "not json"), ("stop", "")]
)
def test_other_parse_failures_keep_retrying(
    pipeline, monkeypatch, caplog, finish_reason, content
):
    storage, window_id, dispatch = _run(
        pipeline, monkeypatch, _failing_extraction(finish_reason, content)
    )
    cursor, run, checkpoint = _state(storage, window_id)
    assert "truncated_requests" not in checkpoint
    assert max(_request_counts(dispatch).values()) > MAX_TRUNCATED_REQUEST_DISPATCHES
    assert "OutputTruncatedRequestError" not in caplog.text


def test_request_that_finishes_after_transient_truncation_is_learned(
    pipeline, monkeypatch, caplog
):
    """Reasoning truncation is often transient; a later identical dispatch can finish."""
    truncate = _failing_extraction("length", "")
    sent: dict[str, int] = {}

    def respond(params, hard_timeout):
        if _extraction_request(params):
            key = json.dumps(params.get("messages"), sort_keys=True, default=str)
            sent[key] = sent.get(key, 0) + 1
            # Every request is cut off until its last allowed dispatch.
            if sent[key] < MAX_TRUNCATED_REQUEST_DISPATCHES:
                return truncate(params, hard_timeout)
        return _respond(params, hard_timeout)

    storage, window_id, dispatch = _run(pipeline, monkeypatch, respond)
    assert max(_request_counts(dispatch).values()) == MAX_TRUNCATED_REQUEST_DISPATCHES
    assert MAX_TRUNCATED_REQUEST_DISPATCHES > 2
    assert storage.extraction_status("u", "truncated")["status"] == "done"
    assert storage.count_all_profiles() > 0
    assert "OutputTruncatedRequestError" not in caplog.text
