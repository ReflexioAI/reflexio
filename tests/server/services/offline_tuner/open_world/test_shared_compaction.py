"""Native guarded dispatch, restart accounting and bounded shared selections."""

import base64
import json
from copy import deepcopy
from hashlib import sha256

import litellm
import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.input_compression import CompressionBudget
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    MAX_SELECTION_BYTES,
    PART_BYTES,
    Attempt,
    Part,
    Root,
    Selection,
    SharedReceipt,
    decode_selection,
    digest,
    encode_selection,
    receipt,
)
from reflexio.server.services.offline_tuner.open_world.hierarchical_compaction import (
    compress_session_evidence,
)
from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
    JournaledCompressionClient,
    SharedCompactionJournal,
)
from reflexio.test_support.open_world_rule_evidence import (
    MemoryCheckpoints as MemoryCheckpoints,
)
from reflexio.test_support.open_world_rule_evidence import no_network as no_network
from tests.server.services.offline_tuner.open_world.test_compaction import (
    document,
)
from tests.server.services.offline_tuner.open_world.test_hierarchical_compaction import (
    _freeze_document,
)


def run(storage, source, trajectory):
    journal = SharedCompactionJournal(
        storage,
        job_id=7,
        fence=1,
        identity="a" * 64,
        before_call=lambda: None,
        now=lambda: 1,
        limit=128,
    )
    client = JournaledCompressionClient(
        LiteLLMConfig(
            model="minimax/MiniMax-M3",
            max_tokens=8192,
            max_retries=1,
            fallback_models=[],
        ),
        journal,
    )
    result = compress_session_evidence(
        source,
        project_id="project",
        through_request_id="request",
        frozen_trajectory=trajectory,
        governance_epochs=(("owner", 1),),
        budget=CompressionBudget(
            output_tokens=3000, input_tokens=8000, section_tokens=8000
        ),
        client=client,
        context_limits={"minimax/MiniMax-M3": 1_000_000},
        before_call=lambda: None,
        journal=journal,
    )
    return journal, result


def fake_transport(calls, *, usage=True, fail=False, repair=False):
    def complete(_client, params, _timeout):
        calls.append(deepcopy(params))
        if fail:
            raise RuntimeError("scripted transport interruption")
        data = json.loads(params["messages"][1]["content"])
        refs = [
            s["source_id"]
            for packet in data["input_packets"]
            for s in packet["evidence"]
        ]
        content = json.dumps({"points": [{"evidence_refs": refs[:1]}]})
        if repair and len(calls) == 1:
            content = "{"
        response = litellm.ModelResponse(
            model="MiniMax-M3",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
            if usage
            else None,
        )
        if not usage:
            response.usage = None  # pyright: ignore[reportAttributeAccessIssue]
        return response

    return complete


@pytest.mark.parametrize("usage", [False, True])
def test_native_compression_without_usage_resumes_from_committed_choices(
    monkeypatch, usage
):
    calls = []
    monkeypatch.setattr(
        LiteLLMClient,
        "_completion_with_hard_timeout",
        fake_transport(calls, usage=usage),
    )
    source, trajectory = _freeze_document(
        document("User correction: require explicit approval. " + "noise " * 6000)
    )
    storage = MemoryCheckpoints()
    first, (prepared, result) = run(storage, source, trajectory)
    assert result.calls > 0
    assert all(
        span.immutable_text in source.spans[0].immutable_text for span in prepared.spans
    )
    root = SharedReceipt.model_validate(storage.rows[(7, "a" * 64)])
    assert isinstance(root.payload, Root) and root.complete
    assert root.payload.calls == len(calls)
    attempts = [
        SharedReceipt.model_validate(row).payload
        for row in storage.rows.values()
        if row.get("payload", {}).get("kind") == "attempt"
    ]
    assert all(isinstance(a, Attempt) and a.response is not None for a in attempts)
    assert all(
        (a.response.prompt_tokens is None) == (not usage)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        for a in attempts
    )
    previous = len(calls)
    _, (again, replay) = run(storage, source, trajectory)
    assert len(calls) == previous and replay.cache_hits > 0
    assert again == prepared
    assert first.consumed_attempt is not None


def test_unknown_dispatch_remains_spent_after_restart(monkeypatch):
    calls = []
    storage = MemoryCheckpoints()
    source, trajectory = _freeze_document(
        document("material user correction. " + "noise " * 6000)
    )
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", fake_transport(calls, fail=True)
    )
    with pytest.raises(Exception, match="scripted transport interruption"):
        run(storage, source, trajectory)
    spent = len(calls)
    before = SharedReceipt.model_validate(storage.rows[(7, "a" * 64)]).payload
    assert isinstance(before, Root) and before.calls == spent and spent > 0
    old_attempts = [
        row
        for row in storage.rows.values()
        if row.get("payload", {}).get("kind") == "attempt"
    ]
    assert old_attempts and all(
        row["payload"]["response"] is None for row in old_attempts
    )
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", fake_transport(calls)
    )
    run(storage, source, trajectory)
    after = SharedReceipt.model_validate(storage.rows[(7, "a" * 64)]).payload
    assert isinstance(after, Root) and after.calls == len(calls) and after.calls > spent
    assert all(storage.rows[(7, row["identity"])] == row for row in old_attempts)


def test_small_input_uses_no_model_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", fake_transport(calls)
    )
    source, trajectory = _freeze_document(document())
    _, (prepared, result) = run(MemoryCheckpoints(), source, trajectory)
    assert calls == [] and result.passthrough and prepared == source


def test_parse_repair_records_each_actual_response(monkeypatch):
    calls = []
    monkeypatch.setattr(
        LiteLLMClient,
        "_completion_with_hard_timeout",
        fake_transport(calls, usage=False, repair=True),
    )
    source, trajectory = _freeze_document(
        document("material correction. " + "noise " * 6000)
    )
    storage = MemoryCheckpoints()
    run(storage, source, trajectory)
    first = SharedReceipt.model_validate(
        storage.rows[(7, digest({"root": "a" * 64, "attempt": 1}))]
    ).payload
    second = SharedReceipt.model_validate(
        storage.rows[(7, digest({"root": "a" * 64, "attempt": 2}))]
    ).payload
    assert isinstance(first, Attempt) and isinstance(second, Attempt)
    assert first.response is not None and second.response is not None
    assert first.response.content_digest != second.response.content_digest
    assert len(calls) >= 2


def test_large_choice_is_lossless_and_each_record_is_bounded():
    choice = {
        "points": [{"category": "selected_evidence", "evidence_refs": ["e1"] * 100_000}]
    }
    body, encoded = encode_selection(choice)
    manifest = Selection(
        root="a" * 64,
        node="b" * 64,
        input_refs_digest="c" * 64,
        request_contract_digest="d" * 64,
        body_digest=sha256(body).hexdigest(),
        body_bytes=len(body),
        parts=len(encoded),
        attempt="e" * 64,
        response_digest="f" * 64,
    )
    parts = [
        Part(
            root=manifest.root,
            node=manifest.node,
            body_digest=manifest.body_digest,
            index=i,
            data=data,
        )
        for i, data in enumerate(encoded)
    ]
    assert len(parts) > 1
    assert decode_selection(manifest, parts) == choice
    assert all(
        len(canonical_json_bytes(receipt(digest({"part": i}), p, complete=True)))
        < 22000
        for i, p in enumerate(parts)
    )
    with pytest.raises(ValueError, match="missing parts"):
        decode_selection(manifest, parts[:-1])
    with pytest.raises(ValueError, match="another source"):
        decode_selection(manifest, list(reversed(parts)))
    changed = parts.copy()
    changed[0] = changed[0].model_copy(
        update={"data": base64.b64encode(b"x" * PART_BYTES).decode()}
    )
    with pytest.raises(ValueError, match="digest changed"):
        decode_selection(manifest, changed)


def test_exact_logical_byte_limit_and_one_byte_over():
    choice = {
        "points": [{"category": "selected_evidence", "evidence_refs": []}],
        "overview": "",
    }
    base = len(canonical_json_bytes(choice))
    count, remainder = divmod(
        MAX_SELECTION_BYTES - base, 5
    )  # each added "e1" costs 5 bytes, except the first
    choice["points"][0]["evidence_refs"] = ["e1"] * count
    remainder = MAX_SELECTION_BYTES - len(canonical_json_bytes(choice))
    choice["overview"] = "x" * remainder
    body, parts = encode_selection(choice)
    assert len(body) == MAX_SELECTION_BYTES and len(parts) == 88
    choice["overview"] += "x"
    with pytest.raises(ValueError, match="1 MiB"):
        encode_selection(choice)


def test_salvaged_selection_binds_the_single_response_consumed(monkeypatch):
    calls = []
    original = fake_transport(calls)

    def partial(client, params, timeout):
        response = original(client, params, timeout)
        selection = json.loads(response.choices[0].message.content)  # pyright: ignore[reportArgumentType]
        selection["points"].append({"evidence_refs": [f"unseen-{len(calls)}"]})
        response.choices[0].message.content = json.dumps(selection)
        return response

    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", partial)
    source, trajectory = _freeze_document(
        document("Require approval. " + "noise " * 6000)
    )
    storage = MemoryCheckpoints()
    journal, (_, result) = run(storage, source, trajectory)
    assert result.provider_reference_recoveries > 0
    # Selection rule violations are settled by the host, never re-asked.
    assert result.provider_attempts == len(calls) >= 1
    observed = {
        key[1]: SharedReceipt.model_validate(value).payload
        for key, value in storage.rows.items()
        if value["payload"]["kind"] == "attempt"
    }
    selections = [
        SharedReceipt.model_validate(value).payload
        for value in storage.rows.values()
        if value["payload"]["kind"] == "selection"
    ]
    assert selections
    for selection in selections:
        attempt = observed[selection.attempt]  # pyright: ignore[reportAttributeAccessIssue]
        assert attempt.response.content_digest == selection.response_digest  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    assert len(selections) == len(observed) == len(calls)
    assert journal.consumed_attempt in observed


def test_checkpoint_failure_cannot_skip_native_usage_accounting(monkeypatch):
    from reflexio.server.llm.input_budget import extraction_usage_observer

    calls, metered = [], []
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", fake_transport(calls)
    )

    class FailingResponseCheckpoint(MemoryCheckpoints):
        def session_compaction_checkpoint(self, **kwargs):
            value = kwargs.get("receipt")
            if (
                value
                and value["payload"]["kind"] == "attempt"
                and value["payload"].get("response")
            ):
                raise RuntimeError("scripted response persistence failure")
            return super().session_compaction_checkpoint(**kwargs)

    source, trajectory = _freeze_document(
        document("Require approval. " + "noise " * 6000)
    )
    storage = FailingResponseCheckpoint()
    token = extraction_usage_observer.set(lambda *args, **_kwargs: metered.append(args))
    try:
        with pytest.raises(RuntimeError, match="response persistence"):
            run(storage, source, trajectory)
    finally:
        extraction_usage_observer.reset(token)
    assert len(metered) == len(calls) >= 1
    root = next(
        value["payload"]
        for value in storage.rows.values()
        if value["payload"]["kind"] == "root"
    )
    assert root["calls"] == len(calls)
    assert not any(
        value["payload"]["kind"] == "selection" for value in storage.rows.values()
    )
