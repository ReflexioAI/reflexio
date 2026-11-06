"""Shared compressor to real analyst-rendered evidence, with frozen authority."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
)
from reflexio.server.services.offline_tuner.open_world.hierarchical_compaction import (
    compress_session_evidence,
)
from tests.server.services.offline_tuner.open_world.test_compaction import (
    document,
)


def _freeze_document(value, *, empty_terminal: str | None = None):
    import hashlib

    from reflexio.models.api_schema.service_schemas import Interaction, Request
    from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
        project_trajectory_spans,
    )
    from reflexio.server.services.storage.session_outcome_identity import (
        canonical_session_trajectory,
        canonical_trajectory_bytes,
    )

    requests = {}
    interactions = {}
    for span in value.spans:
        requests.setdefault(
            span.request_id,
            Request(
                request_id=span.request_id,
                user_id=value.user_id,
                session_id=value.session_id,
                created_at=0,
            ).model_dump(mode="json"),
        )
        interactions.setdefault(span.request_id, []).append(
            Interaction(
                interaction_id=span.interaction_id,
                request_id=span.request_id,
                user_id=value.user_id,
                role={"user": "User", "agent": "Assistant", "other": "Tool"}[
                    span.speaker
                ],
                created_at=0,
                content=span.immutable_text,
            ).model_dump(mode="json")
        )
    if empty_terminal:
        requests[empty_terminal] = Request(
            request_id=empty_terminal,
            user_id=value.user_id,
            session_id=value.session_id,
            created_at=1,
        ).model_dump(mode="json")
        interactions[empty_terminal] = []
    for row in requests.values():
        row.update(
            evaluation_only=False,
            retrieval_experiment_id=None,
            retrieval_experiment_arm=None,
        )
    trajectory = canonical_session_trajectory(
        value.session_id, list(requests.values()), interactions
    )
    digest = hashlib.sha256(canonical_trajectory_bytes(trajectory)).hexdigest()
    spans = project_trajectory_spans(
        trajectory,
        trajectory_digest=digest,
        tenant_id=value.spans[0].tenant_id,
        role=value.manifest_role,
        user_id=value.user_id,
        session_id=value.session_id,
        include_tools=True,
    )
    return replace(value, trajectory_digest=digest, spans=spans), trajectory


def test_session_adapter_preserves_outcome_prefix_scope_and_citable_originals(
    monkeypatch,
):
    # Use the same real core/wrapper as the extraction adapter, and a REAL
    # client: the compression admission path (reflexio#649 onward) builds and
    # checks native request params through the client, so only the provider
    # transport is stubbed here, as the OSS adapter tests do.
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    requests: list[dict] = []

    def transport(params, _hard_timeout):
        requests.append(params)
        content = next(
            message["content"]
            for message in params["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith("{")
            and "input_packets" in json.JSONDecoder().raw_decode(message["content"])[0]
        )
        payload = json.JSONDecoder().raw_decode(content)[0]
        assert "outcome" not in payload and "manifest_role" not in payload
        assert "failure" not in content
        refs = [
            s["source_id"]
            for p in payload["input_packets"]
            for s in p["evidence"]
            if "approval" in s.get("text", "")
        ]
        choice = {"points": [{"evidence_refs": refs[:1]}] if refs else []}
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": json.dumps(choice)},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    original = document(
        "Staging requires approval; production requires separate approval. "
        + "log " * 5000
    )
    original, trajectory = _freeze_document(original)
    before = Mock()
    compacted, result = compress_session_evidence(
        original,
        project_id="project",
        through_request_id="request",
        frozen_trajectory=trajectory,
        governance_epochs=(("subject", 1),),
        budget=CompressionBudget(output_tokens=4096, section_tokens=1500),
        client=client,
        context_limits={client.config.model: 512000},
        before_call=before,
    )
    assert result.provider_attempts is not None
    assert before.call_count == result.provider_attempts + 2
    assert result.provider_attempts > 0
    assert len(requests) == result.provider_attempts
    assert all(request["model"] == client.config.model for request in requests)
    assert compacted.session_outcome == original.session_outcome == "failure"
    assert compacted.trajectory_digest == original.trajectory_digest
    assert compacted.manifest_role == original.manifest_role
    assert compacted.user_id == original.user_id
    assert compacted.spans
    assert compacted.compaction_summary is not None
    summary = json.loads(compacted.compaction_summary)
    assert summary["omission_counts"] == result.packet.omission_counts.model_dump()
    assert all(
        0 <= index < len(compacted.spans)
        for groups in summary["evidence_groups"].values()
        for group in groups
        for index in group
    )
    assert all(
        span.immutable_text in original.spans[0].immutable_text
        for span in compacted.spans
    )
    assert "production requires separate approval" in compacted.spans[0].immutable_text


def test_mixed_tenant_and_missing_prefix_rejected_before_model():
    original, trajectory = _freeze_document(document())
    mixed = replace(
        original,
        spans=(original.spans[0], replace(original.spans[0], tenant_id="other")),
    )
    for value, prefix in ((mixed, "request"), (original, "")):
        with pytest.raises(CompressionError):
            compress_session_evidence(
                value,
                project_id="project",
                through_request_id=prefix,
                frozen_trajectory=trajectory,
                governance_epochs=(),
                budget=CompressionBudget(output_tokens=4096),
                client=Mock(),
                context_limits={},
                before_call=Mock(),
            )


def test_session_passthrough_checks_encoded_consumer_instead_of_core_packet(
    monkeypatch,
):
    from reflexio.server.metering_signals import count_input_tokens
    from reflexio.server.services.input_compression import (
        CompressionChoice,
        EvidencePacket,
        EvidenceSource,
        compress_input,
        packet_tokens,
    )
    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )
    from reflexio.server.services.offline_tuner.open_world.analyst import (
        _canonical_user_message,
    )
    from reflexio.server.services.offline_tuner.open_world.identity import (
        canonical_payload,
    )

    original, trajectory = _freeze_document(
        document("Repeated diagnostic output. " * 500)
    )
    span = original.spans[0]
    core = EvidencePacket(
        evidence=[
            EvidenceSource(
                source_id=adapter._source_key(span),
                request_id=span.request_id,
                interaction_id=span.interaction_id,
                field=span.field_path,
                speaker=span.speaker,
                source_order=0,
                text=span.immutable_text,
            )
        ]
    )
    raw_size = packet_tokens(core)
    consumer_size = (
        count_input_tokens(
            _canonical_user_message({"document": canonical_payload(original)})
        )
        + 750
    )
    assert consumer_size > raw_size
    target = (raw_size + consumer_size) // 2
    completions = []

    def wrapper(sources, **kwargs):
        assert kwargs["measure_output"](core) == consumer_size > target

        def complete(call):
            completions.append(call)
            return CompressionChoice(
                overview="",
                points=[],
                unresolved=[],
                omissions=["Diagnostic output omitted."],
            )

        return compress_input(
            sources,
            scope=kwargs["scope"],
            policy=kwargs["policy"],
            budget=kwargs["budget"],
            model_contract="fixture",
            complete=complete,
            measure_output=kwargs["measure_output"],
            output_contract=kwargs["output_contract"],
            estimate_output=kwargs.get("estimate_output"),
        )

    monkeypatch.setattr(adapter, "compress_with_client", wrapper)
    compacted, result = adapter.compress_session_evidence(
        original,
        project_id="project",
        through_request_id="request",
        frozen_trajectory=trajectory,
        governance_epochs=(),
        budget=CompressionBudget(output_tokens=target),
        client=Mock(),
        context_limits={},
        before_call=lambda: None,
    )
    assert completions and not result.passthrough
    assert result.output_tokens <= target
    assert compacted.session_outcome == original.session_outcome


def test_session_append_reuses_leaves_and_rebinds_new_citable_ids(monkeypatch):
    from reflexio.server.services.input_compression import (
        CompressionChoice,
        CompressionPoint,
        compress_input,
    )
    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )

    original = document("log " * 2000)
    original = replace(
        original,
        spans=tuple(
            replace(
                original.spans[0],
                span_id=f"old-{i}",
                interaction_id=i,
                request_id=f"request-{i}",
                immutable_text=f"source {i}: " + "log " * 2000,
            )
            for i in range(1, 4)
        ),
    )
    original, trajectory = _freeze_document(original)
    cache = {}

    def complete(call):
        payload = json.loads(call.messages[1]["content"])
        refs = [
            record["source_id"]
            for packet in payload["input_packets"]
            for record in packet["evidence"]
            if record.get("text", "").startswith("source")
        ]
        return CompressionChoice(
            overview="",
            points=[
                CompressionPoint(
                    category="action",
                    detail="Retained original source.",
                    evidence_refs=[ref],
                )
                for ref in refs
            ],
            unresolved=[],
            omissions=[],
        )

    def wrapper(sources, **kwargs):
        return compress_input(
            sources,
            scope=kwargs["scope"],
            policy=kwargs["policy"],
            budget=kwargs["budget"],
            model_contract="fixture",
            complete=complete,
            load=kwargs["load"],
            save=kwargs["save"],
            measure_output=kwargs["measure_output"],
            output_contract=kwargs["output_contract"],
            estimate_output=kwargs.get("estimate_output"),
        )

    monkeypatch.setattr(adapter, "compress_with_client", wrapper)
    kwargs = {
        "project_id": "project",
        "through_request_id": "request-3",
        "frozen_trajectory": trajectory,
        "governance_epochs": (),
        "budget": CompressionBudget(output_tokens=4096, section_tokens=1500),
        "client": Mock(),
        "context_limits": {},
        "before_call": lambda: None,
        "load": cache.get,
        "save": lambda k, v: cache.update({k: v}),
    }
    _, first = adapter.compress_session_evidence(original, **kwargs)
    appended = replace(
        original,
        trajectory_digest="b" * 64,
        spans=tuple(
            replace(s, span_id=f"new-{i}") for i, s in enumerate(original.spans)
        )
        + (
            replace(
                original.spans[-1],
                span_id="appended",
                interaction_id=4,
                request_id="request-4",
            ),
        ),
    )
    appended, appended_trajectory = _freeze_document(appended)
    kwargs["through_request_id"] = "request-4"
    kwargs["frozen_trajectory"] = appended_trajectory
    compacted, second = adapter.compress_session_evidence(appended, **kwargs)
    assert first.calls > 1 and second.cache_hits >= first.sections - 1
    assert second.source_digest != first.source_digest
    assert compacted.trajectory_digest == appended.trajectory_digest
    assert compacted.spans
    assert all(
        span.span_id not in {old.span_id for old in original.spans}
        for span in compacted.spans
    )
    assert all(
        any(span.immutable_text in source.immutable_text for source in appended.spans)
        for span in compacted.spans
    )


@pytest.mark.parametrize("role", ["held_out", "safety_only"])
def test_unlabelled_session_pricing_and_passthrough_preserve_role(monkeypatch, role):
    from reflexio.server.services.input_compression import compress_input
    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )

    original = document()
    original = replace(
        original,
        manifest_role=role,
        session_outcome=None,
        spans=tuple(replace(span, manifest_role=role) for span in original.spans),
    )

    original, trajectory = _freeze_document(original)

    def wrapper(sources, **kwargs):
        return compress_input(
            sources,
            scope=kwargs["scope"],
            policy=kwargs["policy"],
            budget=kwargs["budget"],
            model_contract="fixture",
            complete=Mock(
                side_effect=AssertionError("Small document must pass through")
            ),
            measure_output=kwargs["measure_output"],
            estimate_output=kwargs["estimate_output"],
            output_contract=kwargs["output_contract"],
        )

    monkeypatch.setattr(adapter, "compress_with_client", wrapper)
    compacted, result = adapter.compress_session_evidence(
        original,
        project_id="project",
        through_request_id="request",
        frozen_trajectory=trajectory,
        governance_epochs=(),
        budget=CompressionBudget(output_tokens=4096),
        client=Mock(),
        context_limits={},
        before_call=lambda: None,
    )
    assert result.passthrough and result.calls == 0
    assert compacted == original
    assert compacted.session_outcome is None


def test_frozen_empty_terminal_is_validated_and_bound_without_invalidating_leaf_scope(
    monkeypatch,
):
    from reflexio.server.services.input_compression import compress_input
    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )

    scopes = []
    source_keys = []

    def wrapper(sources, **kwargs):
        scopes.append(kwargs["scope"])
        source_keys.append([s.source_id for s in sources])
        return compress_input(
            sources,
            scope=kwargs["scope"],
            policy=kwargs["policy"],
            budget=kwargs["budget"],
            model_contract="fixture",
            complete=Mock(side_effect=AssertionError("Small input must pass through")),
            measure_output=kwargs["measure_output"],
            estimate_output=kwargs["estimate_output"],
            output_contract=kwargs["output_contract"],
        )

    dispatch = Mock(side_effect=wrapper)
    monkeypatch.setattr(adapter, "compress_with_client", dispatch)
    plain, plain_trajectory = _freeze_document(document())
    empty, empty_trajectory = _freeze_document(document(), empty_terminal="empty-final")
    kwargs = {
        "project_id": "project",
        "governance_epochs": (),
        "budget": CompressionBudget(output_tokens=4096),
        "client": Mock(),
        "context_limits": {},
        "before_call": Mock(),
    }
    with pytest.raises(CompressionError, match="terminal prefix"):
        adapter.compress_session_evidence(
            empty,
            frozen_trajectory=empty_trajectory,
            through_request_id="request",
            **kwargs,
        )
    dispatch.assert_not_called()
    kwargs["before_call"].assert_not_called()
    _, first = adapter.compress_session_evidence(
        plain,
        frozen_trajectory=plain_trajectory,
        through_request_id="request",
        **kwargs,
    )
    output, second = adapter.compress_session_evidence(
        empty,
        frozen_trajectory=empty_trajectory,
        through_request_id="empty-final",
        **kwargs,
    )
    assert output == empty
    assert scopes[0] == scopes[1] and source_keys[0] == source_keys[1]
    assert first.source_digest != second.source_digest


@pytest.mark.parametrize("tamper", ["trajectory", "document_text", "scope"])
def test_frozen_prefix_proof_rejects_mismatch_before_cache_or_provider(
    monkeypatch, tamper
):
    from copy import deepcopy

    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )

    original, trajectory = _freeze_document(document())
    if tamper == "trajectory":
        trajectory = deepcopy(trajectory)
        trajectory["requests"][0]["interactions"][0]["content"] += "changed"
    else:
        span = replace(
            original.spans[0],
            **(
                {"immutable_text": "changed"}
                if tamper == "document_text"
                else {"user_id": "other"}
            ),
        )
        original = replace(original, user_id=span.user_id, spans=(span,))
    dispatch = Mock(
        side_effect=AssertionError("Invalid frozen proof must not dispatch")
    )
    monkeypatch.setattr(adapter, "compress_with_client", dispatch)
    before = Mock()
    with pytest.raises(CompressionError, match="exact frozen trajectory"):
        adapter.compress_session_evidence(
            original,
            frozen_trajectory=trajectory,
            through_request_id="request",
            project_id="project",
            governance_epochs=(),
            budget=CompressionBudget(output_tokens=4096),
            client=Mock(),
            context_limits={},
            before_call=before,
        )
    dispatch.assert_not_called()
    before.assert_not_called()


@pytest.mark.parametrize("field", ["request_session", "interaction_request"])
def test_self_consistent_digest_cannot_authorize_foreign_row_binding(
    monkeypatch, field
):
    import hashlib

    from reflexio.server.services.offline_tuner.open_world import (
        hierarchical_compaction as adapter,
    )
    from reflexio.server.services.storage.session_outcome_identity import (
        canonical_trajectory_bytes,
    )

    original, trajectory = _freeze_document(document())
    if field == "request_session":
        trajectory["requests"][0]["request"]["session_id"] = "foreign-session"
    else:
        trajectory["requests"][0]["interactions"][0]["request_id"] = "foreign-request"
    digest = hashlib.sha256(canonical_trajectory_bytes(trajectory)).hexdigest()
    original = replace(
        original,
        trajectory_digest=digest,
        spans=adapter.project_trajectory_spans(
            trajectory,
            trajectory_digest=digest,
            tenant_id=original.spans[0].tenant_id,
            role=original.manifest_role,
            user_id=original.user_id,
            session_id=original.session_id,
            include_tools=True,
        ),
    )
    dispatch = Mock(
        side_effect=AssertionError("Foreign row binding must never dispatch")
    )
    monkeypatch.setattr(adapter, "compress_with_client", dispatch)
    with pytest.raises(CompressionError, match="exact frozen trajectory"):
        adapter.compress_session_evidence(
            original,
            frozen_trajectory=trajectory,
            through_request_id="request",
            project_id="project",
            governance_epochs=(),
            budget=CompressionBudget(output_tokens=4096),
            client=Mock(),
            context_limits={},
            before_call=Mock(),
        )
    dispatch.assert_not_called()
