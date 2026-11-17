"""Physical transport admission scope across every v2 actor stage, no network."""

import json
from dataclasses import replace

import pytest

from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    PROTOCOL,
    PROTOCOL_KEY,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    ObservedStructuredCompletion,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    qualify_rule_evidence_method,
)
from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
    shared_consumer_admission,
)
from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
    TriggerBatch,
    TriggerGate,
)
from reflexio.test_support.open_world_rule_evidence import (
    no_network as no_network,
)
from reflexio.test_support.open_world_rule_evidence import (
    runtime as runtime,
)
from reflexio.test_support.open_world_rule_evidence import (
    source as source,
)


def _configured(runtime, *, shared_protocol: bool, fail_stage: str | None):
    runner, transport, storage, kwargs = runtime
    # The historical fixture deliberately supplies its original two-component
    # method. Use the current qualified method while retaining its memory port.
    method = qualify_rule_evidence_method(transport)
    assert method.is_v2
    runner.store.method = method
    storage.authority = replace(
        storage.authority,
        authority_digest=method.digest,
        method_authority_json=method.canonical_json,
    )
    frozen = kwargs["frozen"]
    metadata = json.loads(frozen.job.metadata_json or "{}")
    if shared_protocol:
        metadata[PROTOCOL_KEY] = PROTOCOL
    else:
        metadata.pop(PROTOCOL_KEY, None)
    metadata_json = json.dumps(metadata)
    kwargs = {
        **kwargs,
        "frozen": replace(
            frozen, job=frozen.job.model_copy(update={"metadata_json": metadata_json})
        ),
    }
    storage.job.metadata_json = metadata_json
    original_complete = transport.complete
    observed = []

    def complete(**request):
        schema = request["response_format"]
        # Observe at the physical transport boundary rather than inferring
        # admission from the portable method's proxy-token limit.
        observed.append((schema, shared_consumer_admission.get()))
        if schema not in (TriggerGate, TriggerBatch):
            return original_complete(**request)
        stage = "gate" if schema is TriggerGate else "builder"
        if stage == fail_stage:
            raise TimeoutError("scripted Trigger transport timeout")
        payload = json.loads(request["messages"][1].content)
        if schema is TriggerGate:
            frame = payload["source"]
            raw = {
                "contrast": {
                    "gap": "opposed_consultation_state",
                    "missed_task": "Inspect report state",
                    "original_excludes_because": "Different report state",
                    "content_actually_provides": "Actual report-state guidance",
                    "revision_admits_because": "Consult the current state",
                    "support": [
                        {"field": "trigger", "quote": frame["trigger"]},
                        {"field": "content", "quote": frame["content"]},
                    ],
                    "missing_facts": [],
                },
                "revision": {
                    "action": "rewrite",
                    "trigger": "Consult for report state",
                    "reason": "Scripted supported shape; not a semantic grade",
                    "support": [{"field": "content", "quote": frame["content"]}],
                },
            }
        else:
            raw = {
                "revisions": [
                    {
                        "index": 0,
                        "revision": {
                            "action": "rewrite",
                            "trigger": "Consult for report state",
                            "reason": "Scripted source-only construction",
                            "support": [
                                {
                                    "field": "content",
                                    "quote": payload["sources"][0]["content"],
                                }
                            ],
                        },
                    }
                ]
            }
        return ObservedStructuredCompletion(
            value=schema.model_validate_json(json.dumps(raw)),
            observed_provider="test",
            observed_model_revision="frozen-model",
            called_at=200,
            latency_ms=10,
            usage_json=None,
        )

    transport.complete = complete
    return runner, storage, kwargs, observed


@pytest.mark.parametrize("shared_protocol", [False, True])
def test_v2_every_transport_stage_observes_the_frozen_admission_scope(
    runtime, shared_protocol
):
    runner, storage, kwargs, observed = _configured(
        runtime, shared_protocol=shared_protocol, fail_stage=None
    )
    token = shared_consumer_admission.set(False)
    try:
        result = runner.run(**kwargs)
        assert result.decision.decision == "accept"
        assert len(observed) == 4
        assert [schema for schema, _ in observed][-2:] == [TriggerGate, TriggerBatch]
        assert all(flag is shared_protocol for _, flag in observed)
        assert shared_consumer_admission.get() is False
        assert storage.job.stage == "held_out_analyzed"
    finally:
        shared_consumer_admission.reset(token)


@pytest.mark.parametrize("shared_protocol", [False, True])
@pytest.mark.parametrize("fail_stage", ["gate", "builder"])
def test_v2_transport_failure_does_not_leak_admission_or_publish(
    runtime, shared_protocol, fail_stage
):
    runner, storage, kwargs, observed = _configured(
        runtime, shared_protocol=shared_protocol, fail_stage=fail_stage
    )
    token = shared_consumer_admission.set(False)
    try:
        result = runner.run(**kwargs)
        assert result.reason == "infrastructure_failure"
        assert len(observed) == (3 if fail_stage == "gate" else 4)
        assert all(flag is shared_protocol for _, flag in observed)
        assert shared_consumer_admission.get() is False
        assert storage.job.stage == "failed"
        assert "open_world_candidate" not in storage.artifacts
        assert "open_world_attempt_decision" not in storage.artifacts
    finally:
        shared_consumer_admission.reset(token)
