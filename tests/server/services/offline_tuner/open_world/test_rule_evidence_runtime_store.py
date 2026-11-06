"""Durable-binding refusals of the rule-evidence analysis store.

`RuleEvidenceAnalysisStore` is the boundary between the promoted method and the
invocation / artifact ledger. Each test here completes one real (fake-transport)
method run through `test_runner.py`'s in-memory store, then tampers with exactly
one durable binding -- authority, attempt context, artifact bytes, invocation
row, receipt or publication input -- and asserts the store refuses with its
specific error instead of returning unbound output or making a provider call.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
)
from reflexio.server.llm import LiteLLMClientError
from reflexio.server.llm.litellm_client import StructuredOutputParseError
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSchemaUnsupportedError,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    candidate_payload,
    rejected_model_output,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationConflictError,
    OpenWorldInvocationIdentityBoundError,
    OpenWorldInvocationLeaseConflictError,
    OpenWorldInvocationSlotExhaustedError,
    OpenWorldQualificationAuthorityStaleError,
)
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)
from tests.server.services.offline_tuner.open_world.test_runner import (
    runtime as runtime,
)

CLAIM = OptimizationJobClaim(job_id=1, owner="worker", fence=1, expires_at=4000)


def _parse_error(**changes: Any) -> StructuredOutputParseError:
    values: dict[str, Any] = {
        "finish_reason": "stop",
        "validation_errors": ("observations: field required",),
    }
    values.update(changes)
    return StructuredOutputParseError("schema rejected the reply", **values)


def _wrapped(inner: BaseException) -> LiteLLMClientError:
    try:
        raise LiteLLMClientError("structured output failed") from inner
    except LiteLLMClientError as exc:
        return exc


# --- rejected_model_output --------------------------------------------------


def test_a_complete_reply_rejected_by_the_schema_is_output_rejected() -> None:
    assert rejected_model_output(_parse_error()) is True
    assert rejected_model_output(_wrapped(_parse_error())) is True


@pytest.mark.parametrize(
    "error",
    [
        _parse_error(finish_reason="length"),
        _parse_error(validation_errors=()),
        _parse_error(empty_response=True),
        RuntimeError("transport reset"),
    ],
)
def test_cut_empty_or_non_parse_failures_are_not_output_rejected(
    error: BaseException,
) -> None:
    assert rejected_model_output(_wrapped(error)) is False


def test_a_handled_rejection_in_the_implicit_context_does_not_relabel() -> None:
    # The corrective retry runs inside the first rejection's handler, so a
    # timeout on that retry carries the rejection only as __context__.
    try:
        try:
            raise _parse_error()
        except StructuredOutputParseError:
            raise TimeoutError("retry timed out")  # noqa: B904
    except TimeoutError as exc:
        assert exc.__context__ is not None
        assert rejected_model_output(exc) is False


def test_the_cause_walk_is_bounded() -> None:
    error: BaseException = _parse_error()
    for depth in range(8):
        outer = RuntimeError(f"wrap {depth}")
        outer.__cause__ = error
        error = outer
    # The parse error sits nine links down; the walk stops at eight.
    assert rejected_model_output(error) is False


def test_a_cyclic_cause_chain_terminates() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert rejected_model_output(first) is False


# --- helpers over a completed run ---------------------------------------------


def _completed(runtime: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    runner, transport, storage, kwargs = runtime
    assert runner.run(**kwargs).decision.decision == "accept"
    return runner, transport, storage, kwargs


def _call(transport: Any, index: int) -> tuple[str, dict[str, Any], Any]:
    request = transport.calls[index]
    return (
        request["messages"][0].content,
        json.loads(request["messages"][1].content),
        request["response_format"],
    )


def _rewrite_payload(storage: Any, kind: str, mutate: Any) -> None:
    """Replace an artifact with a canonically re-encoded, re-digested payload."""
    artifact = storage.artifacts[kind]
    raw = json.loads(artifact.content_json)
    mutate(raw["payload"])
    text = canonical_json_bytes(raw).decode()
    storage.artifacts[kind] = artifact.model_copy(
        update={
            "content_json": text,
            "content_digest": sha256(text.encode()).hexdigest(),
        }
    )


def _diagnostic_row_key(storage: Any) -> str:
    return next(
        key for key, row in storage.rows.items() if row.component_role == "discovery"
    )


# --- authority -------------------------------------------------------------------


def test_reconciled_active_authority_yields_the_method_reference(runtime) -> None:
    runner, _, storage, _ = runtime
    method = runner.store.method
    storage.reconcile_rule_evidence_authority = lambda **kwargs: (
        storage.reconciled.append(kwargs) or storage.authority
    )
    storage.reconciled = []
    ref = runner.store.reconcile_authority()
    assert ref == method.authority_ref(epoch=1)
    assert storage.reconciled == [
        {
            "authority_json": method.canonical_json,
            "authority_digest": method.digest,
            "now": 200,
        }
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "unavailable"},
        {"held_out_component_identity_digest": "e" * 64},
        {"candidate_generator_identity_digest": "e" * 64},
        {"method_authority_json": "{}"},
    ],
)
def test_stale_or_incomplete_authority_is_refused(runtime, changes) -> None:
    runner, _, storage, _ = runtime
    stale = replace(storage.authority, **changes)
    storage.reconcile_rule_evidence_authority = lambda **_kwargs: stale
    with pytest.raises(OpenWorldQualificationAuthorityStaleError):
        runner.store.reconcile_authority()


def test_missing_authority_is_refused(runtime) -> None:
    runner, _, storage, kwargs = runtime
    storage.authority = None
    with pytest.raises(OpenWorldQualificationAuthorityStaleError):
        runner.store.bind_context(kwargs["frozen"].bundle, CLAIM)


@pytest.mark.parametrize(
    "ceiling", ["PAYLOAD_PROXY_TOKEN_LIMIT", "GENERATED_CONTEXT_TOKEN_LIMIT"]
)
def test_changed_input_budget_cannot_reuse_prior_method_authority(
    runtime, monkeypatch: pytest.MonkeyPatch, ceiling: str
) -> None:
    from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
        qualify_rule_evidence_method,
    )
    from reflexio.server.services.playbook import rule_evidence

    runner, transport, _, kwargs = _completed(runtime)
    old_method = runner.store.method
    calls = len(transport.calls)
    monkeypatch.setattr(rule_evidence, ceiling, getattr(rule_evidence, ceiling) - 1)
    new_method = qualify_rule_evidence_method(transport)
    for old, new in (
        (old_method.diagnostic, new_method.diagnostic),
        (old_method.proposal, new_method.proposal),
    ):
        assert old.prompt_template_digest == new.prompt_template_digest
        assert old.output_schema_digest == new.output_schema_digest
        assert old.generation_settings_digest != new.generation_settings_digest
        assert old.digest != new.digest
    assert old_method.digest != new_method.digest
    runner.store.method = new_method
    with pytest.raises(OpenWorldQualificationAuthorityStaleError):
        runner.store.bind_context(kwargs["frozen"].bundle, CLAIM)
    assert len(transport.calls) == calls


def test_changed_wire_normalization_cannot_reuse_prior_method_authority(
    runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.server.services.offline_tuner.open_world import (
        rule_evidence_authority,
    )

    runner, transport, _, kwargs = _completed(runtime)
    old_method = runner.store.method
    calls = len(transport.calls)
    monkeypatch.setattr(
        rule_evidence_authority, "WIRE_NORMALIZATION_VERSION", "next-normalizer"
    )
    new_method = rule_evidence_authority.qualify_rule_evidence_method(transport)
    for old, new in (
        (old_method.diagnostic, new_method.diagnostic),
        (old_method.proposal, new_method.proposal),
    ):
        assert old.output_schema_digest == new.output_schema_digest
        assert old.generation_settings_digest != new.generation_settings_digest
        assert old.digest != new.digest
    assert old_method.digest != new_method.digest
    runner.store.method = new_method
    with pytest.raises(OpenWorldQualificationAuthorityStaleError):
        runner.store.bind_context(kwargs["frozen"].bundle, CLAIM)
    assert len(transport.calls) == calls


# --- attempt context and artifacts ----------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"job_id": 2},
        {"bundle_digest": "f" * 64},
        {"qualification_authority_epoch": 9},
        {"context_digest": "f" * 64},
    ],
)
def test_a_changed_attempt_context_binding_is_refused(runtime, changes) -> None:
    runner, _, storage, kwargs = runtime
    bind = storage.bind_open_world_attempt_context
    storage.bind_open_world_attempt_context = lambda **request: replace(
        bind(**request), **changes
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="context binding"):
        runner.store.bind_context(kwargs["frozen"].bundle, CLAIM)


def test_a_persisted_artifact_with_other_bytes_is_refused(runtime) -> None:
    runner, _, storage, _ = runtime
    write = storage.write_open_world_artifact
    # The ledger answers with an artifact already bound to other content.
    storage.write_open_world_artifact = lambda **request: write(**request).model_copy(
        update={"content_digest": "f" * 64}
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="persisted artifact"):
        runner.store.write("open_world_discovery_memo", {"a": 1}, CLAIM)


def test_an_absent_artifact_reads_as_none(runtime) -> None:
    runner, *_ = runtime
    assert runner.store.read("open_world_candidate", "x", CLAIM) is None


def _store_artifact(storage: Any, raw: Any, *, text: str | None = None, **changes):
    text = text if text is not None else canonical_json_bytes(raw).decode()
    values: dict[str, Any] = {
        "job_id": 1,
        "artifact_kind": "open_world_candidate",
        "content_json": text,
        "content_digest": sha256(text.encode()).hexdigest(),
        "created_at": 200,
        "updated_at": 200,
    }
    values.update(changes)
    # model_construct: the tampered shapes are exactly what validation rejects,
    # so they can only arrive from a ledger that bypassed it.
    storage.artifacts["open_world_candidate"] = (
        PlaybookOptimizationArtifact.model_construct(**values)
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"contract_type": "other", "schema_version": "v1", "payload": {}},
        {"contract_type": "rule_evidence_candidate", "schema_version": "v2"},
    ],
)
def test_an_unsupported_artifact_schema_is_refused(runtime, raw) -> None:
    runner, _, storage, _ = runtime
    _store_artifact(storage, raw)
    with pytest.raises(OpenWorldEvidenceSchemaUnsupportedError):
        runner.store.read("open_world_candidate", "rule_evidence_candidate", CLAIM)


def test_artifact_canonical_binding_is_enforced(runtime) -> None:
    runner, _, storage, _ = runtime
    good = {"contract_type": "rule_evidence_candidate", "schema_version": "v1"}
    cases = [
        ({**good, "payload": {}, "extra": 1}, {}),
        ({**good, "payload": []}, {}),
        ({**good, "payload": {}}, {"text": json.dumps({**good, "payload": {}})}),
        ({**good, "payload": {}}, {"content_digest": "f" * 64}),
        ({**good, "payload": {}}, {"job_id": 2}),
    ]
    for raw, changes in cases:
        _store_artifact(storage, raw, **changes)
        with pytest.raises(OpenWorldInvocationConflictError, match="canonical"):
            runner.store.read("open_world_candidate", "rule_evidence_candidate", CLAIM)
    _store_artifact(storage, {**good, "payload": {"k": "v"}})
    artifact, payload = runner.store.read(
        "open_world_candidate", "rule_evidence_candidate", CLAIM
    )
    assert payload == {"k": "v"} and artifact.job_id == 1


# --- invocation ledger -------------------------------------------------------------


def test_reinvoking_a_completed_call_replays_without_a_provider_call(
    runtime,
) -> None:
    runner, transport, _, kwargs = _completed(runtime)
    system, payload, schema = _call(transport, 0)
    calls = len(transport.calls)
    value, receipt = runner.store.invoke(
        bundle=kwargs["frozen"].bundle,
        claim=CLAIM,
        system=system,
        payload=payload,
        schema=schema,
    )
    assert len(transport.calls) == calls
    assert receipt["invocation"]["observed_provider"] == "test"
    assert value.model_dump(mode="json") == receipt["observed_output"]


def test_prompt_drift_is_refused_before_any_ledger_write(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    system, payload, schema = _call(transport, 0)
    rows = dict(storage.rows)
    with pytest.raises(OpenWorldQualificationAuthorityStaleError, match="drift"):
        runner.store.invoke(
            bundle=kwargs["frozen"].bundle,
            claim=CLAIM,
            system=system + " ",
            payload=payload,
            schema=schema,
        )
    assert storage.rows == rows


def test_a_diagnostic_for_a_session_outside_the_frozen_source_is_refused(
    runtime,
) -> None:
    runner, transport, _, kwargs = _completed(runtime)
    system, payload, schema = _call(transport, 0)
    with pytest.raises(OpenWorldInvocationConflictError, match="not frozen"):
        runner.store.invoke(
            bundle=kwargs["frozen"].bundle,
            claim=CLAIM,
            system=system,
            payload={**payload, "session_id": "session-unfrozen"},
            schema=schema,
        )


def test_bound_slots_advance_to_the_next_attempt_then_exhaust(runtime) -> None:
    runner, transport, storage, kwargs = runtime
    prepare = storage.prepare_open_world_invocation
    attempts: list[str] = []

    def bound_twice(*, binding, expected, **rest):
        attempts.append(binding.analyst_input_identity)
        if len(attempts) <= 2:
            raise OpenWorldInvocationIdentityBoundError("bound to another job")
        return prepare(binding=binding, expected=expected, **rest)

    storage.prepare_open_world_invocation = bound_twice
    result = runner.run(**kwargs)
    assert result.decision.decision == "accept"
    # Three distinct slot identities were tried for the first question.
    assert len(set(attempts[:3])) == 3

    storage.prepare_open_world_invocation = lambda **_kw: (_ for _ in ()).throw(
        OpenWorldInvocationIdentityBoundError("bound")
    )
    system, payload, schema = _call(transport, 0)
    with pytest.raises(OpenWorldInvocationSlotExhaustedError):
        runner.store.invoke(
            bundle=kwargs["frozen"].bundle,
            claim=CLAIM,
            system=system,
            payload=payload,
            schema=schema,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"user_id": "user-other"}, "binding changed"),
        ({"input_digest": "f" * 64}, "input changed"),
        ({"observed_provider": "other"}, "provenance/output changed"),
        ({"output_json": None}, "provenance/output changed"),
    ],
)
def test_a_tampered_invocation_row_is_refused(runtime, changes, message) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    key = _diagnostic_row_key(storage)
    storage.rows[key] = replace(storage.rows[key], **changes)
    system, payload, schema = _call(transport, 0)
    with pytest.raises(OpenWorldInvocationConflictError, match=message):
        runner.store.invoke(
            bundle=kwargs["frozen"].bundle,
            claim=CLAIM,
            system=system,
            payload=payload,
            schema=schema,
        )


def test_a_non_canonical_but_digest_consistent_output_is_refused(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    key = _diagnostic_row_key(storage)
    row = storage.rows[key]
    # Same JSON value, different bytes: the digest matches the stored bytes,
    # but the bytes are not the canonical encoding of the observed output.
    loose = json.dumps(json.loads(row.output_json), indent=1)
    storage.rows[key] = replace(
        row, output_json=loose, output_digest=sha256(loose.encode()).hexdigest()
    )
    system, payload, schema = _call(transport, 0)
    with pytest.raises(OpenWorldInvocationConflictError, match="contract changed"):
        runner.store.invoke(
            bundle=kwargs["frozen"].bundle,
            claim=CLAIM,
            system=system,
            payload=payload,
            schema=schema,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "prepared"},
        {"called_at": None},
        {"latency_ms": None},
        {"observed_model_revision": None},
    ],
)
def test_an_incomplete_invocation_record_is_refused(runtime, changes) -> None:
    runner, *_ = runtime
    row = next(iter(_completed(runtime)[2].rows.values()))
    with pytest.raises(OpenWorldInvocationConflictError, match="not complete"):
        runner.store._record(replace(row, **changes))


# --- receipt verification ------------------------------------------------------------


def _diagnostic_receipt(storage: Any) -> dict[str, Any]:
    raw = json.loads(storage.artifacts["open_world_discovery_memo"].content_json)
    session = raw["payload"]["sessions"][0]
    return {key: session[key] for key in ("prepared", "invocation", "observed_output")}


def _verify(runner: Any, transport: Any, kwargs: Any, receipt: dict[str, Any]):
    system, payload, schema = _call(transport, 0)
    return runner.store.verify_receipt(
        receipt,
        bundle=kwargs["frozen"].bundle,
        claim=CLAIM,
        schema=schema,
        system=system,
        payload=payload,
    )


def test_an_intact_receipt_verifies_to_the_observed_output(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    receipt = _diagnostic_receipt(storage)
    output = _verify(runner, transport, kwargs, receipt)
    assert output.model_dump(mode="json") == receipt["observed_output"]


def test_a_receipt_for_another_component_is_stale(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    receipt = _diagnostic_receipt(storage)
    receipt["prepared"]["component_identity"] = {
        **receipt["prepared"]["component_identity"],
        "provider": "other",
    }
    with pytest.raises(OpenWorldQualificationAuthorityStaleError, match="component"):
        _verify(runner, transport, kwargs, receipt)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda prepared: prepared.update(request_json=prepared["request_json"] + " "),
        lambda prepared: prepared["binding"].update(job_id=2),
        lambda prepared: prepared["analyst_input_identity"].update(attempt=2),
        lambda prepared: prepared["expected"].update(expected_provider="other"),
    ],
)
def test_a_receipt_that_does_not_bind_the_exact_request_is_refused(
    runtime, mutate
) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    receipt = _diagnostic_receipt(storage)
    mutate(receipt["prepared"])
    with pytest.raises(OpenWorldInvocationConflictError):
        _verify(runner, transport, kwargs, receipt)


def test_a_receipt_whose_invocation_record_differs_is_refused(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    receipt = _diagnostic_receipt(storage)
    receipt["invocation"] = {**receipt["invocation"], "latency_ms": 999}
    with pytest.raises(OpenWorldInvocationConflictError, match="durable invocation"):
        _verify(runner, transport, kwargs, receipt)


def test_a_receipt_whose_observed_output_differs_is_refused(runtime) -> None:
    runner, transport, storage, kwargs = _completed(runtime)
    receipt = _diagnostic_receipt(storage)
    receipt["observed_output"] = {
        **receipt["observed_output"],
        "missing_information": ["an invented gap"],
    }
    with pytest.raises(OpenWorldInvocationConflictError, match="observed output"):
        _verify(runner, transport, kwargs, receipt)


# --- accepted publication inputs ----------------------------------------------------


def _publication_ready(runtime: Any) -> tuple[Any, Any, dict[str, Any]]:
    runner, _, storage, kwargs = _completed(runtime)
    frozen = kwargs["frozen"]
    storage.job.target_id = frozen.bundle.target.user_playbook_id
    storage.job.expected_population_manifest_digest = frozen.bundle.expected.digest
    storage.job.attempt_key = "attempt-1"
    storage.load_open_world_evidence_bundle = lambda _job_id: frozen
    storage.get_user_playbook_by_id = lambda _id: kwargs["incumbent"]
    return runner, storage, kwargs


def _load(runner: Any, claim: OptimizationJobClaim = CLAIM, now: int = 200) -> Any:
    return runner.store.load_accepted_publication_inputs(job_id=1, claim=claim, now=now)


def test_an_accepted_exact_edit_yields_bound_publication_inputs(runtime) -> None:
    runner, storage, kwargs = _publication_ready(runtime)
    inputs = _load(runner)
    bundle = kwargs["frozen"].bundle
    assert inputs.attempt_key == "attempt-1"
    assert inputs.evidence_bundle_digest == bundle.digest
    assert inputs.candidate.revision.revised_content == (
        "Send once. Report queued as queued. Keep the recipient."
    )
    assert (
        inputs.candidate_digest
        == storage.artifacts["open_world_candidate"].content_digest
    )
    assert inputs.decision.decision.decision == "accept"
    assert inputs.qualification_authority == runner.store.method.authority_ref(epoch=1)


@pytest.mark.parametrize(
    ("claim", "now", "job_changes"),
    [
        (CLAIM.model_copy(update={"job_id": 2}), 200, {}),
        (CLAIM, 200, {"status": "failed"}),
        (CLAIM, 200, {"stage": "candidate_generated"}),
        (CLAIM, 200, {"lease_owner": "peer"}),
        (CLAIM, 200, {"lease_fence": 2}),
        (CLAIM, 200, {"lease_expires_at": 5000}),
        (CLAIM, 4000, {}),
    ],
)
def test_a_stale_publication_lease_or_stage_is_refused(
    runtime, claim, now, job_changes
) -> None:
    runner, storage, _ = _publication_ready(runtime)
    for name, value in job_changes.items():
        setattr(storage.job, name, value)
    with pytest.raises(OpenWorldInvocationLeaseConflictError, match="stale"):
        _load(runner, claim, now)


def test_a_missing_job_is_a_stale_publication(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    storage.get_playbook_optimization_job = lambda _job_id: None
    with pytest.raises(OpenWorldInvocationLeaseConflictError):
        _load(runner)


@pytest.mark.parametrize(
    "change",
    [
        lambda storage: setattr(storage, "org_id", "tenant-other"),
        lambda storage: setattr(storage.job, "target_id", 999),
        lambda storage: setattr(
            storage.job, "expected_population_manifest_digest", "f" * 64
        ),
    ],
)
def test_a_publication_source_bundle_that_changed_is_refused(runtime, change):
    runner, storage, _ = _publication_ready(runtime)
    change(storage)
    with pytest.raises(OpenWorldInvocationConflictError, match="source bundle"):
        _load(runner)


def test_a_frozen_artifact_digest_mismatch_is_refused(runtime) -> None:
    runner, storage, kwargs = _publication_ready(runtime)
    frozen = kwargs["frozen"]
    tampered = SimpleNamespace(
        artifact=frozen.artifact.model_copy(update={"content_digest": "f" * 64}),
        bundle=frozen.bundle,
    )
    storage.load_open_world_evidence_bundle = lambda _job_id: tampered
    with pytest.raises(OpenWorldInvocationConflictError, match="source bundle"):
        _load(runner)


def test_a_missing_incumbent_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    storage.get_user_playbook_by_id = lambda _id: None
    with pytest.raises(OpenWorldInvocationConflictError, match="incumbent is missing"):
        _load(runner)


def test_a_missing_diagnostic_artifact_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    del storage.artifacts["open_world_discovery_memo"]
    with pytest.raises(OpenWorldInvocationConflictError, match="diagnostic artifact"):
        _load(runner)


@pytest.mark.parametrize(
    "field", ["job_id", "bundle_digest", "authority_epoch", "authority_digest"]
)
def test_a_diagnostic_collection_for_another_context_is_refused(runtime, field) -> None:
    runner, storage, _ = _publication_ready(runtime)
    _rewrite_payload(
        storage,
        "open_world_discovery_memo",
        lambda payload: payload.update(
            {field: 2 if field.endswith(("id", "epoch")) else "f" * 64}
        ),
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="diagnostic context"):
        _load(runner)


def test_a_rewritten_diagnostic_population_digest_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    _rewrite_payload(
        storage,
        "open_world_discovery_memo",
        lambda payload: payload.update(diagnostic_population_digest="f" * 64),
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="population changed"):
        _load(runner)


def test_a_rewritten_candidate_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    _rewrite_payload(
        storage,
        "open_world_candidate",
        lambda payload: payload.update(revised_content="Report queued as sent."),
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="exact-edit artifact"):
        _load(runner)


def test_a_receipt_missing_from_the_collection_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    # Without the candidate the method still asks for its proposal, and no
    # durable output is left to answer that call.
    del storage.artifacts["open_world_candidate"]
    with pytest.raises(OpenWorldInvocationConflictError, match="output is missing"):
        _load(runner)


def test_a_duplicated_source_session_in_the_collection_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    _rewrite_payload(
        storage,
        "open_world_discovery_memo",
        lambda payload: payload["sessions"].append(payload["sessions"][0]),
    )
    with pytest.raises(ValueError, match="exact frozen source population"):
        _load(runner)


def test_a_missing_or_changed_decision_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    decision = storage.artifacts.pop("open_world_attempt_decision")
    with pytest.raises(OpenWorldInvocationConflictError, match="accepted exact edit"):
        _load(runner)
    storage.artifacts["open_world_attempt_decision"] = decision
    _rewrite_payload(
        storage,
        "open_world_attempt_decision",
        lambda payload: payload.update(candidate_digest="f" * 64),
    )
    with pytest.raises(OpenWorldInvocationConflictError, match="accepted decision"):
        _load(runner)


def test_a_publication_without_an_attempt_identity_is_refused(runtime) -> None:
    runner, storage, _ = _publication_ready(runtime)
    storage.job.attempt_key = ""
    with pytest.raises(OpenWorldInvocationConflictError, match="attempt identity"):
        _load(runner)


def test_candidate_payload_requires_an_admitted_exact_edit(source) -> None:
    for result in (
        SimpleNamespace(decision="abstain", revised_content=None, rule_audit=None),
        SimpleNamespace(
            decision="proposal",
            revised_content=None,
            rule_audit=SimpleNamespace(edits=[1]),
        ),
        SimpleNamespace(
            decision="proposal",
            revised_content="x",
            rule_audit=SimpleNamespace(edits=[]),
        ),
    ):
        with pytest.raises(ValueError, match="admitted exact edit"):
            candidate_payload(
                job_id=1,
                bundle=source["bundle"],
                incumbent=source["incumbent"],
                diagnostics_artifact_digest="a" * 64,
                diagnostics=source["diagnostics"],
                result=result,
                citations=[],
                receipt={},
            )
