"""Inactive outer2 has one durable ledger; old readers cannot consume it."""

import ast
import copy
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.lib._base import create_generation_litellm_client
from reflexio.server.llm._litellm_types import LiteLLMConfig, ProviderRequestGuardError
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning import window_executor
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_protocol import package_stage_caps
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowAttemptBudgetExceededError,
    OverflowCheckpointConflictError,
    validate_extraction_overflow_manifest,
)
from tests.server.services import test_durable_window_pipeline as durable

package_pipeline = durable.pipeline
DIGEST = "a" * 64
NODE = "b" * 64


def protocol(*, sections=1, rungs=1, kind="profile", gate=True):
    consumer = {
        "kind": kind,
        "gate_enabled": gate if kind == "profile" else False,
        "ladders": {
            name: [f"{name}-model-{i}" for i in range(rungs)]
            if (name != "intent") == (kind == "profile")
            else []
            for name in ("gate", "agent", "dedup", "intent")
        },
    }
    return {
        "name": "source-package",
        "version": 1,
        "plan_digest": DIGEST,
        "source_cache_digest": DIGEST,
        "contract_digest": DIGEST,
        "scope": {
            "org_id": "org",
            "project_id": "",
            "user_id": "u",
            "window_id": "w",
            "request_id": "r",
            "agent_version": None,
            "kind": "profile" if kind == "profile" else "playbook",
        },
        "section_digests": [
            hashlib.sha256(str(i).encode()).hexdigest() for i in range(sections)
        ],
        "consumer": consumer,
        "stage_caps": package_stage_caps(consumer),
    }


def attempt(identity, stage="compress"):
    return {"id": identity, "stage": stage, "usage": None, "model_provenance": None}


def selection(manifest, *, ids=None):
    return {
        "kind": "source_selection",
        "protocol_version": 1,
        "source_digest": manifest["source_digest"],
        "plan_digest": DIGEST,
        "node_key": NODE,
        "input_refs_digest": DIGEST,
        "request_contract_digest": DIGEST,
        "attempt_ids": ids if ids is not None else ["paid"],
        "output": {
            "overview": "",
            "unresolved": [],
            "points": [
                {
                    "category": "selected_evidence",
                    "detail": "Original evidence.",
                    "evidence_refs": ["source-1:0:4"],
                }
            ],
            "omission_counts": {"capacity_points": 3, "unseen_reference_groups": 2},
            "omissions": [
                "Host discarded 2 entire groups with unseen references after bounded repair; source selection is incomplete."
            ],
        },
    }


@pytest.mark.parametrize(
    ("rungs", "sections"), [(1, 48), (2, 45), (3, 36), (4, 26), (5, 17), (6, 7)]
)
def test_actual_ladder_arithmetic_preserves_follow_on_budget(rungs, sections):
    descriptor = protocol(rungs=rungs, sections=sections)
    caps = descriptor["stage_caps"]
    assert sum(caps.values()) == 128
    assert caps["gate"] == rungs
    assert caps["profile_agent"] == 16 * rungs
    assert caps["profile_dedup"] == 2 * rungs
    manifest = OverflowCoordinator.initial_package_manifest(DIGEST, descriptor)
    assert validate_extraction_overflow_manifest(manifest) is manifest
    if sections < 48:
        with pytest.raises(OverflowCheckpointConflictError):
            OverflowCoordinator.initial_package_manifest(
                DIGEST, protocol(rungs=rungs, sections=sections + 1)
            )


def test_gate_skipping_and_playbook_intent_are_distinct_frozen_caps():
    assert protocol(gate=False)["stage_caps"] == {
        "compress": 110,
        "gate": 0,
        "profile_agent": 16,
        "profile_dedup": 2,
        "playbook_intent": 0,
    }
    assert protocol(kind="source_quoted_playbook")["stage_caps"] == {
        "compress": 126,
        "gate": 0,
        "profile_agent": 0,
        "profile_dedup": 0,
        "playbook_intent": 2,
    }
    with pytest.raises(ValueError):
        protocol(rungs=7)
    assert OverflowCoordinator.initial_package_manifest(DIGEST, protocol(sections=0))


def test_selection_cache_retains_exact_host_omission_metadata_without_author_provenance():
    manifest = OverflowCoordinator.initial_package_manifest(DIGEST, protocol())
    manifest["attempts"] = [attempt("paid")]
    receipt = selection(manifest)
    manifest["checkpoint"]["completed_steps"][NODE] = receipt
    before = json.dumps(manifest, sort_keys=True)
    assert validate_extraction_overflow_manifest(manifest) is manifest
    assert json.dumps(manifest, sort_keys=True) == before
    assert "model_provenance" not in receipt


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "prep",
        "bool_version",
        "stage",
        "cap",
        "source",
        "node",
        "attempt",
        "repeat_attempt",
        "text",
        "size",
        "repartition",
    ],
)
def test_damaged_package_or_selection_cannot_be_accepted(change):
    manifest = OverflowCoordinator.initial_package_manifest(DIGEST, protocol())
    manifest["attempts"] = [attempt("paid")]
    receipt = selection(manifest)
    manifest["checkpoint"]["completed_steps"][NODE] = receipt
    if change == "unknown":
        manifest["input_protocol"]["name"] = "unknown"
    elif change == "prep":
        manifest["preparation_version"] = 3
    elif change == "bool_version":
        manifest["version"] = True
    elif change == "stage":
        manifest["attempts"][0]["stage"] = "unknown"
    elif change == "cap":
        manifest["input_protocol"]["stage_caps"]["compress"] += 1
    elif change == "source":
        receipt["source_digest"] = "c" * 64
    elif change == "node":
        receipt["node_key"] = "c" * 64
    elif change == "attempt":
        receipt["attempt_ids"] = ["unreserved"]
    elif change == "repeat_attempt":
        other = copy.deepcopy(receipt)
        other["node_key"] = "c" * 64
        manifest["checkpoint"]["completed_steps"][other["node_key"]] = other
    elif change == "text":
        receipt["output"]["overview"] = "Invented instructions"
    elif change == "size":
        receipt["output"]["points"][0]["evidence_refs"] = ["x" * 1024] * 65
    else:
        manifest["checkpoint"]["repartition"] = {}
    with pytest.raises(OverflowCheckpointConflictError):
        validate_extraction_overflow_manifest(manifest)


def seed(package_pipeline, *, attempts=()):
    storage, window, token = durable._overflow_coordinator(
        package_pipeline, actual_source=True
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    descriptor = protocol()
    descriptor["consumer"]["ladders"]["gate"] = ["gpt-4o"]
    descriptor["scope"] = {
        "org_id": storage.org_id,
        "project_id": window.project_id,
        "user_id": window.user_id,
        "window_id": window.window_id,
        "request_id": parent.binding.request_id,
        "agent_version": parent.binding.agent_version,
        "kind": window.kind,
    }
    manifest = OverflowCoordinator.initial_package_manifest(
        parent.generation_request_snapshot["overflow"]["source_digest"], descriptor
    )
    manifest["attempts"] = list(attempts)
    snapshot = copy.deepcopy(parent.generation_request_snapshot)
    snapshot["overflow"] = manifest
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    return storage, window, token, manifest


def coordinator(storage, window, token, manifest):
    return OverflowCoordinator(
        storage,
        window,
        token,
        manifest,
        source_digest=manifest["source_digest"],
        context_limits={"gpt-4o": 128000},
    )


@pytest.mark.integration
def test_stage_caps_and_remaining_work_survive_restart_and_stale_cas(package_pipeline):
    storage, window, token, manifest = seed(package_pipeline)
    first = coordinator(storage, window, token, manifest)
    stale = coordinator(storage, window, token, manifest)
    first._advance(
        reserve_attempt_id="gate-paid-before-crash", reserve_attempt_stage="gate"
    )
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert saved["attempts"] == [attempt("gate-paid-before-crash", "gate")]
    with pytest.raises(OverflowCheckpointConflictError):
        stale._advance(
            reserve_attempt_id="stale", reserve_attempt_stage="profile_agent"
        )
    restarted = coordinator(storage, window, token, saved)
    with pytest.raises(OverflowAttemptBudgetExceededError):
        restarted._advance(
            reserve_attempt_id="gate-retry", reserve_attempt_stage="gate"
        )
    with pytest.raises(OverflowAttemptBudgetExceededError):
        restarted._advance(
            reserve_attempt_id="spends-future-work",
            reserve_attempt_stage="compress",
            minimum_remaining_attempts=109,
        )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == saved
    )
    assert storage.count_all_profiles() == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "field",
    ["org_id", "project_id", "user_id", "window_id", "request_id", "agent_version"],
)
def test_locked_parent_scope_refuses_descriptor_drift_without_reserving(
    package_pipeline, field
):
    storage, window, token, manifest = seed(package_pipeline)
    manifest["input_protocol"]["scope"][field] = "other"
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = parent.generation_request_snapshot
    snapshot["overflow"] = manifest
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    with pytest.raises(OverflowCheckpointConflictError):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest=manifest["source_digest"],
            reserve_attempt_id="must-not-reserve",
            reserve_attempt_stage="compress",
        )
    assert storage.get_agent_run(parent.id).generation_request_snapshot == snapshot


@pytest.mark.integration
def test_selection_reload_preserves_native_choice_and_immutable_usage_and_receipts(
    package_pipeline,
):
    from reflexio.server.services.input_compression import CompressionChoice

    storage, window, token, manifest = seed(
        package_pipeline, attempts=[attempt("paid")]
    )
    current = coordinator(storage, window, token, manifest)
    receipt = selection(manifest)
    current.complete_selection_step(NODE, receipt)
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    reloaded = coordinator(storage, window, token, saved)
    choice = CompressionChoice.model_validate(reloaded.completed_step(NODE)["output"])
    assert choice.model_dump(mode="json") == receipt["output"]
    assert choice.omission_counts.capacity_points == 3
    assert choice.omission_counts.unseen_reference_groups == 2
    changed = copy.deepcopy(receipt)
    changed["output"]["omissions"] = []
    with pytest.raises(OverflowCheckpointConflictError):
        reloaded.complete_selection_step(NODE, changed)
    with pytest.raises(OverflowCheckpointConflictError):
        reloaded.save_checkpoint({"stage": "prepare", "completed_steps": {}})
    with pytest.raises(OverflowCheckpointConflictError):
        reloaded._advance(
            observed_attempt={
                "id": "paid",
                "stage": "gate",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                },
                "model_provenance": {},
            }
        )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == saved
    )


@pytest.mark.integration
@pytest.mark.parametrize("change", ["descriptor", "outer1"])
def test_adoption_never_rebinds_or_downgrades_package(package_pipeline, change):
    storage, window, token, manifest = seed(
        package_pipeline, attempts=[attempt("charged")]
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    proposed = copy.deepcopy(manifest)
    if change == "descriptor":
        proposed["input_protocol"]["contract_digest"] = "c" * 64
    else:
        proposed = OverflowCoordinator.initial_manifest(manifest["source_digest"])
    with pytest.raises(OverflowCheckpointConflictError):
        storage.adopt_extraction_overflow(
            window,
            token,
            source_digest=manifest["source_digest"],
            source_digests=parent.generation_request_snapshot[
                "source_interaction_digests"
            ],
            manifest=proposed,
        )
    assert (
        storage.get_agent_run(parent.id).generation_request_snapshot
        == parent.generation_request_snapshot
    )


@pytest.mark.integration
def test_marker_free_legacy_output_cannot_be_adopted_as_package(package_pipeline):
    storage, window, token, manifest = seed(package_pipeline)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = parent.generation_request_snapshot
    del snapshot["overflow"]
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=?,status='agent_completed',committed_output=? WHERE id=?",
            (json.dumps(snapshot), '{"profiles":[]}', parent.id),
        )
    with pytest.raises(OverflowCheckpointConflictError):
        storage.adopt_extraction_overflow(
            window,
            token,
            source_digest=manifest["source_digest"],
            source_digests=snapshot["source_interaction_digests"],
            manifest=manifest,
        )
    assert storage.get_agent_run(parent.id).generation_request_snapshot == snapshot


@pytest.mark.integration
def test_one_128_ledger_enforces_every_frozen_stage(package_pipeline):
    storage, window, token, manifest = seed(package_pipeline)
    current = coordinator(storage, window, token, manifest)
    for stage, limit in manifest["input_protocol"]["stage_caps"].items():
        for index in range(limit):
            current._advance(
                reserve_attempt_id=f"{stage}-{index}", reserve_attempt_stage=stage
            )
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert len(saved["attempts"]) == 128
    assert all(item["usage"] is None for item in saved["attempts"])
    for stage in manifest["input_protocol"]["stage_caps"]:
        with pytest.raises(OverflowAttemptBudgetExceededError):
            current._advance(
                reserve_attempt_id=f"extra-{stage}", reserve_attempt_stage=stage
            )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == saved
    )


@pytest.mark.integration
@pytest.mark.parametrize("descriptor_drift", [False, True])
def test_actual_native_dispatch_requires_stage_and_reloads_paid_cap(
    package_pipeline, monkeypatch, descriptor_drift
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    storage, window, token, manifest = seed(package_pipeline)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"profiles":[]}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    wire = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    current = coordinator(storage, window, token, manifest)

    def call():
        return client.generate_chat_response_with_provenance(
            messages=[{"role": "user", "content": "Original source"}],
            response_format=StructuredProfilesOutput,
        )

    with current.provider_scope():
        with pytest.raises(ProviderRequestGuardError):
            call()
        wire.assert_not_called()
        if descriptor_drift:
            parent = storage.get_agent_run(f"window:{window.window_id}")
            snapshot = parent.generation_request_snapshot
            snapshot["overflow"]["input_protocol"]["contract_digest"] = "c" * 64
            with storage._stream_sql() as db:
                db.query(
                    "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
                    (json.dumps(snapshot), parent.id),
                )
            with (
                current.attempt_stage("gate"),
                pytest.raises(ProviderRequestGuardError),
            ):
                call()
            wire.assert_not_called()
            assert (
                storage.get_agent_run(parent.id).generation_request_snapshot == snapshot
            )
            return
        with current.attempt_stage("gate") as ids:
            assert call().value == StructuredProfilesOutput(profiles=[])
        assert len(ids) == 1
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert saved["attempts"][0]["stage"] == "gate"
    assert saved["attempts"][0]["usage"]["prompt_tokens"] == 10
    restarted = coordinator(storage, window, token, saved)
    with (
        restarted.provider_scope(),
        restarted.attempt_stage("gate"),
        pytest.raises(ProviderRequestGuardError),
    ):
        call()
    assert wire.call_count == 1
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == saved
    )


def old_snapshot():
    path = Path(__file__).parent / "fixtures/overflow_manifest_v1_snapshot.py.txt"
    source = path.read_text()
    functions = [
        ast.get_source_segment(source, item)
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef)
    ]
    namespace = runpy.run_path(str(path))
    assert (
        hashlib.sha256(("\n\n\n".join(functions) + "\n").encode()).hexdigest()
        == namespace["SNAPSHOT_FUNCTIONS_SHA256"]
    )
    return namespace


def old_reader():
    return old_snapshot()["validate_extraction_overflow_manifest"]


def test_outer1_frozen_and_current_readers_preserve_exact_legacy_payload():
    manifest = OverflowCoordinator.initial_manifest("legacy-source")
    manifest["input_protocol"] = {"future_unknown_field": "ignored by outer1"}
    manifest["attempts"] = [{"id": "paid", "usage": None, "model_provenance": None}]
    manifest["checkpoint"]["completed_steps"] = {
        "done": {"output": False, "model_provenance": {}}
    }
    before = json.dumps(manifest)
    assert old_reader()(manifest) is manifest
    assert validate_extraction_overflow_manifest(manifest) is manifest
    assert json.dumps(manifest) == before


@pytest.mark.integration
@pytest.mark.parametrize("waiting", [False, True])
def test_frozen_old_reader_failure_preserves_new_journal_and_cached_output(
    package_pipeline, monkeypatch, waiting
):
    engine, _, _ = package_pipeline
    storage, window, token, manifest = seed(
        package_pipeline, attempts=[attempt("already-paid")]
    )
    current = coordinator(storage, window, token, manifest)
    current.complete_selection_step(NODE, selection(manifest, ids=["already-paid"]))
    if waiting:
        checkpoint = current.checkpoint
        checkpoint["waiting_child"] = {
            "key": "saved-child",
            "messages": [{"role": "user", "content": "Owned fixture source"}],
        }
        current.save_checkpoint(checkpoint)
    storage.save_extraction_outcome(window, token, {"version": 1, "skipped": True})
    window = storage.prepare_extraction(window.user_id, token)
    before = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = copy.deepcopy(before.generation_request_snapshot)
    frozen = old_reader()
    assert frozen(OverflowCoordinator.initial_manifest("legacy-source"))["version"] == 1
    wire = Mock(side_effect=AssertionError("Unsupported worker must not dispatch"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", wire)
    executor = window_executor.WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(
            window_executor, "validate_extraction_overflow_manifest", frozen
        )
        with pytest.raises(InputBudgetConfigurationError) as failure:
            executor.execute(window, token)
        assert (
            old_snapshot()["_handle_window_failure"](
                storage, window, token, failure.value
            )
            == "input_budget_failure"
        )
    after = storage.get_agent_run(before.id)
    assert after.generation_request_snapshot == snapshot
    assert after.committed_output == before.committed_output
    assert after.status.value == "failed"
    assert storage.extraction_status(window.user_id, "overflow-checkpoint") == {
        "status": "failed",
        "reason": "input_budget_configuration_error",
    }
    # The unchanged CURRENT executor also refuses prep4 before legacy geometry.
    monkeypatch.setattr(
        window_executor,
        "prepare_extraction_input",
        lambda *_a, **_k: pytest.fail("Cannot prepare package as legacy chunks"),
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    with pytest.raises(InputBudgetConfigurationError):
        executor.execute(window, token)
    # Failure releases ownership and applies the existing bounded config
    # backoff. Resume must win a NEW real lease after that delay.
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    with storage._stream_sql(read_only=True) as db:
        resumed_at = db.now() + 301
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: resumed_at)
    claimed = storage.claim_extraction("new-package-worker", 300)
    assert claimed is not None
    user, token = claimed
    window = storage.prepare_extraction(user, token)
    restarted = coordinator(
        storage, window, token, after.generation_request_snapshot["overflow"]
    )
    restarted._advance(
        reserve_attempt_id="new-stage", reserve_attempt_stage="profile_agent"
    )
    final = storage.get_agent_run(before.id).generation_request_snapshot["overflow"]
    assert final["attempts"] == [
        attempt("already-paid"),
        attempt("new-stage", "profile_agent"),
    ]
    assert final["checkpoint"] == snapshot["overflow"]["checkpoint"]
    assert storage.count_all_profiles() == 0
    wire.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("stage", "ladder"),
    [
        ("gate", ("wrong-primary",)),
        ("gate", ("gpt-4o", "extra-rung")),
        ("profile_agent", ("gpt-4o",)),
        ("profile_dedup", ("agent-model-0",)),
    ],
)
def test_consumer_ladder_mismatch_never_reserves(package_pipeline, stage, ladder):
    storage, window, token, manifest = seed(package_pipeline)
    current = coordinator(storage, window, token, manifest)
    before = copy.deepcopy(
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot
    )
    with (
        current.attempt_stage(stage),
        pytest.raises(InputBudgetConfigurationError, match="model ladder changed"),
    ):
        current._guard(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Original source"}],
                "max_tokens": 1024,
            },
            1,
            ladder,
            False,
        )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot
        == before
    )


@pytest.mark.integration
def test_consumer_ladder_order_is_frozen(package_pipeline):
    storage, window, token, manifest = seed(package_pipeline)
    descriptor = manifest["input_protocol"]
    descriptor["consumer"]["ladders"]["gate"] = ["gpt-4o", "gpt-4o-mini"]
    descriptor["stage_caps"] = package_stage_caps(descriptor["consumer"])
    current = coordinator(storage, window, token, manifest)
    with (
        current.attempt_stage("gate"),
        pytest.raises(InputBudgetConfigurationError, match="model ladder changed"),
    ):
        current._guard(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Original source"}],
                "max_tokens": 1024,
            },
            1,
            ("gpt-4o-mini", "gpt-4o"),
            False,
        )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]["attempts"]
        == []
    )
