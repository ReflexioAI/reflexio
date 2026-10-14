"""Real HTTP admission, native requests, SQLite publication and durable retry."""

import copy
import json
from unittest.mock import Mock

import litellm
import pytest

from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.profile_package_executor import (
    ProfilePackageWindowExecutor,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def setup(pipeline, monkeypatch, *, gate=True, profiles=True):
    engine, _, http = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o-mini":128000}')
    for module in ("_litellm_text_generation", "tools"):
        monkeypatch.setattr(
            f"reflexio.server.llm.{module}.resolve_model_name",
            lambda **_: "gpt-4o-mini",
        )
    monkeypatch.setattr(
        "reflexio.server.services.deduplication_utils.resolve_model_name",
        lambda *_, **__: "gpt-4o-mini",
    )
    monkeypatch.setattr(
        ProfileGenerationService, "_resolve_should_run_model", lambda _: "gpt-4o-mini"
    )
    assert http.post(
        "/api/publish_interaction", json=durable.payload("package-parent")
    ).json()["success"]
    storage = engine.get_storage()
    monkeypatch.setattr(storage, "_get_embedding", Mock(return_value=[0.0] * 512))
    user, token = storage.claim_extraction("package-parent-test", 600)
    window = storage.prepare_extraction(user, token)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def wire(params, _timeout):
        calls.append(copy.deepcopy(params))
        schema = params.get("response_format", {})
        if "ProfileDeduplicationOutput" in str(schema):
            content = json.dumps(
                {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
            )
        elif schema:
            content = json.dumps(
                {
                    "profiles": [
                        {
                            "content": "Prefers concise answers",
                            "time_to_live": "infinity",
                            "source_span": "I prefer concise answers with details 0",
                        }
                    ]
                    if profiles
                    else []
                }
            )
        else:
            content = str(gate).lower()
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    return engine, storage, window, token, client, calls


@pytest.mark.parametrize(
    ("gate", "profiles", "expected"),
    [(True, True, 1), (True, False, 0), (False, True, 0)],
)
def test_actual_package_parent_publishes_once(
    pipeline, monkeypatch, gate, profiles, expected
):
    engine, storage, window, token, client, calls = setup(
        pipeline, monkeypatch, gate=gate, profiles=profiles
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    executor.execute(window, token)
    rows = storage.get_user_profile("u")
    assert len(rows) == expected
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent.status.value == ("agent_completed" if gate else "finalized")
    journal = parent.generation_request_snapshot["overflow"]
    assert journal["version"] == 2
    assert (
        len(journal["attempts"]) == len(calls) == (3 if expected else 2 if gate else 1)
    )
    assert storage.extraction_status("u", "package-parent")["status"] == "done"
    effects = storage.pending_extraction_effects()
    assert len(effects) == 1
    assert effects[0][1]["billing"]["prompt_tokens"] == len(calls) * 100
    if not gate:
        assert effects[0][1]["billing"]["ids"] == []
        assert effects[0][1]["billing"]["input_tokens"] == 0


def test_cached_write_plan_cannot_change_profile_content(pipeline, monkeypatch):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    original = storage.complete_extraction
    monkeypatch.setattr(
        storage, "complete_extraction", Mock(side_effect=RuntimeError("commit failure"))
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="commit failure"):
        executor.execute(window, token)
    assert storage.get_user_profile("u") == []

    monkeypatch.setattr(storage, "complete_extraction", original)
    resumed = storage.prepare_extraction("u", token)
    assert resumed.outcome is not None
    resumed.outcome["write_plan"]["new_profiles"][0]["content"] = "tampered fact"
    with pytest.raises(InputBudgetConfigurationError, match="write plan changed"):
        executor.execute(resumed, token)
    assert len(calls) == 3
    assert storage.get_user_profile("u") == []


@pytest.mark.parametrize(
    "boundary", ["candidates", "parent_completion", "outcome", "commit"]
)
def test_restart_preserves_candidates_and_paid_calls(pipeline, monkeypatch, boundary):
    from reflexio.server.services.extraction.profile_package_consolidator import (
        PackageProfileConsolidator,
    )
    from reflexio.server.services.storage.storage_base import AgentRunStatus

    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    if boundary == "candidates":
        target, name = PackageProfileConsolidator, "deduplicate"
        original = target.deduplicate
        failure = InputBudgetConfigurationError("injected pause")
    elif boundary == "parent_completion":
        target, name = storage, "update_agent_run_status"
        original = target.update_agent_run_status

        def interrupted(*args, **kwargs):
            if args[1] == AgentRunStatus.AGENT_COMPLETED:
                raise RuntimeError("injected pause")
            return original(*args, **kwargs)

        failure = None
    else:
        target, name = (
            storage,
            "save_extraction_outcome"
            if boundary == "outcome"
            else "complete_extraction",
        )
        original = getattr(target, name)
        failure = RuntimeError("injected pause")
    monkeypatch.setattr(
        target, name, Mock(side_effect=failure) if failure else interrupted
    )
    with pytest.raises(
        (RuntimeError, InputBudgetConfigurationError), match="injected pause"
    ):
        executor.execute(window, token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    checkpoint = parent.generation_request_snapshot["overflow"]["checkpoint"]
    frozen = copy.deepcopy(checkpoint["profile_package_candidates"])
    assert storage.get_user_profile("u") == []
    monkeypatch.setattr(target, name, original)
    executor.execute(storage.prepare_extraction("u", token), token)
    rows = storage.get_user_profile("u")
    assert len(rows) == 1
    assert rows[0].profile_id == frozen[0]["profile_id"]
    assert rows[0].last_modified_timestamp == frozen[0]["last_modified_timestamp"]
    assert rows[0].expiration_timestamp == frozen[0]["expiration_timestamp"]
    assert len(calls) == 3
    assert len(storage.pending_extraction_effects()) == 1


@pytest.mark.parametrize("failure", ["invalid_first_parsed", "transport"])
def test_native_dedup_host_fallback_is_durable(pipeline, monkeypatch, failure):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    transport = client._completion_with_hard_timeout

    def wire(params, timeout):
        response = transport(params, timeout)
        if "ProfileDeduplicationOutput" in str(params.get("response_format")):
            if failure == "transport":
                raise RuntimeError("fixture provider unavailable")
            response.choices[
                0
            ].message.content = (
                '{"duplicate_groups":[],"unique_ids":["NEW-999"],"deletions":[]}'
            )
        return response

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    original = storage.precompute_profile_embeddings
    monkeypatch.setattr(
        storage,
        "precompute_profile_embeddings",
        Mock(side_effect=RuntimeError("after dedup")),
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="after dedup"):
        executor.execute(window, token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    decision = parent.generation_request_snapshot["overflow"]["checkpoint"][
        "profile_dedup_decision"
    ]
    assert decision["disposition"] == "host_fallback"
    assert len(decision["attempt_ids"]) == (
        2 if failure == "invalid_first_parsed" else 1
    )
    before = len(calls)
    monkeypatch.setattr(storage, "precompute_profile_embeddings", original)
    executor.execute(storage.prepare_extraction("u", token), token)
    assert len(storage.get_user_profile("u")) == 1
    assert len(calls) == before


@pytest.mark.parametrize("finish", [True, False])
def test_native_eight_turns_keep_all_fallback_attempts(pipeline, monkeypatch, finish):
    from reflexio.models.config_schema import PendingToolCallConfig

    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    config = engine.request_context.configurator.get_config()
    config.pending_tool_call_config = PendingToolCallConfig(enabled=True)
    engine.request_context.configurator.set_config(config)
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.profile_package_executor.pending_tool_calls_enabled",
        lambda _: True,
    )
    client.config.fallback_models = ["gpt-4.1-mini"]
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"gpt-4o-mini":128000,"gpt-4.1-mini":128000}',
    )
    transport = client._completion_with_hard_timeout
    agent_calls = 0

    def wire(params, timeout):
        nonlocal agent_calls
        response = transport(params, timeout)
        if "StructuredProfilesOutput" not in str(params.get("response_format")):
            return response
        agent_calls += 1
        if params["model"] == "gpt-4o-mini":
            raise RuntimeError("fixture primary unavailable")
        if agent_calls < 16 or not finish:
            response = litellm.ModelResponse(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{agent_calls}",
                                    "type": "function",
                                    "function": {
                                        "name": "attach_pending_info_request",
                                        "arguments": json.dumps(
                                            {
                                                "pending_tool_call_id": "missing",
                                                "why_relevant": "check prerequisite",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            )
        return response

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    if finish:
        executor.execute(window, token)
    else:
        with pytest.raises(InputBudgetExceededError, match="step budget exhausted"):
            executor.execute(window, token)
        with pytest.raises(InputBudgetExceededError, match="step budget exhausted"):
            executor.execute(storage.prepare_extraction("u", token), token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    journal = parent.generation_request_snapshot["overflow"]
    assert agent_calls == 16
    assert sum(item["stage"] == "profile_agent" for item in journal["attempts"]) == 16
    assert len(calls) == len(journal["attempts"]) == (18 if finish else 17)
    assert len(storage.get_user_profile("u")) == int(finish)
    if not finish:
        assert journal["checkpoint"]["active_child"]["max_steps_remaining"] == 0


@pytest.mark.parametrize("corruption", ["envelope", "output", "provenance", "attempts"])
def test_malformed_saved_dedup_cannot_publish_through_fallback(
    pipeline, monkeypatch, corruption
):
    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    original = storage.precompute_profile_embeddings
    monkeypatch.setattr(
        storage,
        "precompute_profile_embeddings",
        Mock(side_effect=RuntimeError("after dedup")),
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="after dedup"):
        executor.execute(window, token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = parent.generation_request_snapshot
    decision = snapshot["overflow"]["checkpoint"]["profile_dedup_decision"]
    if corruption == "envelope":
        decision.pop("output")
    elif corruption == "output":
        decision["output"] = {"unique_ids": 123}
    elif corruption == "provenance":
        decision["provenance"] = {"unknown": "forged"}
    else:
        decision["attempt_ids"] = ["unknown"]
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    monkeypatch.setattr(storage, "precompute_profile_embeddings", original)
    with pytest.raises(InputBudgetConfigurationError):
        executor.execute(storage.prepare_extraction("u", token), token)
    assert len(calls) == 3
    assert storage.get_user_profile("u") == []


def test_frozen_old_projection_replays_without_vectors_or_new_search(
    pipeline, monkeypatch
):
    from reflexio.models.api_schema.service_schemas import UserProfile
    from reflexio.server.services.extraction.profile_package_state import (
        PROFILE_WITNESS_FIELDS,
    )

    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    old = UserProfile(
        profile_id="old",
        user_id="u",
        content="An existing preference",
        last_modified_timestamp=100,
        generated_from_request_id="package-parent",
        embedding=[0.0] * 512,
        custom_features={"condition": "only at work"},
        notes="Keep the condition",
        source_span="Original OLD quotation",
        source_interaction_ids=[1],
    )
    storage.add_user_profile("u", [old], skip_embedding=True)
    search = Mock(return_value=[old])
    monkeypatch.setattr(storage, "search_user_profile", search)
    original = storage.precompute_profile_embeddings
    monkeypatch.setattr(
        storage,
        "precompute_profile_embeddings",
        Mock(side_effect=RuntimeError("after dedup")),
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="after dedup"):
        executor.execute(window, token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = parent.generation_request_snapshot["overflow"]["checkpoint"][
        "profile_dedup_inputs"
    ]["old"]
    assert set(frozen[0]) == set(PROFILE_WITNESS_FIELDS)
    assert frozen[0]["notes"] == old.notes
    assert frozen[0]["custom_features"] == old.custom_features
    assert search.call_count == 1
    monkeypatch.setattr(storage, "precompute_profile_embeddings", original)
    executor.execute(storage.prepare_extraction("u", token), token)
    assert search.call_count == 1
    assert len(calls) == 3
    assert len(storage.get_user_profile("u")) == 2


def test_nonfinite_embedding_never_reaches_publication(pipeline, monkeypatch):
    engine, storage, window, token, client, _ = setup(pipeline, monkeypatch)

    def invalid_embeddings(profiles):
        for profile in profiles:
            profile.embedding = [float("nan")] * 512

    monkeypatch.setattr(storage, "precompute_profile_embeddings", invalid_embeddings)
    with pytest.raises(InputBudgetConfigurationError, match="not finite JSON"):
        ProfilePackageWindowExecutor(engine.request_context, client).execute(
            window, token
        )
    assert storage.get_user_profile("u") == []
    assert storage.pending_extraction_effects() == []


def test_source_tokenization_keeps_existing_billing_event_clock(pipeline, monkeypatch):
    from types import SimpleNamespace

    from reflexio.server import metering_signals

    engine, storage, window, token, client, _ = setup(pipeline, monkeypatch)
    before = 1_800_000_000.0 - 0.1
    clock = [before]
    counter = metering_signals.count_input_tokens

    def crossing_boundary(text):
        clock[0] += 1
        return counter(text)

    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.window_executor.time",
        SimpleNamespace(time=lambda: clock[0]),
    )
    monkeypatch.setattr(metering_signals, "count_input_tokens", crossing_boundary)
    ProfilePackageWindowExecutor(engine.request_context, client).execute(window, token)
    assert storage.pending_extraction_effects()[0][1]["billing"]["created_at"] == before


def test_shared_sql_cannot_replace_or_remove_frozen_profile_decisions(
    pipeline, monkeypatch
):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowCheckpointConflictError,
    )

    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    monkeypatch.setattr(
        storage,
        "save_extraction_outcome",
        Mock(side_effect=RuntimeError("before publication")),
    )
    executor = ProfilePackageWindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="before publication"):
        executor.execute(window, token)
    service, request, config = executor.service(window)
    current = executor._coordinator(window, token, service, request, config)
    original = current.checkpoint
    revision = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]["revision"]
    for field in (
        "profile_package_candidates",
        "profile_dedup_inputs",
        "profile_dedup_decision",
        "profile_package_plan",
        "profile_package_plan_digest",
    ):
        assert field in original
        for remove in (False, True):
            altered = copy.deepcopy(original)
            if remove:
                altered.pop(field)
            else:
                altered[field] = {"forged": True}
            with pytest.raises(OverflowCheckpointConflictError, match="immutable"):
                current.save_checkpoint(altered)
    actual = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert actual["revision"] == revision
    assert actual["checkpoint"] == original
    assert len(calls) == len(actual["attempts"]) == 3
    assert storage.get_user_profile("u") == []
