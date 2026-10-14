"""Complete native preparation, without provider, admission, or publication."""

import copy
import hashlib
import json
import shutil
from typing import Any
from unittest.mock import Mock

import litellm
import pytest
from litellm.exceptions import APIConnectionError

from reflexio.models.config_schema import (
    APIKeyConfig,
    CustomEndpointConfig,
    PendingToolCallConfig,
    ProfileExtractorConfig,
)
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    extraction_provider_guard,
    measure_request_input_tokens,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.profile_package_preparation import (
    build_profile_package_request,
    prepare_profile_package,
    profile_package_original_seed,
)
from reflexio.server.services.extraction.source_package import render_source_package
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import canonical_json
from tests.server.services.extraction import test_prior_answer_search as prior_tests
from tests.server.services.test_input_compression_adapters import group

prior_storage = prior_tests.storage

pytestmark = pytest.mark.unit


def empty_prior_capture(monkeypatch, *, groups=None):
    """Exercise real capture/query binding with a deliberately empty lookup."""
    from reflexio.server.services.extraction.prior_answer_search import (
        capture_prior_knowledge_context,
    )

    if groups is None:
        original = group()
        original.interactions[0].content = "Production requires separate approval."
        groups = [original]
    config = ProfileExtractorConfig(extraction_definition_prompt="Stable preferences")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            "reflexio.server.services.extraction.prior_answer_search.find_prior_answer_matches",
            Mock(return_value=[]),
        )
        return capture_prior_knowledge_context(
            messages=profile_package_original_seed(
                prompt_manager=PromptManager(),
                groups=groups,
                existing_profiles=[],
                agent_context="Agent context",
                extractor_config=config,
            ),
            storage=Mock(),
            org_id="org",
            extractor_kind="profile",
            extractor_config=config,
            source=groups[-1].request.source,
            agent_version=None,
            similarity_threshold=PendingToolCallConfig()
            .for_tool("attach_pending_info_request")
            .similarity_threshold,
        )


@pytest.mark.parametrize(
    "model", ["gpt-4o-mini", "minimax/MiniMax-M3", "anthropic/claude-sonnet-4-5"]
)
@pytest.mark.parametrize("with_tools", [False, True])
def test_pure_build_matches_scoped_native_build_and_restores_guard(
    monkeypatch, model, with_tools
):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        Mock(return_value={"max_output_tokens": 4096, "max_input_tokens": 131072}),
    )
    wire = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(litellm, "completion", wire)
    client = LiteLLMClient(LiteLLMConfig(model=model, fallback_models=[]))
    messages = [
        {"role": "system", "content": "Policy"},
        {"role": "user", "content": "source"},
    ]
    original = copy.deepcopy(messages)
    options = {
        "response_format": StructuredProfilesOutput,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "attach_pending_info_request",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        if with_tools
        else None,
        "tool_choice": "auto" if with_tools else None,
    }
    outer = Mock(side_effect=AssertionError("no quota"))
    token = extraction_provider_guard.set(outer)
    try:
        expected = client._build_completion_params(
            messages=copy.deepcopy(messages), **copy.deepcopy(options)
        )[0]
        actual = build_profile_package_request(client, messages, **options)
        assert actual == expected
        output_cap = actual.get("max_completion_tokens", actual.get("max_tokens"))
        assert isinstance(output_cap, int) and output_cap > 0
        if model == "gpt-4o-mini":
            assert actual["max_tokens"] == 4096
        assert messages == original
        assert extraction_provider_guard.get() is outer
        with monkeypatch.context() as scoped:
            scoped.setattr(
                client,
                "_build_completion_params",
                Mock(side_effect=ValueError("builder failure")),
            )
            with pytest.raises(ValueError, match="builder failure"):
                build_profile_package_request(client, messages)
        assert extraction_provider_guard.get() is outer
    finally:
        extraction_provider_guard.reset(token)
    outer.assert_not_called()
    wire.assert_not_called()


def preparation(monkeypatch, **overrides: Any):
    model = "gpt-4o-mini"
    client = overrides.pop(
        "client",
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=8192, fallback_models=[])),
    )
    model = client.config.model
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda **_kwargs: model,
    )
    original = group()
    original.interactions[0].content = "Production requires separate approval."
    return prepare_profile_package(
        **{
            "groups": [original],
            "org_id": "org",
            "project_id": "project",
            "user_id": "alice",
            "source": original.request.source,
            "client": client,
            "context_limits": {model: 131072},
            "prompt_manager": PromptManager(),
            "extractor_config": ProfileExtractorConfig(
                extraction_definition_prompt="Stable preferences"
            ),
            "existing_profiles": [],
            "agent_context": "Agent context",
            "gate_enabled": True,
            "gate_model": model,
            "dedup_model": model,
            **overrides,
        }
    ), client


def test_preparation_prices_actual_full_native_requests_and_is_provider_free(
    monkeypatch,
):
    wire = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(litellm, "completion", wire)
    prepared, client = preparation(monkeypatch)
    assert prepared.compression.native.compression.passthrough
    result = prepared.compression.execute()
    source = render_source_package(
        result.packet, source_digest=prepared.compression.source_digest
    )
    messages = prepared.agent_messages(source)
    assert source in str(messages)
    assert "exact original quotation" in str(messages)
    agent = build_profile_package_request(
        client,
        messages,
        model="gpt-4o-mini",
        response_format=StructuredProfilesOutput,
        tools=None,
        tool_choice=None,
        fallback_models=[],
    )
    gate = build_profile_package_request(
        client, prepared.gate_messages(source), model="gpt-4o-mini", fallback_models=[]
    )
    assert result.output_tokens == max(
        measure_request_input_tokens(agent), measure_request_input_tokens(gate)
    )
    snapshot = prepared.snapshot
    assert snapshot["input_target"] == 32768
    assert snapshot["stage_caps"] == {
        "compress": 109,
        "gate": 1,
        "profile_agent": 16,
        "profile_dedup": 2,
        "playbook_intent": 0,
    }
    assert (
        prepared.contract_digest
        == hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
    )
    snapshot["agent_context"] = "tampered"
    assert prepared.snapshot["agent_context"] == "Agent context"
    wire.assert_not_called()


@pytest.mark.parametrize("repetitions", [1, 5000])
def test_original_lookup_seed_matches_existing_extractor_before_overflow_policy(
    repetitions,
):
    from reflexio.server.services.extraction.prior_answer_search import (
        build_knowledge_need_query,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        construct_profile_extraction_messages_from_sessions,
    )

    original = group()
    original.interactions[0].content = "Preserve approval conditions. " * repetitions
    config = ProfileExtractorConfig(
        extraction_definition_prompt=" Stable preferences ", context_prompt=" Context "
    )
    manager = PromptManager()
    native = construct_profile_extraction_messages_from_sessions(
        prompt_manager=manager,
        request_interaction_data_models=[original],
        existing_profiles=[],
        agent_context_prompt="Agent context",
        context_prompt=(config.context_prompt or "").strip(),
        extraction_definition_prompt=config.extraction_definition_prompt.strip(),
    )
    seed = profile_package_original_seed(
        prompt_manager=manager,
        groups=[original],
        existing_profiles=[],
        agent_context="Agent context",
        extractor_config=config,
    )
    assert seed == native
    query_options = {
        "extractor_kind": "profile",
        "extractor_config": config,
        "source": original.request.source,
        "agent_version": None,
    }
    assert build_knowledge_need_query(messages=seed, **query_options) == (
        build_knowledge_need_query(messages=native, **query_options)
    )


def test_pending_tools_are_real_and_counted_with_the_actual_schema(monkeypatch):
    plain, _ = preparation(monkeypatch, gate_enabled=False)
    prepared, client = preparation(
        monkeypatch,
        gate_enabled=False,
        pending_tools_active=True,
        pending_tool_config=PendingToolCallConfig(),
        prior_context=empty_prior_capture(monkeypatch),
    )
    assert prepared.gate_messages("source") == []
    assert [spec["function"]["name"] for spec in prepared.snapshot["tools"]] == [
        "attach_pending_info_request"
    ]
    assert (
        prepared.compression.execute().output_tokens
        > plain.compression.execute().output_tokens
    )
    assert prepared.contract_digest != plain.contract_digest
    assert prepared.snapshot["stage_caps"]["compress"] == 110
    with pytest.raises(InputBudgetConfigurationError, match="configuration"):
        preparation(monkeypatch, pending_tools_active=True)
    with pytest.raises(InputBudgetConfigurationError, match="prior capture"):
        preparation(
            monkeypatch,
            pending_tools_active=True,
            pending_tool_config=PendingToolCallConfig(),
        )


def test_fixed_native_request_overflow_refuses_before_provider(monkeypatch):
    wire = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(litellm, "completion", wire)
    with pytest.raises(ValueError, match="fixed|Fixed"):
        preparation(monkeypatch, agent_context="fixed instructions " * 20000)
    wire.assert_not_called()


def test_empty_sqlite_prior_capture_is_distinct_from_skipped_lookup(
    monkeypatch, prior_storage
):
    from reflexio.server.services.extraction.prior_answer_search import (
        capture_prior_knowledge_context,
    )

    original = group()
    config = ProfileExtractorConfig(extraction_definition_prompt="Stable preferences")
    tools_config = PendingToolCallConfig()
    captured = capture_prior_knowledge_context(
        messages=profile_package_original_seed(
            prompt_manager=PromptManager(),
            groups=[original],
            existing_profiles=[],
            agent_context="Agent context",
            extractor_config=config,
        ),
        storage=prior_storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_config=config,
        source=original.request.source,
        agent_version=None,
        similarity_threshold=tools_config.for_tool(
            "attach_pending_info_request"
        ).similarity_threshold,
    )
    assert captured.suffix == []
    assert captured.answer_witnesses == {}
    prepared, _ = preparation(
        monkeypatch,
        groups=[original],
        org_id="org_1",
        prior_context=captured,
        pending_tools_active=True,
        pending_tool_config=tools_config,
    )
    assert prepared.snapshot["prior_context"] == captured.snapshot


@pytest.mark.parametrize("pending_active", [False, True])
def test_custom_route_credentials_are_not_in_preparation_snapshot(
    monkeypatch, pending_active
):
    route = "https://example.invalid/v1?token=private-route-secret"
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-4o-mini",
            max_tokens=8192,
            api_key_config=APIKeyConfig(
                custom_endpoint=CustomEndpointConfig.model_validate(
                    {
                        "model": "openai/private",
                        "api_key": "private-api-key",
                        "api_base": route,
                    }
                )
            ),
        )
    )
    prepared, _ = preparation(
        monkeypatch,
        client=client,
        context_limits={f"openai/private@{route}": 60000},
        pending_tools_active=pending_active,
        prior_context=empty_prior_capture(monkeypatch) if pending_active else None,
        pending_tool_config=PendingToolCallConfig(
            hmac_secrets=["configured-hmac-secret"]
        ),
    )
    snapshot = canonical_json(prepared.snapshot)
    assert "private-route-secret" not in snapshot
    assert "private-api-key" not in snapshot
    assert "configured-hmac-secret" not in snapshot
    if pending_active:
        assert "hmac_secrets" not in prepared.snapshot["pending_tool_config"]
        assert (
            len(prepared.snapshot["pending_tool_config"]["hmac_secrets_digest"]) == 64
        )
    else:
        assert prepared.snapshot["pending_tool_config"] is None
    assert prepared.snapshot["consumer"]["ladders"]["agent"] == ["openai/private"]


def test_signing_secret_rotation_changes_only_its_frozen_digest(monkeypatch):
    original = group()
    options = {
        "groups": [original],
        "pending_tools_active": True,
        "prior_context": empty_prior_capture(monkeypatch, groups=[original]),
    }
    first, _ = preparation(
        monkeypatch,
        **options,
        pending_tool_config=PendingToolCallConfig(
            hmac_secrets=["first-signing-secret"]
        ),
    )
    second, _ = preparation(
        monkeypatch,
        **options,
        pending_tool_config=PendingToolCallConfig(
            hmac_secrets=["rotated-signing-secret", "first-signing-secret"]
        ),
    )
    assert first.agent_messages("source") == second.agent_messages("source")
    before, after = first.snapshot, second.snapshot
    assert before["pending_tool_config"].pop("hmac_secrets_digest") != after[
        "pending_tool_config"
    ].pop("hmac_secrets_digest")
    assert before == after
    assert first.contract_digest != second.contract_digest


@pytest.mark.parametrize(
    "model", ["gpt-4o-mini", "minimax/MiniMax-M3", "anthropic/claude-sonnet-4-5"]
)
@pytest.mark.parametrize("long_echo", [False, True])
def test_actual_native_agent_and_semantic_repair_pass_the_frozen_checker(
    monkeypatch, model, long_echo
):
    prepared, client = preparation(
        monkeypatch,
        client=LiteLLMClient(
            LiteLLMConfig(model=model, max_tokens=8192, fallback_models=[])
        ),
    )
    messages = prepared.agent_messages("original source package")
    reserve = Mock()
    seen = []

    def guard(params, _timeout, ladder, parse):
        prepared.check_request("agent", messages, params, ladder, parse)
        reserve()

    def transport(params, _timeout):
        seen.append(copy.deepcopy(params))
        content = (
            json.dumps(
                {"profiles": [{"content": "x" * 5000, "time_to_live": "infinity"}]}
            )
            if long_echo and len(seen) == 1
            else '{"profiles":[]}'
        )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        )

    validator = Mock(
        side_effect=[["Try again\n" + "E" * 10000 if long_echo else "Try again"], []]
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    token = extraction_provider_guard.set(guard)
    try:
        result = client.generate_chat_response(
            messages,
            model_role=ModelRole.EXTRACTION_AGENT,
            response_format=StructuredProfilesOutput,
            structured_output_validator=validator,
            tools=None,
            tool_choice=None,
        )
    finally:
        extraction_provider_guard.reset(token)
    assert isinstance(result, StructuredProfilesOutput)
    assert reserve.call_count == len(seen) == 2
    assert seen[0]["messages"] != seen[1]["messages"]
    assert len(messages) == 3


@pytest.mark.parametrize(
    "mutation",
    [
        "messages",
        "schema",
        "tools",
        "temperature",
        "output",
        "retries",
        "fallbacks",
        "ladder",
        "parse",
    ],
)
def test_request_changes_refuse_before_attempt_reservation(monkeypatch, mutation):
    prepared, client = preparation(monkeypatch)
    messages = prepared.agent_messages("original source package")
    params = build_profile_package_request(
        client,
        messages,
        model="gpt-4o-mini",
        response_format=StructuredProfilesOutput,
        tools=None,
        tool_choice=None,
        fallback_models=[],
    )
    params["num_retries"] = 0
    ladder = ("gpt-4o-mini",)
    parse = True
    if mutation == "messages":
        params["messages"] = [{"role": "user", "content": "different question"}]
    elif mutation == "schema":
        params["response_format"] = {"type": "json_object"}
    elif mutation == "tools":
        params["tools"] = [{"type": "function", "function": {"name": "other"}}]
    elif mutation == "temperature":
        params["temperature"] = 0.123
    elif mutation == "output":
        params["max_tokens"] = 4096
    elif mutation == "retries":
        params["num_retries"] = 1
    elif mutation == "fallbacks":
        params["fallbacks"] = []
    elif mutation == "ladder":
        ladder = ("gpt-4o",)
    else:
        parse = False
    reserve = Mock()
    with pytest.raises(InputBudgetConfigurationError):
        prepared.check_request("agent", messages, params, ladder, parse)
        reserve()
    reserve.assert_not_called()


def test_fitting_initial_request_refuses_oversized_native_repair_before_reservation(
    monkeypatch,
):
    from reflexio.server.services.profile.components.extractor import (
        _profile_evidence_validation_errors,
    )

    prepared, client = preparation(monkeypatch)

    def initial(count):
        messages = prepared.agent_messages(" original evidence" * count)
        params = build_profile_package_request(
            client,
            messages,
            model="gpt-4o-mini",
            response_format=StructuredProfilesOutput,
            tools=None,
            tool_choice=None,
            fallback_models=[],
        )
        return messages, measure_request_input_tokens(params)

    # Find a fitting complete initial request using the real native counter.
    low, high = 0, 32768
    while low + 1 < high:
        middle = (low + high) // 2
        if initial(middle)[1] <= 32768:
            low = middle
        else:
            high = middle
    messages, tokens = initial(low)
    assert 32760 <= tokens <= 32768 < initial(high)[1]
    original = copy.deepcopy(messages)
    reserve = Mock()
    checked = []

    def guard(params, _timeout, ladder, parse):
        checked.append(copy.deepcopy(params))
        prepared.check_request("agent", messages, params, ladder, parse)
        reserve()

    wire = Mock(
        return_value=litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "profiles": [
                                    {
                                        "content": "unsupported preference",
                                        "time_to_live": "infinity",
                                        "source_span": "absent quotation",
                                    }
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ]
        )
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    token = extraction_provider_guard.set(guard)
    try:
        with pytest.raises(InputBudgetExceededError):
            client.generate_chat_response(
                messages,
                model_role=ModelRole.EXTRACTION_AGENT,
                response_format=StructuredProfilesOutput,
                structured_output_validator=lambda output: (
                    _profile_evidence_validation_errors(output, ())
                ),
                tools=None,
                tool_choice=None,
            )
    finally:
        extraction_provider_guard.reset(token)
    assert reserve.call_count == wire.call_count == 1
    assert len(checked) == 2
    assert measure_request_input_tokens(checked[0]) <= 32768
    assert measure_request_input_tokens(checked[1]) > 32768
    assert messages == original


def test_long_complete_continuation_rejects_at_32k_without_clipping(monkeypatch):
    prepared, client = preparation(monkeypatch)
    answer = "complete human answer " * 20000 + "ONLY with approval"
    messages = [
        *prepared.agent_messages("original source package"),
        {"role": "user", "content": answer},
    ]
    params = build_profile_package_request(
        client,
        messages,
        model="gpt-4o-mini",
        response_format=StructuredProfilesOutput,
        tools=None,
        tool_choice=None,
        fallback_models=[],
    )
    params["num_retries"] = 0
    with pytest.raises(InputBudgetExceededError):
        prepared.check_request("agent", messages, params, ("gpt-4o-mini",), True)
    assert messages[-1]["content"] == params["messages"][-1]["content"] == answer


def test_resolved_fallbacks_reserve_stage_caps_and_smaller_context_lowers_ceiling(
    monkeypatch,
):
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=["gpt-4o"])
    )
    prepared, _ = preparation(
        monkeypatch,
        client=client,
        context_limits={"gpt-4o-mini": 14000, "gpt-4o": 128000},
    )
    assert prepared.snapshot["input_ceiling"] == 5808
    assert prepared.snapshot["consumer"]["ladders"]["agent"] == [
        "gpt-4o-mini",
        "gpt-4o",
    ]
    assert prepared.snapshot["stage_caps"] == {
        "compress": 90,
        "gate": 2,
        "profile_agent": 32,
        "profile_dedup": 4,
        "playbook_intent": 0,
    }


@pytest.mark.parametrize("trigger_source", ["coding_agent", "historical_source", None])
def test_captured_full_prior_suffix_is_priced_and_never_looked_up_again(
    monkeypatch, prior_storage, trigger_source
):
    from reflexio.server.services.extraction.prior_answer_search import (
        capture_prior_knowledge_context,
    )

    prior_tests._capture_record(prior_storage)
    original = group()
    original.interactions[0].content = "Production requires separate approval."
    older = original.model_copy(deep=True)
    older.request.request_id = "older-request"
    older.request.source = "historical_source"
    older.interactions[0].request_id = "older-request"
    older.interactions[0].interaction_id = 2
    groups = [older, original]
    config = ProfileExtractorConfig(extraction_definition_prompt="Stable preferences")
    tools_config = PendingToolCallConfig()
    seed = profile_package_original_seed(
        prompt_manager=PromptManager(),
        groups=groups,
        existing_profiles=[],
        agent_context="Agent context",
        extractor_config=config,
    )
    captured = capture_prior_knowledge_context(
        messages=seed,
        storage=prior_storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_config=config,
        source=trigger_source,
        agent_version=None,
        similarity_threshold=tools_config.for_tool(
            "attach_pending_info_request"
        ).similarity_threshold,
    )
    monkeypatch.setattr(
        prior_storage,
        "search_prior_tool_calls",
        Mock(side_effect=AssertionError("no second lookup")),
    )
    with pytest.raises(InputBudgetConfigurationError, match="prior capture"):
        preparation(
            monkeypatch,
            groups=groups,
            org_id="org_1",
            source=trigger_source,
            extractor_config=config,
            pending_tools_active=True,
            pending_tool_config=tools_config,
        )
    prepared, _ = preparation(
        monkeypatch,
        groups=groups,
        org_id="org_1",
        source=trigger_source,
        extractor_config=config,
        prior_context=captured,
        pending_tools_active=True,
        pending_tool_config=tools_config,
    )
    result = prepared.compression.execute()
    assert result.passthrough
    assert (
        "ONLY with customer approval"
        in prepared.agent_messages("package")[-1]["content"]
    )
    assert prepared.snapshot["prior_context"] == captured.snapshot
    with pytest.raises(InputBudgetConfigurationError, match="lookup contract"):
        preparation(
            monkeypatch,
            groups=groups,
            org_id="org_1",
            source=trigger_source,
            agent_context="different context",
            prior_context=captured,
            pending_tools_active=True,
            pending_tool_config=tools_config,
        )
    with pytest.raises(InputBudgetConfigurationError, match="lookup contract"):
        preparation(
            monkeypatch,
            groups=groups,
            org_id="org_1",
            source="wrong-trigger",
            prior_context=captured,
            pending_tools_active=True,
            pending_tool_config=tools_config,
        )


def test_actual_fallback_uses_its_own_frozen_native_contract(monkeypatch):
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=["gpt-4o"])
    )
    prepared, _ = preparation(monkeypatch, client=client)
    messages = prepared.agent_messages("original source package")
    seen = []

    def guard(params, _timeout, ladder, parse):
        prepared.check_request("agent", messages, params, ladder, parse)

    def transport(params, _timeout):
        seen.append(params["model"])
        if params["model"] == "gpt-4o-mini":
            raise APIConnectionError(
                message="synthetic outage", model="gpt-4o-mini", llm_provider="openai"
            )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": '{"profiles":[]}'},
                    "finish_reason": "stop",
                }
            ]
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    token = extraction_provider_guard.set(guard)
    try:
        result = client.generate_chat_response(
            messages,
            model_role=ModelRole.EXTRACTION_AGENT,
            response_format=StructuredProfilesOutput,
            tools=None,
            tool_choice=None,
        )
    finally:
        extraction_provider_guard.reset(token)
    assert isinstance(result, StructuredProfilesOutput)
    assert seen == ["gpt-4o-mini", "gpt-4o"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unrelated_question",
        "different_schema",
        "oversized_echo",
        "empty_echo",
        "oversized_errors",
        "extra_keys",
    ],
)
def test_unrelated_or_unbounded_validator_suffix_refuses(monkeypatch, mutation):
    from reflexio.server.llm._litellm_text_generation import (
        _validation_repair_correction,
    )

    prepared, client = preparation(monkeypatch)
    messages = prepared.agent_messages("original source package")
    params = build_profile_package_request(
        client,
        messages,
        model="gpt-4o-mini",
        response_format=StructuredProfilesOutput,
        tools=None,
        tool_choice=None,
        fallback_models=[],
    )
    params["num_retries"] = 0
    params["messages"].extend(
        [
            {"role": "assistant", "content": "bounded model reply"},
            {
                "role": "user",
                "content": _validation_repair_correction(
                    "StructuredProfilesOutput", ["field: missing"]
                ),
            },
        ]
    )
    if mutation == "unrelated_question":
        params["messages"][-1]["content"] = "Answer an unrelated question instead"
    elif mutation == "different_schema":
        params["messages"][-1]["content"] = _validation_repair_correction(
            "OtherSchema", ["field: missing"]
        )
    elif mutation == "oversized_echo":
        params["messages"][-2]["content"] = "echo " * 10000
    elif mutation == "empty_echo":
        params["messages"][-2]["content"] = " "
    elif mutation == "oversized_errors":
        params["messages"][-1]["content"] = params["messages"][-1]["content"].replace(
            "field: missing", "E" * 10000
        )
    else:
        params["messages"][-2]["tool_calls"] = []
    with pytest.raises(InputBudgetConfigurationError, match="messages changed"):
        prepared.check_request("agent", messages, params, ("gpt-4o-mini",), True)


@pytest.mark.parametrize("passthrough", [True, False])
@pytest.mark.parametrize("drift", ["template", "schema", "consumer_ladder"])
def test_consumer_drift_refuses_before_cache_or_provider(
    monkeypatch, tmp_path, passthrough, drift
):
    from reflexio.server.services.profile.profile_generation_service_utils import (
        ProfileGenerationServiceConstants as Prompts,
    )

    manager = PromptManager()
    for prompt_id in (
        Prompts.PROFILE_UPDATE_INSTRUCTION_START_PROMPT_ID,
        Prompts.PROFILE_UPDATE_MAIN_PROMPT_ID,
        Prompts.PROFILE_SHOULD_GENERATE_PROMPT_ID,
    ):
        origin = next(
            path / prompt_id
            for path in manager.prompt_bank_paths
            if (path / prompt_id).is_dir()
        )
        shutil.copytree(origin, tmp_path / prompt_id)
    manager = PromptManager(prompt_bank_path=tmp_path)
    original = group()
    original.interactions[0].content = (
        "Short original statement."
        if passthrough
        else "Operational history with conditions. " * 3000
    )
    prepared, client = preparation(
        monkeypatch,
        groups=[original],
        prompt_manager=manager,
        gate_model="gpt-4o",
        context_limits={"gpt-4o-mini": 131072, "gpt-4o": 14000},
    )
    assert prepared.compression.native.compression.passthrough is passthrough
    if drift == "template":
        identity = prepared.snapshot["templates"][Prompts.PROFILE_UPDATE_MAIN_PROMPT_ID]
        path = (
            tmp_path
            / Prompts.PROFILE_UPDATE_MAIN_PROMPT_ID
            / f"v{identity['active_version']}.prompt.md"
        )
        path.write_text(path.read_text() + "\nChanged instructions\n")
    elif drift == "schema":
        monkeypatch.setattr(
            StructuredProfilesOutput,
            "model_json_schema",
            Mock(return_value={"type": "null"}),
        )
    else:
        client.config.fallback_models.append("gpt-4o")
    load = Mock(side_effect=AssertionError("cache must not run"))
    before = Mock(side_effect=AssertionError("provider admission must not run"))
    wire = Mock(side_effect=AssertionError("provider must not run"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    with pytest.raises(
        (ValueError, InputBudgetConfigurationError),
        match="drifted|schema changed|contract changed",
    ):
        prepared.compression.execute(load=load, before_call=before)
    load.assert_not_called()
    before.assert_not_called()
    wire.assert_not_called()
