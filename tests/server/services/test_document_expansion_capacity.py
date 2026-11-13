"""Complete expansion input crosses the actual renderer/client admission seam."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.config_schema import APIKeyConfig, LLMConfig
from reflexio.server.llm import input_budget
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval import DocumentExpander
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.unit

CONTENT = "begin\n" + "文😀" * 2500 + "\nimportant-middle-and-tail"


@pytest.fixture
def capacity(monkeypatch: pytest.MonkeyPatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"tail":["synonym"]}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="gpt-4o",
        _hidden_params={},
    )
    dispatch = Mock(return_value=response)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    counter = Mock(return_value=5000)
    info = Mock(return_value={"max_input_tokens": 100_000, "max_output_tokens": 200})
    monkeypatch.setattr(input_budget.litellm, "token_counter", counter)
    monkeypatch.setattr(input_budget.litellm, "get_model_info", info)
    return dispatch, counter, info


def _expander(
    *,
    max_tokens: int | None = 32,
    model: str = "gpt-4o",
    keys: APIKeyConfig | None = None,
) -> DocumentExpander:
    return DocumentExpander(
        LiteLLMClient(
            LiteLLMConfig(
                model=model,
                max_tokens=max_tokens,
                api_key_config=keys,
                fallback_models=[],
            )
        ),
        PromptManager(),
    )


def test_complete_unicode_input_reaches_real_client_dispatch(capacity):
    dispatch, counter, _ = capacity
    assert _expander().expand(CONTENT).expanded_text == "tail, synonym"
    params = dispatch.call_args.args[0]
    assert CONTENT in params["messages"][-1]["content"]
    assert params["max_tokens"] == 32
    counter.assert_called()
    assert input_budget._model_capacity_limits.get() is None


def test_capacity_refusal_returns_empty_without_dispatch_or_leaked_scope(capacity):
    dispatch, _, info = capacity
    info.return_value = {"max_input_tokens": 100, "max_output_tokens": 200}
    assert _expander().expand(CONTENT).expanded_text == ""
    dispatch.assert_not_called()
    assert input_budget._model_capacity_limits.get() is None


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("output", [None, 32])
def test_custom_gateway_requires_existing_context_and_explicit_output_policy(
    capacity, monkeypatch, declared, output
):
    dispatch, _, _ = capacity
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        json.dumps({"openai/opaque@https://gateway.invalid/v1": 100_000})
        if declared
        else "{}",
    )
    keys = APIKeyConfig.model_validate(
        {
            "custom_endpoint": {
                "api_base": "https://gateway.invalid/v1",
                "api_key": "fixture-only",
            }
        }
    )
    result = _expander(max_tokens=output, model="openai/opaque", keys=keys).expand(
        CONTENT
    )
    if declared and output is not None:
        assert result.expanded_text == "tail, synonym"
        assert CONTENT in dispatch.call_args.args[0]["messages"][-1]["content"]
        assert dispatch.call_args.args[0]["max_tokens"] == output
    else:
        assert result.expanded_text == ""
        dispatch.assert_not_called()
    assert input_budget._model_capacity_limits.get() is None


@pytest.mark.parametrize("provider", ["claude-code", "openclaw"])
def test_host_capacity_remains_delegated(capacity, provider):
    dispatch, counter, _ = capacity
    assert (
        _expander(max_tokens=None, model=f"{provider}/default")
        .expand(CONTENT)
        .expanded_text
        == "tail, synonym"
    )
    assert CONTENT in dispatch.call_args.args[0]["messages"][-1]["content"]
    counter.assert_not_called()


def test_input_admission_does_not_rewrite_an_explicit_output_cap(capacity):
    dispatch, _, info = capacity
    info.return_value = {"max_input_tokens": 100_000, "max_output_tokens": 20}
    assert _expander(max_tokens=1000).expand(CONTENT).expanded_text == "tail, synonym"
    # Input admission reserves the requested output; the provider owns accepting
    # its output maximum. This change promises no new output-cap normalization.
    assert dispatch.call_args.args[0]["max_tokens"] == 1000


@pytest.mark.parametrize(
    "policy",
    [
        None,
        LLMConfig(),
        LLMConfig(generation_max_output_tokens=None),
        LLMConfig(generation_max_output_tokens=37),
    ],
)
def test_sqlite_constructor_preserves_explicit_output_policy(tmp_path: Path, policy):
    storage = SQLiteStorage(
        org_id="capacity-policy", db_path=str(tmp_path / "policy.db"), llm_config=policy
    )
    try:
        assert storage.llm_client.config.max_tokens == (
            policy.generation_max_output_tokens if policy else None
        )
    finally:
        storage.conn.close()


@pytest.mark.parametrize("fits", [False, True])
def test_sqlite_storage_keeps_source_and_indexes_after_expansion_admission(
    tmp_path: Path, capacity, monkeypatch: pytest.MonkeyPatch, fits: bool
):
    dispatch, _, info = capacity
    if not fits:
        info.return_value = {"max_input_tokens": 100, "max_output_tokens": 200}
    storage = SQLiteStorage(
        org_id="capacity-storage",
        db_path=str(tmp_path / "source.db"),
        llm_config=LLMConfig(
            generation_model_name="gpt-4o", generation_max_output_tokens=32
        ),
        enable_document_expansion=True,
    )
    monkeypatch.setattr(storage, "_get_embedding", lambda _text: [])
    profile = UserProfile(
        profile_id="profile",
        user_id="user",
        content=CONTENT,
        last_modified_timestamp=1,
        generated_from_request_id="request",
    )
    try:
        storage.add_user_profile("user", [profile])
        row = storage.conn.execute(
            "SELECT content, expanded_terms FROM profiles WHERE profile_id = ?",
            ("profile",),
        ).fetchone()
        assert row is not None and row["content"] == CONTENT
        assert row["expanded_terms"] == ("tail, synonym" if fits else None)
        if fits:
            assert CONTENT in dispatch.call_args.args[0]["messages"][-1]["content"]
        else:
            dispatch.assert_not_called()
        # Source text remains searchable when optional enrichment is refused.
        assert (
            storage.conn.execute(
                "SELECT count(*) FROM profiles_fts WHERE profiles_fts MATCH 'important' "
            ).fetchone()[0]
            == 1
        )
    finally:
        storage.conn.close()
