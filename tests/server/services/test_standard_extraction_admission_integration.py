"""The completed rollout is standard execution, not a deployment-mode choice."""

import litellm
import pytest

from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline


@pytest.mark.parametrize("mode", ["platform", "self_host"])
@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("obsolete_value", [None, "false", "true"])
def test_new_text_uses_saved_protocol_regardless_of_obsolete_rollout_assignment(
    pipeline, monkeypatch, mode, kind, obsolete_value
):
    engine, _, http = pipeline
    monkeypatch.setenv("DEPLOYMENT_MODE", mode)
    monkeypatch.setenv(
        "REFLEXIO_GOVERNANCE_REF_SECRET", "isolated-standard-admission-" + "x" * 40
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    if obsolete_value is None:
        monkeypatch.delenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", obsolete_value)
    if kind == "playbook":
        configurator = engine.request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Learn recurring practices"
                    ),
                }
            )
        )
    calls = []

    def respond(_client, params, _timeout):
        calls.append(params)
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {"role": "assistant", "content": "false"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
        )

    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", respond)
    assert http.post("/api/publish_interaction", json=payload("standard-text")).json()[
        "success"
    ]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("standard-admission", 300)
    window = storage.prepare_extraction(user, token)
    assert window.kind == kind
    WindowExecutor(
        engine.request_context,
        LiteLLMClient(
            LiteLLMConfig(model="gpt-4o", max_tokens=8192, fallback_models=[])
        ),
    ).execute(window, token)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    manifest = parent.generation_request_snapshot["overflow"]
    assert manifest["version"] == 1
    assert manifest["checkpoint"]["stage"] == "gated_false"
    assert len(manifest["attempts"]) == len(calls) == 1
    assert storage.extraction_status("u", "standard-text") == {
        "status": "done",
        "reason": "covered",
    }
    assert storage.count_all_profiles() == 0
    assert not storage.get_user_playbooks(user_id="u")
