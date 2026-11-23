"""Four current settings drive actual HTTP admission and SQLite selection."""

import pytest

from reflexio.models.config_schema import Config, UserPlaybookExtractorConfig
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("finished", [False, True])
def test_manual_only_http_publish_persists_source_without_provider_or_eligible_window(
    pipeline, monkeypatch, force, finished
):
    from unittest.mock import Mock

    from reflexio.server.llm.litellm_client import LiteLLMClient

    engine, worker, http = pipeline
    forbidden = Mock(
        side_effect=AssertionError("Manual-only publish dispatched a model")
    )
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", forbidden)
    configurator = engine.request_context.configurator
    configurator.set_config_by_name("profile_stride_size", 0)
    request = durable.payload("manual-only", count=4, force=force)
    request["session_finished"] = finished
    response = http.post("/api/publish_interaction", json=request)
    assert response.status_code == 200 and response.json()["success"]
    storage = engine.get_storage()
    assert storage.get_request("manual-only") is not None
    assert worker.drain_org("stream-test", 1, 300) == 0
    forbidden.assert_not_called()

    # Deliberate re-enablement admits new source only; the stored manual-only
    # publication must not be enrolled into the later automatic window.
    configurator.set_config_by_name("profile_stride_size", 3)
    response = http.post(
        "/api/publish_interaction",
        json=durable.payload("explicitly-enabled", count=4, force=True),
    )
    assert response.status_code == 200 and response.json()["success"]
    user, token = storage.claim_extraction("manual-policy-owner", 300)
    assert user == "u" and token
    window = storage.prepare_extraction(user, token)
    assert window is not None and window.kind == "profile"
    assert len(window.manifest) == 4
    assert window.policy["stride_size"] == 3
    forbidden.assert_not_called()


@pytest.mark.parametrize("kind,width,stride", [("profile", 16, 4), ("playbook", 24, 2)])
def test_per_kind_policy_drives_the_selected_window(pipeline, kind, width, stride):
    engine, _worker, http = pipeline
    configurator = engine.request_context.configurator
    policy = configurator.get_config().model_dump(mode="python")
    policy.update(
        profile_window_size=16,
        profile_stride_size=4,
        playbook_window_size=24,
        playbook_stride_size=2,
    )
    if kind == "playbook":
        policy.update(
            profile_extractor_config=None,
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Synthetic playbook policy"
            ),
        )
    configurator.set_config(Config.model_validate(policy))
    response = http.post(
        "/api/publish_interaction", json=durable.payload("four-field-policy", count=32)
    )
    assert response.status_code == 200
    assert response.json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("four-field-policy-worker", 600)
    assert user == "u" and token
    window = storage.prepare_extraction(user, token)
    assert window is not None and window.kind == kind
    assert len(window.manifest) == width
    assert window.policy["window_size"] == width
    assert window.policy["stride_size"] == stride


def test_finished_session_closes_under_actual_large_workspace_window(pipeline):
    engine, _worker, http = pipeline
    configurator = engine.request_context.configurator
    source = configurator.get_config().model_dump(mode="python")
    source.update(
        profile_window_size=100,
        profile_stride_size=80,
        user_playbook_extractor_config=None,
    )
    configurator.set_config(Config.model_validate(source))
    request = durable.payload("finite-core-session", count=4)
    request.update(session_finished=True, force_extraction=False)
    response = http.post("/api/publish_interaction", json=request)
    assert response.status_code == 200 and response.json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("finite-core-owner", 600)
    assert user == "u" and token
    window = storage.prepare_extraction(user, token)
    assert window is not None and window.kind == "profile"
    assert len(window.manifest) == 4 and window.force is False
    assert (window.policy["window_size"], window.policy["stride_size"]) == (100, 80)


def test_excluded_source_can_publish_under_a_manual_usable_small_window(pipeline):
    engine, _worker, http = pipeline
    configurator = engine.request_context.configurator
    policy = configurator.get_config().model_dump(mode="python")
    policy.update(
        profile_window_size=2,
        profile_stride_size=8,
        user_playbook_extractor_config=None,
    )
    policy["profile_extractor_config"]["request_sources_enabled"] = ["allowed"]
    configurator.set_config(Config.model_validate(policy))
    response = http.post(
        "/api/publish_interaction",
        json=durable.payload("excluded-small-window", count=4),
    )
    assert response.status_code == 200 and response.json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("excluded-window-owner", 600)
    assert user == "u" and token
    # The cursor still inspects accepted source; it cannot create an eligible
    # extraction window from a request filtered out at admission.
    assert storage.prepare_extraction(user, token) is None

    from reflexio.models.api_schema.domain.entities import PublishUserInteractionRequest
    from reflexio.server.services.durable_learning.admission import extraction_admission

    eligible = PublishUserInteractionRequest.model_validate(
        {**durable.payload("eligible-small-window"), "source": "allowed"}
    )
    with pytest.raises(ValueError):
        extraction_admission(configurator.get_config(), eligible)
