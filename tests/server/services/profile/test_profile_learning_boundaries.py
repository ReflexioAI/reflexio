"""Source handoff and provider policy stay inside native ownership boundaries."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.profile.learning.client_policy import (
    CURRENT_COMPONENT,
    ComponentClient,
)
from reflexio.server.services.profile.learning.extractor import (
    SourceOnlyProfileExtractor,
    collect_review_source,
)
from reflexio.server.services.profile.service import ProfileGenerationServiceConfig


class Output(BaseModel):
    result: str


def session(text="I prefer tea.", identity=1):
    return RequestInteractionDataModel(
        session_id="session",
        request=Request(
            request_id="request",
            user_id="owner",
            session_id="session",
            created_at=10,
            source="test",
        ),
        interactions=[
            Interaction(
                interaction_id=identity,
                user_id="owner",
                request_id="request",
                content=text,
                role="user",
                created_at=10,
            )
        ],
    )


def test_manual_sources_union_without_changing_native_selection_or_aliasing():
    config = ProfileGenerationServiceConfig("owner", "request")
    first, second = session(), session("I also like hiking.", 2)
    before = deepcopy([first, second])
    collect_review_source(config, [first])
    collect_review_source(config, [second, first])
    assert config.window_interactions is None
    assert len(config.review_interactions) == 1
    assert [item.content for item in config.review_interactions[0].interactions] == [
        "I prefer tea.",
        "I also like hiking.",
    ]
    assert [first, second] == before
    assert config.review_interactions[0] is not first
    with pytest.raises(ValueError, match="Conflicting original source"):
        collect_review_source(config, [session("Conflicting text.")])


def test_explicit_durable_window_is_authoritative_and_unchanged():
    explicit = [session()]
    config = ProfileGenerationServiceConfig(
        "owner", "request", window_interactions=explicit
    )
    collect_review_source(config, [session("Different selected text.", 2)])
    assert config.window_interactions is explicit
    assert config.review_interactions == []


def test_extractor_reads_only_conversation_and_keeps_native_null_result():
    extractor = object.__new__(SourceOnlyProfileExtractor)
    extractor.service_config = ProfileGenerationServiceConfig("owner", "request")
    selected = [session()]
    with patch.object(
        ProfileExtractor, "_generate_raw_updates_from_sessions", return_value=None
    ) as native:
        assert (
            extractor._generate_raw_updates_from_sessions(selected, ["incumbent"])
            is None
        )
    native.assert_called_once_with(selected, existing_profiles=[])
    assert extractor.service_config.review_interactions == selected
    assert extractor.service_config.window_interactions is None


@pytest.mark.parametrize(
    "model", ["minimax/MiniMax-M3", "gpt-6-sol", "anthropic/claude-sonnet-5"]
)
def test_producing_policy_preserves_native_provider_compatibility_and_resets_context(
    model,
):
    seen = []

    def produce(**kwargs):
        seen.append((CURRENT_COMPONENT.get(), kwargs))
        raise RuntimeError("native provider failure")

    client = SimpleNamespace(
        get_completion_model_ladder=lambda **_kwargs: [model],
        generate_chat_response_with_provenance=produce,
    )
    view = ComponentClient(client, "native_profile_extraction", Output)
    original = {"metadata": {"key": "value"}}
    with pytest.raises(RuntimeError, match="native provider failure"):
        view.generate_chat_response_with_provenance(
            response_format=Output, extra_body=original
        )
    assert CURRENT_COMPONENT.get() is None
    assert seen[0][0]["component"] == "native_profile_extraction"
    assert original == {"metadata": {"key": "value"}}
    body = seen[0][1]["extra_body"]
    if model == "minimax/MiniMax-M3":
        assert body["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in body
    with pytest.raises(RuntimeError):
        view.generate_chat_response_with_provenance(
            messages=[{"role": "user", "content": "ordinary request"}]
        )
    assert seen[-1][0] is None
    assert "extra_body" not in seen[-1][1]


def test_conflicting_m3_policy_refuses_dispatch():
    client = MagicMock()
    client.get_completion_model_ladder.return_value = ["minimax/MiniMax-M3"]
    view = ComponentClient(client, "unified_change_planner", Output)
    with pytest.raises(ValueError, match="Conflicting component thinking"):
        view.generate_chat_response_with_provenance(
            response_format=Output, extra_body={"thinking": {"type": "adaptive"}}
        )
    client.generate_chat_response_with_provenance.assert_not_called()


def test_mixed_native_fallback_ladder_does_not_leak_minimax_options():
    client = MagicMock()
    client.get_completion_model_ladder.return_value = [
        "minimax/MiniMax-M3",
        "gpt-6-sol",
    ]
    view = ComponentClient(client, "unified_change_planner", Output)
    view.generate_chat_response_with_provenance(response_format=Output)
    assert (
        "extra_body"
        not in client.generate_chat_response_with_provenance.call_args.kwargs
    )


@pytest.mark.parametrize(
    "raw,canonical", [("User", "user"), ("ASSISTANT", "assistant"), ("Tool", "tool")]
)
def test_known_native_role_case_preserves_authentic_source_metadata(raw, canonical):
    from reflexio.server.services.profile.learning.compiler import original_window

    selected = session()
    selected.interactions[0].role = raw
    row = original_window([selected])[0]["interactions"][0]
    assert row["role"] == canonical
    assert row["source_role"] == raw
    assert selected.interactions[0].role == raw


def test_unknown_speaker_role_is_never_invented_as_user():
    from reflexio.server.services.profile.learning.compiler import original_window

    selected = session()
    selected.interactions[0].role = "Financial advisor"
    row = original_window([selected])[0]["interactions"][0]
    assert row["role"] == "Financial advisor"
    assert "source_role" not in row


def test_configured_eligibility_is_fused_into_producing_definition():
    from reflexio.server.services.profile.learning.extractor import (
        profile_extraction_definition,
    )
    from reflexio.server.services.profile.learning.linked_plan import (
        EXTRACTION_GUIDANCE,
    )

    plain = profile_extraction_definition("Extract preferences")
    fused = profile_extraction_definition(
        "Extract preferences", "Only user-endorsed preferences"
    )
    assert plain.startswith("Extract preferences\n\n" + EXTRACTION_GUIDANCE)
    assert "Only user-endorsed preferences" in fused
    assert "return null profiles" in fused
    assert EXTRACTION_GUIDANCE in fused
    assert profile_extraction_definition("Extract preferences", "  ") == plain
