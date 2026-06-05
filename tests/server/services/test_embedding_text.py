import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    UserProfile,
)
from reflexio.server.services.embedding_text import embedding_input, embedding_text


def test_user_profile_embedding_text_omits_empty_custom_features() -> None:
    profile = UserProfile(
        profile_id="p1",
        user_id="u1",
        content="likes dark mode",
        last_modified_timestamp=1,
        generated_from_request_id="r1",
    )

    assert embedding_text(profile) == "likes dark mode"

    profile.custom_features = {}
    assert embedding_text(profile) == "likes dark mode"


def test_user_profile_embedding_text_includes_custom_features_when_present() -> None:
    profile = UserProfile(
        profile_id="p1",
        user_id="u1",
        content="likes dark mode",
        last_modified_timestamp=1,
        generated_from_request_id="r1",
        custom_features={"theme": "dark"},
    )

    assert embedding_text(profile) == "likes dark mode\n{'theme': 'dark'}"


def test_agent_success_embedding_text_omits_missing_failure_fields() -> None:
    assert (
        embedding_text(
            AgentSuccessEvaluationResult(
                agent_version="v1",
                session_id="s1",
                is_success=False,
                failure_type=None,
                failure_reason="agent stalled",
            )
        )
        == "agent stalled"
    )
    assert (
        embedding_text(
            AgentSuccessEvaluationResult(
                agent_version="v1",
                session_id="s1",
                is_success=True,
                failure_type=None,
                failure_reason=None,
            )
        )
        == ""
    )


def test_embedding_input_applies_asymmetric_prefixes() -> None:
    assert embedding_input("hello") == "search_document: hello"
    assert embedding_input("hello", purpose="query") == "search_query: hello"


def test_embedding_input_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError, match="Unknown embedding purpose"):
        embedding_input("hello", purpose="documnt")  # type: ignore[arg-type]
