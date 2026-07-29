import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.services.embedding_text import (
    embedding_input,
    embedding_text,
    playbook_trigger_embedding_text,
    resolve_clustering_similarity,
    resolve_retrieval_threshold,
)


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


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        (
            "When a user requests a long-form lecture without AI safety framing",
            "long-form lecture without AI safety framing",
        ),
        (
            "User asks the agent to generate a 7-minute course",
            "generate 7-minute course",
        ),
        (
            "After the first attempt fails, do not retry before validation",
            "After first attempt fails, do not retry before validation",
        ),
        (
            "Please preserve Project Zephyr's status=error",
            "preserve Project Zephyr's status=error",
        ),
        (
            "When user asks to compare Plan A with Plan B",
            "compare Plan A with Plan B",
        ),
        (
            "When user requests an A/B test",
            "A/B test",
        ),
    ],
)
def test_playbook_trigger_embedding_text_removes_only_low_signal_language(
    trigger: str, expected: str
) -> None:
    assert playbook_trigger_embedding_text(trigger) == expected


def test_playbook_embeddings_use_trigger_only_without_content_fallback() -> None:
    user_playbook = UserPlaybook(
        agent_version="v1",
        request_id="r1",
        content="content must not be embedded",
        trigger=None,
    )
    agent_playbook = AgentPlaybook(
        agent_version="v1",
        content="content must not be embedded",
        trigger=None,
    )

    assert embedding_text(user_playbook) == ""
    assert embedding_text(agent_playbook) == ""


def test_embedding_input_applies_nomic_asymmetric_prefixes() -> None:
    model = "local/nomic-embed-text-v1.5"
    assert embedding_input("hello", model_name=model) == "search_document: hello"
    assert (
        embedding_input("hello", model_name=model, purpose="query")
        == "search_query: hello"
    )


@pytest.mark.parametrize(
    "model_name",
    ["local/minilm-l6-v2", "text-embedding-3-small", "unknown-model"],
)
def test_embedding_input_leaves_non_nomic_models_unprefixed(model_name: str) -> None:
    assert embedding_input("hello", model_name=model_name) == "hello"
    assert embedding_input("hello", model_name=model_name, purpose="query") == "hello"


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("local/minilm-l6-v2", 0.30),
        ("local/nomic-embed-text-v1.5", 0.70),
        ("local/nomic-embed-v1.5", 0.70),
        ("unknown-model", 0.45),
    ],
)
def test_retrieval_threshold_defaults_by_model(
    model_name: str, expected: float
) -> None:
    assert resolve_retrieval_threshold(None, model_name=model_name) == expected


def test_explicit_retrieval_threshold_wins_including_zero() -> None:
    assert (
        resolve_retrieval_threshold(
            0.0,
            model_name="local/nomic-embed-text-v1.5",
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("local/minilm-l6-v2", 0.30),
        ("local/nomic-embed-text-v1.5", 0.85),
        ("local/nomic-embed-v1.5", 0.85),
        ("unknown-model", 0.30),
    ],
)
def test_clustering_similarity_defaults_by_model(
    model_name: str, expected: float
) -> None:
    assert resolve_clustering_similarity(None, model_name=model_name) == expected


def test_explicit_clustering_similarity_wins_including_zero() -> None:
    assert (
        resolve_clustering_similarity(
            0.0,
            model_name="local/nomic-embed-text-v1.5",
        )
        == 0.0
    )


def test_embedding_input_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError, match="Unknown embedding purpose"):
        embedding_input(
            "hello",
            model_name="local/minilm-l6-v2",
            purpose="documnt",  # type: ignore[arg-type]
        )
