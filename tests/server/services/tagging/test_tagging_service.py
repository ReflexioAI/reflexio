from __future__ import annotations

from typing import Any

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    Config,
    LLMConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.tagging.tagging_service import TaggingService, TagsOutput


class FakeStorage:
    def __init__(self) -> None:
        self.profiles = [
            UserProfile(
                profile_id="profile-1",
                user_id="user-1",
                content="User prefers concise answers.",
                last_modified_timestamp=1,
                generated_from_request_id="request-1",
            )
        ]
        self.user_playbooks = [
            UserPlaybook(
                user_playbook_id=10,
                user_id="user-1",
                agent_version="agent-1",
                request_id="request-1",
                content="Ask a clarifying question before destructive actions.",
                trigger="Dangerous command requested",
                rationale="The user values explicit confirmation.",
            )
        ]
        self.agent_playbooks = [
            AgentPlaybook(
                agent_playbook_id=20,
                agent_version="agent-1",
                content="Summarize validation evidence in final responses.",
                trigger="After code changes",
                rationale="It helps reviewers trust the change.",
            )
        ]
        self.profile_updates: list[tuple[str, str, list[str]]] = []
        self.user_playbook_updates: list[tuple[int, list[str] | None]] = []
        self.agent_playbook_updates: list[tuple[int, list[str] | None]] = []

    def get_user_profile(self, user_id: str) -> list[UserProfile]:
        assert user_id == "user-1"
        return self.profiles

    def update_user_profile_tags(
        self, user_id: str, profile_id: str, tags: list[str]
    ) -> None:
        self.profile_updates.append((user_id, profile_id, tags))

    def get_user_playbooks(
        self,
        *,
        limit: int = 100,
        user_id: str,
        agent_version: str,
        status_filter: list[Any],
    ) -> list[UserPlaybook]:
        assert (user_id, agent_version, status_filter) == ("user-1", "agent-1", [None])
        return self.user_playbooks

    def update_user_playbook(
        self, user_playbook_id: int, *, tags: list[str] | None = None, **_: Any
    ) -> None:
        self.user_playbook_updates.append((user_playbook_id, tags))

    def get_agent_playbooks(
        self, *, limit: int = 100, agent_version: str, status_filter: list[Any]
    ) -> list[AgentPlaybook]:
        assert (agent_version, status_filter) == ("agent-1", [None])
        return self.agent_playbooks

    def update_agent_playbook(
        self, agent_playbook_id: int, *, tags: list[str] | None = None, **_: Any
    ) -> None:
        self.agent_playbook_updates.append((agent_playbook_id, tags))


class FakeConfigurator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def get_config(self) -> Config:
        return self.config


class FakePromptManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def render_prompt(self, prompt_id: str, variables: dict[str, str]) -> str:
        self.calls.append((prompt_id, variables))
        return f"{variables['tagging_definition_prompt']}::{variables['content']}"


class FakeRequestContext:
    def __init__(self, config: Config, storage: FakeStorage) -> None:
        self.configurator = FakeConfigurator(config)
        self.storage = storage
        self.prompt_manager = FakePromptManager()


class FakeLLMClient:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_chat_response(self, **kwargs: Any) -> TagsOutput:
        self.calls.append(kwargs)
        return TagsOutput(tags=self.responses.pop(0))


def make_config(
    *,
    profile_tagging_prompt: str | None = "profile tagging rules",
    playbook_tagging_prompt: str | None = "playbook tagging rules",
) -> Config:
    return Config(
        storage_config=StorageConfigSQLite(db_path=":memory:"),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="profile extraction rules",
            tagging_definition_prompt=profile_tagging_prompt,
        ),
        user_playbook_extractor_config=UserPlaybookExtractorConfig(
            extraction_definition_prompt="playbook extraction rules",
            tagging_definition_prompt=playbook_tagging_prompt,
        ),
        llm_config=LLMConfig(generation_model_name="test-model"),
    )


def test_tagging_service_updates_profiles_and_playbooks(monkeypatch: Any) -> None:
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_service.SiteVarManager.get_site_var",
        lambda *_: {},
    )
    storage = FakeStorage()
    context = FakeRequestContext(make_config(), storage)
    client = FakeLLMClient(
        [["profile-tag"], ["user-playbook-tag"], ["agent-playbook-tag"]]
    )

    TaggingService(client, context).run(user_id="user-1", agent_version="agent-1")  # type: ignore[arg-type]

    assert storage.profile_updates == [("user-1", "profile-1", ["profile-tag"])]
    assert storage.user_playbook_updates == [(10, ["user-playbook-tag"])]
    assert storage.agent_playbook_updates == [(20, ["agent-playbook-tag"])]
    assert [call[0] for call in context.prompt_manager.calls] == [
        "tagging",
        "tagging",
        "tagging",
    ]
    assert [
        call[1]["tagging_definition_prompt"] for call in context.prompt_manager.calls
    ] == [
        "profile tagging rules",
        "playbook tagging rules",
        "playbook tagging rules",
    ]
    assert [call["model"] for call in client.calls] == [
        "test-model",
        "test-model",
        "test-model",
    ]


def test_tagging_service_skips_entity_types_without_tagging_prompts(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_service.SiteVarManager.get_site_var",
        lambda *_: {},
    )
    storage = FakeStorage()
    context = FakeRequestContext(
        make_config(profile_tagging_prompt=None, playbook_tagging_prompt=None), storage
    )
    client = FakeLLMClient([])

    TaggingService(client, context).run(user_id="user-1", agent_version="agent-1")  # type: ignore[arg-type]

    assert storage.profile_updates == []
    assert storage.user_playbook_updates == []
    assert storage.agent_playbook_updates == []
    assert context.prompt_manager.calls == []
    assert client.calls == []


def test_tagging_service_can_scope_to_profiles_only(monkeypatch: Any) -> None:
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_service.SiteVarManager.get_site_var",
        lambda *_: {},
    )
    storage = FakeStorage()
    context = FakeRequestContext(make_config(), storage)
    client = FakeLLMClient([["profile-tag"]])

    TaggingService(client, context).run(  # type: ignore[arg-type]
        user_id="user-1",
        agent_version="agent-1",
        tag_playbooks=False,
    )

    assert storage.profile_updates == [("user-1", "profile-1", ["profile-tag"])]
    assert storage.user_playbook_updates == []
    assert storage.agent_playbook_updates == []
    assert len(context.prompt_manager.calls) == 1


def test_tagging_service_skips_already_tagged_entities(monkeypatch: Any) -> None:
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_service.SiteVarManager.get_site_var",
        lambda *_: {},
    )
    storage = FakeStorage()
    # An empty list means tagging already ran and matched nothing — a final state.
    # The service must NOT re-tag these (only tags is None means "never tagged").
    storage.profiles[0].tags = []
    storage.user_playbooks[0].tags = ["existing"]
    storage.agent_playbooks[0].tags = []
    context = FakeRequestContext(make_config(), storage)
    client = FakeLLMClient([])

    TaggingService(client, context).run(user_id="user-1", agent_version="agent-1")  # type: ignore[arg-type]

    assert storage.profile_updates == []
    assert storage.user_playbook_updates == []
    assert storage.agent_playbook_updates == []
    assert context.prompt_manager.calls == []
    assert client.calls == []
