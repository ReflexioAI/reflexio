from types import SimpleNamespace

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import GetLearningProvenanceRequest
from reflexio.server.api import (
    _get_agent_playbook_learning_provenance,
    _get_profile_learning_provenance,
    _get_user_playbook_learning_provenance,
)


def _interaction(
    interaction_id: int,
    request_id: str,
    *,
    user_id: str = "user-1",
    created_at: int,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        created_at=created_at,
        content=f"interaction {interaction_id}",
    )


class FakeStorage:
    def __init__(self) -> None:
        self.profiles: dict[str, UserProfile] = {}
        self.requests: dict[str, Request] = {}
        self.interactions: list[Interaction] = []
        self.user_playbooks: dict[int, UserPlaybook] = {}
        self.agent_playbooks: dict[int, AgentPlaybook] = {}
        self.agent_source_windows: dict[int, list[AgentPlaybookSourceWindow]] = {}
        self.interactions_by_ids_calls = 0
        self.interactions_by_request_ids_calls = 0

    def get_profile_by_id(
        self, profile_id: str, *, include_tombstones: bool = False
    ) -> UserProfile | None:
        del include_tombstones
        return self.profiles.get(profile_id)

    def get_request(self, request_id: str) -> Request | None:
        return self.requests.get(request_id)

    def get_interactions_by_request_ids(
        self, request_ids: list[str]
    ) -> list[Interaction]:
        self.interactions_by_request_ids_calls += 1
        request_id_set = set(request_ids)
        return sorted(
            [i for i in self.interactions if i.request_id in request_id_set],
            key=lambda i: i.created_at,
        )

    def get_interactions_by_ids(self, interaction_ids: list[int]) -> list[Interaction]:
        self.interactions_by_ids_calls += 1
        id_set = set(interaction_ids)
        return sorted(
            [i for i in self.interactions if i.interaction_id in id_set],
            key=lambda i: i.created_at,
        )

    def get_last_k_interactions_grouped(
        self,
        user_id: str | None,
        k: int,
        sources: list[str] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        agent_version: str | None = None,
    ) -> tuple[list[object], list[Interaction]]:
        del sources, agent_version
        filtered = [
            i for i in self.interactions if user_id is None or i.user_id == user_id
        ]
        if start_time is not None:
            filtered = [i for i in filtered if i.created_at >= start_time]
        if end_time is not None:
            filtered = [i for i in filtered if i.created_at <= end_time]
        return [], sorted(filtered, key=lambda i: i.created_at, reverse=True)[:k]

    def get_user_playbook_by_id(
        self, user_playbook_id: int, *, include_tombstones: bool = False
    ) -> UserPlaybook | None:
        del include_tombstones
        return self.user_playbooks.get(user_playbook_id)

    def get_agent_playbook_by_id(
        self, agent_playbook_id: int, *, include_tombstones: bool = False
    ) -> AgentPlaybook | None:
        del include_tombstones
        return self.agent_playbooks.get(agent_playbook_id)

    def get_source_windows_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[AgentPlaybookSourceWindow]:
        return self.agent_source_windows.get(agent_playbook_id, [])

    def get_user_playbooks_by_ids_any_user(
        self,
        user_playbook_ids: list[int],
        status_filter: list[object] | None = None,
    ) -> list[UserPlaybook]:
        del status_filter
        return [
            self.user_playbooks[user_playbook_id]
            for user_playbook_id in user_playbook_ids
            if user_playbook_id in self.user_playbooks
        ]


def _reflexio(storage: FakeStorage) -> SimpleNamespace:
    config = SimpleNamespace(
        window_size=2,
        stride_size=1,
        profile_extractor_config=SimpleNamespace(
            window_size_override=None,
            stride_size_override=None,
            request_sources_enabled=None,
        ),
    )
    return SimpleNamespace(
        request_context=SimpleNamespace(
            storage=storage,
            configurator=SimpleNamespace(get_config=lambda: config),
        )
    )


def test_profile_provenance_returns_exact_interactions_when_source_ids_exist() -> None:
    storage = FakeStorage()
    storage.interactions = [
        _interaction(2, "req-2", created_at=20),
        _interaction(1, "req-1", created_at=10),
    ]
    storage.profiles["profile-1"] = UserProfile(
        profile_id="profile-1",
        user_id="user-1",
        content="profile",
        last_modified_timestamp=30,
        generated_from_request_id="req-2",
        source_interaction_ids=[2, 1],
    )

    response = _get_profile_learning_provenance(
        GetLearningProvenanceRequest(kind="profile", id="profile-1"),
        _reflexio(storage),
    )

    assert response.success is True
    assert response.provenance_status == "exact"
    assert [i.interaction_id for i in response.interactions] == [1, 2]


def test_profile_provenance_reconstructs_best_effort_window_without_source_ids() -> (
    None
):
    storage = FakeStorage()
    storage.requests["req-3"] = Request(
        request_id="req-3",
        user_id="user-1",
        created_at=30,
        source="web",
        session_id="session-1",
    )
    storage.interactions = [
        _interaction(1, "req-1", created_at=10),
        _interaction(2, "req-2", created_at=20),
        _interaction(3, "req-3", created_at=30),
    ]
    storage.profiles["profile-1"] = UserProfile(
        profile_id="profile-1",
        user_id="user-1",
        content="profile",
        last_modified_timestamp=30,
        generated_from_request_id="req-3",
    )

    response = _get_profile_learning_provenance(
        GetLearningProvenanceRequest(kind="profile", id="profile-1"),
        _reflexio(storage),
    )

    assert response.success is True
    assert response.provenance_status == "best_effort"
    assert [i.interaction_id for i in response.interactions] == [2, 3]


def test_user_playbook_provenance_uses_source_ids_and_falls_back_to_request() -> None:
    storage = FakeStorage()
    storage.interactions = [
        _interaction(1, "req-1", created_at=10),
        _interaction(2, "req-2", created_at=20),
    ]
    storage.user_playbooks[7] = UserPlaybook(
        user_playbook_id=7,
        user_id="user-1",
        agent_version="v1",
        request_id="req-2",
        content="playbook",
        source_interaction_ids=[],
    )

    response = _get_user_playbook_learning_provenance(
        GetLearningProvenanceRequest(kind="user_playbook", id="7"),
        _reflexio(storage),
    )

    assert response.success is True
    assert response.provenance_status == "best_effort"
    assert [i.interaction_id for i in response.interactions] == [2]

    storage.user_playbooks[7].source_interaction_ids = [1]
    response = _get_user_playbook_learning_provenance(
        GetLearningProvenanceRequest(kind="user_playbook", id="7"),
        _reflexio(storage),
    )

    assert response.provenance_status == "exact"
    assert [i.interaction_id for i in response.interactions] == [1]


def test_agent_playbook_provenance_groups_by_source_user_playbook() -> None:
    storage = FakeStorage()
    storage.interactions = [
        _interaction(1, "req-1", created_at=10),
        _interaction(2, "req-1", created_at=20),
        _interaction(3, "req-2", created_at=30),
    ]
    storage.user_playbooks[11] = UserPlaybook(
        user_playbook_id=11,
        user_id="user-1",
        agent_version="v1",
        request_id="req-1",
        content="source 1",
    )
    storage.user_playbooks[12] = UserPlaybook(
        user_playbook_id=12,
        user_id="user-1",
        agent_version="v1",
        request_id="req-2",
        content="source 2",
    )
    storage.agent_playbooks[3] = AgentPlaybook(
        agent_playbook_id=3,
        agent_version="v1",
        content="agent playbook",
    )
    storage.agent_source_windows[3] = [
        AgentPlaybookSourceWindow(user_playbook_id=11, source_interaction_ids=[2, 1]),
        AgentPlaybookSourceWindow(user_playbook_id=12, source_interaction_ids=[3]),
    ]

    response = _get_agent_playbook_learning_provenance(
        GetLearningProvenanceRequest(kind="agent_playbook", id="3"),
        _reflexio(storage),
    )

    assert response.success is True
    assert response.provenance_status == "exact"
    assert [
        g.user_playbook.user_playbook_id for g in response.source_user_playbooks
    ] == [
        11,
        12,
    ]
    assert [
        [i.interaction_id for i in group.interactions]
        for group in response.source_user_playbooks
    ] == [[1, 2], [3]]
    assert storage.interactions_by_ids_calls == 1
    assert storage.interactions_by_request_ids_calls == 0


def test_missing_learning_target_returns_structured_failure() -> None:
    response = _get_profile_learning_provenance(
        GetLearningProvenanceRequest(kind="profile", id="missing"),
        _reflexio(FakeStorage()),
    )

    assert response.success is False
    assert response.provenance_status == "unavailable"
    assert response.msg == "Profile not found"
