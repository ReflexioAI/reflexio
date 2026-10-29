"""Learning provenance over HTTP against real SQLite storage.

The sibling ``tests/server/test_learning_provenance_api.py`` drives the
handlers with an in-memory fake. These tests publish real rows (requests,
interactions, profiles, user/agent playbooks and source windows) and read
them back through ``POST /api/get_learning_provenance``.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

ROUTE = "/api/get_learning_provenance"


class Harness:
    def __init__(self, engine: Reflexio, client: TestClient) -> None:
        self.engine = engine
        self.client = client

    @property
    def storage(self) -> SQLiteStorage:
        storage = self.engine.request_context.storage
        assert isinstance(storage, SQLiteStorage)
        return storage

    def set_profile_config(self, profile_config: ProfileExtractorConfig | None):
        configurator = self.engine.request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={"profile_extractor_config": profile_config}
            )
        )

    def publish(
        self,
        request_id: str,
        *,
        user_id: str = "u1",
        source: str = "chat",
        created_at: int = 1_000,
        contents: tuple[str, ...] = ("hello",),
    ) -> list[int]:
        self.storage.add_request(
            Request(
                request_id=request_id,
                user_id=user_id,
                created_at=created_at,
                source=source,
                session_id=f"session-{request_id}",
            )
        )
        for offset, content in enumerate(contents):
            self.storage.add_user_interaction(
                user_id,
                Interaction(
                    user_id=user_id,
                    request_id=request_id,
                    created_at=created_at + offset,
                    content=content,
                ),
            )
        return [
            i.interaction_id
            for i in self.storage.get_interactions_by_request_ids([request_id])
        ]

    def post(self, kind: str, target_id: object) -> dict:
        response = self.client.post(ROUTE, json={"kind": kind, "id": str(target_id)})
        assert response.status_code == 200, response.text
        return response.json()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    org_id = f"prov-{uuid.uuid4().hex[:12]}"
    configurator = DefaultConfigurator(org_id=org_id, base_dir=str(tmp_path))
    configurator.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "prov.db")),
            window_size=10,
            stride_size=5,
            profile_extractor_config=None,
            user_playbook_extractor_config=None,
            agent_success_config=None,
        )
    )
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        engine = Reflexio(
            org_id=org_id, storage_base_dir=str(tmp_path), configurator=configurator
        )
        monkeypatch.setattr(
            "reflexio.server.routes.provenance.reflexio_cache.get_reflexio",
            lambda **_: engine,
        )
        client = TestClient(
            create_app(get_org_id=lambda: org_id), client=(tmp_path.name, 50000)
        )
        yield Harness(engine, client)


def _contents(body: dict) -> list[str]:
    return [i["content"] for i in body.get("interactions", [])]


@pytest.mark.parametrize("kind", ["user_playbook", "agent_playbook"])
def test_non_integer_playbook_id_is_a_structured_failure(harness, kind):
    body = harness.post(kind, "not-a-number")
    assert body["success"] is False
    assert body["target_kind"] == kind
    assert body["target_id"] == "not-a-number"
    assert body["provenance_status"] == "unavailable"
    assert body["msg"] == f"{kind} id must be an integer"


@pytest.mark.parametrize(
    ("kind", "target_id", "msg"),
    [
        ("profile", "missing-profile", "Profile not found"),
        ("user_playbook", 987654, "User playbook not found"),
        ("agent_playbook", 987654, "Agent playbook not found"),
    ],
)
def test_missing_targets_report_not_found(harness, kind, target_id, msg):
    body = harness.post(kind, target_id)
    assert body["success"] is False
    assert body["target_kind"] == kind
    assert body["target_id"] == str(target_id)
    assert body["provenance_status"] == "unavailable"
    assert body["msg"] == msg


def test_profile_with_source_ids_returns_exact_interactions_in_time_order(harness):
    ids = harness.publish("req-exact", contents=("first", "second", "third"))
    harness.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="p-exact",
                user_id="u1",
                content="prefers terse replies",
                last_modified_timestamp=2_000,
                generated_from_request_id="req-exact",
                source_interaction_ids=[ids[2], ids[0]],
            )
        ],
    )

    body = harness.post("profile", "p-exact")

    assert body["success"] is True
    assert body["provenance_status"] == "exact"
    assert body["trigger_request_id"] == "req-exact"
    assert _contents(body) == ["first", "third"]


def test_profile_without_generation_request_cannot_be_reconstructed(harness):
    harness.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="p-orphan",
                user_id="u1",
                content="no trail",
                last_modified_timestamp=2_000,
                generated_from_request_id="",
            )
        ],
    )

    body = harness.post("profile", "p-orphan")

    assert body["success"] is False
    assert body["msg"] == (
        "Profile has no generation request for provenance reconstruction"
    )


def test_profile_whose_generation_request_is_gone_is_unavailable(harness):
    harness.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="p-gone",
                user_id="u1",
                content="request deleted",
                last_modified_timestamp=2_000,
                generated_from_request_id="req-never-stored",
            )
        ],
    )

    body = harness.post("profile", "p-gone")

    assert body["success"] is False
    assert body["msg"] == "Profile generation request was not found"


def test_profile_reconstructs_window_up_to_trigger_without_extractor_config(harness):
    harness.set_profile_config(None)
    harness.publish("req-early", created_at=1_000, contents=("early",))
    harness.publish("req-trigger", created_at=2_000, contents=("trigger",))
    harness.publish("req-late", created_at=3_000, contents=("after the fact",))
    harness.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="p-window",
                user_id="u1",
                content="reconstructed",
                last_modified_timestamp=4_000,
                generated_from_request_id="req-trigger",
            )
        ],
    )

    body = harness.post("profile", "p-window")

    assert body["success"] is True
    assert body["provenance_status"] == "best_effort"
    assert body["trigger_request_id"] == "req-trigger"
    # The window ends at the trigger's newest interaction: later rows are excluded.
    assert _contents(body) == ["early", "trigger"]


def test_profile_window_honours_the_extractor_source_filter(harness):
    harness.set_profile_config(
        ProfileExtractorConfig(
            extraction_definition_prompt="Extract preferences",
            request_sources_enabled=["chat"],
        )
    )
    harness.publish("req-email", source="email", created_at=1_000, contents=("mail",))
    harness.publish("req-chat", source="chat", created_at=2_000, contents=("chat",))
    harness.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="p-filtered",
                user_id="u1",
                content="from chat only",
                last_modified_timestamp=3_000,
                generated_from_request_id="req-chat",
            )
        ],
    )

    body = harness.post("profile", "p-filtered")

    assert body["provenance_status"] == "best_effort"
    assert _contents(body) == ["chat"]


def _save_user_playbook(harness, **fields) -> int:
    playbook = UserPlaybook(agent_version="v1", content="a rule", **fields)
    harness.storage.save_user_playbooks([playbook])
    assert playbook.user_playbook_id
    return playbook.user_playbook_id


def test_user_playbook_with_source_ids_is_exact(harness):
    ids = harness.publish("req-up", contents=("a", "b"))
    playbook_id = _save_user_playbook(
        harness, user_id="u1", request_id="req-up", source_interaction_ids=[ids[1]]
    )

    body = harness.post("user_playbook", playbook_id)

    assert body["success"] is True
    assert body["provenance_status"] == "exact"
    assert body["trigger_request_id"] == "req-up"
    assert _contents(body) == ["b"]


def test_user_playbook_falls_back_to_its_request_as_best_effort(harness):
    harness.publish("req-fallback", contents=("x", "y"))
    playbook_id = _save_user_playbook(harness, user_id="u1", request_id="req-fallback")

    body = harness.post("user_playbook", playbook_id)

    assert body["provenance_status"] == "best_effort"
    assert _contents(body) == ["x", "y"]


def test_user_playbook_without_source_or_request_is_unavailable(harness):
    playbook_id = _save_user_playbook(harness, user_id="u1", request_id="")

    body = harness.post("user_playbook", playbook_id)

    assert body["success"] is True
    assert body["provenance_status"] == "unavailable"
    assert body["msg"] == "No interactions found for this user playbook"
    assert _contents(body) == []


def _save_agent_playbook(harness) -> int:
    [saved] = harness.storage.save_agent_playbooks(
        [AgentPlaybook(agent_version="v1", content="aggregated rule")]
    )
    assert saved.agent_playbook_id
    return saved.agent_playbook_id


def test_agent_playbook_without_source_windows_is_unavailable(harness):
    agent_id = _save_agent_playbook(harness)

    body = harness.post("agent_playbook", agent_id)

    assert body["success"] is True
    assert body["provenance_status"] == "unavailable"
    assert body["msg"] == (
        "No source user playbooks are recorded for this agent playbook"
    )


def test_agent_playbook_exact_when_every_window_records_its_interactions(harness):
    ids = harness.publish("req-w", contents=("one", "two", "three"))
    first = _save_user_playbook(harness, user_id="u1", request_id="req-w")
    second = _save_user_playbook(harness, user_id="u1", request_id="req-w")
    agent_id = _save_agent_playbook(harness)
    harness.storage.set_source_windows_for_agent_playbook(
        agent_id,
        [
            AgentPlaybookSourceWindow(
                user_playbook_id=first, source_interaction_ids=[ids[0]]
            ),
            AgentPlaybookSourceWindow(
                user_playbook_id=second, source_interaction_ids=[ids[2], ids[1]]
            ),
        ],
    )

    body = harness.post("agent_playbook", agent_id)

    assert body["success"] is True
    assert body["provenance_status"] == "exact"
    groups = {
        g["user_playbook"]["user_playbook_id"]: g for g in body["source_user_playbooks"]
    }
    assert set(groups) == {first, second}
    assert [i["content"] for i in groups[first]["interactions"]] == ["one"]
    assert [i["content"] for i in groups[second]["interactions"]] == ["two", "three"]
    assert groups[second]["source_interaction_ids"] == [ids[2], ids[1]]


def test_agent_playbook_window_falls_back_to_user_playbook_sources(harness):
    ids = harness.publish("req-src", contents=("kept", "ignored"))
    harness.publish("req-other", contents=("from request",))
    via_ids = _save_user_playbook(
        harness, user_id="u1", request_id="req-src", source_interaction_ids=[ids[0]]
    )
    via_request = _save_user_playbook(harness, user_id="u1", request_id="req-other")
    via_nothing = _save_user_playbook(harness, user_id="u1", request_id="")
    agent_id = _save_agent_playbook(harness)
    harness.storage.set_source_windows_for_agent_playbook(
        agent_id,
        [
            AgentPlaybookSourceWindow(user_playbook_id=via_ids),
            AgentPlaybookSourceWindow(user_playbook_id=via_request),
            AgentPlaybookSourceWindow(user_playbook_id=via_nothing),
        ],
    )

    body = harness.post("agent_playbook", agent_id)

    # Request-level fallback and the empty window make the answer best effort.
    assert body["success"] is True
    assert body["provenance_status"] == "best_effort"
    groups = {
        g["user_playbook"]["user_playbook_id"]: g for g in body["source_user_playbooks"]
    }
    assert [i["content"] for i in groups[via_ids]["interactions"]] == ["kept"]
    assert groups[via_ids]["source_interaction_ids"] == [ids[0]]
    assert [i["content"] for i in groups[via_request]["interactions"]] == [
        "from request"
    ]
    assert groups[via_nothing]["interactions"] == []


def test_agent_playbook_with_no_reachable_interactions_is_unavailable(harness):
    playbook_id = _save_user_playbook(harness, user_id="u1", request_id="")
    agent_id = _save_agent_playbook(harness)
    harness.storage.set_source_windows_for_agent_playbook(
        agent_id, [AgentPlaybookSourceWindow(user_playbook_id=playbook_id)]
    )

    body = harness.post("agent_playbook", agent_id)

    assert body["success"] is True
    assert body["provenance_status"] == "unavailable"
    assert body["msg"] == "No source interactions were found"
    assert len(body["source_user_playbooks"]) == 1


def test_agent_playbook_whose_recorded_sources_are_all_missing_fails(harness):
    agent_id = _save_agent_playbook(harness)
    # A dangling linkage row: the source user playbook no longer exists.
    with harness.storage._lock:
        harness.storage.conn.execute(
            """INSERT INTO agent_playbook_source_user_playbooks
               (agent_playbook_id, user_playbook_id, source_interaction_ids)
               VALUES (?, ?, '[]')""",
            (agent_id, 424242),
        )
        harness.storage.conn.commit()

    body = harness.post("agent_playbook", agent_id)

    assert body["success"] is False
    assert body["msg"] == "Recorded source user playbooks were not found"
