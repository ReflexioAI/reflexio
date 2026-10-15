"""Zero-output CLI regeneration through real HTTP routes, SDK polling and SQLite."""

import json
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from reflexio import ReflexioClient
from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import (
    Request,
    Status,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["profiles", "playbooks"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_completed_zero_output_preserves_rows_through_cli(
    tmp_path, monkeypatch, runner, app, request, kind, json_mode
):
    org = "zero-output"
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_: [0.1] * 512)
    monkeypatch.setattr(SQLiteStorage, "_should_expand_documents", lambda _: False)
    # Control only model-backed per-user computation. Admission, batch state,
    # SQLite counts, HTTP submission and SDK attempt polling all remain real.
    monkeypatch.setattr(ProfileGenerationService, "run", lambda *_: None)
    monkeypatch.setattr(PlaybookGenerationService, "run", lambda *_: None)
    configurator = DefaultConfigurator(org_id=org, base_dir=str(tmp_path))
    configurator.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "data.db")),
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract preferences"
            ),
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Extract instructions"
            ),
        )
    )
    engine = Reflexio(
        org_id=org, storage_base_dir=str(tmp_path), configurator=configurator
    )
    storage = engine.get_storage()
    assert isinstance(storage, SQLiteStorage)
    request.addfinalizer(storage.conn.close)
    storage.add_request(
        Request(
            request_id="source", user_id="alice", session_id="s", agent_version="v1"
        )
    )
    for index, status in enumerate((None, Status.ARCHIVED, Status.PENDING)):
        user = "bob" if status == Status.PENDING else "alice"
        storage.add_user_profile(
            user,
            [
                UserProfile(
                    user_id=user,
                    profile_id=f"p{index}",
                    content="Existing evidence",
                    generated_from_request_id="source",
                    last_modified_timestamp=1,
                    status=status,
                )
            ],
            skip_embedding=True,
        )
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    user_id=user,
                    agent_version="v2" if status == Status.PENDING else "v1",
                    request_id="source",
                    content="Existing instruction",
                    status=status,
                )
            ],
            skip_embedding=True,
        )

    def rows():
        return {
            "profiles": [
                tuple(row)
                for row in storage._fetchall(
                    "SELECT profile_id, content, status FROM profiles ORDER BY profile_id LIMIT 10"
                )
            ],
            "playbooks": [
                tuple(row)
                for row in storage._fetchall(
                    "SELECT user_playbook_id, content, status FROM user_playbooks ORDER BY user_playbook_id LIMIT 10"
                )
            ],
        }

    before = rows()
    monkeypatch.setattr(
        "reflexio.server.cache.reflexio_cache.get_reflexio", lambda **_: engine
    )
    with TestClient(
        create_app(get_org_id=lambda: org), client=(tmp_path.name, 50000)
    ) as http:
        sdk = ReflexioClient(url_endpoint="http://testserver")
        sdk.session.close()
        monkeypatch.setattr(sdk, "session", http)
        monkeypatch.setattr("reflexio.cli.commands.profiles.get_client", lambda _: sdk)
        monkeypatch.setattr(
            "reflexio.cli.commands.user_playbooks.get_client", lambda _: sdk
        )
        command = (
            ["user-profiles", "regenerate", "--user-id", "alice"]
            if kind == "profiles"
            else ["user-playbooks", "regenerate", "--agent-version", "v1"]
        )
        result = runner.invoke(
            app, (["--json"] if json_mode else []) + command + ["--wait"]
        )
        assert result.exit_code == 0, result.output
        if json_mode:
            output = json.loads(result.output)
            assert output["ok"] is True
            operation_id = output["data"]["operation_id"]
        else:
            assert "complete (0" in result.output
            operation_id = None
        service = "profile_generation" if kind == "profiles" else "playbook_generation"
        state = sdk._make_request(
            "GET", "/api/get_operation_status", params={"service_name": service}
        )["operation_status"]
        assert state["status"] == "completed"
        assert state["processed_users"] == 1 and state["failed_users"] == 0
        assert state["stats"]["total_generated"] == 0
        actual_id = state["request_params"]["operation_id"]
        assert str(UUID(actual_id)) == actual_id
        if operation_id:
            assert operation_id == actual_id
        assert rows() == before
