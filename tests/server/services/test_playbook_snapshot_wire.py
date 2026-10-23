"""Pin the deployed v1 job layout independently from public configuration."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.domain.entities import PublishUserInteractionRequest
from reflexio.models.config_schema import Config, UserPlaybookExtractorConfig
from reflexio.server.services.durable_learning.admission import extraction_admission
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
    snapshot_extractor_config,
)
from reflexio.server.services.storage.storage_base import build_scope_hash

pytestmark = pytest.mark.unit
V1 = json.loads(
    (Path(__file__).parents[2] / "fixtures/playbook_snapshot_v1.json").read_text()
)


def test_new_admission_and_parent_keep_the_deployed_v1_job_layout() -> None:
    config = UserPlaybookExtractorConfig(extraction_definition_prompt="Extract rules")
    assert snapshot_extractor_config(config) == V1
    request = PublishUserInteractionRequest.model_validate(
        {
            "user_id": "user",
            "session_id": "session",
            "interaction_data_list": [{"role": "user", "content": "Hello"}],
        }
    )
    admission = extraction_admission(
        Config(storage_config=None, user_playbook_extractor_config=config), request
    )
    assert admission["playbook"]["extractor"] == V1
    run = build_extractor_agent_run_record(
        org_id="org",
        extractor_kind="playbook",
        user_id="user",
        agent_version="v1",
        source="api",
        request_interaction_data_models=[],
        extractor_config=config,
        service_config={},
        agent_context="",
        generation_request_id="request",
    )
    assert run.generation_request_snapshot["extractor_config"] == V1
    assert run.binding.extractor_config_hash == build_scope_hash(V1)
    assert "context_prompt" not in config.model_dump()


@pytest.mark.parametrize("context", [None, "Historical inert context"])
def test_historical_parent_keeps_its_witnessed_snapshot_value(context) -> None:
    original = {**deepcopy(V1), "context_prompt": context}
    before = deepcopy(original)
    config = UserPlaybookExtractorConfig.model_validate(original)
    run = build_extractor_agent_run_record(
        org_id="org",
        extractor_kind="playbook",
        user_id="user",
        agent_version="v1",
        source="api",
        request_interaction_data_models=[],
        extractor_config=config,
        service_config=SimpleNamespace(frozen_extractor_config=original),
        agent_context="",
        generation_request_id="request",
    )
    assert run.generation_request_snapshot["extractor_config"] == original
    assert run.binding.extractor_config_hash == build_scope_hash(original)
    assert original == before
