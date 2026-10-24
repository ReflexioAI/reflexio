"""``service_config_snapshot`` never persists copied user content.

The snapshot is audit metadata only (its sole reader is recovery's
``payload_fingerprint``, which hashes whatever a row stores). Window
interactions, every existing profile of the user and a third copy of the
extractor config used to be written into it on every run; the builder now
drops them for every run shape, and these tests fail if any path brings one
back.
"""

from __future__ import annotations

import ast
import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
from reflexio.server.services.extraction import resumable_agent
from reflexio.server.services.extraction.agent_run_records import (
    UNPERSISTED_SERVICE_CONFIG_FIELDS,
    build_extractor_agent_run_record,
    service_config_snapshot,
)
from reflexio.server.services.playbook.service import PlaybookGenerationServiceConfig
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.service import ProfileGenerationServiceConfig
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.usage_metrics import (
    configure_usage_event_recorder,
    exempt_usage_event_recorder,
)

EXCLUDED = {"window_interactions", "existing_data", "frozen_extractor_config"}
PRIVATE = "PRIVATE-USER-CONTENT"


def _groups() -> list[RequestInteractionDataModel]:
    return [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s", source="api"),
            interactions=[
                Interaction(
                    interaction_id=1,
                    user_id="u",
                    request_id="r",
                    role="user",
                    content=PRIVATE,
                )
            ],
        )
    ]


def _profile_config() -> ProfileGenerationServiceConfig:
    return ProfileGenerationServiceConfig(
        user_id="u",
        request_id="r",
        source="api",
        existing_data=[{"content": PRIVATE}],
        window_interactions=_groups(),
        extraction_window_id="w",
        frozen_extractor_config={"extraction_definition_prompt": PRIVATE},
    )


def _playbook_config() -> PlaybookGenerationServiceConfig:
    return PlaybookGenerationServiceConfig(
        request_id="r",
        agent_version="v1",
        user_id="u",
        source="api",
        window_interactions=_groups(),
        extraction_window_id="w",
        frozen_extractor_config={"extraction_definition_prompt": PRIVATE},
    )


class _ModelConfig(BaseModel):
    request_id: str = "r"
    existing_data: list[str] = [PRIVATE]
    window_interactions: list[str] = [PRIVATE]
    frozen_extractor_config: dict[str, str] = {"prompt": PRIVATE}


@pytest.mark.parametrize(
    "config",
    [
        _profile_config(),
        _playbook_config(),
        {"request_id": "r", **dict.fromkeys(EXCLUDED, PRIVATE)},
        _ModelConfig(),
    ],
    ids=["profile", "playbook", "dict", "model"],
)
def test_builder_output_never_contains_unpersisted_fields(config):
    snapshot = service_config_snapshot(config)
    assert isinstance(snapshot, dict)
    assert EXCLUDED.isdisjoint(snapshot)
    assert PRIVATE not in repr(snapshot)
    # The remaining runtime config is still recorded for audit.
    assert snapshot["request_id"] == "r"


def test_unpersisted_fields_are_exactly_the_reviewed_set():
    """A rename of a config field must not silently disable the exclusion."""
    assert UNPERSISTED_SERVICE_CONFIG_FIELDS == EXCLUDED
    profile = {field.name for field in fields(ProfileGenerationServiceConfig)}
    playbook = {field.name for field in fields(PlaybookGenerationServiceConfig)}
    assert profile >= EXCLUDED
    assert {"window_interactions", "frozen_extractor_config"} <= playbook


@pytest.mark.parametrize("config", [_profile_config(), _playbook_config()])
def test_built_record_snapshot_excludes_content(config):
    run = build_extractor_agent_run_record(
        org_id="org",
        extractor_kind="profile",
        user_id="u",
        agent_version=None,
        source="api",
        request_interaction_data_models=_groups(),
        extractor_config=ProfileExtractorConfig(extraction_definition_prompt="x"),
        service_config=config,
        agent_context="",
        generation_request_id="r",
    )
    assert run.service_config_snapshot is not None
    assert EXCLUDED.isdisjoint(run.service_config_snapshot)
    # The binding witness resume checks stays in the generation snapshot.
    assert (
        run.generation_request_snapshot["extractor_config"][
            "extraction_definition_prompt"
        ]
        == "x"
    )


@pytest.mark.parametrize("window_id", [None, "w"], ids=["ad_hoc", "window"])
def test_classic_run_row_persists_trimmed_snapshot(monkeypatch, window_id):
    """Classic runs (ad-hoc and ordinary window) write the trimmed snapshot."""
    output = StructuredProfilesOutput(profiles=[])
    monkeypatch.setattr(
        resumable_agent,
        "run_tool_loop",
        MagicMock(
            return_value=ToolLoopResult(
                ctx=None,
                trace=ToolLoopTrace(finished=True),
                finished_reason="structured_output",
                messages=[],
                structured_output=output,
                max_steps_remaining=2,
            )
        ),
    )
    monkeypatch.setattr(resumable_agent, "pending_tool_calls_enabled", lambda _: False)
    config = _profile_config()
    config.extraction_window_id = window_id
    configure_usage_event_recorder(exempt_usage_event_recorder)
    try:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
        ):
            storage = SQLiteStorage(org_id="org", db_path=f"{temp_dir}/r.db")
            context = MagicMock(org_id="org", storage=storage)
            result = resumable_agent.run_resumable_extraction_agent(
                request_context=context,
                client=MagicMock(),
                extractor_kind="profile",
                user_id="u",
                request_id="r",
                agent_version=None,
                source="api",
                request_interaction_data_models=_groups(),
                extractor_config=ProfileExtractorConfig(
                    extraction_definition_prompt="x"
                ),
                service_config=config,
                agent_context="",
                messages=[{"role": "user", "content": "source"}],
                output_schema=StructuredProfilesOutput,
                log_label="classic",
            )
            stored = storage.get_agent_run(result.run_id)
            assert stored is not None
            assert stored.id == (f"window:{window_id}" if window_id else result.run_id)
            assert stored.service_config_snapshot is not None
            assert EXCLUDED.isdisjoint(stored.service_config_snapshot)
            assert PRIVATE not in repr(stored.service_config_snapshot)
            storage.conn.close()
    finally:
        configure_usage_event_recorder(None)


# Files allowed to set ``service_config_snapshot`` on a record: the builder,
# and the storage row decoders that read back what a row already stores.
_SNAPSHOT_WRITERS = {
    "reflexio/server/services/extraction/agent_run_records.py",
    "reflexio/server/services/storage/sqlite_storage/_agent_run.py",
}


def _snapshot_writes(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            lines.extend(
                node.lineno
                for keyword in node.keywords
                if keyword.arg == "service_config_snapshot"
            )
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"update", "setdefault", "__setitem__"}
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "service_config_snapshot"
            ):
                lines.append(node.lineno)
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            while isinstance(target, ast.Subscript):
                target = target.value
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "service_config_snapshot"
            ):
                lines.append(target.lineno)
    return lines


def test_only_the_builder_sets_service_config_snapshot():
    """A new path cannot write a snapshot that bypasses the builder's exclusion."""
    root = Path(__file__).resolve().parents[4]
    package = root / "reflexio"
    found = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        lines = _snapshot_writes(ast.parse(path.read_text(), filename=relative))
        if lines:
            found[relative] = lines
    assert set(found) == _SNAPSHOT_WRITERS, found


def test_snapshot_write_scan_detects_bypasses():
    """The scan above must recognise each bypass shape it exists to catch."""
    source = (
        "replace(run, service_config_snapshot={})\n"
        "run.service_config_snapshot = {}\n"
        "run.service_config_snapshot['existing_data'] = 1\n"
        "run.service_config_snapshot.update(existing_data=1)\n"
    )
    assert sorted(_snapshot_writes(ast.parse(source))) == [1, 2, 3, 4]
