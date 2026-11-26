"""Previous exact native readers hold new retained work before models or writes.

The fixture contains public source functions frozen from both the selected
enterprise SDK pin and initial remote-main SDK, not an alternate runtime engine.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from reflexio.server.env_utils import env_bool
from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.services.playbook.procedural.native import PREPARATION_VERSION
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
    validate_extraction_overflow_manifest,
)


def previous(commit, group, **namespace):
    path = (
        Path(__file__).parents[3] / "fixtures/procedural/previous_native_readers.json"
    )
    record = json.loads(path.read_text())[commit][group]
    for name, source in record["functions"].items():
        assert (
            hashlib.sha256(source.encode()).hexdigest()
            == record["function_sha256"][name]
        )
    source = "from __future__ import annotations\n" + "\n\n".join(
        record["functions"].values()
    )
    # Frozen, digest-checked test source from our own previous public releases.
    exec(compile(source, f"{commit}:{record['source_path']}", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("commit", ["88accf44", "f850e9a2"])
@pytest.mark.parametrize("cached", [False, True])
def test_old_window_consumer_refuses_v5_before_models_or_cached_publication(
    monkeypatch, commit, cached
):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    original_prepare = Mock(
        side_effect=AssertionError("Old source preparation cannot run")
    )
    ns = previous(
        commit,
        "WindowExecutor",
        env_bool=env_bool,
        ExtractionInputIntegrityError=ExtractionInputIntegrityError,
        OverflowCheckpointConflictError=OverflowCheckpointConflictError,
        validate_extraction_overflow_manifest=validate_extraction_overflow_manifest,
        prepare_extraction_input=original_prepare,
        prepare_initial_extraction_input=original_prepare,
    )
    manifest = {
        "version": 1,
        "revision": 0,
        "source_digest": "a" * 64,
        "attempt_limit": 128,
        "attempts": [],
        "preparation_version": PREPARATION_VERSION,
        "input_chunk_tokens": 10000,
        "checkpoint": {"stage": "prepare", "completed_steps": {}},
    }
    run = SimpleNamespace(generation_request_snapshot={"overflow": manifest})
    storage, client = Mock(), Mock()
    storage.get_agent_run.return_value = run
    owner = SimpleNamespace(
        storage=storage,
        client=client,
        _overflow_manifest=ns["_overflow_manifest"],
        _requires_compatibility_execution=ns["_requires_compatibility_execution"],
        _prepare_coordinator_sources=ns["_prepare_coordinator_sources"],
    )
    window = SimpleNamespace(
        window_id="v5-window", user_id="u", outcome={} if cached else None
    )
    service = SimpleNamespace(service_config=SimpleNamespace(window_interactions=[]))
    with pytest.raises(
        ExtractionInputIntegrityError, match="Unknown extraction preparation version"
    ):
        ns["_coordinator"](owner, window, "token", service, None, None)
    original_prepare.assert_not_called()
    assert client.mock_calls == []
    assert [call[0] for call in storage.mock_calls] == ["get_agent_run"]


@pytest.mark.parametrize("commit", ["88accf44", "f850e9a2"])
def test_old_cached_inline_preparation_refuses_version2_before_decode_or_apply(commit):
    decode = Mock(side_effect=AssertionError("Old plan decoder must not run"))
    ns = previous(
        commit,
        "SavedPublicationPreparation",
        ExtractionInputIntegrityError=ExtractionInputIntegrityError,
        decode_write_plan=decode,
    )
    run = SimpleNamespace(committed_output={"publication": {"id": "output"}})
    payload = dict.fromkeys(
        {
            "version",
            "output_id",
            "output_digest",
            "kind",
            "write_plan",
            "history_inputs",
            "bookmark",
        }
    )
    payload["version"] = 2
    with pytest.raises(ExtractionInputIntegrityError, match="binding changed"):
        ns["from_payload"](object, run, payload)
    decode.assert_not_called()


@pytest.mark.parametrize("commit", ["88accf44", "f850e9a2"])
def test_old_unprepared_inline_output_cannot_validate_as_old_candidates(commit):
    from reflexio.models.config_schema import PlaybookConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.playbook.playbook_evidence import (
        build_playbook_extraction_contract,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredExtractedPlaybookList,
        StructuredPlaybookList,
        StructuredReferencedExtractedPlaybookList,
        build_playbook_prompt_context,
        has_expert_content,
        uses_evidence_grounded_extraction,
    )
    from reflexio.server.services.service_utils import (
        extract_interactions_from_request_interaction_data_models,
    )
    from tests.server.services.playbook.test_source_quoted_overflow import sources

    extractor = Mock(side_effect=AssertionError("Old extractor must not run"))
    ns = previous(
        commit,
        "ExtractionResumeWorker",
        PlaybookConfig=PlaybookConfig,
        ResumeWorkerError=RuntimeError,
        has_expert_content=has_expert_content,
        extract_interactions_from_request_interaction_data_models=extract_interactions_from_request_interaction_data_models,
        StructuredExtractedPlaybookList=StructuredExtractedPlaybookList,
        StructuredPlaybookList=StructuredPlaybookList,
        StructuredReferencedExtractedPlaybookList=StructuredReferencedExtractedPlaybookList,
        uses_evidence_grounded_extraction=uses_evidence_grounded_extraction,
        build_playbook_prompt_context=build_playbook_prompt_context,
        build_playbook_extraction_contract=build_playbook_extraction_contract,
        PlaybookExtractor=extractor,
    )
    owner = SimpleNamespace(
        request_context=SimpleNamespace(prompt_manager=PromptManager())
    )
    run = SimpleNamespace(
        binding=SimpleNamespace(user_id="u"),
        generation_request_snapshot={"output_schema_name": "ProceduralOutput"},
    )
    output = {
        "version": 1,
        "source_digest": "a" * 64,
        "source_coverage": ["a" * 64],
        "historical_playbooks": {},
        "receipt": {},
        "write_plan": None,
        "plan_sha256": "b" * 64,
    }
    with pytest.raises(ValidationError):
        ns["_playbook_items_from_output"](
            owner,
            run,
            PlaybookConfig(extraction_definition_prompt="Learn useful procedures."),
            sources(),
            output,
        )
    extractor.assert_not_called()
