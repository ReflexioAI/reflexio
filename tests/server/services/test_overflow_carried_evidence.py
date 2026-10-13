"""Final extraction may reuse rendered proof, never global unshown source text."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm._litellm_types import (
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.extraction.input_preparation import (
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_pipeline import (
    OverflowExtractionCall,
    _carried_evidence_errors,
    extract_overflow_window,
)
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    protected_quote_view,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedEvidence,
    ProtectedRecord,
)
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    encode_committed_output,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)


def profiles(quote):
    return StructuredProfilesOutput.model_validate(
        {
            "profiles": [
                {
                    "content": "A supported preference",
                    "time_to_live": "infinity",
                    "source_span": quote,
                }
            ]
        }
    )


def test_profile_final_quote_must_remain_one_carried_subquotation():
    raw = [
        {"source_span": "first visible quote"},
        {"source_span": "second visible quote"},
    ]
    for quote, expected in (
        ("visible quote", False),
        ("hidden remainder", True),
        ("quote second", True),
    ):
        errors = _carried_evidence_errors(
            profiles(quote),
            field="profiles",
            candidates=raw,
            protected=ProtectedRecord(),
            protected_spans=(),
        )
        assert bool(errors) is expected


def test_playbook_carried_refs_and_quotes_keep_original_turn_and_typography():
    raw = [
        {
            "evidence_refs": ["i1:content:0:5"],
            "evidence": [
                {
                    "turn_ref": "T1",
                    "source_span": "User says “draft only”.",
                }
            ],
        }
    ]
    for ref, turn, quote, expected in (
        ("i1:content:0:5", "T1", 'User says "draft only".', False),
        ("i1:content:5:10", "T1", 'User says "draft only".', True),
        ("i1:content:0:5", "T2", 'User says "draft only".', True),
    ):
        output = StructuredPlaybookList.model_validate(
            {
                "playbooks": [
                    {
                        "content": "Prepare a draft",
                        "evidence_refs": [ref],
                        "evidence": [{"turn_ref": turn, "source_span": quote}],
                    }
                ]
            }
        )
        assert (
            bool(
                _carried_evidence_errors(
                    output,
                    field="playbooks",
                    candidates=raw,
                    protected=ProtectedRecord(),
                    protected_spans=(),
                )
            )
            is expected
        )


def test_protected_proof_grants_only_the_rendered_physical_quote():
    text = "shown quote hidden material!"
    span = SourceSpan(1, "request", "user", "content", 0, len(text), text, 1)
    prepared = PreparedExtractionInput("source", (PreparedChunk(0, (span,)),))
    protected, visible = protected_quote_view(
        prepared,
        ProtectedRecord(
            evidence=(
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(span.evidence_ref,),
                    quote="shown quote",
                ),
            )
        ),
    )
    assert (
        visible[0].text == "shown quote"
        and visible[0].start == 0
        and visible[0].end == 11
    )
    assert not _carried_evidence_errors(
        profiles("shown quote"),
        field="profiles",
        candidates=[],
        protected=protected,
        protected_spans=visible,
    )
    assert _carried_evidence_errors(
        profiles("hidden material"),
        field="profiles",
        candidates=[],
        protected=protected,
        protected_spans=visible,
    )


def coordinator_fixture():
    first_text = "seen quotation. hidden qualification."
    second_text = "second visible passage"
    prepared = PreparedExtractionInput(
        "source",
        (
            PreparedChunk(
                0,
                (
                    SourceSpan(
                        1, "r1", "user", "content", 0, len(first_text), first_text, 1
                    ),
                ),
            ),
            PreparedChunk(
                1,
                (
                    SourceSpan(
                        2, "r2", "user", "content", 0, len(second_text), second_text, 2
                    ),
                ),
            ),
        ),
    )
    coordinator = MagicMock(spec=OverflowCoordinator)
    coordinator.source_digest = "source"
    coordinator.storage = MagicMock()
    coordinator.saved_run = None
    coordinator.record = ProtectedRecord()
    coordinator.window = SimpleNamespace(
        kind="profile", window_id="window", user_id="user"
    )
    coordinator.token = "token"
    coordinator.require_prepared.return_value = prepared
    coordinator.candidate_key = OverflowCoordinator.candidate_key
    coordinator.step_key = lambda stage, value: OverflowCoordinator.step_key(
        coordinator, stage, value
    )
    coordinator.checkpoint = {"completed_steps": {}}
    coordinator.save_checkpoint.side_effect = lambda value: setattr(
        coordinator, "checkpoint", value
    )
    coordinator.completed_step.side_effect = lambda key: coordinator.checkpoint.get(
        "completed_steps", {}
    ).get(key)

    def complete(key, output, provenance):
        checkpoint = coordinator.checkpoint.copy()
        checkpoint.setdefault("completed_steps", {})[key] = {
            "output": output,
            "model_provenance": {"model_name": None, "provider": None},
        }
        coordinator.save_checkpoint(checkpoint)

    coordinator.complete_step.side_effect = complete

    def build():
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        return OverflowExtractionCall(
            [{"role": "user", "content": presentation.text}],
            StructuredProfilesOutput,
        )

    def execute(call):
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        output = (
            profiles("seen quotation")
            if presentation.chunk.index == 0
            else StructuredProfilesOutput(profiles=[])
        )
        key = f"child-{presentation.chunk.index}"
        checkpoint = coordinator.checkpoint
        checkpoint.setdefault("extraction_children", {})[
            coordinator.candidate_key(presentation.chunk.text)
        ] = {
            "key": key,
            "policy": checkpoint["extraction_policy"],
        }
        complete(
            key,
            {"output": output.model_dump(mode="json"), "trace": {}},
            ModelProvenance(),
        )
        return AgentRunResult(
            "window:window",
            output,
            [],
            [],
            ToolLoopTrace(finished=True),
            "structured_output",
        )

    return coordinator, build, execute


def test_reconciliation_uses_actual_bounded_repair_for_unshown_evidence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-provider")
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    coordinator, build, execute = coordinator_fixture()
    calls = []

    def respond(params, hard_timeout):
        calls.append(params)
        output = (
            profiles("hidden qualification")
            if len(calls) == 1
            else profiles("seen quotation")
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=output.model_dump_json(), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="gpt-4o",
            _hidden_params={"custom_llm_provider": "openai"},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", respond)
    result = extract_overflow_window(
        coordinator=coordinator,
        client=client,
        build_call=build,
        execute_call=execute,
        final_validator=None,
    )
    assert result.output == profiles("seen quotation")
    assert len(calls) == 2 and calls[0]["model"] == calls[1]["model"]
    assert [span.text for span in coordinator.carried_source_spans] == [
        "seen quotation"
    ]


@pytest.mark.parametrize("parent_recovery", [False, True])
def test_cached_reconciliation_and_parent_output_cannot_introduce_unshown_proof(
    parent_recovery,
):
    coordinator, build, execute = coordinator_fixture()
    if parent_recovery:
        client = MagicMock()
        client.generate_chat_response_with_provenance.return_value = SimpleNamespace(
            value=profiles("seen quotation"),
            provenance=ModelProvenance(),
        )
        extract_overflow_window(
            coordinator=coordinator,
            client=client,
            build_call=build,
            execute_call=execute,
            final_validator=None,
        )
        coordinator.saved_run = SimpleNamespace(
            id="window:window",
            pending_tool_call_ids=[],
            generation_request_snapshot={
                "output_schema_name": "StructuredProfilesOutput"
            },
            committed_output=encode_committed_output(
                profiles("hidden qualification"), None
            ),
        )
    else:
        complete = coordinator.completed_step.side_effect
        coordinator.completed_step.side_effect = lambda key: (
            complete(key)
            or {
                "output": profiles("hidden qualification").model_dump(mode="json"),
                "model_provenance": {"model_name": None, "provider": None},
            }
        )
    coordinator.storage.update_agent_run_status.reset_mock()
    with pytest.raises(
        ProviderRequestGuardError, match="carried evidence|original evidence"
    ):
        extract_overflow_window(
            coordinator=coordinator,
            client=MagicMock(),
            build_call=build,
            execute_call=execute,
            final_validator=None,
        )
    coordinator.storage.update_agent_run_status.assert_not_called()
