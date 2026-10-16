"""The complete extraction request must agree with its actual evidence map."""

from contextlib import nullcontext

import pytest

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.overflow_presentation import (
    present_extraction_chunk,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord
from reflexio.server.services.playbook.playbook_evidence import (
    build_playbook_extraction_contract,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
    construct_playbook_extraction_messages_from_sessions,
)
from tests.server.services.playbook.test_source_quoted_overflow import sources


@pytest.mark.parametrize("prepared_mode", [False, True])
def test_full_reference_request_has_one_citation_convention(prepared_mode):
    groups = sources()
    manager = PromptManager()
    prepared = prepare_extraction_input(groups, user_id="u")
    with (
        present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord())
        if prepared_mode
        else nullcontext()
    ):
        context = build_playbook_prompt_context(groups, label_turns=True)
        contract = build_playbook_extraction_contract(
            manager,
            expert=False,
            evidence_sources=context.evidence_sources,
            evidence_units=context.evidence_units,
        )
        messages = construct_playbook_extraction_messages_from_sessions(
            manager, groups, "Agent", "Supported procedures", prompt_context=context
        )
        text = "\n".join(m["content"] for m in messages)
        schema = contract.schema.model_json_schema()
        description = schema["$defs"]["StructuredReferencedExtractedPlaybookContent"][
            "properties"
        ]["evidence_refs"]["description"]
        assert "such as T1" not in description
        assert "containing at least one provided `[T#]` id" not in text
        assert "Cite the smallest set of provided `[T#]` turns" not in text
        assert "preference" in text and "User" in text
        assert ("exact i<id>:<field>:<start>:<end>" in text) == prepared_mode
        assert ("provided T# IDs" in text) != prepared_mode


@pytest.mark.parametrize("prepared_mode", [False, True])
@pytest.mark.parametrize(
    "expert,strict,adopted",
    [
        (False, True, None),
        (True, False, None),
        (True, True, None),
        (False, True, "quoted"),
        (False, True, "legacy"),
    ],
)
def test_sdk_schema_and_repair_use_selected_contract(
    monkeypatch, prepared_mode, expert, strict, adopted
):
    import json

    import litellm

    from reflexio.models.config_schema import APIKeyConfig
    from reflexio.server.llm.input_budget import measure_request_input_tokens
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredExtractedPlaybookList,
        StructuredPlaybookList,
        construct_expert_playbook_extraction_messages,
    )

    groups = sources()
    if expert:
        groups[0].interactions[
            1
        ].expert_content = "Inspect the receipt before retrying."
    manager = PromptManager()
    prepared = prepare_extraction_input(groups, user_id="u")
    with (
        present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord())
        if prepared_mode
        else nullcontext()
    ):
        context = build_playbook_prompt_context(
            groups, expert=expert, label_turns=strict
        )
        contract = build_playbook_extraction_contract(
            manager,
            expert=expert,
            evidence_sources=context.evidence_sources,
            evidence_units=context.evidence_units,
            strict_override=strict,
        )
        schema = {
            "quoted": StructuredExtractedPlaybookList,
            "legacy": StructuredPlaybookList,
        }.get(adopted, contract.schema)
        instruction = contract.citation_instruction(schema=schema)
        constructor = (
            construct_expert_playbook_extraction_messages
            if expert
            else construct_playbook_extraction_messages_from_sessions
        )
        messages = constructor(
            manager,
            groups,
            "Agent",
            "Supported procedures",
            prompt_context=context,
            citation_instruction=instruction,
        )
    config = LiteLLMConfig(
        model="minimax/MiniMax-M3",
        max_tokens=8192,
        max_retries=0,
        fallback_models=[],
        api_key_config=APIKeyConfig.model_validate(
            {
                "custom_endpoint": {
                    "model": "minimax/MiniMax-M3",
                    "api_base": "https://api.minimax.io/v1",
                    "api_key": "sk-cp-scripted-only",
                }
            }
        ),
    )
    client = LiteLLMClient(config)
    calls = []
    supported = {
        "rationale": "The source requires scoped approval.",
        "trigger": "Before deployment",
        "content": "Preserve the required approval.",
        "evidence_kind": "correction",
    }
    if strict and not expert and adopted is None:
        valid_ref = next(iter(context.evidence_units))
        first = {
            "playbooks": [
                {**supported, "evidence_refs": ["T1" if prepared_mode else "unseen"]}
            ]
        }
        corrected = {"playbooks": [{**supported, "evidence_refs": [valid_ref]}]}
    elif strict and adopted != "legacy":
        turn, source = next(iter(context.evidence_sources.items()))
        first = {
            "playbooks": [
                {
                    **supported,
                    "evidence": [{"turn_ref": turn, "source_span": "not visible"}],
                }
            ]
        }
        corrected = {
            "playbooks": [
                {
                    **supported,
                    "evidence": [
                        {"turn_ref": turn, "source_span": source.evidence_texts[0]}
                    ],
                }
            ]
        }
    else:
        first, corrected = None, {"playbooks": []}

    def transport(params, hard_timeout):
        calls.append(params)
        return litellm.ModelResponse(
            id="scripted",
            created=0,
            model=params["model"],
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "not-json"
                        if len(calls) == 1 and first is None
                        else json.dumps(first if len(calls) == 1 else corrected),
                    },
                }
            ],
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    monkeypatch.setattr(
        litellm, "completion", lambda **_kwargs: pytest.fail("No provider dispatch")
    )
    result = client.generate_chat_response_with_provenance(
        messages=messages,
        response_format=schema,
        structured_output_validator=contract.validator if adopted != "legacy" else None,
    )
    assert isinstance(result.value, schema)
    assert len(calls) == 2
    for call in calls:
        text = json.dumps(call["messages"])
        assert call["model"] == "minimax/MiniMax-M3" and call["max_tokens"] == 8192
        assert measure_request_input_tokens(call) <= 32768
        if instruction:
            assert sum(m.get("content") == instruction for m in call["messages"]) == 1
        assert "MUST include exactly non-empty" not in text
        assert "do not reproduce" not in text
        assert '"evidence_refs" is REQUIRED' not in text
        if not expert and adopted is None:
            assert "Prompt-local turn ids such as T1" not in text
            assert "containing at least one provided `[T#]` id" not in text
            assert ("exact i<id>:<field>:<start>:<end>" in text) == prepared_mode
        elif strict and adopted != "legacy":
            assert "For evidence, use the provided turn_ref" in text
            assert "For evidence_refs, use exact" not in text
        else:
            assert "For evidence_refs," not in text
            assert "For evidence, use" not in text


def test_pinned_legacy_contract_and_adopted_schema_keep_their_shapes():
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )

    manager = PromptManager(version_override={"playbook_extraction_context": "4.5.0"})
    contract = build_playbook_extraction_contract(
        manager, expert=False, evidence_sources={}
    )
    assert not contract.strict and contract.schema is StructuredPlaybookList
    assert contract.citation_instruction() == ""
    prepared = prepare_extraction_input(sources(), user_id="u")
    with present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord()):
        modern = build_playbook_extraction_contract(
            PromptManager(), expert=False, evidence_sources={}
        )
        legacy = modern.citation_instruction(schema=StructuredPlaybookList)
        assert "2048 tokens" in legacy
        assert "For evidence_refs," not in legacy
        assert "For evidence, use" not in legacy


def test_prepared_role_and_identity_validation_remains_strict():
    from reflexio.server.services.playbook.playbook_evidence import (
        candidate_rejection_reason,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredReferencedExtractedPlaybookContent,
    )

    groups = sources()
    prepared = prepare_extraction_input(groups, user_id="u")
    with present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord()):
        context = build_playbook_prompt_context(groups, label_turns=True)
    refs = list(context.evidence_units)
    candidate = StructuredReferencedExtractedPlaybookContent(
        rationale="The User corrected deployment behavior.",
        trigger="Before deployment",
        content="Preserve the required approval.",
        evidence_kind="preference",
        evidence_refs=refs,
    )
    assert (
        candidate_rejection_reason(
            candidate, context.evidence_sources, context.evidence_units
        )
        == "preference_without_direct_user_evidence"
    )
    correction = candidate.model_copy(update={"evidence_kind": "correction"})
    assert (
        candidate_rejection_reason(
            correction, context.evidence_sources, context.evidence_units
        )
        is None
    )
    for invalid in ["T1", "i1:content:0:999999", "foreign-project:source"]:
        wrong = correction.model_copy(update={"evidence_refs": [invalid]})
        assert (
            candidate_rejection_reason(
                wrong, context.evidence_sources, context.evidence_units
            )
            == "unknown_evidence_ref"
        )
    user_only = candidate.model_copy(update={"evidence_refs": [refs[0]]})
    assert (
        candidate_rejection_reason(
            user_only, context.evidence_sources, context.evidence_units
        )
        is None
    )
