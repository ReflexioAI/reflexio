"""Provider dispatch authority and server-owned excerpt fidelity."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm import ProviderRequestGuardError
from reflexio.server.llm.llm_utils import make_strict_json_schema
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _canonical_user_message,
)
from reflexio.server.services.offline_tuner.open_world.compaction import (
    SessionCompactor,
    _SelectedCompaction,
)
from reflexio.server.services.offline_tuner.open_world.identity import canonical_payload
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceDocument,
    EvidenceSpan,
)


def document(text: str = "Do not use UTC. Continue in local time.") -> EvidenceDocument:
    return EvidenceDocument(
        "discovery",
        "user",
        "session",
        "a" * 64,
        (
            EvidenceSpan(
                "span",
                "133",
                "discovery",
                "user",
                "session",
                "request",
                1,
                "requests[0].interactions[0].content",
                text,
                "user",
            ),
        ),
        "failure",
    )


def client(callback=None) -> Any:
    import json

    fake = SimpleNamespace(config=SimpleNamespace(model="minimax/MiniMax-M2.7"))

    def generate(messages, **kwargs):
        if callback:
            callback(kwargs)
        kwargs["provider_request_guard"](
            {
                "model": fake.config.model,
                "messages": messages,
                "extra_body": kwargs.get("extra_body", {}),
            }
        )
        payload = json.loads(messages[0]["content"])
        assert "outcome" not in payload and "playbook" not in payload
        assert "manifest_role" not in payload
        value = _SelectedCompaction.model_validate(
            {
                f"p{index:02d}": {
                    "category": "constraint",
                    "detail": "Keep the user constraint",
                    "evidence_id": payload["source_records"][0]["evidence_id"],
                }
                if index == 1
                else None
                for index in range(1, 25)
            }
        )
        assert not kwargs["structured_output_validator"](value)
        return value

    fake.generate_chat_response = Mock(side_effect=generate)
    return fake


@pytest.mark.parametrize("model", ["minimax/MiniMax-M3", "zai/glm-5.1"])
def test_direct_answer_mode_is_preserved_and_guarded(model: str) -> None:
    adapter = client()
    adapter.config.model = model
    SessionCompactor(adapter, project_id="project").compact(
        document(),
        through_request_id=None,
        governance_epochs=(),
        before_call=lambda: None,
    )
    kwargs = adapter.generate_chat_response.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["max_tokens"] == 16_384
    with pytest.raises(ProviderRequestGuardError, match="thinking mode changed"):
        kwargs["provider_request_guard"](
            {
                "model": adapter.config.model,
                "messages": [],
                "extra_body": {"thinking": {"type": "adaptive"}},
            }
        )


def test_provider_prompt_schema_preserves_output_bounds_in_descriptions() -> None:
    schema = make_strict_json_schema(_SelectedCompaction.model_json_schema())
    # Provider-safe normalization drops maxItems/maxLength. The descriptions
    # must still teach a prompt-JSON provider the locally enforced limits.
    assert len(schema["properties"]) == 24
    assert len(schema["required"]) == 24
    assert (
        "512"
        in schema["$defs"]["_SelectedPoint"]["properties"]["detail"]["description"]
    )


def test_extra_slot_is_rejected_instead_of_truncating_evidence() -> None:
    from pydantic import ValidationError

    proposal = {f"p{index:02d}": None for index in range(1, 26)}
    with pytest.raises(ValidationError) as error:
        _SelectedCompaction.model_validate(proposal)
    assert any(
        e["loc"] == ("p25",) and e["type"] == "extra_forbidden"
        for e in error.value.errors()
    )


def test_server_attaches_original_excerpt_and_summary_is_separate() -> None:
    original = document()
    adapter = client()
    result = SessionCompactor(adapter, project_id="project").compact(
        original,
        through_request_id="request",
        governance_epochs=(("subject", 0),),
        before_call=lambda: None,
    )
    assert result.spans[0].immutable_text == original.spans[0].immutable_text
    assert result.compaction_summary is not None
    assert result.session_outcome == "failure"
    assert (
        count_input_tokens(
            _canonical_user_message({"document": canonical_payload(result)})
        )
        < 32000
    )
    assert original.compaction_summary is None


def test_provider_retry_reauthorizes_after_revocation() -> None:
    authorized = True
    dispatches = 0

    def authority():
        if not authorized:
            raise ProviderRequestGuardError("subject erased")

    def retry(kwargs):
        nonlocal authorized, dispatches
        kwargs["provider_request_guard"](
            {"model": "minimax/MiniMax-M2.7", "messages": []}
        )
        dispatches += 1
        authorized = False
        kwargs["provider_request_guard"](
            {"model": "minimax/MiniMax-M2.7", "messages": []}
        )
        dispatches += 1

    with pytest.raises(ProviderRequestGuardError, match="erased"):
        SessionCompactor(client(retry), project_id="project").compact(
            document(),
            through_request_id="request",
            governance_epochs=(),
            before_call=authority,
        )
    assert dispatches == 1


def test_complete_receipt_replays_without_model_call_but_checks_authority() -> None:
    fake = client()
    receipts = []
    compactor = SessionCompactor(fake, project_id="project")
    result = compactor.compact(
        document(),
        through_request_id="request",
        governance_epochs=(),
        before_call=lambda: None,
        checkpoint=lambda identity, batches, value: receipts.append(
            {
                "identity": identity,
                "batches": batches,
                "complete": batches == -1,
                "compaction": value.model_dump(mode="json"),
            }
        ),
    )
    fake.generate_chat_response.reset_mock()
    authority = Mock()
    assert (
        compactor.compact(
            document(),
            through_request_id="request",
            governance_epochs=(),
            before_call=authority,
            cached=receipts[-1],
        )
        == result
    )
    fake.generate_chat_response.assert_not_called()
    assert authority.call_count == 2
    compactor.compact(
        document(),
        through_request_id="request",
        governance_epochs=(("subject", 1),),
        before_call=lambda: None,
        cached=receipts[-1],
    )
    assert fake.generate_chat_response.call_count == 1


def test_repeated_excerpt_keeps_selected_original_offset() -> None:
    from reflexio.server.services.offline_tuner.evidence.compaction import (
        CompactionPoint,
        SessionCompaction,
    )
    from reflexio.server.services.offline_tuner.open_world.compaction import (
        _apply,
        _source,
    )

    original = document("repeat first; repeat second")
    point = CompactionPoint(
        category="action",
        detail="The later occurrence",
        source_id=_source(original.spans[0]).source_id,
        exact_quote="repeat",
        start_offset=14,
    )
    result = _apply(original, SessionCompaction(points=[point]))
    assert result.spans[0].field_path.endswith("[14:20]")
    assert result.spans[0].immutable_text == "repeat"


def test_truncated_provider_list_cannot_silently_salvage_partial_evidence() -> None:
    import json

    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMConfig,
        StructuredOutputParseError,
    )

    point = {
        "category": "constraint",
        "detail": "Never delete evidence",
        "evidence_id": "e1",
    }
    truncated = '{"p01":' + json.dumps(point) + ',"p02":{"category":'
    parser = LiteLLMClient(LiteLLMConfig(model="zai/glm-5.1"))
    with pytest.raises(StructuredOutputParseError):
        parser._maybe_parse_structured_output(truncated, _SelectedCompaction, True)


@pytest.mark.parametrize("oversized", [False, True])
def test_empty_or_oversized_detail_proposal_gets_actionable_repair(oversized) -> None:
    def check_repair(kwargs):
        slots: dict[str, dict[str, str] | None] = {
            f"p{index:02d}": None for index in range(1, 25)
        }
        if oversized:
            slots["p01"] = {
                "category": "constraint",
                "detail": "x" * 513,
                "evidence_id": "e1",
            }
        proposal = _SelectedCompaction.model_validate(slots)
        errors = kwargs["structured_output_validator"](proposal)
        assert len(errors) == 1
        if oversized:
            assert "513" in errors[0] and "512" in errors[0]
            assert len(proposal.points[0].detail) == 513
        else:
            assert "at least one" in errors[0]

    SessionCompactor(client(check_repair), project_id="project").compact(
        document(),
        through_request_id=None,
        governance_epochs=(),
        before_call=lambda: None,
    )


def test_missing_slot_is_not_treated_as_unused_null() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as error:
        _SelectedCompaction.model_validate(
            {f"p{index:02d}": None for index in range(1, 24)}
        )
    assert any(
        e["loc"] == ("p24",) and e["type"] == "missing" for e in error.value.errors()
    )
