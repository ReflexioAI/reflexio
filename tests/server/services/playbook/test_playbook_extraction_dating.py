"""The dating instruction and the session date reach the model together.

``playbook_extraction_main`` v1.6.0 asks the model to prefix the ``rationale`` of a
capability-grounded avoidance rule with ``As of YYYY-MM-DD``, taking the date verbatim
from the ``=== Session N (date: ...) ===`` header. That only works if BOTH halves land
in the same rendered prompt: the instruction (from the template) and the date (from the
interactions the extractor renders into it).

This test proves that on the real extractor path with a real ``PromptManager``, mocking
only the ``litellm.completion`` seam. It is deliberately cheap -- no API key, no Docker,
no storage beyond a tmp dir -- so it runs in CI on every change.

It does NOT test whether a model obeys the instruction. That is a real-LLM question and
belongs to the manifest test (T1); asserting a substring of the template file would be
vacuous, and asserting model behaviour here would need a paid call.

Regression this guards: the date is filtered by a FALSY check
(``[i.created_at for i in interactions if i.created_at]``), so ``created_at=0`` yields no
header at all. An extractor path that lost real timestamps -- or a template edit that
dropped the instruction -- would leave the model unable to comply, silently.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
from reflexio.server.services.playbook.service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.test_support.llm_mock import make_structured_finish

# 2026-03-02 UTC -- an explicit, non-zero instant so the falsy filter keeps it.
_OBSERVED_AT = int(datetime(2026, 3, 2, 12, 0, tzinfo=UTC).timestamp())
_EXPECTED_DATE = "2026-03-02"


def _ridm() -> RequestInteractionDataModel:
    """One request whose interactions carry a real, non-zero ``created_at``."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="dating-user",
            request_id="dating-request",
            role="user",
            content="Export invoice INV-4471 as a PDF.",
            created_at=_OBSERVED_AT,
        ),
        Interaction(
            interaction_id=2,
            user_id="dating-user",
            request_id="dating-request",
            role="assistant",
            content=(
                "The invoice export returned a server error (500), so I built "
                "the PDF manually from the line items instead."
            ),
            created_at=_OBSERVED_AT,
        ),
    ]
    return RequestInteractionDataModel(
        session_id="dating-request",
        request=Request(
            request_id="dating-request",
            user_id="dating-user",
            session_id="dating-session",
            created_at=_OBSERVED_AT,
            source="api",
        ),
        interactions=interactions,
    )


# A well-formed single candidate, so the forced-tool pass finishes instead of
# looping. An empty list keeps the resumable agent asking for more.
_CANNED_FINISH = {
    "playbooks": [
        {
            "trigger": "user asks to export an invoice",
            "content": "Do not call the invoice export; build the PDF manually.",
            "rationale": "The invoice export returned a server error.",
            "evidence_kind": "observed-failure",
            "evidence_refs": ["T1"],
        }
    ]
}

# The loop should need one call. Cap it so a future regression fails fast and
# loudly rather than hanging a CI worker -- this test hung at 300s while being
# written, which is a far worse failure mode than an assertion.
_MAX_COMPLETION_CALLS = 4


def _run_extractor_capturing_prompt(tmp_path) -> str:
    """Drive the real extractor, returning the prompt text sent to the model."""
    captured: list[str] = []
    calls = 0

    def _capture(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > _MAX_COMPLETION_CALLS:
            raise RuntimeError(
                f"extractor exceeded {_MAX_COMPLETION_CALLS} completion calls -- "
                "the agent loop is not terminating on the canned finish"
            )
        messages = kwargs.get("messages") or []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                captured.append(content)
            elif isinstance(content, list):
                captured.extend(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
        return make_structured_finish(_CANNED_FINISH)

    ctx = RequestContext(org_id="dating-test", storage_base_dir=str(tmp_path))
    extractor = PlaybookExtractor(
        request_context=ctx,
        llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
        extractor_config=PlaybookConfig(
            extractor_name="dating_playbook",
            extraction_definition_prompt="Extract playbooks",
        ),
        service_config=PlaybookGenerationServiceConfig(
            agent_version="1.0.0",
            request_id="dating-request",
            user_id="dating-user",
            source="api",
        ),
        agent_context="",
    )

    with (
        patch("litellm.completion", side_effect=_capture),
        patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key", "MOCK_LLM_RESPONSE": "false"},
        ),
    ):
        os.environ.pop("CLAUDE_SMART_USE_LOCAL_CLI", None)
        extractor.extract_playbook_entries([_ridm()])

    assert captured, "extractor made no completion call, so nothing was rendered"
    return "\n".join(captured)


@pytest.fixture
def rendered_prompt(tmp_path) -> str:
    return _run_extractor_capturing_prompt(tmp_path)


def test_session_date_reaches_the_model(rendered_prompt: str) -> None:
    """The header the instruction tells the model to read is actually present."""
    assert f"(date: {_EXPECTED_DATE})" in rendered_prompt, (
        "no session date header reached the model -- the instruction to copy it "
        "verbatim cannot be followed. Check that interactions carry a non-zero "
        "created_at: the date builder filters falsy timestamps."
    )


def test_dating_instruction_reaches_the_model(rendered_prompt: str) -> None:
    """v1.6.0's instruction is in the prompt the extractor actually sends."""
    assert "As of YYYY-MM-DD" in rendered_prompt, (
        "the dating instruction is missing from the rendered extraction prompt -- "
        "playbook_extraction_main v1.6.0 may be inactive or the clause removed"
    )
    assert "capability or environment fact" in rendered_prompt, (
        "the scoping clause is missing; without it the model has no rule for "
        "WHICH avoidance rules to date"
    )


def test_instruction_and_date_arrive_together(rendered_prompt: str) -> None:
    """Both halves land in one request -- the property the feature depends on.

    Asserted separately from the two above because they can fail independently:
    a template edit loses the instruction, a lost timestamp loses the date, and
    either one alone leaves the model unable to comply.
    """
    assert "As of YYYY-MM-DD" in rendered_prompt
    assert f"(date: {_EXPECTED_DATE})" in rendered_prompt


def test_normative_carve_out_is_stated(rendered_prompt: str) -> None:
    """The model is told which rules must NOT be dated.

    Without this the instruction reads as "date every avoidance rule", which would
    put a date on policy and preference rules that cannot go stale.
    """
    assert "takes no date" in rendered_prompt
