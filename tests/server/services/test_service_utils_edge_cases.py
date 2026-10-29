"""Edge cases of service_utils image encoding, budgets, history and log rendering."""

import base64
import logging
import sys
from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services import service_utils
from reflexio.server.services.service_utils import (
    _format_response_for_logging,
    _image_data_url_from_encoding,
    _resolve_max_interaction_content_tokens,
    format_messages_for_logging,
    format_sessions_to_history_string,
    visible_interaction_evidence_texts,
)

# ---------------------------------------------------------------------------
# _image_data_url_from_encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "mime"),
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 12, "image/jpeg"),
        (b"GIF87a" + b"\x00" * 10, "image/gif"),
        (b"GIF89a" + b"\x00" * 10, "image/gif"),
    ],
)
def test_image_encoding_is_labelled_with_the_signature_mime_type(header, mime):
    encoded = base64.b64encode(header).decode("ascii")
    # Line-wrapped base64 (as some clients send it) is compacted.
    wrapped = f"  {encoded[:8]}\n{encoded[8:]}  "

    assert _image_data_url_from_encoding(wrapped) == f"data:{mime};base64,{encoded}"


def test_existing_data_url_is_returned_unchanged():
    url = "data:image/png;base64,iVBORw0KGgo="
    assert _image_data_url_from_encoding(f"  {url}\n") == url


def test_non_base64_image_encoding_is_rejected():
    with pytest.raises(ValueError, match="must be valid base64 image data"):
        _image_data_url_from_encoding("!!!!not-base64!!!!")


# ---------------------------------------------------------------------------
# _resolve_max_interaction_content_tokens
# ---------------------------------------------------------------------------


def test_invalid_token_budget_warns_once_and_keeps_default(monkeypatch, caplog):
    monkeypatch.setattr(service_utils, "_INVALID_MAX_TOKENS_WARNED", False)
    monkeypatch.setenv("REFLEXIO_MAX_INTERACTION_CONTENT_TOKENS", "lots")

    with caplog.at_level(logging.WARNING, logger=service_utils.__name__):
        first = _resolve_max_interaction_content_tokens()
        second = _resolve_max_interaction_content_tokens()

    default = service_utils.DEFAULT_MAX_INTERACTION_CONTENT_TOKENS
    assert first == second == default
    warnings = [r for r in caplog.records if "REFLEXIO_MAX_INTERACTION" in r.message]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# visible_interaction_evidence_texts / format_sessions_to_history_string
# ---------------------------------------------------------------------------


def test_evidence_of_a_contentless_expert_turn_is_only_the_expert_text():
    interaction = Interaction(
        interaction_id=1,
        user_id="u",
        request_id="r",
        role="assistant",
        content="",
        expert_content="Run the staging rehearsal first.",
    )
    assert visible_interaction_evidence_texts(interaction) == (
        "Run the staging rehearsal first.",
    )


def test_session_header_omits_date_when_timestamp_is_out_of_range():
    huge = 10**18
    session = RequestInteractionDataModel(
        session_id="s1",
        request=Request(request_id="r1", user_id="u", session_id="s1", created_at=1),
        interactions=[
            Interaction(
                interaction_id=1,
                user_id="u",
                request_id="r1",
                role="user",
                content="hello",
                created_at=huge,
            )
        ],
    )

    assert format_sessions_to_history_string([session]) == (
        "=== Session: s1 ===\nuser: ```hello```"
    )


# ---------------------------------------------------------------------------
# format_messages_for_logging / _format_response_for_logging
# ---------------------------------------------------------------------------


def test_tool_call_of_unknown_shape_renders_without_name_or_arguments():
    rendered = format_messages_for_logging(
        [{"role": "assistant", "content": None, "tool_calls": ["opaque-call"]}]
    )
    assert "    - id: None" in rendered
    assert "      name: None" in rendered
    assert "      arguments: null" in rendered


def test_tool_call_arguments_that_are_not_json_serializable_fall_back_to_repr():
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="tag", arguments={"labels": {"a"}}),
    )
    rendered = format_messages_for_logging(
        [{"role": "assistant", "content": "", "tool_calls": [call]}]
    )
    assert "      name: tag" in rendered
    assert "      arguments: {'labels': {'a'}}" in rendered


def test_multimodal_content_renders_non_dict_items_and_non_text_blocks_as_json():
    rendered = format_messages_for_logging(
        [
            {
                "role": "user",
                "content": [
                    "raw part",
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            }
        ]
    )
    assert '    "raw part"' in rendered
    assert '"url": "https://x/y.png"' in rendered


def test_non_string_non_list_content_falls_back_to_json():
    rendered = format_messages_for_logging([{"role": "tool", "content": {"ok": 1}}])
    assert "  role: tool" in rendered
    assert '  content: {\n    "ok": 1\n}' in rendered


def test_response_formatting_passes_through_when_client_cannot_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "reflexio.server.llm.litellm_client", None)
    response = object()
    assert _format_response_for_logging(response) is response
