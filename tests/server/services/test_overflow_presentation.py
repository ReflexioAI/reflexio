"""Extraction-only presentation preserves late evidence and global source labels."""

import json
from unittest.mock import Mock

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.overflow_presentation import (
    current_overflow_presentation,
    present_extraction_chunk,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedEvidence,
    ProtectedRecord,
    ProtectedRecordUpdate,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    construct_profile_extraction_messages_from_sessions,
)


def test_profile_and_playbook_prepared_prompts_keep_late_correction_and_source_labels():
    original = (
        "tool progress " * 2000
        + "Do not deploy before the production check."
        + " more tool progress" * 2000
    )
    sessions = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[
                Interaction(
                    interaction_id=7,
                    request_id="r",
                    user_id="u",
                    content=original,
                    role="user",
                )
            ],
            arrival_order=True,
        )
    ]
    before = sessions[0].model_dump(mode="json")
    prepared = prepare_extraction_input(sessions, user_id="u", chunk_tokens=1500)
    assert len(prepared.chunks) > 1
    manager = Mock()
    manager.render_prompt.side_effect = lambda _prompt, variables: json.dumps(variables)
    chunk = next(
        item
        for item in prepared.chunks
        if "Do not deploy before the production check." in item.text
    )
    with present_extraction_chunk(prepared, chunk, ProtectedRecord()):
        profile = construct_profile_extraction_messages_from_sessions(
            manager, sessions, [], "agent", "context", "definition"
        )
        assert "Do not deploy before the production check." in json.dumps(profile)
        playbook = build_playbook_prompt_context(sessions, label_turns=True)
        assert "Do not deploy before the production check." in playbook.text
        assert set(playbook.evidence_units) == {
            span.evidence_ref for span in (*chunk.neighbors, *chunk.spans)
        }
        assert {unit.turn_ref for unit in playbook.evidence_units.values()} == {"T1"}
        assert all(
            unit.interaction_id == 7 for unit in playbook.evidence_units.values()
        )
    assert current_overflow_presentation.get() is None
    assert sessions[0].model_dump(mode="json") == before
    # The regular formatter keeps its existing bounded presentation.
    assert (
        "Do not deploy before the production check."
        not in build_playbook_prompt_context(sessions).text
    )


def test_protected_neighbor_quotation_remains_resolvable_in_later_chunk():
    sessions = [
        RequestInteractionDataModel(
            session_id="s",
            request=Request(request_id="r", user_id="u", session_id="s"),
            interactions=[
                Interaction(
                    interaction_id=7,
                    request_id="r",
                    user_id="u",
                    role="user",
                    content="原文 🙂 scoped approval " * 1500,
                )
            ],
        )
    ]
    prepared = prepare_extraction_input(
        sessions, user_id="u", chunk_tokens=1800, require_overlap=True
    )
    assert len(prepared.chunks) > 2
    neighbor = prepared.chunks[1].neighbors[0]
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(neighbor.evidence_ref,),
                    quote=neighbor.text,
                )
            ]
        ),
        visible_spans=prepared.chunks[1].neighbors,
    )
    with present_extraction_chunk(prepared, prepared.chunks[-1], record):
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        refs = {span.evidence_ref: span for span in presentation.visible_spans}
        assert refs[neighbor.evidence_ref] == neighbor
        context = build_playbook_prompt_context(sessions, label_turns=True)
        assert context.evidence_units[neighbor.evidence_ref].role == "user"
        assert context.evidence_units[neighbor.evidence_ref].interaction_id == 7
