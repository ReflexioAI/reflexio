"""SQLite source-span admission attributes speakers the way rehydration does."""

import pytest

from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    classify_turn_speaker,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationIntegrityError,
)
from reflexio.server.services.storage.sqlite_storage._open_world_rule_evidence import (
    assert_source_span,
)

pytestmark = pytest.mark.unit


def _trajectory(role: str) -> dict:
    return {
        "requests": [
            {
                "request": {"request_id": "r-1"},
                "interactions": [
                    {"interaction_id": 7, "role": role, "content": "ran the tests"}
                ],
            }
        ]
    }


def _span(speaker: str) -> dict:
    return {
        "field_path": "requests[0].interactions[0].content",
        "request_id": "r-1",
        "interaction_id": 7,
        "speaker": speaker,
        "immutable_text": "ran the tests",
    }


@pytest.mark.parametrize(
    "role", ["Agent", " assistant ", "\tAssistant\n", "USER", " user", "tool"]
)
def test_a_span_labelled_as_rehydration_labels_it_is_admitted(role: str) -> None:
    assert_source_span(_span(classify_turn_speaker(role)), _trajectory(role))


def test_a_span_with_another_speaker_is_refused() -> None:
    with pytest.raises(
        OpenWorldInvocationIntegrityError, match="does not project original"
    ):
        assert_source_span(_span("user"), _trajectory("\tAssistant\n"))
