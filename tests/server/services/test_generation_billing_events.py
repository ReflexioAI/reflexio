"""Tests for per-record ``publish_request_succeeded`` usage events.

Task A2 of the BYOC metering redesign: the publish path must emit ONE
``publish_request_succeeded`` event per published interaction (each
``count_value=1``, keyed by ``event_key=f"pub:{interaction_id}"`` /
``entity_id=interaction_id``) instead of a single aggregate event with
``count_value=len(new_interactions)``. Totals must be preserved: the sum of
``count_value`` across the per-interaction events must equal the old
aggregate count.

Uses a real ``GenerationService.run()`` call over a temp-dir SQLite storage
(mirroring the pattern in ``test_generation_service.py`` /
``test_generation_billing_emission.py``) with ``evaluation_only=True`` so the
publish short-circuits before any LLM extraction (site: generation_service.py
"evaluation_only" branch), keeping the test independent of LLM mocking.

Note on ids: ``Interaction.interaction_id`` is a DB auto-increment ``int``
(0 is a placeholder until insert) -- there is no caller-supplied interaction
id. So unlike illustrative pseudocode that might use string labels like
"i1"/"i2"/"i3" as literal ids, this test publishes 3 interactions into a
fresh per-test SQLite db (autoincrement starts at 1, insertion order
preserved) and asserts against the real assigned ids.
"""

from __future__ import annotations

import datetime
from datetime import UTC

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder


class _PublishFixture:
    """Wraps a real ``GenerationService`` over temp-dir SQLite storage.

    ``publish(interaction_ids=[...])`` publishes one interaction per label
    (labels only vary interaction content -- see module docstring for why the
    real ``event_key``/``entity_id`` won't literally equal the labels) and
    returns every ``UsageEvent`` recorded during the call.
    """

    def __init__(self, tmp_path) -> None:
        self.generation_service = GenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(
                org_id="publish_events_test_org", storage_base_dir=str(tmp_path)
            ),
        )

    def publish(self, interaction_ids: list[str]) -> list[UsageEvent]:
        request = PublishUserInteractionRequest(
            user_id="publish_events_test_user",
            interaction_data_list=[
                InteractionData(
                    content=f"interaction {label}",
                    created_at=int(datetime.datetime.now(UTC).timestamp()),
                )
                for label in interaction_ids
            ],
            session_id="publish_events_test_session",
            evaluation_only=True,
        )
        events: list[UsageEvent] = []
        configure_usage_event_recorder(events.append)
        try:
            self.generation_service.run(request)
        finally:
            configure_usage_event_recorder(None)
        return events


@pytest.fixture
def publish_fixture(tmp_path):
    return _PublishFixture(tmp_path)


def test_publish_emits_one_event_per_interaction(publish_fixture):
    events = publish_fixture.publish(interaction_ids=["i1", "i2", "i3"])
    pub = [e for e in events if e.event_name == "publish_request_succeeded"]

    assert len(pub) == 3

    # Real ids are the DB auto-increment ints assigned in insertion order --
    # fresh per-org sqlite db, so a first publish of 3 interactions is 1,2,3.
    ids = sorted(int(e.entity_id) for e in pub)
    assert ids == [1, 2, 3]
    assert sorted(e.event_key for e in pub) == [f"pub:{i}" for i in ids]

    assert all(e.count_value == 1 for e in pub)
    assert sum(e.count_value for e in pub) == 3  # total unchanged vs old count_value=3
