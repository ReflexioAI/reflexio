"""A dated `rationale` survives storage and reaches the agent-facing view.

``playbook_extraction_main`` v1.6.0 puts an ``As of YYYY-MM-DD`` prefix on the
``rationale`` of a capability-grounded avoidance rule. That is only worth doing if the
date actually reaches the agent, which means it must survive two hops the prompt has no
control over:

    extraction -> STORAGE round-trip -> UserPlaybookView -> agent context

Prompt semantics are mode-independent (one prompt file, every deployment). What differs
per mode is this wiring, so it is tested here -- deterministically, with no LLM and no
API key -- rather than folded into a real-model test where a storage bug would surface
as a confusing model-behaviour failure.

Runs against every locally-testable backend via the parametrized ``storage`` fixture, so
adding a backend covers it automatically. The enterprise siblings (supabase, postgres)
are covered by the enterprise ``contract_storage`` fixture.
"""

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.api_schema.ui.converters import to_user_playbook_view

pytestmark = pytest.mark.integration

_OBSERVED_DATE = "2026-03-02"
_DATED_RATIONALE = (
    f"As of {_OBSERVED_DATE}, the invoice export endpoint returned a server error."
)
_UNDATED_RATIONALE = "The user prefers concise summaries over full transcripts."


def _make_playbook(
    user_playbook_id: int,
    user_id: str,
    rationale: str,
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id=user_id,
        playbook_name="dating",
        agent_version="v1",
        request_id=f"req-{user_playbook_id}",
        content="Do not call the invoice export; build the PDF manually.",
        trigger="user asks to export an invoice",
        rationale=rationale,
        created_at=1_700_000_000 + user_playbook_id,
        source="test",
        source_interaction_ids=[],
    )


def test_dated_rationale_survives_storage_round_trip(storage) -> None:
    """Storage returns the rationale byte-identical, date included.

    Asserts equality rather than a substring: a backend that truncated, normalised
    whitespace, or re-encoded the field would still contain the date while corrupting
    the sentence the agent reads.
    """
    storage.save_user_playbooks([_make_playbook(1, "dating-user", _DATED_RATIONALE)])

    stored = storage.get_user_playbooks(user_id="dating-user")

    assert len(stored) == 1
    assert stored[0].rationale == _DATED_RATIONALE


def test_dated_rationale_reaches_the_agent_facing_view(storage) -> None:
    """The date survives into ``UserPlaybookView`` -- the decisive hop.

    ``rationale`` is what the retrieval routes return and what the agent-side formatter
    renders. A date that survives storage but is dropped by the view would make the
    whole change invisible to the agent while every storage test still passed.
    """
    storage.save_user_playbooks([_make_playbook(2, "dating-user", _DATED_RATIONALE)])

    stored = storage.get_user_playbooks(user_id="dating-user")
    view = to_user_playbook_view(stored[0])

    assert view.rationale == _DATED_RATIONALE
    assert _OBSERVED_DATE in (view.rationale or "")


def test_undated_rationale_is_not_given_a_date(storage) -> None:
    """A rule that should carry no date does not acquire one in transit.

    The extraction prompt dates only capability claims; preference and policy rules take
    none. This pins the storage/view path as neutral, so if a dated preference rule ever
    appears in production the cause is the prompt, not this wiring.
    """
    storage.save_user_playbooks([_make_playbook(3, "dating-user", _UNDATED_RATIONALE)])

    stored = storage.get_user_playbooks(user_id="dating-user")
    view = to_user_playbook_view(stored[0])

    assert view.rationale == _UNDATED_RATIONALE
    assert "As of" not in (view.rationale or "")


def test_content_is_unchanged_by_dating(storage) -> None:
    """Dating touches ``rationale`` only; ``content`` keeps its original force.

    The safety property of this phase is that nothing is loosened -- the rule stays a
    prohibition and only its justification records when it was observed. A change that
    leaked the date into ``content`` would be a behaviour change, not a provenance one.
    """
    playbook = _make_playbook(4, "dating-user", _DATED_RATIONALE)
    storage.save_user_playbooks([playbook])

    view = to_user_playbook_view(storage.get_user_playbooks(user_id="dating-user")[0])

    assert view.content == "Do not call the invoice export; build the PDF manually."
    assert _OBSERVED_DATE not in view.content
