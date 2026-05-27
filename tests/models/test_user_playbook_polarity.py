import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import UserPlaybook


def test_user_playbook_default_polarity_is_positive():
    pb = UserPlaybook(
        user_playbook_id=0,
        agent_version="v0",
        request_id="r1",
    )
    assert pb.polarity == "positive"


def test_user_playbook_negative_polarity_accepts_avoid():
    pb = UserPlaybook(
        user_playbook_id=0,
        agent_version="v0",
        request_id="r1",
        playbook_name="default",
        content="Avoid X when Y.",
        trigger="when Y",
        rationale="user pushed back when X was recommended",
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
        polarity="negative",
    )
    assert pb.polarity == "negative"


def test_user_playbook_polarity_rejects_invalid_value():
    with pytest.raises(ValidationError):
        UserPlaybook(
            user_playbook_id=0,
            agent_version="v0",
            request_id="r1",
            playbook_name="default",
            content="X",
            trigger="Y",
            rationale="Z",
            blocking_issue=None,
            status=None,
            source="chat",
            source_interaction_ids=[],
            polarity="neutral",  # type: ignore[arg-type]
        )
