import logging

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.server.services.polarity_utils import (
    NEGATIVE_PREFIXES,
    infer_playbook_polarity,
    looks_negative,
    warn_if_polarity_content_mismatch,
)


def _make_playbook(content: str, polarity: str = "positive") -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version="v0",
        request_id="r1",
        playbook_name="default",
        content=content,
        trigger="t",
        rationale="r",
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
        polarity=polarity,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Avoid X", True),
        ("Do not X", True),
        ("Don't X", True),
        ("Never X", True),
        ("  Avoid X", True),  # leading whitespace tolerated
        ("Recommend X", False),
        ("Stop X", False),  # not in the whitelist
        ("avoid X", False),  # lowercase — convention is title-cased
    ],
)
def test_looks_negative(content: str, expected: bool) -> None:
    assert looks_negative(content) is expected


def test_polarity_consistent_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    pb = _make_playbook(content="Recommend X when Y.", polarity="positive")
    with caplog.at_level(logging.WARNING):
        warn_if_polarity_content_mismatch(pb)
    assert not any("polarity_content_mismatch" in r.message for r in caplog.records)


def test_polarity_mismatch_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    pb = _make_playbook(content="Recommend X when Y.", polarity="negative")
    with caplog.at_level(logging.WARNING):
        warn_if_polarity_content_mismatch(pb)
    assert any("polarity_content_mismatch" in r.message for r in caplog.records)


def test_infer_playbook_polarity_defaults_positive() -> None:
    assert (
        infer_playbook_polarity(
            "Use the narrow verification before broad checks.",
            "The session succeeded after the focused check.",
        )
        == "positive"
    )


def test_infer_playbook_polarity_negative_requires_avoidance_and_failure_evidence() -> (
    None
):
    assert (
        infer_playbook_polarity(
            "Avoid broad setup before the target behavior is isolated.",
            "The session showed unrelated setup failures consumed extra turns.",
        )
        == "negative"
    )


def test_infer_playbook_polarity_avoidance_without_failure_evidence_stays_positive() -> (
    None
):
    assert (
        infer_playbook_polarity("Avoid broad setup.", "Use the focused path.")
        == "positive"
    )


def test_negative_prefixes_constant_matches_docstring() -> None:
    assert NEGATIVE_PREFIXES == ("Avoid", "Do not", "Don't", "Never")
