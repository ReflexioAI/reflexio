import pytest

from tests._polarity_oracle import (
    NEGATIVE_PREFIXES,
    infer_playbook_polarity,
    looks_negative,
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
