"""Tests for ReflectionDecision and ReflectionResult schema shape."""

from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
    ReflectionResult,
)


def test_reflection_decision_no_change_only_required_fields():
    d = ReflectionDecision(target_kind="playbook", target_id="42")
    assert d.new_content is None
    assert d.new_polarity is None


def test_reflection_decision_has_new_polarity_optional():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="42",
        new_content="Avoid X when Y.",
        new_polarity="negative",
        new_rationale="User pushed back when X was recommended.",
    )
    assert d.new_polarity == "negative"


def test_reflection_decision_no_action_field():
    # The action enum was removed; ensure the model no longer declares it.
    fields = ReflectionDecision.model_fields
    assert "action" not in fields


def test_reflection_result_has_revised_and_flipped_counts():
    r = ReflectionResult()
    assert r.revised_count == 0
    assert r.flipped_count == 0


def test_reflection_result_no_replaced_count_field():
    # The old replaced_count field was renamed/folded into revised_count.
    fields = ReflectionResult.model_fields
    assert "replaced_count" not in fields
