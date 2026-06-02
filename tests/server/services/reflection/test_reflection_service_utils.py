"""Tests for ReflectionDecision and ReflectionResult schema shape."""

from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
    ReflectionResult,
)


def test_reflection_decision_no_change_only_required_fields():
    d = ReflectionDecision(target_kind="playbook", target_id="42")
    assert d.new_content is None
    assert d.new_rationale is None


def test_reflection_decision_no_polarity_field():
    # Polarity is derived from wording at apply time; the declarative
    # new_polarity field was removed from the decision schema.
    fields = ReflectionDecision.model_fields
    assert "new_polarity" not in fields


def test_reflection_decision_revision_fields_optional():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="42",
        new_content="Avoid X when Y.",
        new_rationale="User pushed back when X was recommended.",
    )
    assert d.new_content == "Avoid X when Y."
    assert d.new_rationale == "User pushed back when X was recommended."


def test_reflection_decision_no_action_field():
    # The action enum was removed; ensure the model no longer declares it.
    fields = ReflectionDecision.model_fields
    assert "action" not in fields


def test_reflection_result_has_revised_count():
    r = ReflectionResult()
    assert r.revised_count == 0


def test_reflection_result_has_no_flipped_count_field():
    # flipped_count was retired: flips are LLM-reported and indistinguishable
    # from non-flip content rewrites (both carry new_rationale), so there is
    # no separate flip counter.
    fields = ReflectionResult.model_fields
    assert "flipped_count" not in fields


def test_reflection_result_no_replaced_count_field():
    # The old replaced_count field was renamed/folded into revised_count.
    fields = ReflectionResult.model_fields
    assert "replaced_count" not in fields
