from reflexio.models.api_schema.domain import AgentSuccessEvaluationResult


def test_shadow_fields_default_to_none():
    r = AgentSuccessEvaluationResult(
        session_id="s",
        agent_version="v",
        evaluation_name="overall_success",
        is_success=True,
    )
    assert r.shadow_is_success is None
    assert r.shadow_is_escalated is None


def test_shadow_fields_round_trip_through_model_dump():
    r = AgentSuccessEvaluationResult(
        session_id="s",
        agent_version="v",
        evaluation_name="overall_success",
        is_success=True,
        shadow_is_success=False,
        shadow_is_escalated=True,
    )
    payload = r.model_dump()
    assert payload["shadow_is_success"] is False
    assert payload["shadow_is_escalated"] is True
    r2 = AgentSuccessEvaluationResult(**payload)
    assert r2.shadow_is_success is False
    assert r2.shadow_is_escalated is True
