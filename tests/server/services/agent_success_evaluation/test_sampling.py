"""Per-family sampling: agent-success and retrieved-learning sample independently."""

from __future__ import annotations

from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.services.agent_success_evaluation.sampling import (
    samples_agent_success,
    samples_retrieved_learning,
    stable_group_sampling_fraction,
)

SCOPE = {"org_id": "org-1", "user_id": "user-1", "session_id": "sess-1"}


def _config(sampling_rate: float, retrieved: float | None = None) -> AgentSuccessConfig:
    return AgentSuccessConfig(
        success_definition_prompt="Did the agent succeed?",
        sampling_rate=sampling_rate,
        retrieved_learning_sampling_rate=retrieved,
    )


def test_the_field_default_is_none_when_never_supplied() -> None:
    """Pin the DEFAULT, by never passing the field.

    The earlier version of this test called a helper that passed
    `retrieved_learning_sampling_rate=None` explicitly, so it asserted its own
    argument rather than the field default. It therefore stayed green while the
    default silently drifted to 0.1 — which would have doubled the eval LLM bill
    for every org that never opted in. Construct the config WITHOUT the field or
    this test proves nothing.
    """
    config = AgentSuccessConfig(success_definition_prompt="Did the agent succeed?")

    assert config.retrieved_learning_sampling_rate is None


def test_an_org_that_never_opted_in_keeps_its_previous_behavior() -> None:
    """No opt-in => retrieved-learning samples exactly where agent-success does."""
    for rate in (0.0, 0.05, 1.0):
        config = AgentSuccessConfig(
            success_definition_prompt="Did the agent succeed?",
            sampling_rate=rate,
        )
        assert samples_retrieved_learning(config, **SCOPE) == samples_agent_success(
            config, **SCOPE
        )


def test_retrieved_learning_can_sample_denser_than_agent_success() -> None:
    """The whole point: dense tuner coverage without paying the success judge."""
    config = _config(0.0, retrieved=1.0)

    assert samples_agent_success(config, **SCOPE) is False
    assert samples_retrieved_learning(config, **SCOPE) is True


def test_retrieved_learning_can_sample_sparser_than_agent_success() -> None:
    config = _config(1.0, retrieved=0.0)

    assert samples_agent_success(config, **SCOPE) is True
    assert samples_retrieved_learning(config, **SCOPE) is False


def test_zero_rate_never_samples_and_full_rate_always_samples() -> None:
    assert samples_agent_success(_config(0.0), **SCOPE) is False
    assert samples_agent_success(_config(1.0), **SCOPE) is True
    assert samples_retrieved_learning(_config(1.0, retrieved=0.0), **SCOPE) is False
    assert samples_retrieved_learning(_config(0.0, retrieved=1.0), **SCOPE) is True


def test_missing_agent_success_config_samples_neither() -> None:
    assert samples_agent_success(None, **SCOPE) is False
    assert samples_retrieved_learning(None, **SCOPE) is False


def test_sampling_is_stable_and_scoped_to_the_session() -> None:
    """Both families must agree on WHICH sessions are in a given fraction."""
    first = stable_group_sampling_fraction("org-1", "user-1", "sess-1")
    assert first == stable_group_sampling_fraction("org-1", "user-1", "sess-1")
    assert 0.0 <= first < 1.0
    assert first != stable_group_sampling_fraction("org-1", "user-1", "sess-2")
    assert first != stable_group_sampling_fraction("org-2", "user-1", "sess-1")


def test_a_denser_retrieved_rate_is_a_superset_of_the_agent_success_sample() -> None:
    """Nesting matters: raising only the retrieved rate must not DROP sessions
    that the success judge would have sampled, because both families key off the
    same stable fraction."""
    config = _config(0.1, retrieved=0.9)
    sampled_success = 0
    both = 0
    for n in range(500):
        scope = {"org_id": "org-1", "user_id": "u", "session_id": f"s-{n}"}
        if samples_agent_success(config, **scope):
            sampled_success += 1
            if samples_retrieved_learning(config, **scope):
                both += 1
    assert sampled_success > 0
    assert both == sampled_success
