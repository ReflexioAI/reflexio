"""A storage failure must not be indistinguishable from an empty answer.

Before this (the pooled-connection incident of 2026-09-27) a dead pooled connection
reached ``/api/search`` two ways, and both looked like "this user has nothing":

* a TOTAL failure answered HTTP 200, with ``success=False`` only in the body, so
  a caller checking the status code or reading the result lists saw an empty,
  apparently successful search;
* a PARTIAL failure -- the profiles arm alone, which is the only arm with a
  swallow of its own -- answered 200 ``success=True`` ``degraded=False`` with the
  profiles silently missing. Nothing distinguished it at all.

``None`` from an arm now means "did not answer", and the arity of the Nones tells
the two apart: all three -> total, profiles alone -> partial.

The negative control is ``test_an_empty_result_is_not_degraded``: without it
these would all pass against an implementation that marked every search
degraded.
"""

from typing import cast

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.retriever_schema import (
    ReformulationResult,
    UnifiedSearchEntityType,
    UnifiedSearchRequest,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.storage.storage_base import BaseStorage


class _FakeStorage:
    supports_embedding = False
    embedding_model_name = "local/minilm-l6-v2"


def _playbook(content: str) -> UserPlaybook:
    return UserPlaybook(agent_version="v1", request_id="r1", content=content)


def _stub_phase_a(monkeypatch) -> None:
    monkeypatch.setattr(
        uss,
        "_run_phase_a",
        lambda **_kw: (ReformulationResult(standalone_query="q"), None, False),
    )


def _run(
    monkeypatch,
    phase_b_result,
    entity_types: list[UnifiedSearchEntityType] | None = None,
):
    _stub_phase_a(monkeypatch)
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: phase_b_result)
    return uss.run_unified_search(
        request=UnifiedSearchRequest(
            query="q", user_id="u", top_k=5, entity_types=entity_types
        ),
        org_id="o",
        storage=cast(BaseStorage, _FakeStorage()),
        llm_client=cast(LiteLLMClient, object()),
        prompt_manager=cast(PromptManager, object()),
    )


def test_a_total_failure_is_reported_as_a_failure(monkeypatch) -> None:
    """All three arms None -- the route turns this into a 503."""
    resp = _run(monkeypatch, (None, None, None))

    assert resp.success is False
    assert resp.msg == "Search failed"


def test_a_partial_failure_serves_what_answered_and_says_it_is_degraded(
    monkeypatch,
) -> None:
    """The profiles arm died; the playbook arms answered.

    The caller still gets its playbooks -- refusing the whole search would be a
    worse answer than a marked partial one -- but ``degraded`` says the profile
    set is not trustworthy, so "no profile" cannot be mistaken for a fact.
    """
    resp = _run(monkeypatch, (None, [], [_playbook("kept")]))

    assert resp.success is True, "a partial failure must not fail the whole search"
    assert resp.degraded is True, "a silently-missing profile arm must be declared"
    assert [p.content for p in resp.user_playbooks] == ["kept"]
    assert resp.profiles == []


def test_an_empty_result_is_not_degraded(monkeypatch) -> None:
    """The control: genuinely finding nothing is a healthy, complete answer."""
    resp = _run(monkeypatch, ([], [], []))

    assert resp.success is True
    assert resp.degraded is False, "an honest empty answer must not be marked degraded"


class _ProfilesDownStorage:
    """Only the profiles arm fails; both playbook arms answer normally."""

    supports_embedding = False
    supports_unified_hybrid_search = False
    embedding_model_name = "local/minilm-l6-v2"

    def search_user_profile(self, *_args, **_kwargs):
        raise RuntimeError("dead pooled connection")

    def search_agent_playbooks(self, *_args, **_kwargs):
        return []

    def search_user_playbooks(self, *_args, **_kwargs):
        return [_playbook("kept")]


def test_the_real_phase_b_preserves_a_partial_failure(monkeypatch) -> None:
    """Drive the REAL ``_run_phase_b``, not a stub that returns the answer.

    Every other test here monkeypatches ``_run_phase_b`` and asserts on what
    ``run_unified_search`` does with a hand-written tuple, so all three passed
    while the partial path was unreachable in production: the fan-out's span
    dict called ``len(profiles)``, which raises ``TypeError`` on the very None
    the contract exists to carry, and the broad ``except Exception`` converted
    that into the all-three-None TOTAL failure -- a 503 for a search that had
    playbooks to serve.

    This is the only test in the file that would have caught it.
    """
    monkeypatch.setattr(
        uss,
        "_run_phase_a",
        lambda **_kw: (ReformulationResult(standalone_query="q"), None, False),
    )
    resp = uss.run_unified_search(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
        org_id="o",
        storage=cast(BaseStorage, _ProfilesDownStorage()),
        llm_client=cast(LiteLLMClient, object()),
        prompt_manager=cast(PromptManager, object()),
    )

    assert resp.success is True, (
        "a failed profiles arm collapsed to a TOTAL failure, so the route "
        "answered 503 and threw away the playbooks that did answer"
    )
    assert resp.degraded is True
    assert [p.content for p in resp.user_playbooks] == ["kept"]
    assert resp.profiles == []


def test_a_profiles_only_search_whose_only_arm_failed_is_a_total_failure(
    monkeypatch,
) -> None:
    """Partial vs total is about what the caller ASKED for, not tuple arity.

    An arm that was not requested reports ``[]``, so for
    ``entity_types=["profiles"]`` the tuple is ``(None, [], [])`` -- the exact
    shape of a genuine partial failure. Reading arity alone classified this as
    partial and answered a 200 with an empty, apparently successful result,
    although the only arm asked for had failed. That is the bug this whole file
    exists to prevent, reachable through a request field.
    """
    resp = _run(monkeypatch, (None, [], []), entity_types=["profiles"])

    assert resp.success is False, (
        "no requested arm answered, so there is nothing to serve and nothing "
        "for the caller to retry on unless this is a failure"
    )
    assert resp.msg == "Search failed"


def test_a_profiles_only_search_that_found_nothing_is_a_success(monkeypatch) -> None:
    """The control for the test above.

    Without it, the same assertions would pass against an implementation that
    failed every profiles-only search.
    """
    resp = _run(monkeypatch, ([], [], []), entity_types=["profiles"])

    assert resp.success is True
    assert resp.degraded is False


def test_an_unrequested_arm_does_not_make_a_failed_profiles_arm_survivable(
    monkeypatch,
) -> None:
    """Only a REQUESTED playbook arm counts as having answered.

    ``entity_types=["profiles", "agent_playbooks"]`` with both the profiles arm
    failed and the agent arm answering is a real partial; the unrequested
    ``user_playbooks`` ``[]`` must not be what rescues it.
    """
    resp = _run(
        monkeypatch,
        (None, [], []),
        entity_types=["profiles", "agent_playbooks"],
    )

    assert resp.success is True, "the requested agent_playbooks arm did answer"
    assert resp.degraded is True


def test_the_real_phase_b_fails_a_profiles_only_search_outright(monkeypatch) -> None:
    """The same case, driven through the real fan-out rather than a stub tuple."""
    _stub_phase_a(monkeypatch)
    resp = uss.run_unified_search(
        request=UnifiedSearchRequest(
            query="q", user_id="u", top_k=5, entity_types=["profiles"]
        ),
        org_id="o",
        storage=cast(BaseStorage, _ProfilesDownStorage()),
        llm_client=cast(LiteLLMClient, object()),
        prompt_manager=cast(PromptManager, object()),
    )

    assert resp.success is False
    assert resp.msg == "Search failed"
