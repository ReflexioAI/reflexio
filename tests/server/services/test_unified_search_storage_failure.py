"""A storage failure must not be indistinguishable from an empty answer.

Before this (Sentry PYTHON-FASTAPI-Z0, 2026-09-27) a dead pooled connection
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


def _run(monkeypatch, phase_b_result):
    monkeypatch.setattr(
        uss,
        "_run_phase_a",
        lambda **_kw: (ReformulationResult(standalone_query="q"), None, False),
    )
    monkeypatch.setattr(uss, "_run_phase_b", lambda **_kw: phase_b_result)
    return uss.run_unified_search(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
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
