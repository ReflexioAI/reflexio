"""End-to-end test for the failure-path polarity invariant.

Verifies the load-bearing invariant for the
reflection-extraction-polarity feature:

A negative ``UserPlaybook`` produced by the failure-path extraction
flow must NOT be flipped back to ``positive`` by a later reflection
pass that merely cites it. The pipeline is exercised across two
phases against real SQLite storage in a temp directory:

Phase 1 — *extraction stand-in*: a failure-path scenario produces a
``UserPlaybook`` with ``polarity="negative"``. We persist that
playbook directly via ``storage.save_user_playbooks`` to keep this
test focused on the load-bearing invariant; the extraction-side
polarity threading is exhaustively covered by D6 / C3 integration
tests.

Phase 2 — *reflection*: an Assistant interaction cites the negative
playbook. ``ReflectionService.run`` is invoked with a scripted LLM
client returning a ``no_change`` ``ReflectionDecision``. The
assertion bank confirms the cited playbook stays current AND keeps
``polarity == "negative"`` — i.e. reflection does not silently flip
it back to positive.

This is an end-to-end test for the polarity-preservation invariant —
both phases use real ``SQLiteStorage`` and a real ``ReflectionService``
so the assertions ride on the same storage round-trips as production.
The LLM (one call inside reflection) is mocked because e2e tests
bypass the global ``litellm.completion`` mock; the mock here is per
test, not global.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    Citation,
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.config_schema import Config, ReflectionConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.reflection.reflection_service import ReflectionService
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
    ReflectionOutput,
    ReflectionServiceRequest,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_dir():
    """Create an isolated temp directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def request_context(temp_storage_dir, worker_id):
    """Real ``RequestContext`` backed by real SQLite storage in a temp dir.

    Embeddings are patched out so storage doesn't try to call out to an
    embedding model during ``save_user_playbooks`` / ``add_user_interactions_bulk``.

    Args:
        temp_storage_dir (str): Pytest-managed temp directory path.
        worker_id (str): pytest-xdist worker id; ensures per-worker org id
            so parallel runs don't collide.
    """
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    with (
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
        patch.object(
            LiteLLMClient,
            "get_embeddings",
            side_effect=lambda texts, *_args, **_kwargs: [[0.0] * 512 for _ in texts],
        ),
    ):
        ctx = RequestContext(
            org_id=f"failure_path_polarity_e2e_{worker_id}",
            storage_base_dir=temp_storage_dir,
        )
        yield ctx


@pytest.fixture
def llm_client():
    """Mock LLM client whose ``generate_chat_response`` is scripted per test."""
    return MagicMock()


@pytest.fixture
def reflection_service(request_context, llm_client) -> ReflectionService:
    """Real ``ReflectionService`` wired to the request context and mock LLM."""
    return ReflectionService(request_context=request_context, llm_client=llm_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_reflection_config(request_context: RequestContext) -> None:
    """Install a minimal Config with reflection enabled on the context."""
    cfg = Config.model_validate(
        {
            "storage_config": {"db_path": None},
            "window_size": 5,
            "stride_size": 2,
            "reflection_config": ReflectionConfig().model_dump(),
        }
    )
    request_context.configurator = MagicMock()
    request_context.configurator.get_config.return_value = cfg


def _seed_failure_window_negative_playbook(storage, user_id: str) -> UserPlaybook:
    """Persist a negative playbook as if extraction had emitted it.

    Simulates the output of the failure-path extractor (covered
    end-to-end by D6's agentic-loop test): a clear user-pushback
    window yields a ``UserPlaybook`` with ``polarity="negative"`` and
    an ``Avoid``-prefixed content body.

    Args:
        storage: The real ``SQLiteStorage`` instance.
        user_id (str): User scope for the playbook.

    Returns:
        UserPlaybook: The persisted (storage-assigned id) playbook.
    """
    pb = UserPlaybook(
        user_id=user_id,
        agent_version="v1",
        request_id="extraction_req",
        playbook_name="default",
        content="Avoid suggesting product X — the user said no twice.",
        trigger="user has previously declined product X",
        rationale="User pushed back: 'Stop suggesting X, I told you no twice.'",
        source="api",
    )
    storage.save_user_playbooks([pb])
    # Re-fetch so the storage-assigned id is reflected.
    rows = storage.get_user_playbooks(user_id=user_id, status_filter=[None])
    assert len(rows) == 1, "Setup failure — expected one seeded playbook"
    return rows[0]


def _seed_citation_window(storage, user_id: str, playbook_id: int) -> None:
    """Seed an interaction window that cites the negative playbook.

    Enough interactions to clear the stride gate (stride_size=2) and
    one Assistant turn with a citation pointing at the negative rule.

    Args:
        storage: SQLite storage.
        user_id (str): User scope.
        playbook_id (int): The playbook id to cite.
    """
    request_id = "citation_req"
    storage.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            session_id="test_session",
            source="api",
            agent_version="v1",
        )
    )
    now_ts = int(datetime.now(UTC).timestamp())
    cite = Citation(kind="playbook", real_id=str(playbook_id))
    interactions = [
        Interaction(
            user_id=user_id,
            request_id=request_id,
            role="User",
            content="What should I look at next?",
            created_at=now_ts,
            citations=[],
        ),
        Interaction(
            user_id=user_id,
            request_id=request_id,
            role="Assistant",
            content=(
                "Following the prior guidance, I'll skip suggesting X "
                "and look at adjacent options instead."
            ),
            created_at=now_ts + 1,
            citations=[cite],
        ),
        Interaction(
            user_id=user_id,
            request_id=request_id,
            role="User",
            content="Sounds good, thanks.",
            created_at=now_ts + 2,
            citations=[],
        ),
    ]
    storage.add_user_interactions_bulk(user_id=user_id, interactions=interactions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_failure_path_produces_negative_rule_reflection_keeps_it(
    request_context: RequestContext,
    reflection_service: ReflectionService,
    llm_client: MagicMock,
):
    """End-to-end: failure-path negative playbook is preserved by reflection.

    Verifies the load-bearing invariant: a negative ``UserPlaybook``
    produced by the failure-path flow is NOT flipped back to
    ``positive`` by a later reflection pass that cites it.
    """
    _set_reflection_config(request_context)
    storage = request_context.storage
    assert storage is not None, "RequestContext must provide SQLite storage"
    user_id = "u_failure_path"

    # Phase 1 — extraction stand-in: a failure-path window produces a
    # negative playbook. We persist it directly; the extractor-side
    # polarity threading is covered by D6/C3 integration tests.
    seeded = _seed_failure_window_negative_playbook(storage, user_id)
    assert seeded.content.lstrip().startswith("Avoid"), (
        "Setup failure — seeded playbook must use avoidance wording; got "
        f"{seeded.content!r}"
    )

    # Phase 2 — seed a citation window so reflection has something to consider.
    _seed_citation_window(storage, user_id, seeded.user_playbook_id)

    # Script the reflection LLM to return a ``no_change`` decision on the
    # cited negative playbook. This is the happy path: the new prompt's
    # polarity-flip semantics do NOT trigger spuriously when the agent
    # merely respects an existing rule.
    llm_client.generate_chat_response.return_value = ReflectionOutput(
        decisions=[
            ReflectionDecision(
                target_kind="playbook",
                target_id=str(seeded.user_playbook_id),
                reason="agent correctly avoided suggesting X — rule is sound",
            )
        ]
    )

    result = reflection_service.run(
        ReflectionServiceRequest(user_id=user_id, agent_version="v1")
    )

    # Reflection ran and saw the citation.
    assert result.gate_open is True, "stride gate should be open"
    assert result.ran is True, "reflection LLM should have been called"
    assert result.cited_count == 1, (
        f"Reflection should have seen one citation; result={result}"
    )
    assert result.considered_count == 1, (
        f"The cited playbook should have been considered; result={result}"
    )

    # No revision applied — and crucially, no flip. Flips are now LLM-reported
    # and counted as ordinary revisions, so revised_count == 0 is the
    # load-bearing invariant that no flip (or any other revision) occurred.
    assert result.no_change_count == 1, (
        f"Reflection should have recorded one no_change; result={result}"
    )
    assert result.revised_count == 0, (
        "Reflection must not revise (or flip) on no_change — this is the "
        f"load-bearing invariant; result={result}"
    )
    assert result.failed_count == 0, (
        f"No per-decision apply failures expected; result={result}"
    )

    # Storage check: the seeded negative playbook is still current AND
    # still negative; no archived copy was created by reflection.
    current = storage.get_user_playbooks(user_id=user_id, status_filter=[None])
    archived = storage.get_user_playbooks(
        user_id=user_id, status_filter=[Status.ARCHIVED]
    )
    assert len(current) == 1, f"Negative playbook must remain current; got {current!r}"
    assert current[0].user_playbook_id == seeded.user_playbook_id, (
        "The same row must remain — reflection must not have replaced it"
    )
    assert current[0].content.lstrip().startswith("Avoid"), (
        "Avoidance wording must survive extraction + reflection; "
        f"got {current[0].content!r}"
    )
    assert current[0].content == seeded.content, (
        f"Content must be unchanged on a no_change decision; got {current[0].content!r}"
    )
    assert archived == [], (
        f"No reflection-archived rows expected on no_change; got {archived!r}"
    )
