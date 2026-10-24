"""Integration tests for GET /api/evaluations/shadow_comparisons/recent.

The endpoint powers two surfaces on the F1 /evaluations page:
  - The drill-down drawer triggered from the per-turn comparison tile.
  - The "Top 10 disagreements" widget (the frontend pulls a wider pool and
    filters to ``is_significantly_better=True`` losses).

These tests cover the handler contract end-to-end against the real SQLite
storage backend (no judge mocking needed — verdicts are seeded directly):
  - Empty case → returns ``verdicts: []`` without 5xx.
  - Default limit (10) honoured even when storage has more rows.
  - Hard cap at 100 enforced regardless of the ``limit`` query param.
  - "Newest first" ordering — storage returns ascending, the endpoint flips.
  - Exact active template identity filter excludes historical rubric labels.

Each test owns a temporary SQLite database; rubric labels never substitute for
storage isolation. The endpoint resolves the real request-context prompt manager.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.cache.reflexio_cache import get_reflexio
from reflexio.server.services.shadow_comparison.dispatcher import (
    dispatch_shadow_comparison_judge,
)
from reflexio.server.services.shadow_comparison.rubric import ShadowComparisonRubric
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _owned_shadow_database(client_with_org, tmp_path):
    _, org_id = client_with_org
    context = get_reflexio(org_id=org_id).request_context
    previous = context.storage
    storage = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "shadow.db"))
    context.storage = storage
    try:
        yield
    finally:
        context.storage = previous
        storage.conn.close()


def _legacy_rubric_label() -> str:
    """Represent an unverified historical label that must remain separate."""
    return f"historical-{uuid.uuid4().hex[:12]}"


def _make_verdict(
    *,
    interaction_id: str,
    created_at: datetime,
    judge_prompt_version: str,
    is_significantly_better: bool = True,
    better_request: str = "1",
) -> ShadowComparisonVerdict:
    """Build a minimal ShadowComparisonVerdict for seeding storage.

    Args:
        interaction_id (str): Identifier echoed in the response — tests
            assert ordering and filtering by reading this back.
        created_at (datetime): Tz-aware UTC datetime; storage uses
            ``int(created_at.timestamp())`` as the ordering key.
        judge_prompt_version (str): Pinned prompt version that
            partitions this row from other tests' rows.
        is_significantly_better (bool): Flag the frontend uses to filter
            the Top 10 widget down to confident losses.
        better_request (str): The judge's pick — "1", "2", or "tie".

    Returns:
        ShadowComparisonVerdict: A fully-populated verdict; ``verdict_id``
            is 0 because storage auto-assigns the primary key on insert.
    """
    return ShadowComparisonVerdict(
        verdict_id=0,
        interaction_id=interaction_id,
        session_id=f"sess-{interaction_id}",
        agent_version="v1",
        reflexio_is_request_1=True,
        output=ShadowComparisonOutput(
            better_request=better_request,  # type: ignore[arg-type]
            is_significantly_better=is_significantly_better,
            comparison_reason=f"reason for {interaction_id}",
        ),
        judge_prompt_version=judge_prompt_version,
        created_at=created_at,
    )


def _seed_verdicts(
    storage,
    count: int,
    *,
    base_ts: int,
    judge_prompt_version: str,
) -> None:
    """Seed ``count`` verdicts with strictly increasing ``created_at``.

    Args:
        storage: Active storage instance to write to.
        count (int): How many verdicts to insert.
        base_ts (int): Anchor Unix-seconds timestamp; row ``i`` is offset
            by ``i`` seconds so the ordering is deterministic.
        judge_prompt_version (str): Pinned rubric version for isolation.
    """
    for i in range(count):
        storage.save_shadow_comparison_verdict(
            _make_verdict(
                interaction_id=f"i{i}",
                created_at=datetime.fromtimestamp(base_ts + i, tz=UTC),
                judge_prompt_version=judge_prompt_version,
            )
        )


def _active_rubric_key(org_id: str) -> str:
    context = get_reflexio(org_id=org_id).request_context
    return ShadowComparisonRubric(context.prompt_manager).key


def test_dispatch_storage_recent_and_overview_share_the_actual_rubric(
    client_with_org, monkeypatch
):
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None
    anchor = datetime.now(UTC) - timedelta(seconds=2)
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge.datetime",
        SimpleNamespace(now=lambda *_: anchor),
    )
    llm = MagicMock()
    llm.generate_chat_response.return_value = ShadowComparisonOutput(
        better_request="1",
        is_significantly_better=True,
        comparison_reason="The first response follows the request.",
    )
    dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=[
            Interaction(
                interaction_id=42,
                user_id="rubric-user",
                request_id="rubric-request",
                role="assistant",
                content="reflexio response",
                shadow_content="baseline response",
            )
        ],
        session_id="rubric-session",
        agent_version="agent-version",
        request_context=reflexio.request_context,
        llm_client=llm,
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="historical",
            created_at=anchor,
            judge_prompt_version="v1.1.0",
        )
    )
    expected = _active_rubric_key(org_id)
    response = client.get("/api/evaluations/shadow_comparisons/recent")
    assert response.status_code == 200, response.text
    verdicts = response.json()["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0]["interaction_id"] == "42"
    assert verdicts[0]["judge_prompt_version"] == expected
    overview = reflexio.get_evaluation_overview(
        GetEvaluationOverviewRequest(
            from_ts=int(anchor.timestamp()) - 1, to_ts=int(anchor.timestamp()) + 1
        )
    )
    assert overview.shadow_win_rate_trend.window_total.n == 1
    assert overview.shadow_win_rate_trend.judge_prompt_version == expected
    historical = storage.get_shadow_comparison_verdicts(
        from_ts=int(anchor.timestamp()) - 1,
        to_ts=int(anchor.timestamp()) + 1,
        judge_prompt_version="v1.1.0",
    )
    assert [item.interaction_id for item in historical] == ["historical"]
    llm.generate_chat_response.assert_called_once()


def test_returns_empty_when_no_verdicts(client_with_org):
    """No verdicts → 200 with an empty list, not a 5xx."""
    client, org_id = client_with_org
    # Pin a unique version so any prior dev rows in the shared DB are
    # excluded by the storage-side filter.
    resp = client.get("/api/evaluations/shadow_comparisons/recent")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"verdicts": []}


def test_default_limit_returns_at_most_10_newest_first(client_with_org):
    """Seed 15 verdicts; default limit returns the 10 newest, ordered desc."""
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    version = _active_rubric_key(org_id)
    base_ts = int(datetime.now(UTC).timestamp()) - 1_000
    _seed_verdicts(storage, count=15, base_ts=base_ts, judge_prompt_version=version)

    resp = client.get("/api/evaluations/shadow_comparisons/recent")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["verdicts"]) == 10
    interaction_ids = [v["interaction_id"] for v in body["verdicts"]]
    # Newest first: ts grew with i, so i14 has the latest created_at.
    assert interaction_ids[0] == "i14"
    assert interaction_ids[-1] == "i5"


def test_limit_query_param_honoured_up_to_hard_cap(client_with_org):
    """``?limit=500`` is clamped to the 100-row hard cap."""
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    version = _active_rubric_key(org_id)
    base_ts = int(datetime.now(UTC).timestamp()) - 10_000
    _seed_verdicts(storage, count=150, base_ts=base_ts, judge_prompt_version=version)

    resp = client.get("/api/evaluations/shadow_comparisons/recent?limit=500")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["verdicts"]) == 100


def test_filters_to_pinned_judge_prompt_version(client_with_org):
    """Verdicts produced under a different rubric version are excluded.

    The endpoint resolves the actual request-context rubric identity
    and passes it to storage. Verdicts written
    under an older rubric must stay in storage but be hidden from the
    drawer so the dashboard never silently mixes scoring epochs.
    """
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    pinned_version = _active_rubric_key(org_id)
    other_version = _legacy_rubric_label()

    # Anchor verdicts a few seconds in the past so the integer-second
    # truncation in the endpoint's ``to_ts = int(now.timestamp())`` can't
    # accidentally exclude a verdict that was saved with microseconds.
    anchor = datetime.fromtimestamp(int(datetime.now(UTC).timestamp()) - 5, tz=UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id=f"pinned-{pinned_version[-6:]}",
            created_at=anchor,
            judge_prompt_version=pinned_version,
        )
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id=f"other-{other_version[-6:]}",
            created_at=anchor,
            judge_prompt_version=other_version,
        )
    )

    resp = client.get("/api/evaluations/shadow_comparisons/recent")
    assert resp.status_code == 200, resp.text
    interaction_ids = {v["interaction_id"] for v in resp.json()["verdicts"]}
    assert interaction_ids == {f"pinned-{pinned_version[-6:]}"}


def test_invalid_limit_clamped_to_one(client_with_org):
    """``?limit=0`` (or negative) clamps to 1, never returns 422."""
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    version = _active_rubric_key(org_id)
    base_ts = int(datetime.now(UTC).timestamp()) - 100
    _seed_verdicts(storage, count=5, base_ts=base_ts, judge_prompt_version=version)

    resp = client.get("/api/evaluations/shadow_comparisons/recent?limit=0")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["verdicts"]) == 1
