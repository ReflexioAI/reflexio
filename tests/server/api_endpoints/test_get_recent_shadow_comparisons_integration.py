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
  - Pinned ``shadow_comparison_judge_prompt_version`` filter — verdicts
    produced under another rubric must not bleed into the response.

Isolation: the SQLite ``shadow_comparison_verdicts`` table is not
org-scoped, so multiple parallel tests writing to the same shared DB
would otherwise see each other's rows. Each test pins a unique judge
prompt version (the storage filter is on this field exactly) so its
rows are partitioned from sibling tests and from any prior dev rows.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.cache.reflexio_cache import get_reflexio

pytestmark = pytest.mark.integration


def _unique_judge_version() -> str:
    """Return a per-test pinned judge prompt version for isolation.

    The shared SQLite DB at ``~/.reflexio/data/reflexio.db`` carries a
    single ``shadow_comparison_verdicts`` table. Filtering by
    ``judge_prompt_version`` is the storage layer's only natural
    partition; each test reserves its own version string so its rows
    cannot collide with other tests' rows.
    """
    return f"test-{uuid.uuid4().hex[:12]}"


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


def _pin_judge_version(org_id: str, version: str) -> None:
    """Pin the org's ``shadow_comparison_judge_prompt_version`` config."""
    reflexio = get_reflexio(org_id=org_id)
    reflexio.request_context.configurator.set_config_by_name(
        "shadow_comparison_judge_prompt_version", version
    )


def test_returns_empty_when_no_verdicts(client_with_org):
    """No verdicts → 200 with an empty list, not a 5xx."""
    client, org_id = client_with_org
    # Pin a unique version so any prior dev rows in the shared DB are
    # excluded by the storage-side filter.
    _pin_judge_version(org_id, _unique_judge_version())
    resp = client.get("/api/evaluations/shadow_comparisons/recent")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"verdicts": []}


def test_default_limit_returns_at_most_10_newest_first(client_with_org):
    """Seed 15 verdicts; default limit returns the 10 newest, ordered desc."""
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    version = _unique_judge_version()
    _pin_judge_version(org_id, version)
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

    version = _unique_judge_version()
    _pin_judge_version(org_id, version)
    base_ts = int(datetime.now(UTC).timestamp()) - 10_000
    _seed_verdicts(storage, count=150, base_ts=base_ts, judge_prompt_version=version)

    resp = client.get("/api/evaluations/shadow_comparisons/recent?limit=500")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["verdicts"]) == 100


def test_filters_to_pinned_judge_prompt_version(client_with_org):
    """Verdicts produced under a different rubric version are excluded.

    The endpoint reads ``Config.shadow_comparison_judge_prompt_version``
    and passes it to ``get_shadow_comparison_verdicts``. Verdicts written
    under an older rubric must stay in storage but be hidden from the
    drawer so the dashboard never silently mixes scoring epochs.
    """
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    pinned_version = _unique_judge_version()
    other_version = _unique_judge_version()
    _pin_judge_version(org_id, pinned_version)

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

    version = _unique_judge_version()
    _pin_judge_version(org_id, version)
    base_ts = int(datetime.now(UTC).timestamp()) - 100
    _seed_verdicts(storage, count=5, base_ts=base_ts, judge_prompt_version=version)

    resp = client.get("/api/evaluations/shadow_comparisons/recent?limit=0")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["verdicts"]) == 1
