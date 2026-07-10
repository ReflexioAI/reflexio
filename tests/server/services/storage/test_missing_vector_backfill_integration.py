"""Integration tests for the missing-vector backfill sweep (SQLite, mocked LLM).

Reproduces the durability hole: when an embedding call fails at ingest the
interaction is stored with an empty embedding and no vector row, so it is
invisible to vector search forever. The backfill sweep re-embeds those rows.

Coverage:
- backfill writes the vector AND the row becomes vector-searchable (via the
  search path, not by inspecting the table),
- idempotency (a second sweep backfills 0),
- the per-tick bound is respected (seed > cap => only cap done),
- fail-safe: with the embedder raising EmbeddingUnavailableError the backfill
  returns 0 without raising,
- the sweep closure honours its enable gate and is failure-isolated.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.retriever_schema import SearchInteractionRequest
from reflexio.models.api_schema.service_schemas import Interaction
from reflexio.models.config_schema import SearchMode
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, worker_id: str) -> SQLiteStorage:
    # Unique org id per worker keeps xdist runs isolated.
    s = SQLiteStorage(
        org_id=f"backfill-{worker_id}", db_path=str(tmp_path / "backfill.db")
    )
    s.migrate()
    return s


def _unit_vector(dims: int) -> list[float]:
    """A fixed unit vector so cosine similarity to itself is 1.0."""
    return [1.0] + [0.0] * (dims - 1)


def _install_working_embedder(s: SQLiteStorage) -> list[float]:
    """Replace the storage's llm_client with a mock returning a fixed vector.

    Returns the vector so the caller can reuse it as the query embedding.
    """
    vec = _unit_vector(s.embedding_dimensions)

    def _batch(texts, *_a, **_k):
        return [list(vec) for _ in texts]

    def _single(*_a, **_k):
        return list(vec)

    client = MagicMock()
    client.get_embeddings.side_effect = _batch
    client.get_embedding.side_effect = _single
    s.llm_client = client
    return vec


def _seed_missing_vector_interactions(
    s: SQLiteStorage, user_id: str, count: int, *, start_id: int = 1
) -> list[int]:
    """Seed interactions that have NO embedding (simulate a prior embed failure).

    Uses the durable-path insert (embeddings_prepared=True) with embedding=[],
    exactly the state ingest leaves behind when embedding degrades: '[]' in the
    embedding column and no interactions_vec row. No embedder is called.
    """
    ids = list(range(start_id, start_id + count))
    interactions = [
        Interaction(
            interaction_id=iid,
            user_id=user_id,
            request_id=f"req-{iid}",
            content=f"the quick brown fox number {iid}",
            created_at=int(time.time()) + iid,
            embedding=[],
        )
        for iid in ids
    ]
    s.add_user_interactions_bulk(
        user_id=user_id, interactions=interactions, embeddings_prepared=True
    )
    return ids


def _vec_row_count(s: SQLiteStorage, interaction_id: int) -> int:
    if not s._has_sqlite_vec:
        return 0
    row = s.conn.execute(
        "SELECT COUNT(*) AS cnt FROM interactions_vec WHERE rowid = ?",
        (interaction_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def _embedding_column(s: SQLiteStorage, interaction_id: int) -> str | None:
    row = s.conn.execute(
        "SELECT embedding FROM interactions WHERE interaction_id = ?",
        (interaction_id,),
    ).fetchone()
    return row["embedding"] if row else None


def _vector_search_ids(
    s: SQLiteStorage, user_id: str, query_embedding: list[float]
) -> set[int]:
    req = SearchInteractionRequest(
        user_id=user_id, search_mode=SearchMode.VECTOR, most_recent_k=50
    )
    results = s.search_interaction(req, query_embedding=query_embedding)
    return {i.interaction_id for i in results}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_makes_interactions_vector_searchable(tmp_path, worker_id) -> None:
    s = _store(tmp_path, worker_id)
    vec = _install_working_embedder(s)
    uid = "u-searchable"
    ids = _seed_missing_vector_interactions(s, uid, 3)

    # Precondition: empty embedding column, no vec row, NOT vector-searchable.
    for iid in ids:
        assert _embedding_column(s, iid) == "[]"
        assert _vec_row_count(s, iid) == 0
    assert _vector_search_ids(s, uid, vec) == set()

    backfilled = s.backfill_missing_interaction_vectors(limit=100)

    assert backfilled == 3
    # Vector row now exists (when sqlite-vec is loaded) and the column is filled.
    for iid in ids:
        assert _embedding_column(s, iid) != "[]"
        if s._has_sqlite_vec:
            assert _vec_row_count(s, iid) == 1
    # The rows are now reachable through the vector search path.
    assert _vector_search_ids(s, uid, vec) == set(ids)


def test_backfill_is_idempotent(tmp_path, worker_id) -> None:
    s = _store(tmp_path, worker_id)
    _install_working_embedder(s)
    uid = "u-idem"
    _seed_missing_vector_interactions(s, uid, 4)

    first = s.backfill_missing_interaction_vectors(limit=100)
    second = s.backfill_missing_interaction_vectors(limit=100)

    assert first == 4
    assert second == 0
    assert s.iter_interactions_missing_vectors(100) == []


def test_backfill_respects_the_per_tick_bound(tmp_path, worker_id) -> None:
    s = _store(tmp_path, worker_id)
    _install_working_embedder(s)
    uid = "u-bound"
    _seed_missing_vector_interactions(s, uid, 10)

    done = s.backfill_missing_interaction_vectors(limit=3)

    assert done == 3
    # 7 still awaiting backfill on the next tick.
    assert len(s.iter_interactions_missing_vectors(100)) == 7


def test_backfill_is_fail_safe_when_embedder_unavailable(tmp_path, worker_id) -> None:
    s = _store(tmp_path, worker_id)
    uid = "u-failsafe"
    ids = _seed_missing_vector_interactions(s, uid, 3)

    client = MagicMock()
    client.get_embeddings.side_effect = EmbeddingUnavailableError("provider down")
    s.llm_client = client

    done = s.backfill_missing_interaction_vectors(limit=100)

    # No crash, nothing backfilled, work left for the next tick.
    assert done == 0
    for iid in ids:
        assert _embedding_column(s, iid) == "[]"
    assert len(s.iter_interactions_missing_vectors(100)) == 3


def test_base_default_is_a_safe_no_op() -> None:
    """The base mixin default returns empty/0 so backends lacking an override
    are simply skipped, never broken."""
    from reflexio.server.services.storage.storage_base.profiles._interaction_store import (  # noqa: E501
        InteractionStoreMixin,
    )

    assert InteractionStoreMixin.iter_interactions_missing_vectors(None, 10) == []  # type: ignore[arg-type]
    assert InteractionStoreMixin.backfill_missing_interaction_vectors(None, 10) == 0  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content", "action_desc"),
    [
        ("hello world", "clicked the submit button"),
        ("only content", ""),  # empty action desc -> the `or ""` branch
        ("", "only action desc"),  # empty content
    ],
)
def test_backfill_text_matches_the_ingest_derivation(
    tmp_path, worker_id, content: str, action_desc: str
) -> None:
    """Cross-path guard: the text ``iter_interactions_missing_vectors`` yields for
    an interaction is byte-identical to the text the *ingest* path actually
    derives and hands to the embedder for that same interaction. Fails loudly if
    anyone later changes ingest derivation without updating backfill.

    The mock embedder ignores its input, so this asserts on the derived TEXT that
    ingest passes to ``get_embeddings`` — not on the resulting vector.
    """
    s = _store(tmp_path, worker_id)

    captured_texts: list[str] = []

    def _record(texts, *_a, **_k):
        captured_texts.extend(texts)
        return [_unit_vector(s.embedding_dimensions) for _ in texts]

    client = MagicMock()
    client.get_embeddings.side_effect = _record
    s.llm_client = client

    uid = "u-equivalence"
    # Ingest via the REAL bulk path (embeddings_prepared=False) so
    # add_user_interactions_bulk derives the text and passes it to get_embeddings.
    ingested = Interaction(
        interaction_id=1,
        user_id=uid,
        request_id="req-ingest",
        content=content,
        user_action_description=action_desc,
        created_at=int(time.time()),
    )
    s.add_user_interactions_bulk(user_id=uid, interactions=[ingested])
    assert len(captured_texts) == 1
    ingest_text = captured_texts[0]

    # Seed a distinct row with the SAME content/action but no vector, then read
    # back the text the backfill detector derives for it.
    missing = Interaction(
        interaction_id=2,
        user_id=uid,
        request_id="req-missing",
        content=content,
        user_action_description=action_desc,
        created_at=int(time.time()) + 1,
        embedding=[],
    )
    s.add_user_interactions_bulk(
        user_id=uid, interactions=[missing], embeddings_prepared=True
    )
    pairs = dict(s.iter_interactions_missing_vectors(100))
    assert 2 in pairs
    backfill_text = pairs[2]

    # The load-bearing invariant: single-sourced derivation.
    assert backfill_text == ingest_text


# ---------------------------------------------------------------------------
# Sweep closure
# ---------------------------------------------------------------------------


def test_sweep_disabled_by_default_does_not_touch_storage(
    tmp_path, worker_id, monkeypatch
) -> None:
    from reflexio.server.services.lineage import vector_backfill_sweep as vbs

    monkeypatch.delenv("REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED", raising=False)

    # RequestContext must never be constructed when the flag is off.
    def _boom(*_a, **_k):  # pragma: no cover - asserts it is never called
        raise AssertionError("RequestContext built while sweep disabled")

    monkeypatch.setattr(
        "reflexio.server.api_endpoints.request_context.RequestContext", _boom
    )

    assert vbs.missing_vector_backfill_sweep("some-org", 0) == 0


def test_sweep_enabled_backfills_via_storage(tmp_path, worker_id, monkeypatch) -> None:
    from reflexio.server.services.lineage import vector_backfill_sweep as vbs

    s = _store(tmp_path, worker_id)
    _install_working_embedder(s)
    uid = "u-sweep"
    _seed_missing_vector_interactions(s, uid, 5)

    monkeypatch.setenv("REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED", "true")
    monkeypatch.setenv("REFLEXIO_MISSING_VECTOR_BACKFILL_CAP", "2")

    fake_ctx = MagicMock()
    fake_ctx.storage = s

    def _fake_request_context(**_kwargs):
        return fake_ctx

    monkeypatch.setattr(
        "reflexio.server.api_endpoints.request_context.RequestContext",
        _fake_request_context,
    )

    # Cap=2 => two per tick; the sweep threads the env cap through to storage.
    assert vbs.missing_vector_backfill_sweep(s.org_id, 0) == 2
    assert len(s.iter_interactions_missing_vectors(100)) == 3
