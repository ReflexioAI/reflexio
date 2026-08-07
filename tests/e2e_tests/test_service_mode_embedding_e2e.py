"""Faithful e2e test for service-mode embedding over a real socket.

No other test boots a real embedding service over a socket and points a backend
at it, yet this is the topology creao runs in prod today: an
``internal_service`` / ``local_service`` embedding mode routes every embedding
call to a *separate* ``create_embedding_app`` process. This module protects that
live path.

It boots the real embedding daemon on an ephemeral port via ``uvicorn`` in a
background thread — real HTTP, real chunking, real timeout, real MiniLM ONNX
embeddings — and points a SQLite-backed Reflexio instance at it through
``internal_service`` mode (``REFLEXIO_EMBEDDING_SERVICE_URL``). Nothing here is
mocked or monkeypatched at the transport layer.

Two tests:

* ``test_service_mode_round_trip`` — the faithful happy path: a profile is
  embedded through the HTTP service, persisted, and retrieved by *vector*
  similarity (the query shares no words with the winning profile, so an FTS
  ranking could not have produced it). ``degraded`` is False and the stored
  embedding is non-empty.
* ``test_service_mode_degrades_when_daemon_down`` — the daemon is stopped
  mid-test; search still returns 200, sets ``degraded=True``, serves results
  from FTS, and emits the ``event=search_degraded_to_fts`` WARN. That WARN is
  exactly what the ``reflexio-search-degraded-to-fts`` prod alarm keys on, so
  this test is its regression anchor.

The MiniLM (``local/minilm-l6-v2``) model is used deliberately: it is the OSS
default and runs on chromadb's bundled ONNX runtime (no torch /
sentence-transformers), so it is the lightest real model that still returns
rankable vectors — faithful, but the cheapest faithful. No paid API key is
needed (local model), so ``@skip_low_priority`` is not applied; only
``@skip_in_precommit`` because the one-time model load takes a few seconds.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.domain.entities import ProfileTimeToLive, UserProfile
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import (
    Config,
    LLMConfig,
    SearchMode,
    StorageConfigSQLite,
)
from reflexio.server.llm.embedding_service import MINILM_MODEL, create_embedding_app
from reflexio.server.llm.rerank import score_pairs
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from tests.server.test_utils import skip_in_precommit

pytestmark = pytest.mark.e2e

# Profiles under one user. Vector search needs a ``user_id`` filter, so both the
# ingested profiles and the search request share this id.
_USER_ID = "svc_embed_user"

# The winning profile's content and the round-trip query share NO tokens. A
# pure-FTS ranking therefore cannot surface it — only a vector match can — so a
# top-ranked coffee profile proves the vector path ran end to end.
_COFFEE_ID = "profile_coffee"
_COFFEE_CONTENT = "Weekend mornings I brew espresso and roast beans at home."
_MOTO_ID = "profile_moto"
_MOTO_CONTENT = "I restore vintage motorcycles in my garage on Sundays."
_VECTOR_QUERY = "person passionate about barista culture and cappuccino"

# The degrade query DOES lexically overlap the coffee profile, so FTS can still
# return it once the vector path degrades.
_FTS_QUERY = "espresso beans brew"


class _EmbeddingDaemon:
    """Runs ``create_embedding_app`` on an ephemeral port in a background thread.

    A pre-bound socket (port 0) is handed to uvicorn so there is no
    bind-then-reconnect race on the ephemeral port. ``start`` blocks until
    ``GET /health`` returns 200 — because the app is created with a
    ``default_model``, a healthy response means the model is already warmed, so
    the first real embedding request is not paying the load cost.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_embedding_app(default_model=MINILM_MODEL),
                log_level="warning",
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._sock]},
            daemon=True,
            name="test-embedding-daemon",
        )
        self._stopped = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 90.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return
            except httpx.HTTPError as exc:
                last_err = exc
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(
            f"embedding daemon did not become healthy within {timeout:.0f}s: {last_err}"
        )

    def stop(self) -> None:
        """Signal shutdown and join the server thread. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self._server.should_exit = True
        self._thread.join(timeout=30)
        with contextlib.suppress(OSError):
            self._sock.close()


@pytest.fixture
def embedding_daemon():
    """Boot a real embedding daemon; guarantee teardown even if a test stops it."""
    daemon = _EmbeddingDaemon()
    daemon.start()
    try:
        yield daemon
    finally:
        daemon.stop()


def _build_reflexio(db_path: str, org_id: str) -> Reflexio:
    """A SQLite-backed Reflexio pinned to the local MiniLM embedding model.

    Document expansion is disabled and the generation / pre-retrieval model
    names are inert placeholders: this test exercises only the embedding path,
    so no real generation LLM (and no API key) is ever needed. The placeholders
    exist purely so storage construction and ``unified_search``'s model
    resolution do not fall through to API-key auto-detection.
    """
    config = Config(
        storage_config=StorageConfigSQLite(db_path=db_path),
        enable_document_expansion=False,
        llm_config=LLMConfig(
            embedding_model_name=MINILM_MODEL,
            generation_model_name="placeholder/generation",
            pre_retrieval_model_name="placeholder/pre-retrieval",
        ),
    )
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


def _make_profile(profile_id: str, content: str) -> UserProfile:
    return UserProfile(
        user_id=_USER_ID,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(time.time()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _stored_embedding(storage, profile_id: str) -> list[float]:
    """Read the persisted embedding column straight from SQLite.

    The public profile read paths strip embeddings (size / privacy), so the
    only way to assert the vector was actually persisted is to read the column
    directly from the backend under test.
    """
    row = storage.conn.execute(
        "SELECT embedding FROM profiles WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    assert row is not None, f"profile {profile_id} was not persisted"
    raw = row[0]
    return json.loads(raw) if raw else []


@skip_in_precommit
def test_service_mode_round_trip(
    tmp_path, worker_id: str, monkeypatch, embedding_daemon: _EmbeddingDaemon
) -> None:
    """Real HTTP embedding round-trip: persisted + retrieved by vector similarity."""
    org_id = f"emb_svc_e2e_rt_{worker_id}"
    # internal_service mode: a bare SERVICE_URL (no explicit provider env) routes
    # every embedding call to the HTTP daemon. Explicit timeout so a cold CPU
    # encode never trips the 2s internal-service default and flakes.
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", embedding_daemon.base_url)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", "30000")

    reranker_health = httpx.get(
        f"{embedding_daemon.base_url}/health/rerank", timeout=5.0
    )
    assert reranker_health.status_code == 200
    assert reranker_health.json()["ready"] is True
    rerank_scores = score_pairs(
        "Italian food", ["fresh pasta with tomato sauce", "rain forecast"]
    )
    assert len(rerank_scores) == 2
    assert rerank_scores[0] > rerank_scores[1]

    reflexio = _build_reflexio(str(tmp_path / "reflexio.db"), org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    # Ingest: add_user_profile embeds ``content`` through the HTTP service and
    # persists the vector + FTS + vec rows.
    storage.add_user_profile(_USER_ID, [_make_profile(_COFFEE_ID, _COFFEE_CONTENT)])
    storage.add_user_profile(_USER_ID, [_make_profile(_MOTO_ID, _MOTO_CONTENT)])

    # The stored embedding must be a real, non-empty vector.
    coffee_embedding = _stored_embedding(storage, _COFFEE_ID)
    assert len(coffee_embedding) > 0, "stored embedding is empty"
    assert any(value != 0.0 for value in coffee_embedding), (
        "stored embedding is all-zero (embedding service was not used)"
    )

    response = reflexio.unified_search(
        UnifiedSearchRequest(
            query=_VECTOR_QUERY,
            user_id=_USER_ID,
            search_mode=SearchMode.VECTOR,
            threshold=0.0,
            top_k=5,
        ),
        org_id=org_id,
    )

    assert response.success is True
    # Not degraded: the real vector path ran; the requested mode was honored.
    assert response.degraded is False
    assert response.search_mode_effective is None

    returned_ids = [profile.profile_id for profile in response.profiles]
    # The query shares no tokens with the coffee content, yet the coffee profile
    # is ranked first — only a vector ranking can produce this, not FTS.
    assert returned_ids, "vector search returned no profiles"
    assert returned_ids[0] == _COFFEE_ID, (
        f"expected coffee profile ranked first by vector similarity, got {returned_ids}"
    )
    if _MOTO_ID in returned_ids:
        assert returned_ids.index(_COFFEE_ID) < returned_ids.index(_MOTO_ID), (
            "semantically-matching profile did not outrank the unrelated one"
        )


@skip_in_precommit
def test_service_mode_degrades_when_daemon_down(
    tmp_path,
    worker_id: str,
    monkeypatch,
    caplog,
    embedding_daemon: _EmbeddingDaemon,
) -> None:
    """Daemon killed mid-test: search still 200s, degrades to FTS, emits the WARN."""
    org_id = f"emb_svc_e2e_deg_{worker_id}"
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", embedding_daemon.base_url)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", "30000")

    reflexio = _build_reflexio(str(tmp_path / "reflexio.db"), org_id)
    storage = reflexio.request_context.storage
    assert storage is not None

    # Ingest while the daemon is up so the profile carries a real embedding and
    # a populated FTS row.
    storage.add_user_profile(_USER_ID, [_make_profile(_COFFEE_ID, _COFFEE_CONTENT)])
    assert len(_stored_embedding(storage, _COFFEE_ID)) > 0

    # Kill the embedding service. Every subsequent embedding call now fails.
    embedding_daemon.stop()

    with caplog.at_level(
        logging.WARNING,
        logger="reflexio.server.services.unified_search_service",
    ):
        response = reflexio.unified_search(
            UnifiedSearchRequest(
                query=_FTS_QUERY,
                user_id=_USER_ID,
                # HYBRID wants an embedding, so the down daemon triggers degrade.
                search_mode=SearchMode.HYBRID,
                threshold=0.0,
                top_k=5,
            ),
            org_id=org_id,
        )

    # 200 OK, not a crash: the silent-degrade path is the observed behavior.
    assert response.success is True
    assert response.degraded is True
    assert response.search_mode_effective == SearchMode.FTS.value

    # Results are served from FTS: the coffee profile lexically matches the query.
    returned_ids = [profile.profile_id for profile in response.profiles]
    assert _COFFEE_ID in returned_ids, "FTS fallback returned no lexical match"

    # The WARN the reflexio-search-degraded-to-fts prod alarm keys on.
    degrade_records = [
        record
        for record in caplog.records
        if "event=search_degraded_to_fts" in record.getMessage()
    ]
    assert degrade_records, "expected a search_degraded_to_fts WARN log record"
    assert degrade_records[0].levelno == logging.WARNING
