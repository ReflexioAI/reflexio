"""Task 3 (gate-b F2): explicit ``skip_embedding`` on the SQLite write methods.

V1 override of the plan: the durable compute/persist split must opt OUT of
write-time embedding via an EXPLICIT ``skip_embedding=True`` (the vector was
already populated up front by ``precompute_*_embeddings``), never via an
``if not X.embedding`` guard.

The default (``skip_embedding=False`` — what every current caller gets) MUST
recompute the embedding unconditionally EVEN when ``.embedding`` is already
populated, because several live callers ``model_copy`` a DB-loaded row (embedding
preserved) with CHANGED content and rely on the recompute
(``playbook_optimizer/optimizer.py``, ``offline_tuner/components/apply.py``,
``profile/components/consolidator.py``). An ``if not X.embedding`` guard would
persist a stale vector for the changed content — silent search corruption. These
tests pin both halves of that contract.
"""

from __future__ import annotations

import json
import time

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook, UserProfile
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

# UserProfile/UserPlaybook validate embeddings to exactly 512 dimensions.
_PRESET = [1.0] * 512
_RECOMPUTED = [9.0] * 512


def _store(tmp_path, org_id: str = "precompute-skip-org") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _stored_profile_embedding(s: SQLiteStorage, profile_id: str) -> list | None:
    row = s.conn.execute(
        "SELECT embedding FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    return json.loads(row["embedding"]) if row and row["embedding"] else None


def _stored_playbook_embedding(s: SQLiteStorage, upid: int) -> list | None:
    row = s.conn.execute(
        "SELECT embedding FROM user_playbooks WHERE user_playbook_id = ?", (upid,)
    ).fetchone()
    return json.loads(row["embedding"]) if row and row["embedding"] else None


def _isolate_embedding_path(s: SQLiteStorage, monkeypatch) -> None:
    # Keep the test on the single-embedding branch and off the dimension-coupled
    # vec table so it exercises only the embed-vs-skip decision.
    monkeypatch.setattr(s, "_should_expand_documents", lambda: False)
    monkeypatch.setattr(s, "_vec_upsert", lambda *_a, **_k: None)


class TestAddUserProfileSkipEmbedding:
    def test_skip_embedding_true_does_not_recompute(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        _isolate_embedding_path(s, monkeypatch)

        def _boom(*a, **k):
            raise AssertionError("_get_embedding must not run when skip_embedding=True")

        monkeypatch.setattr(s, "_get_embedding", _boom)

        prof = UserProfile(
            user_id="u1",
            profile_id="p1",
            content="new content",
            last_modified_timestamp=int(time.time()),
            generated_from_request_id="r1",
            embedding=list(_PRESET),
        )
        s.add_user_profile("u1", [prof], skip_embedding=True)

        assert _stored_profile_embedding(s, "p1") == _PRESET

    def test_default_recomputes_even_when_embedding_preset(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        _isolate_embedding_path(s, monkeypatch)

        calls: list[str] = []

        def _spy(text, *a, **k):
            calls.append(text)
            return list(_RECOMPUTED)

        monkeypatch.setattr(s, "_get_embedding", _spy)

        # Mirrors a model_copy caller: content changed, old embedding preserved.
        prof = UserProfile(
            user_id="u1",
            profile_id="p1",
            content="changed content",
            last_modified_timestamp=int(time.time()),
            generated_from_request_id="r1",
            embedding=list(_PRESET),
        )
        s.add_user_profile("u1", [prof])  # default skip_embedding=False

        assert calls, "default path must recompute the embedding"
        assert _stored_profile_embedding(s, "p1") == _RECOMPUTED
        assert _stored_profile_embedding(s, "p1") != _PRESET


class TestSaveUserPlaybooksSkipEmbedding:
    def test_skip_embedding_true_does_not_recompute(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        _isolate_embedding_path(s, monkeypatch)

        def _boom(*a, **k):
            raise AssertionError("_get_embedding must not run when skip_embedding=True")

        monkeypatch.setattr(s, "_get_embedding", _boom)

        pb = UserPlaybook(
            user_id="u1",
            agent_version="v1",
            request_id="r1",
            content="content",
            trigger="do the thing",
            embedding=list(_PRESET),
        )
        s.save_user_playbooks([pb], skip_embedding=True)

        assert _stored_playbook_embedding(s, pb.user_playbook_id) == _PRESET

    def test_default_recomputes_even_when_embedding_preset(self, tmp_path, monkeypatch):
        s = _store(tmp_path)
        _isolate_embedding_path(s, monkeypatch)

        calls: list[str] = []

        def _spy(text, *a, **k):
            calls.append(text)
            return list(_RECOMPUTED)

        monkeypatch.setattr(s, "_get_embedding", _spy)

        # Mirrors a model_copy caller: content changed, old embedding preserved.
        pb = UserPlaybook(
            user_id="u1",
            agent_version="v1",
            request_id="r1",
            content="content",
            trigger="do the thing",
            embedding=list(_PRESET),
        )
        s.save_user_playbooks([pb])  # default skip_embedding=False

        assert calls, "default path must recompute the embedding"
        assert _stored_playbook_embedding(s, pb.user_playbook_id) == _RECOMPUTED
        assert _stored_playbook_embedding(s, pb.user_playbook_id) != _PRESET
