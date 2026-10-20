"""Query-vector reuse preserves the storage owner and the exact model input."""

import gc
import weakref
from typing import cast

import pytest

from reflexio.server.services import unified_search_service as search
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._base import BaseStorageCore


class RecordingStorage(BaseStorageCore):
    embedding_model_name = "custom/model"
    embedding_dimensions = 2

    def __init__(self, result):
        super().__init__("same-org", base_dir="unused")
        self.result = result
        self.calls = []

    def _get_embedding(self, query, *, purpose):
        self.calls.append((query, purpose))
        return list(self.result)


def embed(storage, query):
    return search._get_cached_query_embedding(cast(BaseStorage, storage), query)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(search, "_embedding_cache", search.OrderedDict())
    monkeypatch.setattr(search, "_EMBEDDING_CACHE_TTL_SECONDS", 300)
    monkeypatch.setattr(search, "_EMBEDDING_CACHE_MAX_SIZE", 1024)


def test_storage_instances_with_same_model_and_org_do_not_share_vectors():
    first = RecordingStorage([1.0, 0.0])
    second = RecordingStorage([0.0, 1.0])
    assert embed(first, "query") == [1.0, 0.0]
    assert embed(second, "query") == [0.0, 1.0]
    assert second.calls == [("query", "query")]
    assert embed(first, "query") == [1.0, 0.0]
    assert len(first.calls) == 1


@pytest.mark.parametrize("other", ["US", " us", "us ", "uS", "us\n"])
def test_distinct_raw_model_inputs_are_not_equivalent(other):
    storage = RecordingStorage([1.0, 0.0])
    assert embed(storage, "us") == [1.0, 0.0]
    storage.result = [0.0, 1.0]
    assert embed(storage, other) == [0.0, 1.0]
    assert storage.calls == [("us", "query"), (other, "query")]


def test_cache_does_not_retain_storage_or_share_mutable_results():
    storage = RecordingStorage([1.0, 0.0])
    reference = weakref.ref(storage)
    first = embed(storage, "query")
    first[0] = 99.0
    second = embed(storage, "query")
    assert second == [1.0, 0.0]
    second[0] = 88.0
    assert embed(storage, "query") == [1.0, 0.0]
    del storage
    gc.collect()
    assert reference() is None
    assert len(search._embedding_cache) == 1


def test_cache_does_not_retain_the_callers_raw_query():
    class Query(str):
        pass

    storage = RecordingStorage([1.0, 0.0])
    query = Query("private query " * 1000)
    reference = weakref.ref(query)
    embed(storage, query)
    storage.calls.clear()
    del query
    gc.collect()
    assert reference() is None
    assert len(search._embedding_cache) == 1


def test_expiry_and_disabled_reuse_still_follow_operator_policy(monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(search, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(search, "_EMBEDDING_CACHE_TTL_SECONDS", 8)
    storage = RecordingStorage([1.0, 0.0])
    embed(storage, "query")
    clock[0] = 7.0
    embed(storage, "query")
    assert len(storage.calls) == 1
    clock[0] = 9.0
    embed(storage, "query")
    assert len(storage.calls) == 2
    monkeypatch.setattr(search, "_EMBEDDING_CACHE_TTL_SECONDS", 0)
    embed(storage, "query")
    embed(storage, "query")
    assert len(storage.calls) == 4


def test_lru_capacity_is_shared_across_storage_owners(monkeypatch):
    monkeypatch.setattr(search, "_EMBEDDING_CACHE_MAX_SIZE", 2)
    first = RecordingStorage([1.0, 0.0])
    second = RecordingStorage([0.0, 1.0])
    embed(first, "a")
    embed(second, "b")
    embed(first, "a")  # b is now least recently used.
    embed(second, "c")
    assert len(search._embedding_cache) == 2
    embed(first, "a")
    assert len(first.calls) == 1
    embed(second, "b")
    assert len(second.calls) == 3


def test_sqlite_adapters_use_their_own_embedding_client(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from reflexio.models.config_schema import LLMConfig
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    stores = [
        SQLiteStorage(
            "same-org",
            db_path=str(tmp_path / f"{index}.db"),
            llm_config=LLMConfig(embedding_model_name="text-embedding-3-small"),
        )
        for index in range(2)
    ]
    try:
        clients = []
        for index, storage in enumerate(stores):
            vector = [float(index), 1.0] + [0.0] * (storage.embedding_dimensions - 2)
            client = Mock(return_value=vector)
            monkeypatch.setattr(storage.llm_client, "get_embedding", client)
            clients.append(client)
        assert embed(stores[0], "query")[:2] == [0.0, 1.0]
        assert embed(stores[1], "query")[:2] == [1.0, 1.0]
        assert embed(stores[0], "query")[:2] == [0.0, 1.0]
        from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
        from reflexio.models.config_schema import SearchMode

        for storage in stores:
            result = search.run_unified_search(
                request=UnifiedSearchRequest(
                    query="query",
                    entity_types=["profiles"],
                    search_mode=SearchMode.VECTOR,
                    enable_reformulation=False,
                ),
                org_id="same-org",
                storage=storage,
                llm_client=storage.llm_client,
                prompt_manager=Mock(),
            )
            assert result.success
        for client in clients:
            client.assert_called_once()
    finally:
        for storage in stores:
            storage.conn.close()
