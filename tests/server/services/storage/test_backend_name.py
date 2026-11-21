from types import SimpleNamespace

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    BaseStorage,
    storage_backend_name,
)


def test_sqlite_backend_name_is_explicit():
    assert SQLiteStorage.backend_name == "sqlite"


def test_adapter_named_after_backend_derives_label():
    class PostgresStorage(BaseStorage):
        pass

    class SupabaseStorage(BaseStorage):
        pass

    assert PostgresStorage.backend_name == "postgres"
    assert SupabaseStorage.backend_name == "supabase"


def test_helper_reads_the_class_label_and_tolerates_test_doubles():
    class Fake(SimpleNamespace):
        backend_name = "postgres"

    assert storage_backend_name(Fake()) == "postgres"
    assert storage_backend_name(SimpleNamespace()) == "simplenamespace"
