"""Integration contract for archive-before-delete retention."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserActionType,
)
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.storage.retention_archive import RetentionArchiveFullError
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _interaction(interaction_id: int, request_id: str) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id="u1",
        request_id=request_id,
        content=f"interaction {interaction_id}",
        created_at=interaction_id,
        user_action=UserActionType.NONE,
        user_action_description="",
        interacted_image_url="",
    )


def _request(request_id: str, created_at: int) -> Request:
    return Request(
        request_id=request_id,
        user_id="u1",
        session_id="session",
        created_at=created_at,
        source="test",
        agent_version="v1",
    )


def _archive_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_archive_flag_off_preserves_existing_delete_behavior(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REFLEXIO_RETENTION_ARCHIVE", raising=False)
    monkeypatch.delenv("REFLEXIO_RETENTION_ARCHIVE_DIR", raising=False)
    monkeypatch.setattr(
        storage,
        "_retention_guard",
        lambda: (_ for _ in ()).throw(AssertionError("off path acquired guard")),
    )
    archive_dir = Path(storage.db_path).parent / "archive"  # type: ignore[attr-defined]
    storage.add_user_interaction("u1", _interaction(1, "req1"))

    assert storage.delete_oldest_retention_target_rows("interactions", 1) == 1  # type: ignore[attr-defined]
    assert not archive_dir.exists()


def test_archive_ceiling_stops_deletion_without_exceeding_limit(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES", "1")
    storage.add_user_interaction("u1", _interaction(1, "req1"))

    with pytest.raises(RetentionArchiveFullError):
        storage.delete_oldest_retention_target_rows("interactions", 1)  # type: ignore[attr-defined]

    assert {item.interaction_id for item in storage.get_all_interactions(limit=10)} == {
        1
    }
    archive_dir = Path(storage.db_path).parent / "archive"  # type: ignore[attr-defined]
    assert sum(path.stat().st_size for path in archive_dir.glob("*.jsonl")) <= 1


def test_automatic_cleanup_treats_archive_ceiling_as_nonfatal(
    storage: BaseStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES", "1")
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.get_row_retention_limits",
        lambda: {"interactions": 1},
    )
    storage.add_user_interaction("u1", _interaction(1, "req1"))
    service = GenerationService.__new__(GenerationService)
    service.org_id = "archive-ceiling-test"
    service.storage = storage
    monkeypatch.setattr(service, "_should_check_retention_target", lambda *_: True)

    service._cleanup_storage_tables_if_needed()

    assert {item.interaction_id for item in storage.get_all_interactions(limit=10)} == {
        1
    }
    assert "live-row trimming was stopped" in caplog.text


def test_archive_preserves_deleted_rows_without_embeddings(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "TrUe")
    original_ids = {1, 2, 3}
    for interaction_id in original_ids:
        storage.add_user_interaction(
            "u1", _interaction(interaction_id, f"req{interaction_id}")
        )

    assert storage.delete_oldest_retention_target_rows("interactions", 2) == 2  # type: ignore[attr-defined]

    path = Path(storage.db_path).parent / "archive" / "interactions.jsonl"  # type: ignore[attr-defined]
    records = _archive_records(path)
    archived_ids = {record["row"]["interaction_id"] for record in records}  # type: ignore[index]
    live_ids = {
        interaction.interaction_id
        for interaction in storage.get_all_interactions(limit=10)
    }
    assert archived_ids | live_ids == original_ids
    assert all(record["table"] == "interactions" for record in records)
    assert all(isinstance(record["archived_at"], int) for record in records)
    assert all("embedding" not in record["row"] for record in records)  # type: ignore[operator]


def test_over_limit_cleanup_archives_before_automatic_retention(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.get_row_retention_limits",
        lambda: {"interactions": 2},
    )
    original_ids = {1, 2, 3}
    for interaction_id in original_ids:
        storage.add_user_interaction(
            "u1", _interaction(interaction_id, f"req{interaction_id}")
        )
    service = GenerationService.__new__(GenerationService)
    service.org_id = "archive-cleanup-test"
    service.storage = storage
    monkeypatch.setattr(service, "_should_check_retention_target", lambda *_: True)

    service._cleanup_storage_tables_if_needed()

    archive_path = Path(storage.db_path).parent / "archive" / "interactions.jsonl"  # type: ignore[attr-defined]
    archived_ids = {
        record["row"]["interaction_id"] for record in _archive_records(archive_path)
    }
    live_ids = {
        interaction.interaction_id
        for interaction in storage.get_all_interactions(limit=10)
    }
    assert archived_ids | live_ids == original_ids
    assert len(archived_ids) == 1


def test_large_over_limit_cleanup_caps_live_rows_and_preserves_removed_rows(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES", str(10 * 1024**2))
    monkeypatch.setattr(
        "reflexio.server.services.generation_service.get_row_retention_limits",
        lambda: {"interactions": 1000},
    )
    original_ids = set(range(1, 1201))
    for interaction_id in original_ids:
        storage.add_user_interaction(
            "u1", _interaction(interaction_id, f"req{interaction_id}")
        )
    service = GenerationService.__new__(GenerationService)
    service.org_id = "large-archive-cleanup-test"
    service.storage = storage
    monkeypatch.setattr(service, "_should_check_retention_target", lambda *_: True)

    service._cleanup_storage_tables_if_needed()

    archive_path = Path(storage.db_path).parent / "archive" / "interactions.jsonl"  # type: ignore[attr-defined]
    archived_ids = {
        record["row"]["interaction_id"] for record in _archive_records(archive_path)
    }
    live_ids = {
        interaction.interaction_id
        for interaction in storage.get_all_interactions(limit=2000)
    }
    assert len(live_ids) == 960
    assert len(archived_ids) == 240
    assert archived_ids.isdisjoint(live_ids)
    assert archived_ids | live_ids == original_ids


def test_archive_includes_cascade_deleted_rows(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "1")
    for index in range(1, 4):
        request_id = f"req{index}"
        storage.add_request(_request(request_id, index))
        storage.add_user_interaction("u1", _interaction(index, request_id))

    assert storage.delete_oldest_retention_target_rows("requests", 2) == 2  # type: ignore[attr-defined]

    archive_dir = Path(storage.db_path).parent / "archive"  # type: ignore[attr-defined]
    request_records = _archive_records(archive_dir / "requests.jsonl")
    interaction_records = _archive_records(archive_dir / "interactions.jsonl")
    assert {record["row"]["request_id"] for record in request_records} == {  # type: ignore[index]
        "req1",
        "req2",
    }
    assert {record["row"]["request_id"] for record in interaction_records} == {  # type: ignore[index]
        "req1",
        "req2",
    }
    assert {item.request_id for item in storage.get_all_interactions(limit=10)} == {
        "req3"
    }


def test_archive_and_delete_hold_one_sqlite_writer_guard(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    storage.add_request(_request("req1", 1))
    storage.add_user_interaction("u1", _interaction(1, "req1"))
    cascade_fetched = threading.Event()
    writer_attempting = threading.Event()
    writer_done = threading.Event()
    writer_errors: list[BaseException] = []
    original_fetch = storage._retention_fetch_rows  # type: ignore[attr-defined]
    writer_storage = SQLiteStorage(
        org_id="concurrent-writer",
        db_path=storage.db_path,  # type: ignore[attr-defined]
    )

    def observed_fetch(
        table_name: str,
        key_columns: tuple[str, ...],
        keys: list[tuple[object, ...]],
    ) -> list[dict[str, object]]:
        rows = original_fetch(table_name, key_columns, keys)
        if table_name == "interactions":
            cascade_fetched.set()
            assert writer_attempting.wait(timeout=1)
            time.sleep(0.05)
            assert not writer_done.is_set()
        return rows

    monkeypatch.setattr(storage, "_retention_fetch_rows", observed_fetch)

    def add_late_child() -> None:
        try:
            assert cascade_fetched.wait(timeout=1)
            writer_attempting.set()
            writer_storage.add_user_interaction("u1", _interaction(99, "req1"))
        except BaseException as exc:  # pragma: no cover - asserted in main thread
            writer_errors.append(exc)
        finally:
            writer_done.set()

    writer = threading.Thread(target=add_late_child)
    writer.start()
    storage.delete_oldest_retention_target_rows("requests", 1)  # type: ignore[attr-defined]
    writer.join(timeout=1)
    writer_storage.conn.close()

    assert not writer_errors
    assert writer_done.is_set()
    assert {item.interaction_id for item in storage.get_all_interactions(limit=10)} == {
        99
    }


def test_archive_failure_prevents_deletion(
    storage: BaseStorage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    storage.add_user_interaction("u1", _interaction(1, "req1"))

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "reflexio.server.services.storage.retention_mixin.append_archive_rows", fail
    )
    with pytest.raises(OSError, match="disk full"):
        storage.delete_oldest_retention_target_rows("interactions", 1)  # type: ignore[attr-defined]

    assert {item.interaction_id for item in storage.get_all_interactions(limit=10)} == {
        1
    }
    assert "Failed to archive retention rows" in caplog.text


def test_archive_directory_override_is_respected(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_dir = tmp_path / "custom-archive"
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE", "true")
    monkeypatch.setenv("REFLEXIO_RETENTION_ARCHIVE_DIR", str(archive_dir))
    storage.add_user_interaction("u1", _interaction(1, "req1"))

    storage.delete_oldest_retention_target_rows("interactions", 1)  # type: ignore[attr-defined]

    assert (archive_dir / "interactions.jsonl").is_file()
    assert not (Path(storage.db_path).parent / "archive").exists()  # type: ignore[attr-defined]
