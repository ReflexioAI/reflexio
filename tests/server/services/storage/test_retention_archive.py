"""Integration contract for archive-before-delete retention."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserActionType,
)
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


def _archive_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_archive_flag_off_preserves_existing_delete_behavior(
    storage: BaseStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REFLEXIO_RETENTION_ARCHIVE", raising=False)
    monkeypatch.delenv("REFLEXIO_RETENTION_ARCHIVE_DIR", raising=False)
    archive_dir = Path(storage.db_path).parent / "archive"  # type: ignore[attr-defined]
    storage.add_user_interaction("u1", _interaction(1, "req1"))

    assert storage.delete_oldest_retention_target_rows("interactions", 1) == 1  # type: ignore[attr-defined]
    assert not archive_dir.exists()


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
