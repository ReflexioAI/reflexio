"""Storage-owned accepted-access evidence for native profile ranking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from reflexio.server.services.profile.lifecycle import (
    attention_context,
    has_attention_context,
    metadata,
    whole_epoch,
)

ATTENTION_BATCH_SIZE = 256
_RECORD: ContextVar[bool] = ContextVar("profile_access_recording", default=True)


def bounded_ids(user_id: str, profile_ids: Sequence[str]) -> list[str]:
    """Validate owner/IDs; bound each SQL batch rather than limiting ancestry."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Profile attention requires an explicit owner")
    if any(not isinstance(identity, str) or not identity for identity in profile_ids):
        raise ValueError("Profile attention identities must be nonempty strings")
    return sorted(set(profile_ids))


@dataclass
class _Snapshot:
    storage: Any
    user_id: str | None = None
    values: dict[str, dict[str, int | None]] = field(default_factory=dict)


_SNAPSHOT: ContextVar[_Snapshot | None] = ContextVar(
    "profile_access_snapshot", default=None
)


@contextmanager
def access_recording(enabled: bool) -> Iterator[None]:
    """Assessment/dashboard reads do not become actual access evidence."""
    token = _RECORD.set(enabled)
    try:
        yield
    finally:
        _RECORD.reset(token)


def recording_enabled() -> bool:
    return _RECORD.get()


@contextmanager
def storage_snapshot(storage: Any, user_id: str | None = None) -> Iterator[None]:
    if has_attention_context():
        snapshot_token = _SNAPSHOT.set(None)
        try:
            with access_recording(False):
                yield
        finally:
            _SNAPSHOT.reset(snapshot_token)
        return
    token = _SNAPSHOT.set(_Snapshot(storage, user_id))
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


@contextmanager
def candidate_attention(items: Sequence[Any]) -> Iterator[None]:
    snapshot = _SNAPSHOT.get()
    if snapshot is None:
        yield  # Preserve an explicitly supplied studied assessment snapshot.
        return
    owners: dict[str, set[str]] = {}
    for item in items:
        profile = getattr(item, "item", item)
        value = metadata(profile)
        if value["created_at"] is None:
            continue
        owners.setdefault(profile.user_id, set()).update(
            [profile.profile_id, *value["ancestor_ids"]]
        )
    if snapshot.user_id is not None and set(owners) - {snapshot.user_id}:
        raise ValueError("Profile attention candidates differ from the requested owner")
    if len(owners) > 1:
        raise ValueError("Profile attention candidate arm crossed owners")
    epochs: dict[str, int] = {}
    for user_id, identities in owners.items():
        ids = bounded_ids(user_id, sorted(identities))
        cached = snapshot.values.setdefault(user_id, {})
        missing = [identity for identity in ids if identity not in cached]
        from reflexio.server import search_runtime

        for offset in range(0, len(missing), ATTENTION_BATCH_SIZE):
            search_runtime.checkpoint()
            batch = missing[offset : offset + ATTENTION_BATCH_SIZE]
            found = snapshot.storage.get_profile_access_epochs(
                user_id=user_id, profile_ids=batch
            )
            if set(found) - set(batch):
                raise ValueError(
                    "Profile attention storage returned unrequested identities"
                )
            for epoch in found.values():
                whole_epoch(epoch)
            cached.update({identity: found.get(identity) for identity in batch})
        search_runtime.checkpoint()
        for identity in ids:
            epoch = cached[identity]
            if epoch is not None:
                epochs[identity] = epoch
    with attention_context(epochs):
        yield


def record_accepted_profiles(
    storage: Any, profiles: Sequence[Any], *, user_id: str | None = None
) -> None:
    """Register exact final IDs, never pools; HTTP acceptance is not client ACK.

    Embedded callers have a synchronous acceptance boundary. They count as
    actual access unless enclosed in access_recording(False) or an explicit
    studied attention_context (an assessment snapshot).
    """
    from datetime import UTC, datetime

    from reflexio.server import search_runtime
    from reflexio.server.services.profile.lifecycle import has_attention_context

    if not recording_enabled() or has_attention_context():
        return
    owners: dict[str, list[str]] = {}
    for profile in profiles:
        owners.setdefault(profile.user_id, []).append(profile.profile_id)
    if len(owners) > 1 or (user_id is not None and set(owners) - {user_id}):
        raise ValueError("Profile accepted access crossed the requested owner")
    qualified = {owner: bounded_ids(owner, ids) for owner, ids in owners.items()}
    if not qualified:
        return

    def record() -> None:
        accessed_at = int(datetime.now(UTC).timestamp())
        for owner, ids in qualified.items():
            storage.record_served_profile_access(
                user_id=owner, profile_ids=ids, accessed_at=accessed_at
            )

    search_runtime.on_response_accepted("profile_attention", record)
