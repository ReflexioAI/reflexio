"""TTL-free profile metadata and attention, independent of factual validity.

This native policy has no scheduler or provider. The caller owns
actual-access history and must explicitly supply controlled whole-second clocks.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from reflexio.models.api_schema.common import NEVER_EXPIRES_TIMESTAMP

KEY = "profile_lifecycle_v1"
HALF_LIFE_SECONDS = 30 * 86400
_ACCESS: ContextVar[Mapping[str, int] | None] = ContextVar(
    "profile_actual_access", default=None
)


def whole_epoch(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Lifecycle clocks require nonnegative whole epoch seconds")
    return value


class _StorageDeadline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deadline_utc: str

    @field_validator("deadline_utc")
    @classmethod
    def exact_utc(cls, value: str) -> str:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value or parsed.timestamp() < 0:
            raise ValueError("Storage deadline must be exact UTC YYYY-MM-DDTHH:MM:SSZ")
        return value

    @property
    def epoch(self) -> int:
        return int(
            datetime.strptime(self.deadline_utc, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=UTC)
            .timestamp()
        )


class ExplicitStorageDeadline(_StorageDeadline):
    """Model-authored storage permission, never an estimated useful lifetime."""

    source_quote: str

    @field_validator("source_quote")
    @classmethod
    def nonempty_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Explicit storage deadline needs a verbatim source quote")
        return value


class _InheritedStorageDeadline(_StorageDeadline):
    """Code-owned proof reference: keep permission without copying forgotten text.

    Compiled only from an already validated stored permission during an exact
    partial erasure. This digest is an audit reference, not fresh source evidence
    or an independently sufficient authorization for any new deadline.
    """

    source_quote_sha256: str

    @field_validator("source_quote_sha256")
    @classmethod
    def exact_digest(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Inherited permission requires its exact source digest")
        return value


def storage_deadline(
    value: Mapping[str, Any],
) -> ExplicitStorageDeadline | _InheritedStorageDeadline:
    """Read stored permission; model authoring still requires a verbatim quote."""
    if "source_quote_sha256" in value:
        return _InheritedStorageDeadline.model_validate(value)
    return ExplicitStorageDeadline.model_validate(value)


def new_features(
    features: Mapping[str, Any] | None,
    *,
    now: int,
    deadline: ExplicitStorageDeadline | None,
    source_texts: Sequence[str] = (),
    forget_quote: str | None = None,
) -> dict[str, Any]:
    """Validate provenance membership, without classifying retention semantics."""
    if KEY in (features or {}):
        raise ValueError("Model output cannot author lifecycle clocks or ancestry")
    if deadline is not None and not any(
        deadline.source_quote in text for text in source_texts
    ):
        raise ValueError(
            "Explicit storage deadline quote is absent from original source"
        )
    if forget_quote is not None and (
        not isinstance(forget_quote, str)
        or not forget_quote.strip()
        or not any(forget_quote in text for text in source_texts)
    ):
        raise ValueError("Explicit forget quote is absent from original source")
    return {
        **(features or {}),
        KEY: {
            "created_at": whole_epoch(now),
            "ancestor_ids": [],
            "explicit_forget_quote": forget_quote,
            "explicit_storage_deadline": deadline.model_dump() if deadline else None,
        },
    }


def metadata(profile: Any) -> dict[str, Any]:
    features = profile.custom_features or {}
    if KEY not in features:
        # Existing visibility bounds are compatibility data, never permission.
        return {
            "created_at": None,
            "ancestor_ids": [],
            "explicit_forget_quote": None,
            "explicit_storage_deadline": None,
            "legacy_origin": True,
            "legacy_expiration_timestamp": profile.expiration_timestamp,
        }
    value = features[KEY]
    required = {"created_at", "ancestor_ids", "explicit_storage_deadline"}
    optional = {"explicit_forget_quote", "legacy_origin", "legacy_expiration_timestamp"}
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or set(value) - required - optional
    ):
        raise ValueError("Profile has malformed lifecycle metadata")
    quote = value.get("explicit_forget_quote")
    if quote is not None and (not isinstance(quote, str) or not quote.strip()):
        raise ValueError("Malformed explicit forget authority")
    legacy = value.get("legacy_origin", False)
    if type(legacy) is not bool:
        raise ValueError("Malformed legacy lifecycle origin")
    if legacy:
        if (
            value["created_at"] is not None
            or "legacy_expiration_timestamp" not in value
        ):
            raise ValueError("Unknown lifecycle origin requires its visibility bound")
        whole_epoch(value["legacy_expiration_timestamp"])
    else:
        whole_epoch(value["created_at"])
        if "legacy_expiration_timestamp" in value:
            raise ValueError("Qualified origin cannot carry an unknown legacy bound")
    ids = value["ancestor_ids"]
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Invalid lifecycle ancestry")
    if value["explicit_storage_deadline"] is not None:
        storage_deadline(value["explicit_storage_deadline"])
    return value


def merge_features(
    profiles: Sequence[Any],
    *,
    inherited_features: Mapping[str, Any] | None,
    existing_profiles: Sequence[Any],
    erase_replaced_payloads: bool = False,
) -> dict[str, Any]:
    """Conservative row inheritance: rewrites never refresh old attention.

    A merged row may contain several facts: keep the earliest creation and the
    earliest binding storage deadline. This intentionally does not assert
    independent fact-level clocks. Only actually persisted parents join ancestry.
    """
    if not profiles:
        raise ValueError("Lifecycle merge requires actual input profiles")
    owners = {p.user_id for p in profiles}
    if len(owners) != 1 or any(p.user_id not in owners for p in existing_profiles):
        raise ValueError("Lifecycle merge crossed profile owners")
    values = [metadata(p) for p in profiles]
    deadlines = [
        storage_deadline(v["explicit_storage_deadline"])
        for v in values
        if v["explicit_storage_deadline"] is not None
    ]
    earliest = min(deadlines, key=lambda d: d.epoch) if deadlines else None
    if erase_replaced_payloads and isinstance(earliest, ExplicitStorageDeadline):
        earliest = _InheritedStorageDeadline(
            deadline_utc=earliest.deadline_utc,
            source_quote_sha256=sha256(
                earliest.source_quote.encode("utf-8")
            ).hexdigest(),
        )
    ancestors = {identity for v in values for identity in v["ancestor_ids"]}
    ancestors.update(p.profile_id for p in existing_profiles)
    legacy_values = [v for v in values if v.get("legacy_origin")]
    compatibility = (
        {
            "legacy_origin": True,
            "legacy_expiration_timestamp": min(
                v["legacy_expiration_timestamp"] for v in legacy_values
            ),
        }
        if legacy_values
        else {}
    )
    return {
        **(inherited_features or {}),
        KEY: {
            "created_at": None
            if legacy_values
            else min(v["created_at"] for v in values),
            "ancestor_ids": sorted(ancestors),
            "explicit_forget_quote": None,
            "explicit_storage_deadline": earliest.model_dump() if earliest else None,
            **compatibility,
        },
    }


def expiration_for(features: Mapping[str, Any]) -> int:
    """Encode stop-at equality for native inclusive expiry filters.

    The expiry column is only a storage-permission filter here, never attention
    or factual validity. Filtering still does not establish physical deletion.
    """
    deadline = features[KEY]["explicit_storage_deadline"]
    explicit_bound = (
        storage_deadline(deadline).epoch - 1 if deadline else NEVER_EXPIRES_TIMESTAMP
    )
    return min(
        explicit_bound,
        features[KEY].get("legacy_expiration_timestamp", NEVER_EXPIRES_TIMESTAMP),
    )


@contextmanager
def attention_context(access_epochs: Mapping[str, int]) -> Iterator[None]:
    """Bind an immutable snapshot of one arm's actual served-access log.

    Assessment probes pass their saved per-arm map, but never append to it.
    Actual-access events append final served IDs only after response delivery.
    """
    snapshot = dict(access_epochs)
    if any(not isinstance(identity, str) or not identity for identity in snapshot):
        raise ValueError("Attention identities must be nonempty strings")
    for timestamp in snapshot.values():
        whole_epoch(timestamp)
    token = _ACCESS.set(snapshot)
    try:
        yield
    finally:
        _ACCESS.reset(token)


def has_attention_context() -> bool:
    """Whether the caller explicitly bound an immutable assessment/access map."""
    return _ACCESS.get() is not None


def attention_epoch(profile: Any, *, now: int) -> int:
    value = metadata(profile)
    now = whole_epoch(now)
    if value["created_at"] is None:
        return now  # No qualified origin exists from which to measure attention.
    times = [value["created_at"]]
    access = _ACCESS.get() or {}
    times.extend(
        access[i] for i in [profile.profile_id, *value["ancestor_ids"]] if i in access
    )
    if any(timestamp > now for timestamp in times):
        raise ValueError("Profile attention contains evidence from after the probe")
    return max(times)


def due_profile_ids(profiles: Sequence[Any], *, now: int) -> list[str]:
    """Plan identities with their own due authority, including inactive rows.

    This pure plan is not an erasure executor or proof of a complete census.
    Production-safe locked/versioned execution remains separately qualified.
    """
    whole_epoch(now)
    if len({p.user_id for p in profiles}) > 1:
        raise ValueError("Deadline census crossed profile owners")
    ids = set()
    for profile in profiles:
        value = metadata(profile)
        raw = value["explicit_storage_deadline"]
        if raw is not None and storage_deadline(raw).epoch <= now:
            ids.add(profile.profile_id)
    return sorted(ids)
