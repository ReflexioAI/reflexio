"""Versioned shared-compressor records in the existing fenced checkpoint owner.

This module owns serialization and transitions, never connections or dispatch.
The logical selection bound is separate from the unchanged physical row bound.
"""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.server.services.input_compression import CompressionChoice

from .manifests import canonical_json_bytes

PROTOCOL = "shared-session-compaction-v1"
PROTOCOL_KEY = "session_compaction_protocol"
MAX_SELECTION_BYTES = 1024 * 1024
PART_BYTES = 12_000
MAX_PARTS = (MAX_SELECTION_BYTES + PART_BYTES - 1) // PART_BYTES
ROW_BYTES = 22_000
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Root(RecordModel):
    kind: Literal["root"] = "root"
    source_digest: Digest
    plan_witness: Digest
    request_contract_digest: Digest
    limit: int = Field(ge=1, le=128)
    calls: int = Field(ge=0, le=128)

    @model_validator(mode="after")
    def allowance(self) -> Root:
        if self.calls > self.limit:
            raise ValueError("Shared compaction allowance exceeded")
        return self


class Response(RecordModel):
    model: str = Field(min_length=1, max_length=256)
    provider: str | None = Field(default=None, max_length=256)
    content_digest: Digest
    finish_reason: str | None = Field(default=None, max_length=64)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class Attempt(RecordModel):
    kind: Literal["attempt"] = "attempt"
    root: Digest
    ordinal: int = Field(ge=1, le=128)
    node: Digest
    request_digest: Digest
    response: Response | None = None


class Selection(RecordModel):
    kind: Literal["selection"] = "selection"
    root: Digest
    node: Digest
    input_refs_digest: Digest
    request_contract_digest: Digest
    body_digest: Digest
    body_bytes: int = Field(ge=1, le=MAX_SELECTION_BYTES)
    parts: int = Field(ge=1, le=MAX_PARTS)
    attempt: Digest
    response_digest: Digest

    @model_validator(mode="after")
    def geometry(self) -> Selection:
        if self.parts != (self.body_bytes + PART_BYTES - 1) // PART_BYTES:
            raise ValueError("Shared selection part count differs from byte length")
        return self


class Part(RecordModel):
    kind: Literal["part"] = "part"
    root: Digest
    node: Digest
    body_digest: Digest
    index: int = Field(ge=0, lt=MAX_PARTS)
    data: str = Field(
        min_length=1, max_length=16_000, pattern=r"^[A-Za-z0-9+/]*={0,2}$"
    )


class SharedReceipt(RecordModel):
    protocol: Literal["shared-session-compaction-v1"] = PROTOCOL
    identity: Digest
    # Existing column: request count for a root, ordinal for an attempt, 1 for
    # immutable choice records. The legacy rolling protocol is unchanged.
    batches: int = Field(ge=1, le=128)
    complete: bool
    payload: Annotated[Root | Attempt | Selection | Part, Field(discriminator="kind")]

    @model_validator(mode="after")
    def columns(self) -> SharedReceipt:
        value = self.payload
        expected = (
            max(value.calls, 1)
            if isinstance(value, Root)
            else (value.ordinal if isinstance(value, Attempt) else 1)
        )
        if (
            self.batches != expected
            or (
                isinstance(value, Attempt)
                and self.complete != (value.response is not None)
            )
            or (isinstance(value, Selection | Part) and not self.complete)
        ):
            raise ValueError("Shared checkpoint state/columns differ")
        if len(canonical_json_bytes(self.model_dump(mode="json"))) > ROW_BYTES:
            raise ValueError("Shared checkpoint exceeds physical row bound")
        return self


def digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def receipt(
    identity: str, payload: Root | Attempt | Selection | Part, *, complete: bool
) -> dict[str, Any]:
    batches = (
        max(payload.calls, 1)
        if isinstance(payload, Root)
        else (payload.ordinal if isinstance(payload, Attempt) else 1)
    )
    return SharedReceipt(
        identity=identity, batches=batches, complete=complete, payload=payload
    ).model_dump(mode="json")


def validate_advance(old: SharedReceipt, new: SharedReceipt) -> None:
    if old.identity != new.identity or type(old.payload) is not type(new.payload):
        raise ValueError("Shared checkpoint identity or record kind changed")
    if old.complete:
        raise ValueError("Completed shared checkpoint is immutable")
    before, after = old.payload, new.payload
    if isinstance(before, Root) and isinstance(after, Root):
        if before.model_copy(
            update={"calls": after.calls}
        ) != after or after.calls not in (before.calls, before.calls + 1):
            raise ValueError("Shared checkpoint cannot refund or skip reservations")
        if after.calls == before.calls and not new.complete:
            raise ValueError("Shared checkpoint did not advance")
    elif isinstance(before, Attempt) and isinstance(after, Attempt):
        if (
            before.response is not None
            or after.response is None
            or before.model_copy(update={"response": after.response}) != after
        ):
            raise ValueError("Shared attempt observation changed its reservation")
    else:
        raise ValueError("Shared selection records are immutable")


def encode_selection(choice: dict[str, Any]) -> tuple[bytes, tuple[str, ...]]:
    # Check before schema traversal, part allocation or any storage write.
    blocks, size = [], 0
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    for chunk in encoder.iterencode(choice):
        if len(chunk) > MAX_SELECTION_BYTES:
            raise ValueError("Shared selection exceeds the 1 MiB persistence limit")
        block = chunk.encode()
        size += len(block)
        if size > MAX_SELECTION_BYTES:
            raise ValueError("Shared selection exceeds the 1 MiB persistence limit")
        blocks.append(block)
    body = b"".join(blocks)
    CompressionChoice.model_validate(choice)
    if canonical_json_bytes(choice) != body:
        raise ValueError("Shared selection is not canonical JSON")
    return body, tuple(
        base64.b64encode(body[i : i + PART_BYTES]).decode("ascii")
        for i in range(0, len(body), PART_BYTES)
    )


def decode_selection(manifest: Selection, parts: list[Part]) -> dict[str, Any]:
    if len(parts) != manifest.parts:
        raise ValueError("Shared selection has missing parts")
    blocks = []
    for index, part in enumerate(parts):
        if (part.root, part.node, part.body_digest, part.index) != (
            manifest.root,
            manifest.node,
            manifest.body_digest,
            index,
        ):
            raise ValueError("Shared selection part belongs to another source")
        block = base64.b64decode(part.data, validate=True)
        expected = min(PART_BYTES, manifest.body_bytes - index * PART_BYTES)
        if len(block) != expected:
            raise ValueError("Shared selection part has the wrong byte length")
        blocks.append(block)
    body = b"".join(blocks)
    if sha256(body).hexdigest() != manifest.body_digest:
        raise ValueError("Shared selection body digest changed")
    value = json.loads(body)
    CompressionChoice.model_validate(value)
    return value


def protocol_from_metadata(metadata: dict[str, Any]) -> str | None:
    value = metadata.get(PROTOCOL_KEY)
    if PROTOCOL_KEY in metadata and value != PROTOCOL:
        raise ValueError("Unknown frozen session compaction protocol")
    return value
