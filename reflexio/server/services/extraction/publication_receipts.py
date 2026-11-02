"""Minimal immutable publication receipts; financial facts are not source state."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

PublicationOrigin = Literal["initial_inline", "initial_window", "resume"]


def output_publication_origin(output: Any) -> PublicationOrigin:
    """Read the actual writer's origin; claim counts and run names cannot infer it."""
    publication = output.get("publication") if isinstance(output, dict) else None
    origin = publication.get("origin") if isinstance(publication, dict) else None
    if origin not in ("initial_inline", "initial_window", "resume"):
        raise ValueError("Original publication origin is unavailable")
    return origin


ENTITY_PIPELINES = {"profile": "profile", "user_playbook": "playbook"}
TOKEN_FACT_FIELDS = frozenset(
    {
        "input_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
    }
)


def canonical_receipt_id(value: Any) -> str:
    try:
        return uuid.UUID(str(value)).hex
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Invalid publication receipt identity") from error


def require_output_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or canonical_receipt_id(value) != value
    ):
        raise ValueError("Invalid publication output identity")
    return value


def require_learning_ids(value: Any, *, legacy: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError("Invalid publication learning identities")
    if not legacy and len(set(value)) != len(value):
        raise ValueError("Duplicate publication learning identity")
    return list(value)


def require_financial_facts(kind: str, entity_type: str, value: Any) -> dict[str, Any]:
    if entity_type not in ENTITY_PIPELINES or not isinstance(value, dict):
        raise ValueError("Invalid publication financial facts")
    if value.get("pipeline") != ENTITY_PIPELINES[entity_type]:
        raise ValueError("Publication financial pipeline changed")
    stamp = value.get("created_at")
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
        raise ValueError("Invalid publication financial timestamp")
    # Arbitrarily large Python integers need not be converted to float. Native
    # receipts use the same finite double-precision timestamp representation.
    try:
        valid_stamp = math.isfinite(stamp) and stamp >= 0
    except OverflowError:
        valid_stamp = False
    if not valid_stamp:
        raise ValueError("Invalid publication financial timestamp")
    if kind == "output":
        if set(value) != {"pipeline", "source", "created_at"} or (
            value["source"] is not None and not isinstance(value["source"], str)
        ):
            raise ValueError("Invalid output publication financial facts")
    elif kind == "window_handoff":
        if (
            set(value) != {"pipeline", "created_at", "platform_llm", *TOKEN_FACT_FIELDS}
            or type(value["platform_llm"]) is not bool
        ):
            raise ValueError("Invalid window financial facts")
        if any(
            type(value[key]) is not int or value[key] < 0 for key in TOKEN_FACT_FIELDS
        ):
            raise ValueError("Invalid window token facts")
    else:
        raise ValueError("Historical receipts have no new financial facts")
    return dict(value)


def window_handoff_facts(kind: str, billing: Any) -> tuple[list[str], dict[str, Any]]:
    """Project the original window snapshot without retaining subject metadata."""
    entity_type = {"profile": "profile", "playbook": "user_playbook"}.get(kind)
    if entity_type is None or not isinstance(billing, dict):
        raise ValueError("Invalid window billing snapshot")
    required = {
        "ids",
        "created_at",
        "platform_llm",
        "input_tokens",
        "prompt_tokens",
        "completion_tokens",
    }
    cache_fields = {"cache_read_input_tokens", "cache_write_input_tokens"}
    if not required <= billing.keys() or billing.keys() - required - cache_fields:
        raise ValueError("Invalid window billing snapshot fields")
    ids = require_learning_ids(billing["ids"])
    # Cache counters were absent in predecessor snapshots. This is the same
    # compatibility rule used by the existing window strict emitter.
    facts = require_financial_facts(
        "window_handoff",
        entity_type,
        {
            "pipeline": kind,
            "created_at": billing["created_at"],
            "platform_llm": billing["platform_llm"],
            **{key: billing.get(key, 0) for key in TOKEN_FACT_FIELDS},
        },
    )
    return ids, facts


def window_handoff_matches(
    kind: Any, billing_json: Any, ids_json: Any, facts_json: Any
) -> int:
    """SQLite deletion guard: a key alone cannot prove financial transfer."""
    try:
        ids, facts = window_handoff_facts(kind, json.loads(billing_json))
        return int(ids == json.loads(ids_json) and facts == json.loads(facts_json))
    except (ValueError, TypeError, OverflowError):
        return 0


def valid_receipt_payload(
    kind: Any, entity_type: Any, ids_json: Any, state: Any, facts_json: Any
) -> int:
    """SQLite CHECK adapter for the same typed receipt data contract."""
    try:
        ids = require_learning_ids(json.loads(ids_json))
        if entity_type not in ENTITY_PIPELINES:
            return 0
        if kind not in {"output", "window_handoff"}:
            return 0
        if state == "accepted":
            return int(facts_json is None)
        if state != "pending" or facts_json is None:
            return 0
        if kind == "output" and not ids:
            return 0
        require_financial_facts(kind, entity_type, json.loads(facts_json))
        return 1
    except (ValueError, TypeError, OverflowError):
        return 0


@dataclass(frozen=True)
class PublicationReceipt:
    receipt_id: str
    org_id: str
    project_id: str
    run_id: str | None
    output_id: str | None
    row_kind: str
    entity_type: str
    learning_ids: tuple[str, ...]
    delivery_owner: str
    delivery_state: str
    delivery_key: str | None
    _facts_json: str | None = field(repr=False)

    @property
    def financial_facts(self) -> dict[str, Any] | None:
        # Each caller receives a copy; a delivery callback cannot rewrite the
        # observed immutable descriptor used to acknowledge the actual row.
        return json.loads(self._facts_json) if self._facts_json is not None else None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PublicationReceipt:
        ids = row["learning_ids"]
        if isinstance(ids, str):
            ids = json.loads(ids)
        ids = require_learning_ids(ids)
        facts = row["delivery_facts"]
        if isinstance(facts, str):
            facts = json.loads(facts)
        encoded = (
            json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if facts is not None
            else None
        )
        if not valid_receipt_payload(
            row["row_kind"],
            row["entity_type"],
            json.dumps(ids),
            row["delivery_state"],
            encoded,
        ):
            raise ValueError("Stored publication receipt is invalid")
        return cls(
            canonical_receipt_id(row["receipt_id"]),
            row["org_id"],
            row["project_id"],
            row["run_id"],
            row["output_id"],
            row["row_kind"],
            row["entity_type"],
            tuple(ids),
            row["delivery_owner"],
            row["delivery_state"],
            row["delivery_key"],
            encoded,
        )
