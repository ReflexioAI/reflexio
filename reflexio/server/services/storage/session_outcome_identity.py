"""Canonical identities for immutable session outcomes."""

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TypedDict

from reflexio.server.services.playbook.publication import canonical_json_bytes

__all__ = [
    "CanonicalSessionTrajectory",
    "CanonicalTrajectoryDigestResult",
    "CanonicalTrajectoryDigestAccumulator",
    "OUTCOME_ALLOWED_VALUES",
    "OUTCOME_FINALIZATION_RULE",
    "OUTCOME_SCHEMA_VERSION",
    "canonical_json_bytes",
    "canonical_session_trajectory",
    "canonical_trajectory_bytes",
    "outcome_contract_digest",
    "trajectory_digest",
]

MAX_CANONICAL_TRAJECTORY_JSON_DEPTH = 100
OUTCOME_SCHEMA_VERSION = 1
OUTCOME_FINALIZATION_RULE = "first_write"
OUTCOME_ALLOWED_VALUES = ("success", "failure", "unknown")


class CanonicalRequest(TypedDict):
    request_id: str
    user_id: str
    created_at: str
    source: str
    agent_version: str
    session_id: str
    evaluation_only: bool
    retrieval_experiment_id: str | None
    retrieval_experiment_arm: str | None


class CanonicalInteraction(TypedDict):
    interaction_id: int
    user_id: str
    request_id: str
    created_at: str
    content: str
    role: str
    token_count: int | None
    user_action: str
    user_action_description: str
    interacted_image_url: str
    image_encoding: str
    shadow_content: str
    expert_content: str
    tools_used: object
    citations: object
    retrieved_learnings: object


class CanonicalTrajectoryRequest(TypedDict):
    request: CanonicalRequest
    interactions: list[CanonicalInteraction]


class CanonicalSessionTrajectory(TypedDict):
    session_id: str
    requests: list[CanonicalTrajectoryRequest]


@dataclass(frozen=True)
class CanonicalTrajectoryDigestResult:
    """Digest and bounded context derived from one ordered trajectory stream."""

    digest: str
    first_request: dict[str, object] | None
    request_count: int


def _canonical_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _canonical_json_column(value: object, *, default: object) -> object:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _canonical_int(value: object) -> int:
    if not isinstance(value, int | str):
        raise TypeError("canonical integer field must be an integer or integer text")
    return int(value)


def _canonical_interaction(
    interaction: Mapping[str, object],
) -> CanonicalInteraction:
    return {
        "interaction_id": _canonical_int(interaction["interaction_id"]),
        "user_id": str(interaction["user_id"]),
        "request_id": str(interaction["request_id"]),
        "created_at": _canonical_timestamp(interaction["created_at"]),
        "content": str(interaction["content"]),
        "role": str(interaction["role"]),
        "token_count": (
            _canonical_int(interaction["token_count"])
            if interaction["token_count"] is not None
            else None
        ),
        "user_action": str(interaction["user_action"]),
        "user_action_description": str(interaction["user_action_description"] or ""),
        "interacted_image_url": str(interaction["interacted_image_url"] or ""),
        "image_encoding": str(interaction["image_encoding"] or ""),
        "shadow_content": str(interaction["shadow_content"] or ""),
        "expert_content": str(interaction["expert_content"] or ""),
        "tools_used": _canonical_json_column(interaction["tools_used"], default=[]),
        "citations": _canonical_json_column(interaction["citations"], default=[]),
        "retrieved_learnings": _canonical_json_column(
            interaction["retrieved_learnings"], default=[]
        ),
    }


def _canonical_request(row: Mapping[str, object]) -> CanonicalRequest:
    return {
        "request_id": str(row["request_id"]),
        "user_id": str(row["user_id"]),
        "created_at": _canonical_timestamp(row["created_at"]),
        "source": str(row["source"] or ""),
        "agent_version": str(row["agent_version"] or ""),
        "session_id": str(row["session_id"]),
        "evaluation_only": bool(row["evaluation_only"]),
        "retrieval_experiment_id": (
            str(row["retrieval_experiment_id"])
            if row["retrieval_experiment_id"] is not None
            else None
        ),
        "retrieval_experiment_arm": (
            str(row["retrieval_experiment_arm"])
            if row["retrieval_experiment_arm"] is not None
            else None
        ),
    }


def canonical_session_trajectory(
    session_id: str,
    request_rows: Sequence[Mapping[str, object]],
    interactions_by_request: Mapping[str, Sequence[Mapping[str, object]]],
) -> CanonicalSessionTrajectory:
    """Project adapter-specific durable rows into one trajectory identity shape."""
    requests: list[CanonicalTrajectoryRequest] = []
    for row in request_rows:
        request_id = str(row["request_id"])
        request = _canonical_request(row)
        interactions = [
            _canonical_interaction(interaction)
            for interaction in interactions_by_request.get(request_id, ())
        ]
        requests.append({"request": request, "interactions": interactions})
    return {"session_id": session_id, "requests": requests}


def outcome_contract_digest(
    *,
    source: str,
    schema_version: int | str,
    allowed_values: Collection[str],
    finalization_rule: str,
) -> str:
    """Hash a server-owned structured outcome contract."""
    payload = {
        "allowed_values": sorted(set(allowed_values)),
        "finalization_rule": finalization_rule,
        "schema_version": schema_version,
        "source": source,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _canonical_trajectory_json(value: object, *, depth: int = 0) -> str:
    """Encode trajectory JSON deterministically within a bounded nesting depth."""
    if (
        isinstance(value, tuple | list | Mapping)
        and depth >= MAX_CANONICAL_TRAJECTORY_JSON_DEPTH
    ):
        raise ValueError("canonical trajectory JSON exceeds maximum depth")
    if isinstance(value, float):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    if isinstance(value, tuple | list):
        return (
            "["
            + ",".join(
                _canonical_trajectory_json(item, depth=depth + 1) for item in value
            )
            + "]"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical trajectory object keys must be strings")
        keys = sorted(value, key=lambda key: key.encode("utf-16be"))
        return (
            "{"
            + ",".join(
                f"{canonical_json_bytes(key).decode()}:"
                f"{_canonical_trajectory_json(value[key], depth=depth + 1)}"
                for key in keys
            )
            + "}"
        )
    return canonical_json_bytes(value).decode()


def canonical_trajectory_bytes(trajectory: object) -> bytes:
    """Encode a finalized session trajectory to its canonical UTF-8 bytes."""
    return _canonical_trajectory_json(trajectory).encode("utf-8")


def _canonical_trajectory_bytes_at_depth(value: object, *, depth: int) -> bytes:
    return _canonical_trajectory_json(value, depth=depth).encode("utf-8")


class CanonicalTrajectoryDigestAccumulator:
    """Hash one complete canonical trajectory while retaining only its current row."""

    def __init__(self, session_id: str) -> None:
        self._digest = sha256()
        self._digest.update(b'{"requests":[')
        self._session_id = session_id
        self._request: CanonicalRequest | None = None
        self._request_count = 0
        self._interaction_count = 0
        self._hexdigest: str | None = None
        self._poisoned = False

    def _raise_if_poisoned(self) -> None:
        if self._poisoned:
            raise RuntimeError("canonical trajectory digest accumulator is invalid")

    def start_request(self, row: Mapping[str, object]) -> None:
        self._raise_if_poisoned()
        try:
            if self._hexdigest is not None:
                raise RuntimeError("canonical trajectory digest is already finalized")
            if self._request is not None:
                raise RuntimeError("previous canonical request is not finished")
            request = _canonical_request(row)
            if self._request_count:
                self._digest.update(b",")
            self._digest.update(b'{"interactions":[')
            self._request = request
            self._interaction_count = 0
        except Exception:
            self._poisoned = True
            raise

    def add_interaction(self, row: Mapping[str, object]) -> None:
        self._raise_if_poisoned()
        try:
            if self._request is None:
                raise RuntimeError("canonical interaction has no active request")
            if str(row["request_id"]) != self._request["request_id"]:
                raise ValueError(
                    "canonical interaction does not belong to active request"
                )
            encoded = _canonical_trajectory_bytes_at_depth(
                _canonical_interaction(row), depth=4
            )
            if self._interaction_count:
                self._digest.update(b",")
            self._digest.update(encoded)
            self._interaction_count += 1
        except Exception:
            self._poisoned = True
            raise

    def finish_request(self) -> None:
        self._raise_if_poisoned()
        try:
            if self._request is None:
                raise RuntimeError("canonical trajectory has no active request")
            encoded = _canonical_trajectory_bytes_at_depth(self._request, depth=3)
            self._digest.update(b'],"request":')
            self._digest.update(encoded)
            self._digest.update(b"}")
            self._request = None
            self._request_count += 1
        except Exception:
            self._poisoned = True
            raise

    def hexdigest(self) -> str:
        self._raise_if_poisoned()
        try:
            if self._request is not None:
                raise RuntimeError("canonical trajectory has an unfinished request")
            if self._hexdigest is None:
                encoded_session_id = _canonical_trajectory_bytes_at_depth(
                    self._session_id, depth=1
                )
                self._digest.update(b'],"session_id":')
                self._digest.update(encoded_session_id)
                self._digest.update(b"}")
                self._hexdigest = self._digest.hexdigest()
            return self._hexdigest
        except Exception:
            self._poisoned = True
            raise


def trajectory_digest(trajectory: object) -> str:
    """Hash the canonical finalized session trajectory."""
    return sha256(canonical_trajectory_bytes(trajectory)).hexdigest()
