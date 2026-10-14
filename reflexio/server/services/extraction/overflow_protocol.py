"""Pure, bounded witnesses for the inactive source-package journal protocol.

This module neither plans input nor admits work. The future runtime caller must
rebuild and compare the native plan/contract witnesses and validate selected IDs
against the original source. Digests here bind those proofs; they do not replace
them. No provider text, credentials, or duplicate transcript is stored here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

STAGES = ("compress", "gate", "profile_agent", "profile_dedup", "playbook_intent")
MAX_SELECTION_BYTES = 64 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_HOST_OMISSION = re.compile(
    r"Host discarded [1-9][0-9]* entire groups with unseen references after bounded repair; source selection is incomplete\.\Z"
)
_CAPACITY_OMISSION = "Host: Selection still exceeds capacity after bounded repair; whole points may be omitted."


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def package_stage_caps(consumer: Any) -> dict[str, int]:
    """Freeze actual resolved ladder lengths, including supported fallbacks."""
    _require(isinstance(consumer, dict), "Invalid package consumer")
    _require(
        set(consumer) == {"kind", "gate_enabled", "ladders"}
        and type(consumer["gate_enabled"]) is bool,
        "Invalid package consumer contract",
    )
    ladders = consumer["ladders"]
    _require(
        isinstance(ladders, dict)
        and set(ladders) == {"gate", "agent", "dedup", "intent"},
        "Invalid resolved package ladders",
    )
    for models in ladders.values():
        _require(
            isinstance(models, list)
            and len(models) <= 128
            and all(
                isinstance(model, str) and 0 < len(model) <= 256 for model in models
            )
            and len(models) == len(set(models)),
            "Invalid resolved package ladder",
        )
    caps: dict[str, int] = dict.fromkeys(STAGES, 0)
    if consumer["kind"] == "profile":
        _require(
            bool(ladders["agent"])
            and bool(ladders["dedup"])
            and not ladders["intent"]
            and (not consumer["gate_enabled"] or bool(ladders["gate"])),
            "Invalid profile package ladders",
        )
        caps["gate"] = len(ladders["gate"]) if consumer["gate_enabled"] else 0
        caps["profile_agent"] = 16 * len(ladders["agent"])
        caps["profile_dedup"] = 2 * len(ladders["dedup"])
    else:
        _require(
            consumer["kind"] == "source_quoted_playbook"
            and not consumer["gate_enabled"]
            and not any(ladders[key] for key in ("gate", "agent", "dedup"))
            and bool(ladders["intent"]),
            "Invalid source-quoted package ladders",
        )
        caps["playbook_intent"] = 2 * len(ladders["intent"])
    follow_on = sum(caps.values())
    _require(follow_on <= 128, "Package consumer exceeds durable quota")
    caps["compress"] = 128 - follow_on
    return caps


def validate_package_protocol(protocol: Any) -> dict[str, Any]:
    _require(isinstance(protocol, dict), "Invalid package descriptor")
    _require(
        set(protocol)
        == {
            "name",
            "version",
            "plan_digest",
            "source_cache_digest",
            "scope",
            "contract_digest",
            "section_digests",
            "consumer",
            "stage_caps",
        }
        and protocol["name"] == "source-package"
        and type(protocol["version"]) is int
        and protocol["version"] == 1,
        "Unknown package descriptor",
    )
    _require(
        all(
            _digest(protocol[key])
            for key in ("plan_digest", "source_cache_digest", "contract_digest")
        ),
        "Invalid frozen package digest",
    )
    scope = protocol["scope"]
    _require(
        isinstance(scope, dict)
        and set(scope)
        == {
            "org_id",
            "project_id",
            "user_id",
            "window_id",
            "request_id",
            "agent_version",
            "kind",
        }
        and all(
            isinstance(value, str) and len(value) <= 512
            for key, value in scope.items()
            if key != "agent_version"
        )
        and (
            scope["agent_version"] is None
            or isinstance(scope["agent_version"], str)
            and len(scope["agent_version"]) <= 512
        )
        and all(scope[key] for key in ("org_id", "user_id", "window_id", "request_id"))
        and scope["kind"] in {"profile", "playbook"},
        "Invalid frozen package scope",
    )
    sections = protocol["section_digests"]
    _require(
        isinstance(sections, list)
        and len(sections) <= 48
        and all(_digest(value) for value in sections)
        and len(sections) == len(set(sections)),
        "Invalid frozen package sections",
    )
    caps = package_stage_caps(protocol["consumer"])
    supplied = protocol["stage_caps"]
    _require(
        isinstance(supplied, dict)
        and set(supplied) == set(STAGES)
        and all(_integer(value) for value in supplied.values())
        and supplied == caps,
        "Frozen package stage caps changed",
    )
    _require(
        (scope["kind"] == "profile") == (protocol["consumer"]["kind"] == "profile"),
        "Package consumer scope changed",
    )
    _require(
        max(0, 2 * len(sections) - 1) <= caps["compress"],
        "Package geometry exceeds durable quota",
    )
    return protocol


def validate_package_scope(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    if manifest["version"] == 2:
        scope = manifest["input_protocol"]["scope"]
        _require(
            all(scope[key] == value for key, value in expected.items()),
            "Package parent scope changed",
        )


def validate_selection_receipt(
    key: str, receipt: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Check a server-normalized native cache choice, never authored facts."""
    _require(
        set(receipt)
        == {
            "kind",
            "protocol_version",
            "source_digest",
            "plan_digest",
            "node_key",
            "input_refs_digest",
            "output",
            "request_contract_digest",
            "attempt_ids",
        }
        and receipt["kind"] == "source_selection"
        and type(receipt["protocol_version"]) is int
        and receipt["protocol_version"] == 1,
        "Invalid source selection receipt",
    )
    _require(
        receipt["source_digest"] == manifest["source_digest"]
        and receipt["plan_digest"] == manifest["input_protocol"]["plan_digest"]
        and receipt["node_key"] == key
        and all(
            _digest(receipt[field])
            for field in ("node_key", "input_refs_digest", "request_contract_digest")
        ),
        "Source selection binding changed",
    )
    ids = receipt["attempt_ids"]
    attempts = {attempt["id"]: attempt for attempt in manifest["attempts"]}
    _require(
        isinstance(ids, list)
        and 0 < len(ids) <= 128
        and all(isinstance(identity, str) for identity in ids)
        and len(ids) == len(set(ids))
        and all(
            identity in attempts and attempts[identity]["stage"] == "compress"
            for identity in ids
        ),
        "Source selection lacks durable compression attempts",
    )
    output = receipt["output"]
    _require(
        isinstance(output, dict)
        and set(output)
        == {"overview", "points", "unresolved", "omissions", "omission_counts"}
        and output["overview"] == ""
        and output["unresolved"] == [],
        "Invalid normalized source selection",
    )
    points = output["points"]
    _require(
        isinstance(points, list) and len(points) <= 128,
        "Invalid source selection groups",
    )
    for point in points:
        _require(
            isinstance(point, dict)
            and set(point) == {"category", "detail", "evidence_refs"}
            and point["category"] == "selected_evidence"
            and point["detail"] == "Original evidence."
            and isinstance(point["evidence_refs"], list)
            and bool(point["evidence_refs"])
            and all(
                isinstance(ref, str) and 0 < len(ref) <= 1024
                for ref in point["evidence_refs"]
            ),
            "Invalid source selection references",
        )
    counts = output["omission_counts"]
    _require(
        isinstance(counts, dict)
        and set(counts) == {"capacity_points", "unseen_reference_groups"}
        and all(_integer(value) for value in counts.values()),
        "Invalid local source omission counts",
    )
    omissions = output["omissions"]
    _require(
        isinstance(omissions, list)
        and len(omissions) <= 32
        and all(
            isinstance(value, str)
            and len(value) <= 512
            and (
                value == _CAPACITY_OMISSION
                or _HOST_OMISSION.fullmatch(value) is not None
            )
            for value in omissions
        ),
        "Invalid host source omission diagnostics",
    )
    _require(
        len(
            json.dumps(
                receipt, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
        )
        <= MAX_SELECTION_BYTES,
        "Source selection receipt exceeds durable size bound",
    )


def validate_package_manifest(manifest: dict[str, Any]) -> None:
    _require(
        set(manifest)
        == {
            "version",
            "revision",
            "source_digest",
            "attempt_limit",
            "attempts",
            "checkpoint",
            "preparation_version",
            "input_protocol",
        }
        and manifest.get("preparation_version") == 4
        and type(manifest.get("preparation_version")) is int
        and _integer(manifest["revision"])
        and _digest(manifest["source_digest"])
        and "legacy_observed_usage" not in manifest
        and "chunk_digests" not in manifest
        and "input_chunk_tokens" not in manifest,
        "Invalid source-package journal discriminator",
    )
    protocol = validate_package_protocol(manifest.get("input_protocol"))
    attempts = manifest["attempts"]
    _require(
        all(
            isinstance(attempt, dict)
            and set(attempt) == {"id", "stage", "usage", "model_provenance"}
            and attempt.get("stage") in STAGES
            for attempt in attempts
        ),
        "Unattributed source-package attempt",
    )
    counts = Counter(attempt["stage"] for attempt in attempts)
    _require(
        all(counts[stage] <= cap for stage, cap in protocol["stage_caps"].items()),
        "Source-package stage quota exceeded",
    )
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or "repartition" in checkpoint:
        raise ValueError("Package cannot use legacy repartition")
    steps = checkpoint.get("completed_steps")
    if not isinstance(steps, dict) or len(steps) > 128:
        raise ValueError("Invalid package steps")
    claimed: set[str] = set()
    for key, receipt in steps.items():
        _require(
            isinstance(key, str) and bool(key) and isinstance(receipt, dict),
            "Invalid package receipt",
        )
        if receipt.get("kind") == "source_selection":
            validate_selection_receipt(key, receipt, manifest)
            _require(
                not claimed.intersection(receipt["attempt_ids"]),
                "Compression attempt already bound to another receipt",
            )
            claimed.update(receipt["attempt_ids"])
        else:
            _require(
                set(receipt) == {"kind", "output", "model_provenance"}
                and receipt["kind"] == "model_output",
                "Unknown package receipt kind",
            )
