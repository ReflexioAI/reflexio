"""One native selection plan with optional, request-local judge evidence.

Retrieval owners supply eligible candidates and native anchors from ONE snapshot.
This module does not retrieve, authorize, call a provider on its own, or publish
served results. Native scores/floors remain with their original owners. A judge
is an explicit callable dependency; absent means native selection, not fallback.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any

KINDS = ("profiles", "agent_playbooks", "user_playbooks")
Identity = str | int


@dataclass(frozen=True)
class Candidate:
    identity: Identity
    evidence: str


@dataclass(frozen=True, init=False)
class NativePlan:
    """Immutable minimal view; source/scope validation belongs to retrieval.

    Capture after existing suppression, native scoring and anchor selection, but
    before constructing/recording a served response. No playbook Content enters
    this view. Current comparison scope is at most20 candidates and5 anchors/type.
    """

    request: str
    pools: Mapping[str, tuple[Candidate, ...]]
    anchors: Mapping[str, tuple[Identity, ...]]

    def __init__(
        self,
        request: str,
        *,
        pools: Mapping[str, Sequence[Mapping[str, Any]]],
        anchors: Mapping[str, Sequence[Identity]],
    ) -> None:
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Resolved request required")
        if not isinstance(pools, Mapping) or set(pools) != set(KINDS):
            raise ValueError("All three typed pools required")
        if not isinstance(anchors, Mapping) or set(anchors) != set(KINDS):
            raise ValueError("All three native anchor lists required")
        frozen_pools: dict[str, tuple[Candidate, ...]] = {}
        frozen_anchors: dict[str, tuple[Identity, ...]] = {}
        for kind in KINDS:
            items = _sequence(pools[kind], maximum=20)
            field = "text" if kind == "profiles" else "trigger"
            seen: set[Identity] = set()
            candidates: list[Candidate] = []
            for item in items:
                if not isinstance(item, Mapping) or set(item) != {"id", field}:
                    raise ValueError("Only identity and permitted evidence allowed")
                identity, evidence = item["id"], item[field]
                if (
                    type(identity) not in (str, int)
                    or (isinstance(identity, str) and not identity.strip())
                    or (type(identity) is int and identity <= 0)
                    or identity in seen
                ):
                    raise ValueError("Invalid or duplicate candidate identity")
                if not isinstance(evidence, str) or not evidence.strip():
                    raise ValueError("Missing candidate evidence")
                seen.add(identity)
                candidates.append(Candidate(identity, evidence))
            native = _sequence(anchors[kind], maximum=5)
            if any(type(i) not in (str, int) for i in native):
                raise ValueError("Invalid anchor identity")
            if len(set(native)) != len(native) or not set(native) <= seen:
                raise ValueError(
                    "Unique original anchors inside candidate snapshot required"
                )
            frozen_pools[kind] = tuple(candidates)
            frozen_anchors[kind] = tuple(native)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "pools", MappingProxyType(frozen_pools))
        object.__setattr__(self, "anchors", MappingProxyType(frozen_anchors))

    def projection(self) -> dict[str, list[dict[str, Identity]]]:
        return {
            kind: [
                {
                    "id": c.identity,
                    "text" if kind == "profiles" else "trigger": c.evidence,
                }
                for c in self.pools[kind]
            ]
            for kind in KINDS
        }


def _sequence(value: object, *, maximum: int) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Candidate/anchor sequence required")
    if len(value) > maximum:
        raise ValueError("Candidate/anchor bound exceeded")
    return value


def _probability(value: object) -> Decimal:
    if type(value) not in (int, float) or not Decimal(str(value)).is_finite():
        raise ValueError("Finite numeric probability required")
    number = Decimal(str(value))
    if not 0 <= number <= 1:
        raise ValueError("Probability outside unit interval")
    return number


@dataclass(frozen=True)
class Evidence:
    gain: Decimal
    eligible: bool
    remove_anchor: bool


@dataclass(frozen=True)
class Selection:
    ids: Mapping[str, tuple[Identity, ...]]
    mode: str


def _evidence(
    response: object,
    plan: NativePlan,
    *,
    model: str,
    question_keys: frozenset[str],
) -> dict[str, dict[Identity, Evidence]]:
    """Validate the complete response once, before considering any pruning."""
    if not isinstance(response, Mapping):
        raise ValueError("Judge response object required")
    answers = response.get("answers")
    if response.get("model") != model or not isinstance(answers, dict):
        raise ValueError("Wrong judge response identity")
    if answers.keys() != question_keys:
        raise ValueError("Judge answer set differs from dispatched questions")
    evidence: dict[str, dict[Identity, Evidence]] = {}
    number = 0
    for kind in KINDS:
        evidence[kind] = {}
        for candidate in plan.pools[kind]:
            useful = answers[f"item_{number}_usefulness"]
            conflict = answers[f"item_{number}_conflict"]
            if not isinstance(useful, dict) or useful.get("type") != "score":
                raise ValueError("Ordinal Score answer required")
            if not isinstance(conflict, dict) or conflict.get("type") != "noul":
                raise ValueError("Explicit-conflict Noul answer required")
            values = useful.get("probabilities")
            if not isinstance(values, dict) or set(values) != {"0", "1", "2", "3"}:
                raise ValueError("Complete four-grade probabilities required")
            p = [_probability(values[str(i)]) for i in range(4)]
            if abs(sum(p) - 1) > Decimal("0.02"):
                raise ValueError("Rounded probability mass inconsistent")
            clash = _probability(conflict.get("noul"))
            evidence[kind][candidate.identity] = Evidence(
                gain=p[1] + 3 * p[2] + 7 * p[3],
                eligible=p[2] + p[3] >= Decimal("0.50") and clash < Decimal("0.50"),
                remove_anchor=p[0] + p[1] >= Decimal("0.95")
                or clash >= Decimal("0.90"),
            )
            number += 1
    return evidence


def select(
    plan: NativePlan, *, judge: Callable[[dict[str, Any]], object] | None = None
) -> Selection:
    """Select once; disabled/empty paths never construct a judge request.

    The supplied transport owns admission, timeout and raw response recording.
    An enabled transport/validation failure propagates: no hidden retry/provider
    fallback or failed-judge-as-successful-empty. PROFILE-only bypass is chosen by
    the caller's explicit deployment/eligibility policy, not inferred from a key.
    """
    if not isinstance(plan, NativePlan):
        raise TypeError("NativePlan required")
    if judge is None or not any(plan.pools.values()):
        return Selection(plan.anchors, "native" if judge is None else "native_empty")
    # Reuse the exact frozen questions, not their old three-layer decision stack.
    # Pure module import only: no provider SDK, environment or credentials.
    from reflexio.server.services.retrieval.judge_payload import payload

    sent = payload(plan.request, plan.projection())
    # The transport receives a mutable wire object. Bind validation identity
    # before the call so a mutating adapter cannot redefine what was requested.
    expected_model, expected_keys = sent["model"], frozenset(sent["questions"])
    response = judge(sent)
    evidence = _evidence(
        response, plan, model=expected_model, question_keys=expected_keys
    )
    selected: dict[str, tuple[Identity, ...]] = {}
    for kind in KINDS:
        records = evidence[kind]
        current = [i for i in plan.anchors[kind] if not records[i].remove_anchor]
        original = set(current)
        eligible = sorted(
            (
                (-records[c.identity].gain, index, str(c.identity), c.identity)
                for index, c in enumerate(plan.pools[kind])
                if records[c.identity].eligible
            )
        )[:5]
        for _, _, _, novel in eligible:
            if novel in original:
                continue
            if len(current) < 5:
                current.append(novel)
            else:
                worst = min(
                    range(len(current)), key=lambda i: (records[current[i]].gain, -i)
                )
                if records[novel].gain > records[current[worst]].gain:
                    current[worst] = novel
        selected[kind] = tuple(current)
    return Selection(MappingProxyType(selected), "enhanced")
