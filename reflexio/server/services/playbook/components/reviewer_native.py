"""Native playbook-review evidence contract; binding is not semantic proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import PlaybookReviewReasonCode
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.playbook.components.reviewer import (
    CandidateEvidenceUnit,
    CandidateReviewDecision,
    CandidateRevision,
    PlaybookCandidateReviewer,
    PlaybookCandidateReviewOutput,
)

if TYPE_CHECKING:
    from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
    from reflexio.server.services.playbook.components.reviewer import (
        PlaybookReviewOutcome,
    )


class WireModel(StrictStructuredOutput):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OriginalQuote(WireModel):
    field: Literal["content", "trigger", "rationale"]
    text: str = Field(
        min_length=1,
        strict=True,
        description="Verbatim substring of the selected ORIGINAL candidate field, not a conversation/source excerpt or revision paraphrase.",
    )


class ActionQuote(WireModel):
    """Positive original operation/scope; extractor rationale is not an action."""

    field: Literal["content", "trigger"]
    text: str = Field(
        min_length=1,
        strict=True,
        description="Verbatim substring of the selected ORIGINAL candidate content or trigger. User/source wording is valid here only if it also literally occurs in that original field.",
    )


class SourceQuote(WireModel):
    evidence_id: str = Field(min_length=1, strict=True)
    text: str = Field(
        min_length=1,
        strict=True,
        description="Verbatim substring of the own conversation unit identified by evidence_id. This is source evidence, not a quotation of the ORIGINAL candidate field.",
    )


class PositiveInputFacts(WireModel):
    supply_status: Literal["established"]
    original_parameter_spans: tuple[ActionQuote, ...] = Field(min_length=1)
    supply_excerpts: tuple[SourceQuote, ...] = Field(min_length=1)
    use_correspondence: Literal["complete"]
    original_use_span: ActionQuote
    use_excerpts: tuple[SourceQuote, ...] = Field(min_length=1)


class PositiveCoreWitness(WireModel):
    kind: Literal[
        "user_learning", "assigned_input_use", "necessary_procedure", "observed_failure"
    ]
    original_spans: tuple[ActionQuote, ...] = Field(min_length=1, max_length=16)
    excerpts: tuple[SourceQuote, ...] = Field(min_length=1, max_length=16)
    input_facts: PositiveInputFacts | None


class StoredMatch(WireModel):
    stored_id: str = Field(min_length=1, strict=True)
    content: str = Field(min_length=1, strict=True)
    trigger: str | None = Field(strict=True)


class ActionPremise(WireModel):
    kind: Literal["user_instruction", "assigned_input", "task_assignment"]
    witness_index: int = Field(ge=0, strict=True)
    source: SourceQuote
    original_span: ActionQuote


class SourceFactPremise(WireModel):
    kind: Literal["prerequisite", "reported_limitation"]
    witness_index: int = Field(ge=0, strict=True)
    source: SourceQuote
    fact: str = Field(
        min_length=1,
        strict=True,
        description="One concise source-supported factual premise at its actual scope, related to the complete retained action. A literal source quote does not certify this paraphrase. No execution history, inferred preference, unobserved cause, or broader absence conclusion.",
    )


class SurvivorRevision(WireModel):
    content: str = Field(min_length=1, strict=True)
    trigger: str = Field(min_length=1, strict=True)
    premise: ActionPremise | SourceFactPremise = Field(discriminator="kind")


class SurvivorDecision(WireModel):
    id: str = Field(min_length=1, strict=True)
    decision: Literal["accept", "revise", "reject"]
    reason_code: PlaybookReviewReasonCode
    defect_anchor: OriginalQuote | None
    survivor_witnesses: tuple[PositiveCoreWitness, ...] = Field(max_length=16)
    stored_match: StoredMatch | None
    evidence_ids: tuple[str, ...]
    revision: SurvivorRevision | None
    reason: None


class SurvivorReviewOutput(WireModel):
    decisions: tuple[SurvivorDecision, ...]


@dataclass(frozen=True)
class BoundReviewContext:
    """Canonical bytes from owned preparation, including ordered context/profile."""

    canonical_json: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


def bind_context(
    *,
    originals: dict[str, UserPlaybook],
    units: dict[str, list[CandidateEvidenceUnit]],
    ordered_context: str,
    catalog: dict[str, UserPlaybook],
    profile: str,
) -> BoundReviewContext:
    """Freeze actual prepared data; this factory cannot authenticate forged input."""
    if set(originals) != set(units):
        raise ValueError("Preparation candidate mismatch")
    for candidate_id, own_units in units.items():
        if len({u.evidence_id for u in own_units}) != len(own_units):
            raise ValueError("Duplicate prepared source id")
        if any(
            u.interaction_id not in originals[candidate_id].source_interaction_ids
            for u in own_units
        ):
            raise ValueError("Prepared source outside original citation")
    payload = {
        "originals": {k: v.model_dump(mode="json") for k, v in originals.items()},
        "units": {k: [asdict(u) for u in v] for k, v in units.items()},
        "ordered_context": ordered_context,
        "catalog": {k: v.model_dump(mode="json") for k, v in catalog.items()},
        "profile": profile,
    }
    return BoundReviewContext(json.dumps(payload, sort_keys=True, ensure_ascii=False))


class UnsupportedPremiseModeError(ValueError):
    """A construction fault, never a semantic assertion that no core exists."""


def project(
    raw: SurvivorReviewOutput,
    *,
    prepared: BoundReviewContext,
    current: BoundReviewContext,
) -> PlaybookCandidateReviewOutput:
    """Check mechanical bindings; semantic source approval remains a separate gate.

    Does not mutate, save, reclassify, normalize or certify the actual raw DTO.
    No legacy assessment is fabricated. Public reason prose remains null.
    """
    if prepared != current:
        raise ValueError("Original/source/context/catalog/profile binding changed")
    data = json.loads(prepared.canonical_json)
    originals = data["originals"]
    if len(raw.decisions) != len(originals) or {d.id for d in raw.decisions} != set(
        originals
    ):
        raise ValueError("Exact candidate accounting required")
    public = PlaybookCandidateReviewOutput(
        decisions=[_project_decision(decision, data) for decision in raw.decisions]
    )
    canonical_units = {
        candidate_id: [CandidateEvidenceUnit(**unit) for unit in units]
        for candidate_id, units in data["units"].items()
    }
    errors = PlaybookCandidateReviewer._validation_errors(public, canonical_units)
    if errors:
        raise ValueError("; ".join(errors))
    return public


def _project_decision(
    decision: SurvivorDecision, data: dict[str, Any]
) -> CandidateReviewDecision:
    original = data["originals"][decision.id]
    own = {u["evidence_id"]: u for u in data["units"][decision.id]}
    retained = set(decision.evidence_ids)
    if len(retained) != len(decision.evidence_ids) or not retained.issubset(own):
        raise ValueError("Unique own retained source ids required")

    def original_quote(
        quote: OriginalQuote | ActionQuote, *, defect: bool = False
    ) -> None:
        if (
            (quote.field == "rationale" and not defect)
            or not quote.text.strip()
            or quote.text not in (original.get(quote.field) or "")
        ):
            raise ValueError(
                "Quote the supplied candidate original content/trigger/rationale "
                "literally for the selected field. Conversation/User wording "
                "belongs in source excerpts, not original_spans, original "
                "parameter/use spans, premise.original_span or defect_anchor, "
                "unless it also literally occurs in that original field. "
                "Do not paraphrase or substitute revision text."
            )

    def source_quote(quote: SourceQuote, *, user: bool = False) -> None:
        unit = own.get(quote.evidence_id)
        if (
            unit is None
            or not quote.text.strip()
            or quote.text not in unit["source_span"]
            or (decision.decision != "reject" and quote.evidence_id not in retained)
        ):
            raise ValueError("Exact canonical own retained source required")
        if user and unit["role"].casefold() != "user":
            raise ValueError(
                "User-learning authorization excerpts must cite actual User units only. "
                "Agent or Tool event/status units may remain in evidence_ids as context, "
                "but are not User authorization excerpts. Acceptance must still retain all own evidence_ids."
            )

    _validate_shape(decision, retained, data["catalog"])
    if decision.defect_anchor is not None:
        original_quote(decision.defect_anchor, defect=True)
    for witness in decision.survivor_witnesses:
        for span in witness.original_spans:
            original_quote(span)
        for excerpt in witness.excerpts:
            source_quote(excerpt, user=witness.kind == "user_learning")
        facts = witness.input_facts
        if witness.kind == "assigned_input_use" and facts is None:
            raise ValueError("Assigned-input witness requires positive facts")
        if facts is not None:
            for span in (*facts.original_parameter_spans, facts.original_use_span):
                original_quote(span)
            for excerpt in (*facts.supply_excerpts, *facts.use_excerpts):
                source_quote(excerpt)

    revision = None
    if decision.decision == "revise":
        if decision.revision is None:
            raise UnsupportedPremiseModeError(
                "SurvivorRevision requires represented source premise"
            )
        draft = decision.revision
        premise = draft.premise
        if premise.witness_index >= len(decision.survivor_witnesses):
            raise ValueError("Premise witness reference outside candidate")
        witness = decision.survivor_witnesses[premise.witness_index]
        if isinstance(premise, ActionPremise):
            original_quote(premise.original_span)
        source_quote(premise.source)
        _validate_user_premise_role(premise, own[premise.source.evidence_id]["role"])
        rationale = _render_premise(premise, witness)
        revision = CandidateRevision(
            content=draft.content, trigger=draft.trigger, rationale=rationale
        )
    elif decision.revision is not None:
        raise ValueError("Only revisions construct new guidance")
    return CandidateReviewDecision(
        id=decision.id,
        decision=decision.decision,
        reason_code=decision.reason_code,
        evidence_ids=list(decision.evidence_ids),
        revision=revision,
        reason=None,
    )


def _validate_user_premise_role(
    premise: ActionPremise | SourceFactPremise, role: str
) -> None:
    if (
        premise.kind in {"user_instruction", "assigned_input"}
        and role.casefold() != "user"
    ):
        if premise.kind == "assigned_input":
            raise ValueError(
                "revision.premise.kind=assigned_input asserts User-supplied input "
                "and requires an actual User source. For an actual non-User task "
                "assignment, choose task_assignment with the same complete truthful "
                "positive input_facts. A provider default, event or mere availability "
                "does not establish task assignment. This role mismatch does not "
                "prove Generic or no eligible core: preserve any source-supported "
                "complete original core and its highest original defect; do not "
                "discard them solely to repair the premise mode."
            )
        raise ValueError(
            "user_instruction requires an actual User authorization excerpt. "
            "A Tool requirement or result is not a User request. If its actual "
            "source and complete positive witness establish a prerequisite or "
            "reported limitation, use that truthful source premise instead. "
            "A role mismatch alone proves neither Generic nor absence of an "
            "eligible original core, and cannot lower the original defect."
        )


def _validate_shape(
    decision: SurvivorDecision, retained: set[str], catalog: dict[str, Any]
) -> None:
    if decision.reason_code == "generic" and decision.survivor_witnesses:
        raise ValueError("Generic contradicts a positive eligible core")
    if decision.reason_code == "redundant" and decision.decision != "reject":
        raise ValueError("An actual duplicate is rejected with stored proof")
    if (
        decision.reason_code in {"late_trigger", "compound"}
        and decision.decision != "revise"
    ):
        raise ValueError(
            "Pure trigger or compound defect requires a witnessed revision"
        )
    fatal = decision.reason_code in {"not_agent_decision", "absence_inference"}
    if fatal and (
        decision.decision != "reject"
        or decision.survivor_witnesses
        or decision.stored_match is not None
    ):
        raise ValueError(
            "not_agent_decision and absence_inference require decision=reject, "
            "survivor_witnesses=[], evidence_ids=[], stored_match=null and "
            "revision=null, even if another clause has true supported guidance. "
            "Keep the original fatal reason and exact defect_anchor. "
            "Discarded core is unassessed; do not deny a real User instruction."
        )
    if decision.decision == "accept":
        if decision.reason_code != "grounded_useful" or decision.defect_anchor:
            raise ValueError("Acceptance cannot declare an original defect")
    else:
        if decision.reason_code == "grounded_useful":
            raise ValueError("Grounded useful is acceptance only")
        if decision.reason_code != "redundant" and decision.defect_anchor is None:
            raise ValueError("Original defect anchor required")
    if decision.decision != "reject" and not decision.survivor_witnesses:
        raise ValueError("Every survivor requires positive original/source witnesses")
    if decision.decision == "reject":
        if retained or decision.revision is not None:
            raise ValueError("Rejection emits no guidance or retained evidence")
        if decision.reason_code == "redundant" and decision.stored_match is None:
            raise ValueError("Duplicate rejection requires actual stored match")
        if decision.survivor_witnesses and decision.stored_match is None:
            raise ValueError("Positive nonfatal core requires revision or stored match")
    elif not retained:
        raise ValueError("Survivor retains own source ids")
    if decision.stored_match is not None:
        match = decision.stored_match
        stored = catalog.get(match.stored_id)
        if (
            decision.decision != "reject"
            or not decision.survivor_witnesses
            or stored is None
            or match.content != stored["content"]
            or match.trigger != stored["trigger"]
        ):
            raise ValueError("Exact actual stored entry and positive core required")


def _render_premise(
    premise: ActionPremise | SourceFactPremise, witness: PositiveCoreWitness
) -> str:
    if isinstance(premise, SourceFactPremise):
        allowed = (
            {"necessary_procedure"}
            if premise.kind == "prerequisite"
            else {"necessary_procedure", "observed_failure"}
        )
        if (
            witness.kind not in allowed
            or premise.source not in witness.excerpts
            or not premise.fact.strip()
        ):
            raise ValueError(
                "Source fact requires matching positive witness and own excerpt"
            )
        prefix = (
            "The recorded source establishes this prerequisite: "
            if premise.kind == "prerequisite"
            else "The recorded source reports this limitation: "
        )
        rationale = prefix + premise.fact
    elif premise.kind == "user_instruction":
        if premise.source not in witness.excerpts:
            raise ValueError("Instruction premise must reference its positive witness")
        rationale = f"The user requested this procedure: {premise.original_span.text}"
    else:
        facts = witness.input_facts
        if (
            facts is None
            or premise.original_span not in facts.original_parameter_spans
            or premise.source not in facts.supply_excerpts
        ):
            raise ValueError("Assignment premise requires actual positive supply proof")
        prefix = (
            "For the recorded task, the user supplied a value for this input: "
            if premise.kind == "assigned_input"
            else "The recorded task has an assigned value for this input: "
        )
        rationale = prefix + premise.original_span.text
    if not rationale.endswith((".", "!", "?")):
        rationale += "."
    return rationale


_DISPATCH_FIELDS = (
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "extra_body",
    "response_format",
    "output_config",
    "reasoning_effort",
)


def _dispatch_settings(
    reviewer: PlaybookCandidateReviewer,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Reuse pure SDK policy/transport helpers, without credentials or budget work."""
    from reflexio.server.env_utils import env_str
    from reflexio.server.llm._model_compat import apply_model_request_policy
    from reflexio.server.llm.model_defaults import default_max_tokens_for_model

    client = reviewer.client
    try:
        seed = int(env_str("REFLEXIO_LLM_SEED", "42"))
    except ValueError:
        seed = 42
    expected: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 1.0 if client._is_temperature_restricted_model(model) else 0.0,
        "seed": seed,
    }
    cap = (
        reviewer.max_tokens
        if reviewer.max_tokens is not None
        else client.config.max_tokens
    )
    if cap is None:
        cap = default_max_tokens_for_model(model)
    if cap:
        expected["max_tokens"] = cap
    if client.config.top_p != 1.0:
        expected["top_p"] = client.config.top_p
    if reviewer.reasoning_effort is not None:
        expected["extra_body"] = {
            "reasoning_effort": reviewer.reasoning_effort,
            "thinking": {
                "type": "disabled" if reviewer.reasoning_effort == "none" else "enabled"
            },
        }
    client._apply_structured_output_transport(
        params=expected,
        messages=messages,
        response_format=SurvivorReviewOutput,
        model=model,
        strict_response_format=True,
        tools_available=False,
        allowed_openai_params=[],
    )
    apply_model_request_policy(expected)
    return expected


def _dispatch_identity(params: dict[str, Any]) -> dict[str, Any]:
    result = {key: params[key] for key in _DISPATCH_FIELDS if key in params}
    if isinstance(result.get("response_format"), type):
        result["response_format"] = result["response_format"].model_json_schema()
    result["endpoint_route"] = _endpoint_route(params.get("api_base"))
    result["api_version"] = params.get("api_version")
    return result


def _endpoint_route(
    value: str | None,
) -> tuple[str, str | None, int | None, str] | None:
    # Route identity excludes credentials, query strings and fragments.
    if not value:
        return None
    url = urlsplit(value)
    return url.scheme, url.hostname, url.port, url.path


def _dispatch_route(reviewer: PlaybookCandidateReviewer, model: str) -> dict[str, Any]:
    """Mirror the SDK's nonsecret endpoint precedence, without resolving keys."""
    client = reviewer.client
    config = client.config.api_key_config
    base, version = None, None
    if model == client.config.model:
        base, version = client._api_base, client._api_version
    elif config is not None:
        if config.custom_endpoint is not None:
            base = str(config.custom_endpoint.api_base)
        elif model.lower().startswith("dashscope/") and config.dashscope is not None:
            base = config.dashscope.api_base
        elif (
            model.lower().startswith("azure/")
            and config.openai is not None
            and config.openai.azure_config is not None
        ):
            base = str(config.openai.azure_config.endpoint)
            version = config.openai.azure_config.api_version
    if not base and model.lower().startswith("zai/"):
        base = (
            "https://api.z.ai/api/paas/v4"
            if model == "zai/glm-5.3-flashx"
            else "https://api.z.ai/api/coding/paas/v4"
        )
    return {"endpoint_route": _endpoint_route(base), "api_version": version or None}


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def _assert_dispatch(
    params: dict[str, Any],
    hard_timeout: float,
    ladder: tuple[str, ...],
    parse_structured_output: bool,
    *,
    dispatch_by_model: dict[str, dict[str, Any]],
    bound_profile: dict[str, Any],
) -> None:
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError

    model = params.get("model")
    expected = dispatch_by_model.get(model) if isinstance(model, str) else None
    if expected is None:
        raise ProviderRequestGuardError("Reviewer dispatched an unbound model")
    actual = _dispatch_identity(params)
    wanted = _dispatch_identity(expected)
    wanted.update(bound_profile["routes"][model])
    # JSON canonical binding turns route tuples into lists.
    actual["endpoint_route"] = (
        list(actual["endpoint_route"]) if actual["endpoint_route"] is not None else None
    )
    # Scoped extraction supplies an advertised output reservation only when
    # neither the caller nor provider declares a cap; it cannot replace one.
    if "max_tokens" not in wanted and "max_completion_tokens" not in wanted:
        actual.pop("max_tokens", None)
        actual.pop("max_completion_tokens", None)
    timeout = params.get("timeout")
    actual_messages = params.get("messages", [])
    expected_messages = expected.get("messages", [])
    if (
        actual != wanted
        or list(ladder) != bound_profile["ladder"]
        or parse_structured_output is not True
        or params.get("num_retries") != 0
        or params.get("fallbacks")
        or not isinstance(timeout, (float, int))
        or not 0 < timeout <= bound_profile["effective_timeouts"][model]
        or not 0 < hard_timeout <= timeout + bound_profile["hard_timeout_grace"]
        or len(actual_messages) < len(expected_messages)
        or any(
            actual_message.get("role") != expected_message.get("role")
            or _message_text(actual_message) != _message_text(expected_message)
            for actual_message, expected_message in zip(
                actual_messages, expected_messages, strict=False
            )
        )
    ):
        raise ProviderRequestGuardError(
            "Reviewer dispatch differs from bound inference profile"
        )


def review_native(
    reviewer: PlaybookCandidateReviewer,
    *,
    candidates: list[UserPlaybook],
    request_interaction_data_models: list[RequestInteractionDataModel],
    existing_playbooks: list[UserPlaybook],
    agent_context: str,
    playbook_definition: str,
    tool_context: str,
) -> PlaybookReviewOutcome:
    """Run the inactive witness contract through the existing SDK and checkpoint.

    Binding faults are fatal admission errors, outside the repairable model-value
    boundary. The raw parsed DTO is retained, never synthesized from projections.
    """
    from reflexio.server.llm._litellm_types import (
        ModelProvenance,
        ProviderRequestGuardError,
    )
    from reflexio.server.llm.model_defaults import ModelRole
    from reflexio.server.services.extraction.overflow_coordinator import (
        current_overflow_coordinator,
    )
    from reflexio.server.services.playbook.components.reviewer import (
        PlaybookReviewOutcome,
        _candidate_digest,
        _json_digest,
        _seal_outcome,
    )

    originals = {
        f"C{index}": candidate.model_copy(deep=True)
        for index, candidate in enumerate(candidates, 1)
    }
    original_digest = _candidate_digest(candidates)
    manager = reviewer.request_context.prompt_manager
    policy = manager.render_prompt(reviewer.PROMPT_ID, {})
    coordinator = current_overflow_coordinator.get()
    reviewer._format_existing(existing_playbooks)

    def prepare() -> tuple[
        BoundReviewContext, dict[str, list[CandidateEvidenceUnit]], list[dict[str, str]]
    ]:
        context, actual_agent_context, scope_messages = reviewer._review_context(
            candidates, request_interaction_data_models, agent_context
        )
        units = reviewer._candidate_evidence_units(candidates, context)
        client = reviewer.client
        # Whitelist nonsecret inference identity. Never dump APIKeyConfig or
        # credential values into prompts, caches, logs or repair messages.
        model = client._resolve_primary_model(None, ModelRole.GENERATION)
        profile = json.dumps(
            {
                "model": model,
                "fallback_models": list(client.config.fallback_models),
                "configured_timeout": client.config.timeout,
                "effective_timeouts": {
                    name: client._effective_timeout_for_model(name)
                    for name in [model, *client.config.fallback_models]
                },
                "hard_timeout_grace": client._hard_timeout_grace_seconds(),
                "configured_max_tokens": client.config.max_tokens,
                "max_tokens": reviewer.max_tokens,
                "requested_temperature": 0,
                "effective_temperatures": {
                    name: 1.0 if client._is_temperature_restricted_model(name) else 0.0
                    for name in client._resolve_ladder(model_role=ModelRole.GENERATION)
                },
                "ladder": client._resolve_ladder(model_role=ModelRole.GENERATION),
                "top_p": client.config.top_p,
                "reasoning_effort": reviewer.reasoning_effort,
                "policy": reviewer.request_context.prompt_manager.render_prompt(
                    reviewer.PROMPT_ID, {}
                ),
                "schema": SurvivorReviewOutput.model_json_schema(),
                "dispatch": {
                    name: _dispatch_identity(_dispatch_settings(reviewer, name, []))
                    for name in client._resolve_ladder(model_role=ModelRole.GENERATION)
                },
                "endpoint_route": _endpoint_route(client._api_base),
                "custom_endpoint_route": _endpoint_route(
                    str(client.config.api_key_config.custom_endpoint.api_base)
                )
                if client.config.api_key_config
                and client.config.api_key_config.custom_endpoint
                else None,
                "api_version": client._api_version,
                "routes": {
                    name: _dispatch_route(reviewer, name)
                    for name in client._resolve_ladder(model_role=ModelRole.GENERATION)
                },
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        ordered = json.dumps(
            {
                "interaction_context": context.text,
                "source_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            group.model_dump(mode="json")
                            for group in request_interaction_data_models
                        ],
                        sort_keys=True,
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest(),
                "agent_context": actual_agent_context,
                "playbook_definition": playbook_definition,
                "tool_context": tool_context,
                "scope_messages": scope_messages,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        bound = bind_context(
            originals={f"C{i}": c for i, c in enumerate(candidates, 1)},
            units=units,
            ordered_context=ordered,
            catalog={f"X{i}": p for i, p in enumerate(existing_playbooks, 1)},
            profile=profile,
        )
        return bound, units, scope_messages

    prepared, units, scope_messages = prepare()
    private = json.loads(prepared.canonical_json)
    context_data = json.loads(private["ordered_context"])
    # Only task evidence enters the model. IDs, vectors, notes and private
    # source/profile hashes remain in the binding rather than the prompt.
    source = {
        "chronology": context_data["interaction_context"],
        "agent_context": context_data["agent_context"],
        "playbook_definition": context_data["playbook_definition"],
        "tool_context": context_data["tool_context"],
        "candidates": [
            {
                "id": candidate_id,
                **{
                    key: row[key]
                    for key in ("content", "trigger", "rationale", "reader_angle")
                },
                "own_source_units": [
                    {
                        key: unit[key]
                        for key in (
                            "evidence_id",
                            "turn_ref",
                            "role",
                            "request_source",
                            "source_span",
                        )
                    }
                    for unit in private["units"][candidate_id]
                ],
            }
            for candidate_id, row in private["originals"].items()
        ],
        "existing": [
            {
                "id": stored_id,
                **{key: row[key] for key in ("content", "trigger", "rationale")},
            }
            for stored_id, row in private["catalog"].items()
        ],
    }
    messages = [
        {"role": "system", "content": policy},
        *scope_messages,
        {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
    ]

    dispatch_by_model = {
        model: _dispatch_settings(reviewer, model, messages)
        for model in json.loads(json.loads(prepared.canonical_json)["profile"])[
            "ladder"
        ]
    }
    bound_profile = json.loads(json.loads(prepared.canonical_json)["profile"])

    def assert_current() -> None:
        try:
            current, _, _ = prepare()
        except (ValueError, KeyError) as error:
            raise ProviderRequestGuardError(
                "Reviewer source binding unavailable"
            ) from error
        if (
            current != prepared
            or reviewer.request_context.prompt_manager.get_active_version(
                reviewer.PROMPT_ID
            )
            != "1.51.0"
        ):
            raise ProviderRequestGuardError(
                "Reviewer source/context/catalog/profile binding changed"
            )

    def admission(
        params: dict[str, Any],
        hard_timeout: float,
        ladder: tuple[str, ...],
        parse_structured_output: bool,
    ) -> None:
        assert_current()
        _assert_dispatch(
            params,
            hard_timeout,
            ladder,
            parse_structured_output,
            dispatch_by_model=dispatch_by_model,
            bound_profile=bound_profile,
        )
        if reviewer.provider_request_guard is not None:
            # SDK repair messages are already present at admission. The caller's
            # accounting hook may observe this request, but cannot change it.
            admitted = _json_digest(
                {
                    "settings": _dispatch_identity(params),
                    "messages": params.get("messages"),
                    "timeout": params.get("timeout"),
                }
            )
            reviewer.provider_request_guard(
                params, hard_timeout, ladder, parse_structured_output
            )
            if admitted != _json_digest(
                {
                    "settings": _dispatch_identity(params),
                    "messages": params.get("messages"),
                    "timeout": params.get("timeout"),
                }
            ):
                raise ProviderRequestGuardError(
                    "Reviewer admission hook changed the request"
                )
        # An existing budget/admission hook may change shared source or params.
        assert_current()
        _assert_dispatch(
            params,
            hard_timeout,
            ladder,
            parse_structured_output,
            dispatch_by_model=dispatch_by_model,
            bound_profile=bound_profile,
        )

    def validate(value: object) -> tuple[str, ...]:
        assert_current()
        if not isinstance(value, SurvivorReviewOutput):
            return ("Review requires the native survivor structured type",)
        try:
            project(value, prepared=prepared, current=prepared)
        except ValueError as error:
            return (str(error),)
        return ()

    options: dict[str, Any] = {
        "model_role": ModelRole.GENERATION,
        "max_retries": 0,
        "temperature": 0,
        "response_format": SurvivorReviewOutput,
        "parse_structured_output": True,
        "structured_output_validator": validate,
        "provider_request_guard": admission,
    }
    if reviewer.max_tokens is not None:
        options["max_tokens"] = reviewer.max_tokens
    if reviewer.reasoning_effort is not None:
        options["extra_body"] = {
            "reasoning_effort": reviewer.reasoning_effort,
            "thinking": {
                "type": "disabled" if reviewer.reasoning_effort == "none" else "enabled"
            },
        }
    review_key = None
    provenance = None
    assert_current()
    if coordinator is not None:
        review_key = coordinator.step_key(
            "review-native-v1",
            {
                "messages": messages,
                "source_binding_digest": prepared.digest,
            },
        )
        saved = coordinator.completed_step(review_key)
        if saved is None:
            coordinator.require_remaining_work(1)
            result = reviewer.client.generate_chat_response_with_provenance(
                messages, **options
            )
            raw, provenance = result.value, result.provenance
        else:
            raw = SurvivorReviewOutput.model_validate(saved["output"])
            provenance = ModelProvenance(**saved["model_provenance"])
    else:
        raw = reviewer.client.generate_chat_response(messages, **options)
    assert_current()
    errors = validate(raw)
    if errors:
        raise ValueError("; ".join(errors))
    if not isinstance(raw, SurvivorReviewOutput):
        raise ValueError("Reviewer returned the wrong native structured type")
    output = project(raw, prepared=prepared, current=prepared)
    if _candidate_digest(list(originals.values())) != original_digest:
        raise ProviderRequestGuardError("Reviewer original snapshot changed")
    if coordinator is not None and review_key is not None and provenance is not None:
        coordinator.complete_step(review_key, raw.model_dump(mode="json"), provenance)
        coordinator.review_models[coordinator.playbook_key(candidates[0])] = provenance
    return _seal_outcome(
        PlaybookReviewOutcome(
            output=output,
            units_by_candidate=units,
            native_output=raw,
            original_candidates_sha256=original_digest,
        )
    )
