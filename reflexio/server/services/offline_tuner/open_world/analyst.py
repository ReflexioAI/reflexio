"""Role-separated production adapters for open-world evidence analysis."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import random
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal, Protocol, cast

from pydantic import ValidationError

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm import ProviderRequestGuardError
from reflexio.server.llm.failure import (
    OutageWait,
    Transient,
    classify_llm_failure,
    exception_chain,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ModelProvenance,
    is_structured_output_correction_turn,
    structured_output_correction_turn,
    structured_output_repair_idempotency_key,
)
from reflexio.server.llm.provider_circuit import ProviderCircuitOpenError
from reflexio.server.prompt.prompt_manager import PromptManager

from ..evidence.manifests import canonical_json_bytes
from .identity import (
    canonical_bytes,
    canonical_payload,
    discovery_model_payload,
    held_out_model_input,
)
from .models import (
    CandidateRevisionV1,
    ComponentIdentity,
    DiscoveryMemo,
    DiscoveryRequest,
    HeldOutSessionMemo,
    HeldOutSessionRequest,
    ObservedStructuredCompletion,
    OpenWorldCapabilityUnavailableError,
)
from .qualification import (
    QualificationCase,
    QualificationSuite,
)

type AdapterRole = Literal["discovery", "candidate", "held_out", "confirmation"]

_PROMPT_IDS: dict[AdapterRole, str] = {
    "discovery": "offline_tuner_open_world_discovery",
    "candidate": "offline_tuner_open_world_candidate",
    "held_out": "offline_tuner_open_world_held_out",
}
_OUTPUT_TYPES: dict[AdapterRole, type[StrictStructuredOutput]] = {
    "discovery": DiscoveryMemo,
    "candidate": CandidateRevisionV1,
    "held_out": HeldOutSessionMemo,
}
_OUTPUT_SCHEMA_IDS: dict[AdapterRole, str] = {
    "discovery": "offline_tuner_open_world_discovery_output",
    "candidate": "offline_tuner_open_world_candidate_output",
    "held_out": "offline_tuner_open_world_held_out_output",
}
# The roles that HAVE a production adapter here, which is narrower than
# `AdapterRole`.
#
# "confirmation" is a fourth qualified analyst role --
# `qualification.py`, `identity.py` and `models.py` all carry it -- but this
# module gives it no prompt id, no output type and no output schema id, because
# the confirmation memo is produced elsewhere. The three maps above are
# therefore deliberately three-keyed, and every lookup into them is walled by
# `_require_adapter_role` rather than left to raise a bare `KeyError`: a
# KeyError names a dict, reads as a defect in the caller, and says nothing
# about the role being unsupported here on purpose.
_ADAPTER_ROLES: frozenset[str] = frozenset(_PROMPT_IDS)
# Prompt versions are per ROLE, because the roles version independently: the
# discovery contract gained an explicit byte-for-byte citation clause in 1.1.0,
# per-span speaker attribution in 1.2.0 and weak-finding ranking in 1.3.0, while
# the candidate contract moved separately, to state the citation-ID equality rule
# in 1.1.0 and then to read those IDs from the trusted plaintext block in 1.2.0
# (see `trusted_citation_id_block`). Held-out has never changed. A single shared
# constant
# would have forced unrelated prompts to claim a version they did not change, and
# would have invalidated their packaged qualification fixtures for no reason --
# which is exactly why the two independent lines of work above could be merged by
# taking each role's own advance rather than choosing between them.
_PROMPT_VERSIONS: dict[str, str] = {
    "discovery": "1.5.0",
    "candidate": "1.2.0",
    "held_out": "1.1.0",
}
_OUTPUT_SCHEMA_VERSION = "v1"
_UNTRUSTED_CLOSE = "</open_world_untrusted_data>"
_UNTRUSTED_FRAMING_VERSION = "rfc8785-base64-v1"
# The payload PROJECTION version, and it is deliberately NOT folded into
# `_UNTRUSTED_FRAMING_VERSION` above.
#
# The ENVELOPE did not change: still RFC 8785 canonical JSON, still base64,
# still `rfc8785-base64-v1` in the opening tag -- and every role's prompt names
# that string in its own text ("The delimited user message uses
# rfc8785-base64-v1 framing"), so advancing it would put the prompt and the tag
# the model reads into direct contradiction, and would need a new template
# version for all three roles to repair.
#
# What DID change is the shape of the document INSIDE the envelope: constant
# span identity is now carried once per document rather than repeated on every
# span (`identity._evidence_document_payload`). That is invisible to the
# envelope, but it must still invalidate every CACHED qualification result,
# because a qualification attests that this adapter reproduces frozen fixture
# messages and those messages are rendered from this projection. Qualification
# results are keyed by `ComponentIdentity`, whose
# `generation_settings_digest` hashes the dict below -- so naming the
# projection here is what makes a stale pass unusable.
_ANALYST_PAYLOAD_PROJECTION_VERSION = "hoisted-span-identity-v2-compaction"
# `max_tokens` and `request_timeout_seconds` were chosen from measurement
# against a reasoning model route (`minimax/MiniMax-M3`), not by doubling the
# previous value. What holds, and why the obvious answer is wrong:
#
#  1. REASONING EXPANDS TO FILL THE OFFER on a hard input. A larger cap can be
#     consumed entirely by reasoning, leaving a body of a couple of
#     characters; on a BOUNDED input the same prompt's reasoning stays flat
#     across caps. So the cap is a bound on WASTE, not headroom to be spent.
#  2. THE WALL CLOCK, NOT THE CAP, IS THE BINDING CONSTRAINT. At the route's
#     sustained output rate the per-attempt timeout bounds how many tokens can
#     arrive at all, so raising `max_tokens` alone is decorative: the attempt
#     is killed before reaching it. 12288 tokens with 240s of per-attempt
#     budget keeps the cap reachable with margin.
#  3. THE BODY IS SMALL. A successful discovery memo takes on the order of
#     1000 completion tokens in total.
#
# Termination is BIMODAL and stochastic, not a property of the evidence: the
# same frozen evidence, prompt and cap can either finish in ~1000 completion
# tokens or consume the cap exactly, with no middle outcome. 12288 is already
# an order of magnitude more than success needs; a larger cap only buys a
# longer wait for the same truncated body. Do not raise it in response to a
# non-terminating run.
#
# What a non-terminating run DOES cost is the day. It leaves the invocation
# row in `prepared`, and `analyst_input_identity` carries no attempt number and
# no `job_id` -- see the DELIBERATELY ABSENT note in `identity.py` -- so the next
# job for the same playbook recomputes an identity the stranded row already
# owns and converges on `invocation_slot_pinned` with no provider call. A
# retry is therefore unavailable until the UTC day rolls; changing that needs
# an attempt-scoped row identity and a new SQL preimage `schema_version` in the
# storage schema IN THE SAME CHANGE.
# `reasoning_effort` is NOT an option either: litellm's
# `MinimaxChatConfig.map_openai_params` maps it to `{}` and drops it silently.
# The one reasoning knob it does map is `thinking`, which is an on/off switch
# rather than a budget -- and disabling it degrades exactly the judgement
# discovery exists to make.
_GENERATION_SETTINGS = {
    "fallback_rungs": 0,
    "input_framing": (
        f"{_UNTRUSTED_FRAMING_VERSION}+{_ANALYST_PAYLOAD_PROJECTION_VERSION}"
    ),
    "max_tokens": 12288,
    "provider_attempts": 1,
    "provider_num_retries": 0,
    "provider_request_guard": "offline-tuner-open-world-v1",
    # This explicit per-call timeout belongs to the qualified analyst protocol.
    # Other callers retain their own configured budgets; model selection never
    # raises them to an implicit provider-specific minimum.
    "request_timeout_seconds": 240,
    "seed": 42,
    "shared_max_retries_argument": 1,
    "temperature": 0,
    "tools": None,
    "top_p": 1,
    "transport_admission": "litellm-exact-reviewed-io-v2",
}
# How many times the ONE qualified question may cross the wire. Two, because
# the shared client's `_run_rung_plain` makes exactly one same-model re-issue
# when the first structured body fails to parse. This is a ceiling on
# repetitions of a single attested question, not a licence for a second
# distinct one -- `guard_exact_request` enforces that separately by pinning the
# message list, admitting only two shapes for the second crossing: the
# byte-identical re-issue, and the reviewed client's own bounded corrective
# turn (`is_structured_output_correction_turn`).
#
# It lives here rather than in `_GENERATION_SETTINGS` on purpose: that dict is
# hashed into `ComponentIdentity.generation_settings_digest`, which the packaged
# qualification suite pins, so adding a key there would invalidate every
# reference identity in `qualification_responses/v1.json`.
_MAX_QUALIFIED_PROVIDER_ATTEMPTS = 2
_REVIEWED_GENERATE_METHOD = LiteLLMClient.generate_chat_response_with_provenance
_REVIEWED_COMPLETION_METHOD = LiteLLMClient._completion_with_hard_timeout  # noqa: SLF001
_REVIEWED_PROMPT_RENDER_METHOD = PromptManager.render_prompt_from_identity
_REVIEWED_CASE_REQUEST_METHOD = QualificationCase.to_production_request
# The corrective second crossing is admitted on the say-so of these two, so
# they belong in the reviewed set exactly as much as the generate and I/O
# methods do.
#
# `guard_exact_request` lets a SECOND, non-identical request reach the provider
# whenever `is_structured_output_correction_turn` recognises it. That predicate
# is the whole admission rule, and it was the one link in the chain nothing
# pinned: a swap of it -- or of the builder whose output it is meant to
# recognise -- would widen what counts as "the qualified question" while every
# existing guard kept passing.
#
# The BUILDER is pinned alongside the predicate because the builder is what
# decides the correction body's contents. The reason that matters is stated in
# its own docstring: the turn carries the failing field paths "and NOTHING else
# -- no fragment of the model's own response". A change that put model output
# back into that body would still satisfy the predicate, since the predicate
# only checks the placeholder and the prefix.
_REVIEWED_CORRECTION_TURN_PREDICATE = is_structured_output_correction_turn
_REVIEWED_CORRECTION_TURN_BUILDER = structured_output_correction_turn
_LITELLM_CREDENTIAL_DRIFT_DOMAIN = (
    b"reflexio/offline-tuner/litellm-transport-credential-drift/v1\x00"
)
# Process-local HMAC key for the in-session credential drift fingerprint. It is
# regenerated every boot ON PURPOSE. The fingerprint is only ever compared
# against another fingerprint derived in this same process, so a per-process
# key costs nothing -- and it makes the value structurally incapable of
# becoming a stable cross-process identifier for a credential, including if it
# were ever logged. It must never reach the persisted transport digest; see
# `_litellm_transport_identity`.
_CREDENTIAL_DRIFT_KEY = secrets.token_bytes(32)


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user"]
    content: str


class OpenWorldModelClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        response_format: type[StrictStructuredOutput],
        model_revision: str,
        idempotency_key: str,
    ) -> ObservedStructuredCompletion: ...


class QualificationFixtureModelClient(Protocol):
    """Opaque handle issued only for the packaged qualification gate."""

    @property
    def calls(self) -> list[QualificationFixtureCall]: ...

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        response_format: type[StrictStructuredOutput],
        model_revision: str,
        idempotency_key: str,
    ) -> ObservedStructuredCompletion: ...


type AdapterClient = OpenWorldModelClient | QualificationFixtureModelClient


@dataclass(frozen=True)
class QualificationFixtureCall:
    messages: tuple[ChatMessage, ...]
    response_format: type[StrictStructuredOutput]
    model_revision: str
    idempotency_key: str


@dataclass(frozen=True)
class _FrozenFixtureResponse:
    case_id: str
    role: Literal["discovery", "held_out"]
    case_digest: str
    request_context: dict[str, Any]
    request_canonical: dict[str, Any]
    request_digest: str
    messages: tuple[ChatMessage, ...]
    messages_digest: str
    output: DiscoveryMemo | HeldOutSessionMemo


@dataclass(frozen=True)
class _QualificationCallBinding:
    case_digest: str
    request_digest: str
    messages_digest: str


type _QualificationFixtureOutput = (
    DiscoveryMemo | HeldOutSessionMemo | Mapping[str, object]
)


class _MalformedQualificationFixtureOutputError(ValueError):
    """A sealed fixture emitted raw bytes that failed the production schema path."""


def _ordered_suite_digest(suite: QualificationSuite) -> str:
    return hashlib.sha256(
        canonical_json_bytes(suite.model_dump(mode="json"))
    ).hexdigest()


def _case_digest(case: QualificationCase) -> str:
    return hashlib.sha256(
        canonical_json_bytes(case.model_dump(mode="json"))
    ).hexdigest()


def _request_digest(request: DiscoveryRequest | HeldOutSessionRequest) -> str:
    return hashlib.sha256(canonical_bytes(request)).hexdigest()


def _messages_digest(messages: Sequence[ChatMessage]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            [{"role": message.role, "content": message.content} for message in messages]
        )
    ).hexdigest()


def _load_frozen_fixture_responses() -> tuple[
    str,
    str,
    dict[AdapterRole, ComponentIdentity],
    tuple[_FrozenFixtureResponse, ...],
]:
    raw = json.loads(
        files(__package__)
        .joinpath("qualification_responses", "v1.json")
        .read_text(encoding="utf-8")
    )
    if (
        set(raw)
        != {
            "schema_version",
            "suite_digest",
            "suite_ordered_digest",
            "reference_identities",
            "responses",
        }
        or raw["schema_version"] != "offline-tuner-task5-fixture-expectations-v2"
        or set(raw["reference_identities"]) != {"discovery", "held_out", "candidate"}
    ):
        raise ValueError("qualification response artifact has an invalid contract")
    reference_identities = {
        role: ComponentIdentity(
            **{key: value for key, value in payload.items() if key != "schema_version"}
        )
        for role, payload in raw["reference_identities"].items()
    }
    if (
        reference_identities["discovery"].provider != "deterministic-fixture"
        or reference_identities["held_out"].provider != "deterministic-fixture"
        or reference_identities["discovery"].immutable_model_revision
        != "fixture-model-2026-08-17"
        or reference_identities["held_out"].immutable_model_revision
        != "fixture-model-2026-08-17"
    ):
        raise ValueError("qualification fixture identity is not the packaged gate")
    responses = []
    for item in raw["responses"]:
        if set(item) != {
            "case_id",
            "role",
            "case_digest",
            "request_context",
            "request_canonical",
            "request_digest",
            "messages_canonical",
            "messages_digest",
            "output",
        }:
            raise ValueError("qualification response entry has an invalid contract")
        role = item["role"]
        if role == "discovery":
            output_payload = dict(item["output"])
            for field in (
                "observations",
                "citations",
                "alternative_explanations",
                "counterevidence",
                "uncertainties",
            ):
                output_payload[field] = tuple(output_payload[field])
            output = DiscoveryMemo.model_validate(output_payload)
        elif role == "held_out":
            output_payload = dict(item["output"])
            for field in (
                "observations",
                "citations",
                "alternative_explanations",
                "uncertainties",
            ):
                output_payload[field] = tuple(output_payload[field])
            output = HeldOutSessionMemo.model_validate(output_payload)
        else:
            raise ValueError("qualification response role is invalid")
        messages = tuple(
            ChatMessage(**message) for message in item["messages_canonical"]
        )
        if (
            type(item["request_context"]) is not dict
            or type(item["request_canonical"]) is not dict
            or len(messages) != 2
            or tuple(message.role for message in messages) != ("system", "user")
            or hashlib.sha256(
                canonical_json_bytes(item["request_canonical"])
            ).hexdigest()
            != item["request_digest"]
            or _messages_digest(messages) != item["messages_digest"]
        ):
            raise ValueError("qualification response expectation is invalid")
        responses.append(
            _FrozenFixtureResponse(
                case_id=item["case_id"],
                role=role,
                case_digest=item["case_digest"],
                request_context=item["request_context"],
                request_canonical=item["request_canonical"],
                request_digest=item["request_digest"],
                messages=messages,
                messages_digest=item["messages_digest"],
                output=output,
            )
        )
    return (
        raw["suite_digest"],
        raw["suite_ordered_digest"],
        reference_identities,
        tuple(responses),
    )


(
    _FROZEN_SUITE_DIGEST,
    _FROZEN_SUITE_ORDERED_DIGEST,
    _FROZEN_REFERENCE_IDENTITIES,
    _FROZEN_FIXTURE_RESPONSES,
) = _load_frozen_fixture_responses()


def _expected_identity_payload(identity: ComponentIdentity) -> dict[str, object]:
    return {
        "component_role": identity.component_role,
        "provider": identity.provider,
        "immutable_model_revision": identity.immutable_model_revision,
        "prompt_id": identity.prompt_id,
        "prompt_version": identity.prompt_version,
        "prompt_template_digest": identity.prompt_template_digest,
        "output_schema_id": identity.output_schema_id,
        "output_schema_version": identity.output_schema_version,
        "output_schema_digest": identity.output_schema_digest,
        "generation_settings_digest": identity.generation_settings_digest,
        "qualification_suite_version": identity.qualification_suite_version,
    }


def _expected_identity_digest(identity: ComponentIdentity) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "offline-tuner-open-world-component-identity-v1",
                **_expected_identity_payload(identity),
            }
        )
    ).hexdigest()


def _expected_request_input_digest(
    contract_type: str,
    *,
    model_input: Mapping[str, object],
    semantic_bindings: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_type": contract_type,
                "schema_version": "v1",
                "payload": {
                    "model_input": model_input,
                    "semantic_bindings": semantic_bindings,
                },
            }
        )
    ).hexdigest()


def _expected_contract(
    contract_type: str, payload: Mapping[str, object]
) -> dict[str, object]:
    return {
        "contract_type": contract_type,
        "schema_version": "v1",
        "payload": dict(payload),
    }


def _expected_provisional_analyst_binding(
    analyst: Mapping[str, object],
) -> dict[str, object]:
    return _expected_contract(
        "open_world_provisional_case_analyst",
        {
            "identity": analyst["identity"],
            "qualification": _expected_contract(
                "open_world_analyst_qualification_result_ref",
                cast(Mapping[str, object], analyst["qualification"]),
            ),
        },
    )


def _expected_prepared_invocation(
    *,
    job_id: int,
    bundle_digest: str,
    identity: ComponentIdentity,
    qualification: Mapping[str, object],
    qualification_result_digest: str,
    rendered_input_digest: str,
    outcome_contract_digest: str,
    role: AdapterRole,
    manifest_role: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    trajectory_digest: str | None = None,
    candidate_binding: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    identity_payload = _expected_identity_payload(identity)
    identity_digest = _expected_identity_digest(identity)
    analyst_input = {
        "component_identity_digest": identity_digest,
        "qualification_result_digest": qualification_result_digest,
        "rendered_input_digest": rendered_input_digest,
        "outcome_contract_digest": outcome_contract_digest,
        "role": role,
    }
    analyst_input_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": (
                    "offline-tuner-open-world-analyst-input-identity-v1"
                ),
                **analyst_input,
            }
        )
    ).hexdigest()
    return {
        "job_id": job_id,
        "bundle_digest": bundle_digest,
        "component_identity": identity_payload,
        "qualification": dict(qualification),
        "qualification_result_digest": qualification_result_digest,
        "rendered_input_digest": rendered_input_digest,
        "outcome_contract_digest": outcome_contract_digest,
        "role": role,
        "manifest_role": manifest_role,
        "user_id": user_id,
        "session_id": session_id,
        "trajectory_digest": trajectory_digest,
        "candidate_binding": None
        if candidate_binding is None
        else dict(candidate_binding),
        "analyst_input_identity": analyst_input,
        "intent": {
            "job_id": job_id,
            "analyst_input_identity": analyst_input_digest,
            "component_identity_digest": identity_digest,
            "input_digest": rendered_input_digest,
            "role": role,
            "expected_provider": identity.provider,
            "expected_model_revision": identity.immutable_model_revision,
            "manifest_role": manifest_role,
            "user_id": user_id,
            "session_id": session_id,
        },
    }


def _expected_discovery_envelope(
    *,
    frozen: _FrozenFixtureResponse,
    envelope: dict[str, Any],
    outcome_contract: dict[str, Any],
    discovery_identity: ComponentIdentity,
) -> dict[str, Any]:
    """Return the expected discovery envelope without taking input ownership."""
    envelope = deepcopy(envelope)
    context = frozen.request_context
    qualification = deepcopy(context["discovery_qualification"])
    qualification["component_identity_digest"] = _expected_identity_digest(
        discovery_identity
    )
    analyst = {
        "identity": _expected_identity_payload(discovery_identity),
        "qualification": qualification,
    }
    rendered_input_digest = _expected_request_input_digest(
        "open_world_discovery_request_input",
        model_input={
            "bundle_digest": envelope["bundle_digest"],
            "discovery_sessions": context["discovery_sessions"],
            "outcome_contract": outcome_contract,
        },
        semantic_bindings={
            "analyst": _expected_provisional_analyst_binding(analyst),
        },
    )
    prepared = _expected_prepared_invocation(
        job_id=envelope["prepared_invocation"]["job_id"],
        bundle_digest=envelope["bundle_digest"],
        identity=discovery_identity,
        qualification=qualification,
        qualification_result_digest=qualification["result_digest"],
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract["digest"],
        role="discovery",
    )
    envelope["analyst_identity"] = _expected_identity_payload(discovery_identity)
    envelope["prepared_invocation"] = prepared
    envelope["invocation"].update(
        {
            "analyst_input_identity": prepared["intent"]["analyst_input_identity"],
            "component_identity_digest": _expected_identity_digest(discovery_identity),
            "observed_provider": discovery_identity.provider,
            "observed_model_revision": discovery_identity.immutable_model_revision,
        }
    )
    return envelope


def _expected_generator_authorization(
    candidate_identity: ComponentIdentity,
) -> dict[str, object]:
    canonical_identity = {
        "schema_version": "offline-tuner-open-world-component-identity-v1",
        **_expected_identity_payload(candidate_identity),
    }
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_type": "open_world_pinned_generator_authorization",
                "schema_version": "v1",
                "component_identity": canonical_identity,
            }
        )
    ).hexdigest()
    return {
        "generator_identity": _expected_identity_payload(candidate_identity),
        "digest": digest,
    }


def _expected_candidate_envelope(
    *,
    frozen: _FrozenFixtureResponse,
    envelope: dict[str, Any],
    discovery: dict[str, Any],
    outcome_contract: dict[str, Any],
    candidate_identity: ComponentIdentity,
) -> dict[str, Any]:
    """Return the expected candidate envelope without taking input ownership."""
    envelope = deepcopy(envelope)
    authorization = _expected_generator_authorization(candidate_identity)
    rendered_input_digest = _expected_request_input_digest(
        "open_world_candidate_request_input",
        model_input={
            "bundle_digest": envelope["bundle_digest"],
            "discovery_memo": discovery["memo"],
            "incumbent_content": frozen.request_context["candidate_incumbent_content"],
            "outcome_contract": outcome_contract,
            "verified_discovery_citation_ids": tuple(
                citation["citation_id"] for citation in discovery["verified_citations"]
            ),
        },
        semantic_bindings={
            "discovery_analyst_input_identity": discovery["prepared_invocation"][
                "intent"
            ]["analyst_input_identity"],
            "incumbent_full_version_fingerprint": envelope[
                "incumbent_full_version_fingerprint"
            ],
            "generator_authorization": _expected_contract(
                "open_world_pinned_generator_authorization", authorization
            ),
        },
    )
    prepared = _expected_prepared_invocation(
        job_id=envelope["prepared_invocation"]["job_id"],
        bundle_digest=envelope["bundle_digest"],
        identity=candidate_identity,
        qualification=authorization,
        qualification_result_digest=cast(str, authorization["digest"]),
        rendered_input_digest=rendered_input_digest,
        outcome_contract_digest=outcome_contract["digest"],
        role="candidate",
    )
    envelope["generator_identity"] = _expected_identity_payload(candidate_identity)
    envelope["prepared_invocation"] = prepared
    envelope["invocation"].update(
        {
            "analyst_input_identity": prepared["intent"]["analyst_input_identity"],
            "component_identity_digest": _expected_identity_digest(candidate_identity),
            "observed_provider": candidate_identity.provider,
            "observed_model_revision": candidate_identity.immutable_model_revision,
        }
    )
    return envelope


def _expected_runtime_request_digest(
    *,
    frozen: _FrozenFixtureResponse,
    discovery_identity: ComponentIdentity,
    held_out_identity: ComponentIdentity,
    candidate_identity: ComponentIdentity,
) -> str:
    request = deepcopy(frozen.request_canonical)
    payload = request["payload"]
    outcome_contract = payload["outcome_contract"]
    if frozen.role == "discovery":
        qualification = payload["analyst"]["qualification"]
        qualification["component_identity_digest"] = _expected_identity_digest(
            discovery_identity
        )
        analyst = {
            "identity": _expected_identity_payload(discovery_identity),
            "qualification": qualification,
        }
        payload["analyst"] = analyst
        rendered_input_digest = _expected_request_input_digest(
            "open_world_discovery_request_input",
            model_input={
                "bundle_digest": payload["bundle_digest"],
                "discovery_sessions": payload["discovery_sessions"],
                "outcome_contract": outcome_contract,
            },
            semantic_bindings={
                "analyst": _expected_provisional_analyst_binding(analyst),
            },
        )
        payload["rendered_input_digest"] = rendered_input_digest
        payload["prepared_invocation"] = _expected_prepared_invocation(
            job_id=payload["job_id"],
            bundle_digest=payload["bundle_digest"],
            identity=discovery_identity,
            qualification=qualification,
            qualification_result_digest=qualification["result_digest"],
            rendered_input_digest=rendered_input_digest,
            outcome_contract_digest=outcome_contract["digest"],
            role="discovery",
        )
    else:
        discovery = _expected_discovery_envelope(
            frozen=frozen,
            envelope=payload["discovery"],
            outcome_contract=outcome_contract,
            discovery_identity=discovery_identity,
        )
        candidate = _expected_candidate_envelope(
            frozen=frozen,
            envelope=payload["candidate"],
            discovery=discovery,
            outcome_contract=outcome_contract,
            candidate_identity=candidate_identity,
        )
        payload["discovery"] = discovery
        payload["candidate"] = candidate
        qualification = payload["analyst"]["qualification"]
        qualification["component_identity_digest"] = _expected_identity_digest(
            held_out_identity
        )
        analyst = {
            "identity": _expected_identity_payload(held_out_identity),
            "qualification": qualification,
        }
        payload["analyst"] = analyst
        rendered_input_digest = _expected_request_input_digest(
            "open_world_held_out_request_input",
            model_input={
                "bundle_digest": payload["bundle_digest"],
                "candidate_diff": candidate["content_diff"],
                "hypothesis": discovery["memo"]["hypothesis"],
                "evidence": payload["evidence"],
                "manifest_role": payload["manifest_role"],
                "outcome_contract": outcome_contract,
            },
            semantic_bindings={
                "analyst": _expected_provisional_analyst_binding(analyst),
            },
        )
        payload["rendered_input_digest"] = rendered_input_digest
        payload["prepared_invocation"] = _expected_prepared_invocation(
            job_id=payload["job_id"],
            bundle_digest=payload["bundle_digest"],
            identity=held_out_identity,
            qualification=qualification,
            qualification_result_digest=qualification["result_digest"],
            rendered_input_digest=rendered_input_digest,
            outcome_contract_digest=outcome_contract["digest"],
            role="held_out",
            manifest_role=payload["manifest_role"],
            user_id=payload["evidence"]["user_id"],
            session_id=payload["evidence"]["session_id"],
            trajectory_digest=payload["evidence"]["trajectory_digest"],
            candidate_binding={
                "artifact_digest": hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "contract_type": "open_world_candidate_envelope",
                            "schema_version": "v1",
                            "payload": candidate,
                        }
                    )
                ).hexdigest(),
                "analyst_input_identity": candidate["invocation"][
                    "analyst_input_identity"
                ],
            },
        )
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def _require_exact_frozen_qualification_suite(suite: QualificationSuite) -> None:
    if type(suite) is not QualificationSuite:
        raise TypeError("suite must be the exact qualification contract")
    expected = tuple(
        (item.case_id, item.role, item.case_digest)
        for item in _FROZEN_FIXTURE_RESPONSES
    )
    observed = tuple(
        (case.case_id, case.role, _case_digest(case)) for case in suite.cases
    )
    if (
        suite.digest != _FROZEN_SUITE_DIGEST
        or _ordered_suite_digest(suite) != _FROZEN_SUITE_ORDERED_DIGEST
        or observed != expected
    ):
        raise ValueError("fixture requires the exact canonical qualification suite")


def _build_qualification_fixture_client_authority() -> tuple[  # noqa: C901
    Callable[..., QualificationFixtureModelClient],
    Callable[..., QualificationFixtureModelClient],
    Callable[..., None],
    Callable[..., ObservedStructuredCompletion],
]:
    factory_token = object()

    class _SealedQualificationFixtureClient:
        __slots__ = (
            "_bindings",
            "_case_digests",
            "_case_ids",
            "_identity",
            "_next_index",
            "_allow_malformed_raw_output",
            "_observed_model_revision",
            "_observed_provider",
            "_outputs",
            "_subject_transport",
            "_suite_bytes_digest",
            "_suite_digest",
            "_token",
            "calls",
        )

        def __init__(
            self,
            *,
            suite: QualificationSuite,
            role: Literal["discovery", "held_out"],
            identity: ComponentIdentity,
            outputs: tuple[_QualificationFixtureOutput, ...],
            allow_malformed_raw_output: bool,
            observed_provider: str,
            observed_model_revision: str,
            subject_transport: object | None,
            token: object,
        ) -> None:
            if token is not factory_token:
                raise TypeError(
                    "qualification fixture client must be created by its factory"
                )
            expected_cases = suite.cases_for_role(role)
            self._suite_digest = suite.digest
            self._suite_bytes_digest = _ordered_suite_digest(suite)
            self._case_digests = tuple(_case_digest(case) for case in expected_cases)
            self._case_ids = tuple(case.case_id for case in expected_cases)
            self._outputs = outputs
            self._identity = identity
            self._allow_malformed_raw_output = allow_malformed_raw_output
            self._observed_provider = observed_provider
            self._observed_model_revision = observed_model_revision
            self._subject_transport = subject_transport
            self._bindings: tuple[_QualificationCallBinding, ...] | None = None
            self._next_index = 0
            self._token = token
            self.calls: list[QualificationFixtureCall] = []

        def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            response_format: type[StrictStructuredOutput],
            model_revision: str,
            idempotency_key: str,
        ) -> ObservedStructuredCompletion:
            if self._bindings is None:
                raise ValueError("qualification calls must be bound before execution")
            if self._next_index >= len(self._case_ids):
                raise ValueError("qualification role received an extra model call")
            expected_type = (
                DiscoveryMemo
                if self._identity.component_role == "discovery"
                else HeldOutSessionMemo
            )
            index = self._next_index
            messages_tuple = tuple(messages)
            binding = self._bindings[index]
            if (
                binding.messages_digest != _messages_digest(messages_tuple)
                or response_format is not expected_type
                or model_revision != self._identity.immutable_model_revision
                or len(idempotency_key) != 64
            ):
                raise ValueError("qualification call does not match the canonical case")
            self._next_index += 1
            self.calls.append(
                QualificationFixtureCall(
                    messages=messages_tuple,
                    response_format=response_format,
                    model_revision=model_revision,
                    idempotency_key=idempotency_key,
                )
            )
            output = self._outputs[index]
            if type(output) is response_format:
                value = cast(StrictStructuredOutput, output)
            elif type(output) is dict and self._allow_malformed_raw_output:
                try:
                    value = response_format.model_validate(output)
                except ValidationError as exc:
                    raise _MalformedQualificationFixtureOutputError(
                        "qualification fixture output did not match production schema"
                    ) from exc
                if type(value) is not response_format:
                    raise _MalformedQualificationFixtureOutputError(
                        "qualification fixture output did not match production schema"
                    )
                value = cast(StrictStructuredOutput, value)
            else:
                raise TypeError(
                    "qualification output does not match the production schema"
                )
            return ObservedStructuredCompletion(
                value=value,
                observed_provider=self._observed_provider,
                observed_model_revision=self._observed_model_revision,
                called_at=100 + self._next_index,
                latency_ms=1,
                usage_json=None,
            )

    def build(
        *,
        suite: QualificationSuite,
        role: Literal["discovery", "held_out"],
        identity: ComponentIdentity,
        outputs: Sequence[_QualificationFixtureOutput] | None = None,
        subject_transport: object | None = None,
        _allow_malformed_raw_output: bool = False,
        _observed_provider: str | None = None,
        _observed_model_revision: str | None = None,
    ) -> QualificationFixtureModelClient:
        _require_exact_frozen_qualification_suite(suite)
        _require_qualification_subject_identity(
            role=role, identity=identity, subject_transport=subject_transport
        )
        expected_cases = suite.cases_for_role(role)
        frozen_by_case = {item.case_id: item for item in _FROZEN_FIXTURE_RESPONSES}
        selected_outputs = tuple(
            frozen_by_case[case.case_id].output for case in expected_cases
        )
        if outputs is not None:
            selected_outputs = tuple(outputs)
        expected_type = DiscoveryMemo if role == "discovery" else HeldOutSessionMemo
        if len(selected_outputs) != len(expected_cases):
            raise ValueError(
                "qualification fixture outputs must exactly cover the role cases"
            )
        for output in selected_outputs:
            if type(output) is expected_type:
                continue
            if _allow_malformed_raw_output and type(output) is dict:
                continue
            raise ValueError(
                "qualification fixture outputs must exactly cover the role cases"
            )
        return _SealedQualificationFixtureClient(
            suite=suite,
            role=role,
            identity=identity,
            outputs=selected_outputs,
            allow_malformed_raw_output=_allow_malformed_raw_output,
            observed_provider=_observed_provider or identity.provider,
            observed_model_revision=(
                _observed_model_revision or identity.immutable_model_revision
            ),
            subject_transport=subject_transport,
            token=factory_token,
        )

    def require(
        client: object,
        *,
        suite: QualificationSuite,
        role: Literal["discovery", "held_out"],
        identity: ComponentIdentity,
    ) -> _SealedQualificationFixtureClient:
        if (
            type(client) is not _SealedQualificationFixtureClient
            or client._token is not factory_token
            or client._suite_digest != suite.digest
            or client._suite_bytes_digest != _ordered_suite_digest(suite)
            or client._case_ids
            != tuple(case.case_id for case in suite.cases_for_role(role))
            or client._case_digests
            != tuple(_case_digest(case) for case in suite.cases_for_role(role))
            or client._identity != identity
        ):
            raise ValueError(
                "qualification requires a sealed deterministic qualification client"
            )
        # Re-derive the subject attestation rather than trusting a flag the
        # sealed client set at build time. Two things this buys that a stored
        # boolean would not: the packaged-gate arm is re-checked against
        # `_FROZEN_REFERENCE_IDENTITIES` from scratch, and the production arm
        # re-reads the LIVE LiteLLM configuration, so a transport whose model,
        # endpoint or credential changed between composition and use fails
        # here instead of qualifying an identity the deployment no longer runs.
        _require_qualification_subject_identity(
            role=role,
            identity=identity,
            subject_transport=client._subject_transport,
        )
        return client

    def bind(
        client: object,
        *,
        suite: QualificationSuite,
        role: Literal["discovery", "held_out"],
        identity: ComponentIdentity,
        bindings: tuple[_QualificationCallBinding, ...],
    ) -> None:
        sealed = require(client, suite=suite, role=role, identity=identity)
        if sealed._bindings is not None:
            raise ValueError("qualification fixture calls are already bound")
        if len(bindings) != len(sealed._case_ids):
            raise ValueError("qualification call binding does not cover role cases")
        sealed._bindings = bindings

    def complete_case(
        client: object,
        *,
        suite: QualificationSuite,
        role: Literal["discovery", "held_out"],
        identity: ComponentIdentity,
        case: QualificationCase,
        request: DiscoveryRequest | HeldOutSessionRequest,
        messages: tuple[ChatMessage, ...],
        response_format: type[StrictStructuredOutput],
        model_revision: str,
        idempotency_key: str,
    ) -> ObservedStructuredCompletion:
        sealed = require(client, suite=suite, role=role, identity=identity)
        if sealed._bindings is None:
            raise ValueError("qualification calls must be bound before execution")
        if sealed._next_index >= len(sealed._case_ids):
            raise ValueError("qualification role received an extra model call")
        index = sealed._next_index
        binding = sealed._bindings[index]
        if (
            case.case_id != sealed._case_ids[index]
            or binding.case_digest != _case_digest(case)
            or binding.request_digest != _request_digest(request)
        ):
            raise ValueError("qualification call does not match the canonical case")
        return sealed.complete(
            messages=messages,
            response_format=response_format,
            model_revision=model_revision,
            idempotency_key=idempotency_key,
        )

    return build, require, bind, complete_case


(
    qualification_fixture_model_client,
    _require_qualification_fixture_model_client,
    _bind_qualification_fixture_model_client,
    _complete_qualification_fixture_case,
) = _build_qualification_fixture_client_authority()
del _build_qualification_fixture_client_authority


_CUSTOM_ENDPOINT_ROUTE = "custom-openai-compatible"
_DEFAULT_OPENAI_ROUTE = "openai-default"
# The analyst is deliberately PROVIDER-AGNOSTIC. A floating alias --
# `minimax/MiniMax-M3`, `claude-sonnet-5`, `gpt-4o` -- composes exactly like a
# dated snapshot. An earlier rule refused any model id without a provider-issued
# `-YYYY-MM-DD` / `-YYYYMMDD` stamp, on the theory that it guaranteed "the model
# that runs is the model that was qualified". That guarantee was empty:
# qualification replays packaged frozen responses through the sealed fixture
# client and never contacts a provider at all, so it grades the harness, not the
# model.
#
# What still protects a call is unchanged and lives below: the observed-provider
# and observed-model echo checks reject a substitution WITHIN a request, and
# post-composition drift detection rejects a transport whose live config moved.
# Model-quality risk is carried by the pipeline harness and by the tuner's
# own next evidence pass, which revises a published playbook (the 14-day
# observation window that once auto-restored one is deleted).


def _strip_route_prefix(model: str, *, route: str) -> str:
    """Drop a leading litellm ``route/`` qualifier from a model identifier."""
    prefix = f"{route}/"
    return model.removeprefix(prefix)


def _credential_drift_fingerprint(credential: str | None) -> str | None:
    """Fingerprint a live credential for IN-SESSION drift detection only.

    This value never leaves the process and never enters the transport digest.
    Keeping it out of the digest is the point: a rotated API key must not
    invalidate a cached qualification. Keeping it here is also deliberate --
    ``api_base`` is ``None`` for every provider whose credentials resolve by
    model prefix (MiniMax, Anthropic, Gemini, DeepSeek, ...), so without this
    field nothing at all would notice a credential swapped underneath a
    composed transport.

    Args:
        credential (str | None): The live credential snapshot, or ``None`` when
            the route carries no API key.

    Returns:
        str | None: An opaque per-process fingerprint, or ``None``.
    """
    if credential is None:
        return None
    return hmac.new(
        _CREDENTIAL_DRIFT_KEY,
        _LITELLM_CREDENTIAL_DRIFT_DOMAIN + credential.encode(),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class _LiteLLMTransportIdentity:
    """The composed description of one LiteLLM transport.

    Two of these fields answer different questions and must not be conflated.

    ``provider`` (and the ``digest`` inside it) is the PERSISTED identity: it
    becomes ``ComponentIdentity.provider`` and therefore the qualification
    cache key. It names the transport's *scope* -- route, endpoint, API
    version, model revision, reviewed implementation -- and deliberately says
    nothing about the credential, so rotating an API key against the same
    provider, endpoint and model keeps a cached qualification valid.

    ``credential_fingerprint`` is NOT persisted and is not part of the digest.
    It exists only so dataclass equality still notices a credential that
    changed underneath a live composition; see
    ``_credential_drift_fingerprint``.
    """

    provider: str
    model_revision: str
    digest: str
    route: str
    expected_observed_provider: str
    bare_model_revision: str
    credential_fingerprint: str | None


def _litellm_transport_identity(client: LiteLLMClient) -> _LiteLLMTransportIdentity:
    if type(client) is not LiteLLMClient:
        raise TypeError("production composition requires the exact LiteLLMClient")
    completion_method = client.generate_chat_response_with_provenance
    if (
        LiteLLMClient.generate_chat_response_with_provenance
        is not _REVIEWED_GENERATE_METHOD
        or getattr(completion_method, "__func__", None) is not _REVIEWED_GENERATE_METHOD
    ):
        raise TypeError("production composition requires the reviewed LiteLLM method")
    io_method = client._completion_with_hard_timeout  # noqa: SLF001
    if (
        LiteLLMClient._completion_with_hard_timeout  # noqa: SLF001
        is not _REVIEWED_COMPLETION_METHOD
        or getattr(io_method, "__func__", None) is not _REVIEWED_COMPLETION_METHOD
    ):
        raise TypeError(
            "production composition requires the reviewed LiteLLM I/O method"
        )
    if (
        is_structured_output_correction_turn is not _REVIEWED_CORRECTION_TURN_PREDICATE
        or structured_output_correction_turn is not _REVIEWED_CORRECTION_TURN_BUILDER
    ):
        raise TypeError(
            "production composition requires the reviewed correction-turn contract"
        )
    config = client.config
    if type(config) is not LiteLLMConfig:
        raise TypeError("production composition requires the exact LiteLLMConfig")
    custom = config.api_key_config.custom_endpoint if config.api_key_config else None
    route = (
        "custom-openai-compatible"
        if custom is not None
        else (
            config.model.split("/", 1)[0] if "/" in config.model else "openai-default"
        )
    )
    credential = client._api_key  # noqa: SLF001 - immutable transport snapshot
    # This payload binds credential SCOPE, never the credential secret.
    #
    # The digest below becomes `provider`, which becomes
    # `ComponentIdentity.provider`, which is the qualification cache key.
    # Folding the API key into it made routine key rotation -- same provider,
    # same endpoint, same model -- silently take the capability offline: the
    # cache key moved, the lookup missed, and the gate reported "unavailable"
    # for a change that is not one of the spec's requalification triggers
    # (provider/model revision, prompt, output schema, generation setting,
    # qualification suite).
    #
    # What remains is every non-I/O scope signal that actually exists here.
    # `api_base` and `api_version` come from `LiteLLMClient._resolve_api_key`,
    # which populates them only for Azure (resource endpoint + API version),
    # DashScope (intl vs China endpoint) and the custom gateway; for every
    # provider that resolves by model prefix they are `None`, so two customers
    # of MiniMax, Anthropic or default-route OpenAI are NOT distinguishable at
    # composition time. That is accepted deliberately: whose account pays is
    # not a requalification trigger, and manufacturing an account signal would
    # take a network call, which composition must not make.
    #
    # An Azure *deployment* is still distinguished -- not by `api_base`, which
    # is the shared resource endpoint, but by `model_revision`, because litellm
    # selects the deployment through the model id.
    # The retired Azure credential-level `deployment_name` never described
    # dispatch. Only the selected model id names the actual deployment; legacy
    # payload cleanup must not change this transport identity.
    payload = {
        "api_base": client._api_base,  # noqa: SLF001
        "api_version": client._api_version,  # noqa: SLF001
        "client_implementation": ("reflexio.server.llm.litellm_client.LiteLLMClient"),
        "completion_implementation": (
            "reflexio.server.llm._litellm_text_generation."
            "TextGenerationMixin._completion_with_hard_timeout"
        ),
        "custom_endpoint_model": config.model if custom is not None else None,
        "model_revision": config.model,
        "route": route,
        # v3 drops `credential_digest`. Every previously persisted identity
        # therefore changes exactly once, which is a deliberate one-time
        # requalification rather than drift -- the version bump is what makes
        # the difference legible.
        "schema_version": "offline-tuner-litellm-transport-v3",
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return _LiteLLMTransportIdentity(
        provider=f"litellm:{route}:{digest}",
        model_revision=config.model,
        digest=digest,
        route=route,
        # `route` is litellm's own provider name whenever the configured model
        # is route-qualified (`anthropic/claude-...` -> `custom_llm_provider ==
        # "anthropic"`), and a bare model id is litellm's OpenAI default route.
        # That equality is what lets an observed provider label be checked
        # against the composition rather than trusted.
        expected_observed_provider=(
            "openai" if route == _DEFAULT_OPENAI_ROUTE else route.casefold()
        ),
        bare_model_revision=_strip_route_prefix(config.model, route=route),
        # Excluded from `digest` above by construction; see the dataclass
        # docstring for why this is drift-only state.
        credential_fingerprint=_credential_drift_fingerprint(credential),
    )


# ---------------------------------------------------------------------------
# Provider rate limiting
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT A THROUGHPUT PROBLEM
# ------------------------------------
# A sweep issues a few provider requests per minute, far below published
# provider rate tiers, so nothing here is sized as though the limit were the
# binding constraint -- the ceiling is serialized wall clock. What this exists
# for is the OTHER way a limit is met: deployments that share ONE provider key
# meet a window that rolls at the same instant for all of them.
#
# WHY THE DETECTOR COVERS TWO SHAPES
# ----------------------------------
# MiniMax documents the tiers and does NOT document the error semantics: no
# published 429 contract, no `Retry-After`, no backoff guidance. So this is not
# designed against a published contract, and that is said out loud rather than
# papered over.
#
# UNVERIFIED, AND DELIBERATELY NOT GUESSED AT: nothing here has been measured
# against a live rate-limited MiniMax response, because provoking one needs
# ~200x this project's request rate against the shared production key. Third
# party integration reports describe MiniMax surfacing rate limiting as an
# application-level `base_resp.status_code` INSIDE an HTTP 200 body rather than
# as an HTTP 429 -- which would make a detector keyed on 429 alone fire never.
# Rather than pick whichever shape sounds likelier, the detector covers BOTH:
# the transport shape (HTTP 429 / a `RateLimitError` class) and the application
# shape (a MiniMax `base_resp` rate-limit status code). The asymmetry favours
# that: a false negative costs the whole day, while a false positive costs at
# most the backoff budget below and then reports the same failure it would have
# reported anyway.
#
# HOW A LIMIT ACTUALLY REACHES THIS CODE
# --------------------------------------
# Both shapes arrive as an EXCEPTION, not as a value. The application shape
# carries no `choices`, so the structured parse fails and the ladder wraps a
# final-rung exhaustion into `LiteLLMClientError`. That wrap DROPS the typed
# `upstream_error_type` (`_litellm_text_generation.py`: `wrapped =
# LiteLLMClientError(f"API call failed: {exc}")`) while preserving `__cause__`,
# so the detector reads every link of the chain instead of one object.
_RATE_LIMIT_TRANSPORT_ERROR_NAMES: frozenset[str] = frozenset({"RateLimitError"})
#: HTTP 429 and the phrase, matched as tokens so a `4290` identifier or a
#: 429-millisecond latency in a message cannot be read as a status code.
_RATE_LIMIT_HTTP_STATUS_PATTERN = re.compile(
    r"(?<![0-9])429(?![0-9])|rate[ _-]?limit", re.IGNORECASE
)
#: MiniMax application-level status codes meaning "rate limited", matched ONLY
#: when a `base_resp` envelope appears in the same text. 1002 is the documented
#: "rate limit reached"; 1039 the documented token-rate limit. Matching the
#: bare integers without the envelope would classify any message containing
#: "1002" as a rate limit, which is why the envelope is required.
_MINIMAX_BASE_RESP_RATE_LIMIT_PATTERN = re.compile(
    r"base_resp.{0,160}?(?<![0-9])(?:1002|1039)(?![0-9])", re.IGNORECASE | re.DOTALL
)
#: First backoff, in seconds.
_RATE_LIMIT_BACKOFF_BASE_SECONDS = 2.0
#: Ceiling on ONE sleep. Beyond this a longer single sleep buys nothing the
#: budget below does not already bound, and makes a lease outrun likelier.
_RATE_LIMIT_BACKOFF_CAP_SECONDS = 60.0
#: THE CLIENT-SIDE RETRY BUDGET, and the thing that actually bounds this.
#:
#: A per-request retry COUNT is the wrong control. Every customer deployment
#: shares one key whose window rolls at the same instant, so N deployments each
#: retrying "three times" is 3N requests aimed at one quota inside one window
#: -- the standard amplification failure. A budget in SECONDS, spent across the
#: whole attempt and paired with FULL jitter, bounds the aggregate instead of
#: the per-caller count.
#:
#: SIZED AGAINST THE LEASE, not chosen as an adjective. The attempt lease is
#: `ADMIN_ATTEMPT_LEASE_SECONDS` = 3600 and renewal happens at STAGE boundaries
#: (`runner.LEASE_RENEWAL_FLOOR_SECONDS` = 600), so what has to fit inside the
#: remaining lease is one STAGE, not the attempt. A measured attempt runs 8-11
#: minutes across ALL stages, so its slowest single stage is at most ~660s:
#: 660 + 300 = 960s against the 3000s a fresh lease has before the renewal
#: floor, a margin above 3x. Spending the entire budget inside one stage
#: therefore cannot leave the row in the non-terminal state
#: `20260903010000`'s deploy precondition forbids.
#:
#: The budget is per COMPOSED CLIENT, which in both live callers is per
#: ATTEMPT: `run_open_world_sweep` and the manual route each call
#: `resolve_open_world_pipeline` exactly once, and one pipeline composes one
#: transport shared by all three analysts. So a bundle rate-limited at
#: discovery AND again at held-out spends ONE 300s budget between them, not two
#: -- which is the property that makes this a budget rather than a count.
_RATE_LIMIT_RETRY_BUDGET_SECONDS = 300.0


class OpenWorldProviderRateLimitedError(RuntimeError):
    """The provider rate-limited this attempt and the retry budget is spent.

    Deliberately NOT a subclass of ``LiteLLMClientError``, ``ValueError`` or
    ``TypeError``. The runner's inner stage handlers catch the latter two
    around the very calls that raise this, so inheriting from either would let
    a rate limit be relabelled ``analyst_unqualified`` by a handler with no
    idea what it caught.
    """


def _names_a_rate_limit(link: BaseException) -> bool:
    """Whether one link of an availability failure is a rate limit."""
    upstream = getattr(link, "upstream_error_type", None)
    status = getattr(link, "upstream_status_code", None)
    if status is None:
        status = getattr(link, "status_code", None)
    return (
        type(link).__name__ in _RATE_LIMIT_TRANSPORT_ERROR_NAMES
        or (isinstance(upstream, str) and upstream in _RATE_LIMIT_TRANSPORT_ERROR_NAMES)
        or status == 429
        or bool(_RATE_LIMIT_HTTP_STATUS_PATTERN.search(str(link)))
    )


def _carries_base_resp_limit(link: BaseException) -> bool:
    raw = getattr(link, "raw_content", None)
    return bool(_MINIMAX_BASE_RESP_RATE_LIMIT_PATTERN.search(str(link))) or (
        isinstance(raw, str) and bool(_MINIMAX_BASE_RESP_RATE_LIMIT_PATTERN.search(raw))
    )


def rate_limit_evidence(exc: BaseException) -> str | None:
    """Name the rate-limit shape present in one exception chain, or ``None``.

    Branches on ``classify_llm_failure`` first, then adds the detail the
    verdict deliberately does not carry (decision 1A). The links come from
    ``exception_chain``, the verdict's own walk: the fallback ladder re-wraps a
    final-rung failure into a fresh ``LiteLLMClientError`` and drops the typed
    ``upstream_error_type`` while keeping the chain intact.

    - An availability verdict (``OutageWait`` / ``Transient``) is a rate limit
      when the breaker refused the call (``circuit_open``), a MiniMax
      ``base_resp`` limit is in the chain, or a link is a ``RateLimitError``,
      an HTTP 429, or names one (``transport``). A timeout or an overload is
      not a rate limit.
    - Any other verdict is a rate limit only in the MiniMax application shape:
      a ``base_resp`` limit inside an otherwise-successful body, which reaches
      here as a parse failure the verdict cannot see as availability.

    ``raw_content`` IS READ. ``StructuredOutputParseError``'s own message is
    deliberately content-free -- "(content omitted from logs)" -- and puts the
    provider's body on a separate ``raw_content`` attribute. An empty or
    truncated completion has no ``ValidationError`` beneath it to render the
    body into ``str()``, so ``raw_content`` is the only route for those.
    Reading it is not recording it: nothing here returns, logs or persists any
    fragment of the body -- only which arm matched.

    Args:
        exc (BaseException): Exception raised by the provider call.

    Returns:
        str | None: ``"circuit_open"``, ``"base_resp"``, ``"transport"``, or
        ``None`` when no rate-limit shape is present.
    """
    links = list(exception_chain(exc))
    base_resp = any(_carries_base_resp_limit(link) for link in links)
    # The breaker's state is not evidence of a rate limit; its refusal is.
    verdict = classify_llm_failure(exc, consult_breaker=False)
    if not isinstance(verdict, OutageWait | Transient):
        return "base_resp" if base_resp else None
    # Repeated rate limits open the provider circuit, which then refuses
    # without sending; the same back-off applies, within the same budget.
    if any(isinstance(link, ProviderCircuitOpenError) for link in links):
        return "circuit_open"
    # Before the transport arm: a `base_resp` body's generic text can also
    # match the phrase arm, and the specific shape is the honest evidence.
    if base_resp:
        return "base_resp"
    if any(_names_a_rate_limit(link) for link in links):
        return "transport"
    return None


class _LiteLLMOpenWorldModelClient:
    """The reviewed production transport for qualified open-world calls.

    Any provider and any model id compose, floating aliases included. The one
    route still refused is the custom OpenAI-compatible gateway, and NOT for
    the withdrawn pinning rationale -- it is refused because it defeats two
    checks this class keeps. See ``__init__``.
    """

    def __init__(self, client: LiteLLMClient) -> None:
        transport = _litellm_transport_identity(client)
        # Provider-echo admission remains unqualified for custom gateways:
        # litellm reports an OpenAI-family label, which cannot match the
        # composed "custom-openai-compatible" route. Refuse before the first
        # call even though dispatch now honors the requested model revision.
        if transport.route == _CUSTOM_ENDPOINT_ROUTE:
            raise OpenWorldCapabilityUnavailableError(
                "open-world analysis has no qualified transport for the custom "
                "OpenAI-compatible route: the gateway's observed provider label "
                "cannot match the composed route"
            )
        self._client = client
        self._transport = transport
        # Seconds of sleep this COMPOSED CLIENT may still spend absorbing
        # provider rate limits. Not reset per call: one composition is one
        # attempt in both live callers, and the budget is deliberately shared
        # across every crossing that attempt makes.
        self._rate_limit_budget_remaining = _RATE_LIMIT_RETRY_BUDGET_SECONDS
        # Which detector arm fired on each absorbed crossing, in order.
        #
        # RECORDED HERE AND LOGGED ELSEWHERE, deliberately. A tuner problem
        # line must emit at `TUNER_OUTCOME_LOG_LEVEL` (`offline_tuner/config.py`),
        # because an INFO line from a module logger is invisible under the
        # production logging profile -- but `open_world/` may not import that
        # module: its approved first-party imports are dependency-free leaves,
        # and `config.py` pulls in the config schema and the site-var manager.
        #
        # So the transport reports and `open_world_composition` logs, ONE line
        # per attempt rather than one per backoff. That also bounds the signal:
        # a fleet-wide limit would otherwise emit a WARNING per sleep per org.
        self._absorbed_rate_limits: list[str] = []
        # Injectable ONLY so tests can observe the sleeps rather than serve
        # them. Production always gets the real clock and the real RNG.
        self._sleep: Callable[[float], None] = time.sleep
        self._jitter: Callable[[float, float], float] = random.uniform

    @property
    def rate_limit_budget_remaining(self) -> float:
        """Seconds of rate-limit backoff this composed client may still spend."""
        return self._rate_limit_budget_remaining

    @property
    def absorbed_rate_limits(self) -> tuple[str, ...]:
        """Detector arm of every rate-limited crossing this client slept off."""
        return tuple(self._absorbed_rate_limits)

    def _absorb_rate_limit(self, *, shape: str, crossing: int) -> bool:
        """Sleep off one rate-limited crossing, or report the budget spent.

        FULL jitter -- ``uniform(0, window)`` rather than ``window`` -- and
        that is the half that matters when the key is shared. Every deployment
        sharing one provider key meets the rolling window at the same instant;
        an unjittered (or half-jittered) backoff has them all retry at the same
        instant too, which reproduces the burst that caused the limit. Full
        jitter spreads the same aggregate retry volume across the window.

        Args:
            shape (str): Which detector arm fired, for the log line only.
            crossing (int): 1-based index of the rate-limited crossing, which
                is what the exponential window is computed from.

        Returns:
            bool: ``True`` when a sleep was served and the caller should retry,
            ``False`` when the budget could not fund one.
        """
        window = min(
            _RATE_LIMIT_BACKOFF_BASE_SECONDS * (2 ** max(0, crossing - 1)),
            _RATE_LIMIT_BACKOFF_CAP_SECONDS,
        )
        delay = self._jitter(0.0, window)
        # Clamped to what is LEFT rather than refused outright: a budget with
        # 4s in it should still absorb a limit that clears in 3.
        if delay > self._rate_limit_budget_remaining:
            delay = self._rate_limit_budget_remaining
        if delay <= 0.0:
            return False
        self._rate_limit_budget_remaining -= delay
        self._absorbed_rate_limits.append(shape)
        self._sleep(delay)
        return True

    @property
    def transport_identity(self) -> _LiteLLMTransportIdentity:
        """Return the transport identity sealed when this client was composed."""
        return self._transport

    def current_transport_identity(self) -> _LiteLLMTransportIdentity:
        """Re-derive the transport identity from the live LiteLLM client."""
        return _litellm_transport_identity(self._client)

    def _generation_options(
        self, response_format: type[StrictStructuredOutput], idempotency_key: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        from reflexio.server.services.playbook.rule_evidence.trigger_maintenance import (
            TriggerBatch,
            TriggerGate,
        )
        from reflexio.server.services.playbook.rule_evidence.wire import (
            BoundedTargetDiagnostic,
            GlobalRuleAudit,
            generation_profile,
            output_kind,
        )

        settings = dict(_GENERATION_SETTINGS)
        profile = {}
        if issubclass(response_format, (BoundedTargetDiagnostic, GlobalRuleAudit)):
            profile = generation_profile(
                output_kind(response_format), provider=self._transport.provider
            )
            settings.update(profile)
        elif response_format in (TriggerGate, TriggerBatch):
            profile = generation_profile("tune", provider=self._transport.provider)
            settings.update(profile)
        options = {
            "response_format": response_format,
            "model": self._transport.model_revision,
            "tools": None,
            "fallback_models": [],
            "max_retries": _GENERATION_SETTINGS["shared_max_retries_argument"],
            "max_tokens": settings["max_tokens"],
            "temperature": _GENERATION_SETTINGS["temperature"],
            "top_p": _GENERATION_SETTINGS["top_p"],
            "seed": _GENERATION_SETTINGS["seed"],
            "timeout": _GENERATION_SETTINGS["request_timeout_seconds"],
            "extra_headers": {"Idempotency-Key": idempotency_key},
        }
        if "extra_body" in profile:
            options["extra_body"] = profile["extra_body"]
        return settings, profile, options

    def preview_request(
        self,
        system: str,
        payload: dict[str, object],
        schema: type[StrictStructuredOutput],
    ) -> tuple[int, int, str]:
        """Measure actual native request parameters; no provider or authority call."""
        from reflexio.server.llm.input_budget import (
            measure_request_input_tokens,
            request_input_capacity,
        )

        from ..evidence.native_capacity import verified_native_context

        if _litellm_transport_identity(self._client) != self._transport:
            raise ValueError(
                "LiteLLM transport configuration changed after composition"
            )
        _, _, options = self._generation_options(schema, "0" * 64)
        params = self._client._build_completion_params(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            **options,
        )[0]
        context = verified_native_context(self._client.config.model)
        contract = hashlib.sha256(
            json.dumps(
                {key: value for key, value in params.items() if key != "api_key"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return (
            measure_request_input_tokens(params),
            request_input_capacity(
                params,
                context_limits={self._client.config.model: context},
                prompt_target=context,
            ),
            contract,
        )

    def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        response_format: type[StrictStructuredOutput],
        model_revision: str,
        idempotency_key: str,
    ) -> ObservedStructuredCompletion:
        from reflexio.server.llm.input_budget import extraction_usage_observer
        from reflexio.server.llm.token_accounting import RunTokenTotals

        usage = RunTokenTotals()
        usage_responses = 0
        parent_observer = extraction_usage_observer.get()

        def observe_usage(
            provenance: ModelProvenance,
            totals: RunTokenTotals,
            *,
            finish_reason: str | None = None,
        ) -> None:
            nonlocal usage_responses
            if parent_observer is not None:
                parent_observer(provenance, totals, finish_reason=finish_reason)
            usage.add(**dataclasses.asdict(totals))
            usage_responses += 1

        settings, profile, options = self._generation_options(
            response_format, idempotency_key
        )
        if (
            _litellm_transport_identity(self._client) != self._transport
            or model_revision != self._transport.model_revision
        ):
            raise ValueError(
                "LiteLLM transport configuration changed after composition"
            )
        # Declared here for the closure's ``nonlocal`` binding and for the
        # annotation; both are RE-initialised before every crossing of the
        # retry loop below.
        provider_attempts = 0
        first_messages: list[dict[str, Any]] | None = None

        def guard_exact_request(
            params: dict[str, Any],
            hard_timeout: float,
            ladder: tuple[str, ...],
            parse_structured_output: bool,
        ) -> None:
            """Admit only the one qualified request, and identical re-issues of it.

            This used to refuse outright on the second call, on a literal
            ``provider_attempts != 1``. That contradicted the shared client,
            whose ``_run_rung_plain`` re-issues the request once when the first
            structured body fails to parse (``event=llm_parse_retry``), so ANY
            malformed first response deterministically killed the attempt.

            The two policies disagreed because the count was standing in for a
            property it does not actually express. What qualification attests
            is WHAT was sent -- the exact params, the single-model ladder, no
            fallbacks, no tools, the idempotency key -- not how many times the
            wire carried it. The shared client's parse-retry replays the very
            same ``params`` object at the same model with the same idempotency
            key; nothing about the attested invocation differs. Refusing it
            bought no property and cost an entire attempt.

            What WOULD break the attestation is an unbounded repair turn: the
            validated rung's corrective conversation appends the malformed
            output itself, so the model sees a prompt carrying unframed
            provider content that was never qualified. A bare count could not
            tell that apart from a re-issue. Pinning the message list does, and
            does so directly -- the messages are the thing that must not
            change, so they are what is checked.

            ONE further second-crossing shape is admitted, and only one: the
            shared client's own bounded correction, recognised by
            ``is_structured_output_correction_turn``. It is the qualified
            prompt VERBATIM plus a fixed assistant placeholder and a turn
            listing the pydantic field paths and error tags that failed -- no
            fragment of the model's response, which is what made the validated
            rung's repair inadmissible here. Admitting it is what makes the one
            retry corrective instead of a byte-identical re-ask that, at
            ``temperature=0``, could only fail the same way. A corrective
            crossing must also carry the reviewed client's DERIVED idempotency
            key, never attempt 1's: a provider honouring the header would
            otherwise replay the rejected answer and make the correction a
            silent no-op.

            ``_GENERATION_SETTINGS["provider_attempts"]`` therefore keeps its
            value of 1 and now means what it always should have: at most one
            DISTINCT qualified request. Its value is deliberately unchanged
            because it feeds ``_settings_digest`` -> ``ComponentIdentity``, and
            the packaged qualification suite pins that digest.

            Args:
                params (dict[str, Any]): Exact completion kwargs about to be sent
                hard_timeout (float): Unused; the walk owns attempt timing
                ladder (tuple[str, ...]): Model ladder litellm will walk
                parse_structured_output (bool): Whether the body is parsed strictly

            Raises:
                ProviderRequestGuardError: When the request is not the qualified
                    one, or is a second DIFFERENT request rather than a re-issue
            """
            nonlocal provider_attempts, first_messages
            del hard_timeout
            provider_attempts += 1
            messages = params.get("messages")
            corrective = False
            if provider_attempts == 1:
                first_messages = deepcopy(messages)
            elif provider_attempts > _MAX_QUALIFIED_PROVIDER_ATTEMPTS:
                raise ProviderRequestGuardError(
                    "open-world generation exhausted its re-issue budget"
                )
            elif messages == first_messages:
                pass
            elif is_structured_output_correction_turn(
                first_messages or (), messages or ()
            ):
                corrective = True
            else:
                raise ProviderRequestGuardError(
                    "open-world generation permits a single provider attempt"
                )
            # The corrective crossing is the ONLY one that may carry a
            # different idempotency key, and it must carry exactly the reviewed
            # client's derivation -- not merely "something else".
            expected_key = (
                structured_output_repair_idempotency_key(idempotency_key)
                if corrective
                else idempotency_key
            )
            expected = {
                "extra_headers": {"Idempotency-Key": expected_key},
                "max_tokens": settings["max_tokens"],
                "model": model_revision,
                "num_retries": _GENERATION_SETTINGS["provider_num_retries"],
                "seed": _GENERATION_SETTINGS["seed"],
                "temperature": _GENERATION_SETTINGS["temperature"],
                # Pinned like every other sampling knob: the token cap is only
                # meaningful together with the budget that makes it reachable,
                # and a silently-inherited shorter one would truncate the very
                # completions the cap was raised to admit.
                "timeout": _GENERATION_SETTINGS["request_timeout_seconds"],
                "top_p": _GENERATION_SETTINGS["top_p"],
            }
            if "extra_body" in profile:
                expected["extra_body"] = profile["extra_body"]
            if (
                ladder != (model_revision,)
                or parse_structured_output is not True
                or any(params.get(name) != value for name, value in expected.items())
                or "fallbacks" in params
                or "tools" in params
                or "tool_choice" in params
            ):
                raise ProviderRequestGuardError(
                    "open-world provider request does not match qualified settings"
                )
            from .shared_compaction import shared_consumer_admission

            if shared_consumer_admission.get():
                from reflexio.server.llm.input_budget import check_request_input_budget

                from ..evidence.native_capacity import verified_native_context

                context = verified_native_context(self._client.config.model)
                check_request_input_budget(
                    params,
                    context_limits={self._client.config.model: context},
                    prompt_target=context,
                )

        # THE RATE-LIMIT RETRY LOOP.
        #
        # `provider_attempts` and `first_messages` are reset on every pass, and
        # that is required rather than tidy: `guard_exact_request` caps the
        # re-issue budget at `_MAX_QUALIFIED_PROVIDER_ATTEMPTS`, so without the
        # reset the second rate-limited crossing would be refused by our OWN
        # guard as an over-budget re-ask. Resetting is also the honest
        # accounting -- a rate-limited crossing never reached the model, so the
        # qualified question has not yet been asked, let alone answered.
        #
        # The idempotency key is deliberately NOT re-derived: this is still the
        # same logical attempt, and a provider honouring the header must be
        # free to collapse the retries.
        rate_limited_crossings = 0
        while True:
            # Re-stamped per crossing so `latency_ms` describes the crossing
            # that actually answered. Carrying the first stamp across a backoff
            # would report the sleep as provider latency, on the one record a
            # later reader uses to size timeouts.
            started_at = int(time.time())
            started = time.perf_counter()
            provider_attempts = 0
            first_messages = None
            usage_token = extraction_usage_observer.set(observe_usage)
            try:
                result = self._client.generate_chat_response_with_provenance(
                    [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    provider_request_guard=guard_exact_request,
                    **options,
                )
            except Exception as exc:
                shape = rate_limit_evidence(exc)
                if shape is None:
                    raise
                rate_limited_crossings += 1
                if not self._absorb_rate_limit(
                    shape=shape, crossing=rate_limited_crossings
                ):
                    raise OpenWorldProviderRateLimitedError(
                        "open-world generation was rate limited and the retry "
                        f"budget is spent (shape={shape}, "
                        f"crossings={rate_limited_crossings})"
                    ) from exc
                continue
            finally:
                extraction_usage_observer.reset(usage_token)
            break
        if type(result.value) is not response_format:
            raise TypeError("provider returned the wrong structured output type")
        observed_provider = result.provenance.provider
        observed_model = result.provenance.model_name
        if observed_provider is None or observed_model is None:
            raise ValueError("provider did not return immutable observed provenance")
        # Two separate namespaces meet here. `observed_provider` is the
        # provider's own family label ("openai", "anthropic"); the composed
        # `provider` is Reflexio's attestation of the transport's scope (route,
        # endpoint, API version, model revision, reviewed implementation -- but
        # deliberately not the credential). The provider is never
        # asked to echo the composed string -- no provider could -- so the
        # observed label is checked against the route the composition derived,
        # and only a match licenses stamping the composed identity.
        if (
            observed_provider.strip().casefold()
            != self._transport.expected_observed_provider
        ):
            raise ValueError(
                "observed provider does not match the composed transport route"
            )
        # Silent substitution: the provider answered with a model other than the
        # qualified revision. Refuse rather than record an invocation whose
        # provenance disagrees with the identity that authorized it.
        if (
            _strip_route_prefix(observed_model.strip(), route=self._transport.route)
            != self._transport.bare_model_revision
        ):
            raise ValueError(
                "provider served a different model than the qualified revision"
            )
        raw_usage = result.usage
        final_usage = _final_response_usage(raw_usage)
        return ObservedStructuredCompletion(
            value=cast(StrictStructuredOutput, result.value),
            observed_provider=self._transport.provider,
            observed_model_revision=model_revision,
            called_at=started_at,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            usage_json=(
                canonical_json_bytes(
                    {
                        "scope": "final successful response plus observed-response lower bounds; unavailable usage remains unknown",
                        "final_response": final_usage,
                        "usage_observed_responses": usage_responses,
                        "observed_response_token_lower_bounds": {
                            **dataclasses.asdict(usage),
                            "total_tokens": usage.prompt_tokens
                            + usage.completion_tokens,
                        },
                    }
                ).decode()
                if usage_responses or raw_usage is not None
                else None
            ),
        )


def _final_response_usage(raw_usage: Any) -> dict[str, int | None]:
    """Preserve missing counts instead of turning them into reported zero usage."""
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = (
            raw_usage.get(name)
            if isinstance(raw_usage, dict)
            else getattr(raw_usage, name, None)
        )
        result[name] = count if type(count) is int and count >= 0 else None
    return result


def _assert_transport_conforms(
    transport: _LiteLLMOpenWorldModelClient,
) -> OpenWorldModelClient:
    """Statically assert the production transport satisfies the adapter protocol.

    Never called: the assignment is the assertion, and pyright reports
    ``reportReturnType`` if ``complete``'s signature drifts from
    ``OpenWorldModelClient``. This is the only place that conformance is
    checked -- adapters receive the transport through the ``AdapterClient``
    union, whose fixture arm would absorb a drifted signature, so test call
    sites cannot be relied on to catch it. Do NOT make the class inherit
    ``OpenWorldModelClient`` instead: nominal inheritance makes the assignment
    succeed regardless of signature, which silently disarms this check.
    """
    return transport


def compose_litellm_open_world_model_client(
    client: LiteLLMClient,
) -> _LiteLLMOpenWorldModelClient:
    """Compose the reviewed production transport for open-world analyst calls.

    Args:
        client (LiteLLMClient): The deployment's exact, unmodified LiteLLM
            client, configured with the model the analyst identity qualifies.
            Any provider and any model id are accepted, floating aliases
            (``minimax/MiniMax-M3``, ``claude-sonnet-5``) included.

    Returns:
        _LiteLLMOpenWorldModelClient: A transport that satisfies the
        ``OpenWorldModelClient`` protocol.

    Raises:
        TypeError: ``client`` is not the exact reviewed ``LiteLLMClient``, or
            its reviewed generation or I/O methods have been replaced.
        OpenWorldCapabilityUnavailableError: The client is configured with a
            custom OpenAI-compatible endpoint, whose observed provider label
            and model pin defeat the retained per-request checks.
    """
    return _LiteLLMOpenWorldModelClient(client)


def _require_trusted_provider_composition(
    client: AdapterClient, *, expected_provider: str
) -> None:
    """Verify the caller holds the reviewed transport for ``expected_provider``.

    Args:
        client (AdapterClient): The client an adapter is about to render into.
        expected_provider (str): ``ComponentIdentity.provider`` of the
            qualified identity authorizing this call.

    Raises:
        OpenWorldCapabilityUnavailableError: ``client`` is not the reviewed
            production transport; or its live LiteLLM configuration has drifted
            from the configuration that was composed; or it was composed
            against a provider other than the one the qualified identity names.
    """
    if type(client) is not _LiteLLMOpenWorldModelClient:
        raise OpenWorldCapabilityUnavailableError(
            "a trusted provider composition is required before rendering"
        )
    transport = client.transport_identity
    if client.current_transport_identity() != transport:
        raise OpenWorldCapabilityUnavailableError(
            "LiteLLM transport configuration changed after composition"
        )
    if transport.provider != expected_provider:
        raise OpenWorldCapabilityUnavailableError(
            "composed transport provider does not match the qualified identity"
        )


# The two coordinates of a qualified identity that name the TRANSPORT rather
# than the contract the suite exercises. A qualification subject may vary these
# from the packaged gate identity -- that is the whole point of qualifying a
# production analyst -- but only against a real composed transport that attests
# them. Every other field is pinned by equality below.
_QUALIFICATION_SUBJECT_TRANSPORT_FIELDS = frozenset(
    {"immutable_model_revision", "provider"}
)


def _qualification_subject_contract(identity: ComponentIdentity) -> dict[str, object]:
    """Project the identity fields the qualification suite's meaning depends on."""
    return {
        key: value
        for key, value in _expected_identity_payload(identity).items()
        if key not in _QUALIFICATION_SUBJECT_TRANSPORT_FIELDS
    }


def _require_qualification_subject_identity(
    *,
    role: Literal["discovery", "held_out"],
    identity: ComponentIdentity,
    subject_transport: object | None,
) -> None:
    """Admit the identity a qualification pass will be recorded against.

    Qualification conflates two things if this is left as a single equality:
    WHAT answered the suite (the packaged deterministic fixtures, which must
    stay pinned so a pass means the same thing on every machine) and WHOSE
    qualification it is (the runtime identity being authorized). Deployments
    run the same local suite when enabling the tuner and cache the result by
    exact analyst identity, so the subject is the runtime identity, and requiring
    it to equal the fixture's made production qualification impossible.

    Separating them costs nothing from the gate as long as the subject is still
    *the production adapter for this role*. That is enforced in two parts:

    1. Every field the suite's meaning depends on -- role, prompt id, version
       and template digest, output schema id, version and digest, generation
       settings digest, qualification suite version -- must equal the packaged
       gate identity's exactly. Only ``provider`` and
       ``immutable_model_revision`` are free.
    2. Those two are free only when a real composed transport attests them. A
       ``ComponentIdentity`` is inert data and can be spelled at will; the
       attestation cannot, because ``provider`` is
       ``litellm:{route}:{digest}`` where the digest covers the route, the
       endpoint, the API version, the model revision and the reviewed
       implementation names. Naming a provider string does not produce a
       transport that computes it.

       The digest deliberately does NOT cover the credential: an API key is a
       secret, not a scope, and rotating one must not invalidate a cached
       qualification. A credential swapped underneath a live composition is
       still caught, by ``_LiteLLMTransportIdentity.credential_fingerprint``,
       which ``_require_trusted_provider_composition`` compares but nothing
       persists.

    Args:
        role (Literal["discovery", "held_out"]): The qualification role whose
            packaged gate identity supplies the pinned contract.
        identity (ComponentIdentity): The identity the pass will be sealed
            against.
        subject_transport (object | None): ``None`` to qualify the packaged
            gate identity itself, which needs no provider; otherwise the
            deployment's composed ``_LiteLLMOpenWorldModelClient`` for exactly
            this identity.

    Raises:
        TypeError: ``identity`` is not the exact ``ComponentIdentity`` type.
        ValueError: ``identity`` differs from the packaged gate identity in a
            field the suite's meaning depends on, or it names a non-fixture
            transport without presenting one.
        OpenWorldCapabilityUnavailableError: ``subject_transport`` is not the
            reviewed production transport, its live LiteLLM configuration has
            drifted since composition, or it was composed against a different
            provider or model revision than ``identity`` claims.
    """
    if type(identity) is not ComponentIdentity:
        raise TypeError("qualification subject must be a ComponentIdentity")
    expected = _FROZEN_REFERENCE_IDENTITIES[role]
    if _qualification_subject_contract(identity) != _qualification_subject_contract(
        expected
    ):
        raise ValueError(
            "qualification subject identity does not match the packaged gate contract"
        )
    if subject_transport is None:
        if identity != expected:
            raise ValueError(
                "qualification of a non-fixture identity requires the composed "
                "production transport that attests it"
            )
        return
    _require_trusted_provider_composition(
        cast(AdapterClient, subject_transport), expected_provider=identity.provider
    )
    # `_require_trusted_provider_composition` has established the exact class,
    # so the transport identity below is the composed one, not an attacker's
    # look-alike attribute.
    transport = cast(_LiteLLMOpenWorldModelClient, subject_transport).transport_identity
    if transport.model_revision != identity.immutable_model_revision:
        raise OpenWorldCapabilityUnavailableError(
            "composed transport model revision does not match the qualified identity"
        )


def _require_adapter_role(role: AdapterRole) -> None:
    """Refuse a qualified analyst role that has no production adapter here.

    Args:
        role (AdapterRole): The role a caller is asking this module to render.

    Raises:
        ValueError: ``role`` is admitted by ``AdapterRole`` but this module
            defines no prompt, output type or output schema id for it.
    """
    if role not in _ADAPTER_ROLES:
        raise ValueError(
            f"no open-world production adapter is defined for role {role!r}"
        )


class OpenWorldPromptVersionDriftError(ValueError):
    """The deployed prompt bank does not carry the version this image pins.

    A DEPLOYMENT MISMATCH, and that is the whole reason it has its own class.
    ``_PROMPT_VERSIONS`` ships in the image; the active version ships in the
    prompt bank. An operator who rolls one without the other gets this on every
    composition, for every organization, until the two are put back in step --
    so it is operator-actionable in a way that no other fault reaching the
    sweep's tick boundary is.

    NAMED RATHER THAN MATCHED. The tick boundary has to tell this apart from an
    unknown fault, and the alternative is matching on the message text -- a
    dual-expression identity between a ``raise`` and an ``except`` that a
    reworded message silently breaks, with the failure landing as "unknown
    fault" rather than as a test failure.

    Subclasses ``ValueError`` so every existing caller that catches
    ``ValueError`` around composition behaves exactly as it did before.
    """


def _schema_type(role: AdapterRole) -> type[StrictStructuredOutput]:
    _require_adapter_role(role)
    return _OUTPUT_TYPES[role]


def _schema_digest(role: AdapterRole) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_schema_type(role).model_json_schema())
    ).hexdigest()


def _settings_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(_GENERATION_SETTINGS)).hexdigest()


def generation_settings_digest() -> str:
    """Identity of the reviewed native generation/retry settings."""
    return _settings_digest()


def component_identity_for_adapter(
    *,
    prompt_manager: PromptManager,
    role: AdapterRole,
    provider: str,
    immutable_model_revision: str,
    qualification_suite_version: str,
) -> ComponentIdentity:
    _require_adapter_role(role)
    prompt = prompt_manager.get_prompt_template_identity(_PROMPT_IDS[role])
    if prompt["active_version"] != _PROMPT_VERSIONS[role]:
        raise OpenWorldPromptVersionDriftError(
            "open-world prompt version is not the frozen production version"
        )
    return ComponentIdentity(
        component_role=role,
        provider=provider,
        immutable_model_revision=immutable_model_revision,
        prompt_id=_PROMPT_IDS[role],
        prompt_version=_PROMPT_VERSIONS[role],
        prompt_template_digest=prompt["template_content_digest"],
        output_schema_id=_OUTPUT_SCHEMA_IDS[role],
        output_schema_version=_OUTPUT_SCHEMA_VERSION,
        output_schema_digest=_schema_digest(role),
        generation_settings_digest=_settings_digest(),
        qualification_suite_version=qualification_suite_version,
    )


def _validate_identity(
    identity: ComponentIdentity,
    *,
    prompt_manager: PromptManager,
    role: AdapterRole,
) -> None:
    expected = component_identity_for_adapter(
        prompt_manager=prompt_manager,
        role=role,
        provider=identity.provider,
        immutable_model_revision=identity.immutable_model_revision,
        qualification_suite_version=identity.qualification_suite_version,
    )
    if identity != expected:
        raise ValueError("component identity does not match the production adapter")


def _canonical_user_message(payload: Mapping[str, object]) -> str:
    canonical_json = canonical_json_bytes(canonical_payload(payload))
    encoded = base64.b64encode(canonical_json).decode("ascii")
    framing, version = _UNTRUSTED_FRAMING_VERSION.rsplit("-", maxsplit=1)
    opening = (
        f'<open_world_untrusted_data framing="{framing}" version="{version}" '
        f'byte_length="{len(canonical_json)}">'
    )
    return f"{opening}\n{encoded}\n{_UNTRUSTED_CLOSE}"


_TRUSTED_CITATION_OPEN = "<open_world_trusted_citation_ids>"
_TRUSTED_CITATION_CLOSE = "</open_world_trusted_citation_ids>"
# `fullmatch`, NOT `match` with `^...$`. In Python `$` also matches just before
# a trailing newline, so `match` would have admitted "<64 hex chars>\n" -- a
# string that then goes into a newline-delimited block, one ID per line, where
# a smuggled line break is precisely the thing the character class exists to
# make impossible.
_CITATION_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


def trusted_citation_id_block(citation_ids: Sequence[str]) -> str:
    """Render server-generated citation IDs OUTSIDE the untrusted envelope.

    Why this exists, measured on the live route (``minimax/MiniMax-M3``):
    asked to copy three 64-character hex IDs back out of the
    ``rfc8785-base64-v1`` envelope, the model reproduced **5 of 9** exactly
    across three trials -- it has to decode the base64 mentally first, and
    drops or duplicates a nibble while doing it (observed lengths 63, 65, 66
    against an expected 64). Handed the same three IDs as plaintext it
    reproduced **9 of 9**.

    The candidate role is the only one that needs a byte-exact ID.
    ``EvidenceCitation`` carries an ``exact_quote`` alongside its span ID, so
    ``verify_citations`` repairs a mistranscribed discovery or held-out ID
    against the caller's own text -- resolution by CONTENT, which is what makes
    that repair unforgeable. ``supporting_citation_ids`` is a bare opaque
    token with no second field to resolve by, so one dropped nibble discards
    the entire revision: the rewritten content, the rationale, and every
    correctly copied ID with it.

    Moving these out of the envelope weakens nothing the envelope defends. The
    framing exists so untrusted EVIDENCE TEXT cannot be read as instructions.
    These IDs are server-computed SHA-256 digests over the caller's own spans;
    the guard below admits nothing but ``[0-9a-f]{64}``, so this region cannot
    express an instruction, a link, or a tool request even if a caller tried.
    The evidence itself stays inside the envelope, untouched.

    Args:
        citation_ids (Sequence[str]): The verified discovery citation IDs.

    Returns:
        str: A delimited plaintext block, one ID per line.

    Raises:
        ValueError: If any ID is not a lowercase 64-character hex digest.
    """
    for citation_id in citation_ids:
        if type(citation_id) is not str or not _CITATION_ID_PATTERN.fullmatch(
            citation_id
        ):
            raise ValueError(
                "trusted citation block admits only 64-character hex digests"
            )
    body = "\n".join(citation_ids)
    return f"{_TRUSTED_CITATION_OPEN}\n{body}\n{_TRUSTED_CITATION_CLOSE}"


def _messages_for_adapter(
    *,
    prompt_manager: PromptManager,
    identity: ComponentIdentity,
    role: AdapterRole,
    payload: Mapping[str, object],
) -> tuple[ChatMessage, ...]:
    if (
        type(prompt_manager) is not PromptManager
        or PromptManager.render_prompt_from_identity
        is not _REVIEWED_PROMPT_RENDER_METHOD
        or getattr(
            prompt_manager.render_prompt_from_identity,
            "__func__",
            None,
        )
        is not _REVIEWED_PROMPT_RENDER_METHOD
    ):
        raise TypeError("adapter requires the reviewed PromptManager renderer")
    _validate_identity(identity, prompt_manager=prompt_manager, role=role)
    system = prompt_manager.render_prompt_from_identity(
        _PROMPT_IDS[role],
        {},
        {
            "prompt_id": identity.prompt_id,
            "active_version": identity.prompt_version,
            "template_content_digest": identity.prompt_template_digest,
        },
    )
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=_canonical_user_message(payload)),
    )


def _discovery_payload(request: DiscoveryRequest) -> Mapping[str, object]:
    return discovery_model_payload(
        bundle_digest=request.bundle_digest,
        discovery_sessions=request.discovery_sessions,
        outcome_contract=request.outcome_contract,
        citation_repair=request.citation_repair,
    )


def _held_out_payload(request: HeldOutSessionRequest) -> Mapping[str, object]:
    return held_out_model_input(
        bundle_digest=request.bundle_digest,
        candidate=request.candidate,
        discovery=request.discovery,
        evidence=request.evidence,
        manifest_role=request.manifest_role,
        outcome_contract=request.outcome_contract,
    )
