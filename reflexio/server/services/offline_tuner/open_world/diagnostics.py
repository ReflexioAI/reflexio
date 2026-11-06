"""Content-free attribution for one open-world attempt's terminal failure.

The terminal-failure reason vocabulary (``AttemptTerminalFailureReason``) is a
persisted contract the UI maps to copy, so it stays exactly as it is. Six
distinct exception classes legitimately converge on ``infrastructure_failure``;
what was missing is any record of *which* one fired and what lay beneath it.
This module builds that record.

What is recorded, and why only this
-----------------------------------

For the raised exception and each wrapped cause beneath it:

* the exception class name and its defining module,
* the module and line where that exception surfaced, and
* for a pydantic ``ValidationError`` only, the ``loc``/``type`` pairs of its
  individual errors -- the field path that failed and the fixed pydantic error
  tag (``missing``, ``string_type``, ...), and
* how many errors that ``ValidationError`` carried in total, which is not the
  same as how many are listed: ``MAX_VALIDATION_ERRORS`` truncates the list,
  and without the total a truncated tail is indistinguishable from an absent
  one.

What is deliberately excluded
-----------------------------

**The exception message is never recorded**, in whole or in part. This is not
caution in the abstract: a ``ValidationError`` raised on a provider response
renders as ::

    hypothesis
      Input should be a valid string [type=string_type, input_value=None, ...]
    a
      Field required [type=missing, input_value={'hypothesis': None, 'sec...

The ``input_value=`` fragment is the model's own output, which is derived from
evidence text. ``errors()`` entries carry the same hazard in their ``msg``,
``input`` and ``url`` keys, so only ``loc`` and ``type`` are read. There is no
reliable way to redact arbitrary provider prose, so none is attempted; the
class, the location and the failed field path are enough to attribute a job
without ever touching content.

Field paths are schema identifiers, but an ``extra_forbidden`` error can place
a model-chosen key in ``loc``. Every component is therefore length- and
charset-clamped by :func:`_safe_loc_component`, and a component that fails the
clamp is replaced wholesale rather than truncated.

Volume
------

This record is a fixed-size value written under one reserved key of one
existing job row. It is never appended to and never inserts a row, and the
runner writes it at most once per attempt, so the diagnostic cannot grow a
table the way an unbounded per-exception signal would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from reflexio.server.exception_detail import raised_at
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusalError,
)
from reflexio.server.services.offline_tuner.open_world_job_metadata import (
    ADVISORY_DIAGNOSTIC_METADATA_KEYS,
    CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY,
    TERMINAL_DIAGNOSTIC_METADATA_KEY,
)

__all__ = [
    "ADVISORY_DIAGNOSTIC_METADATA_KEYS",
    "CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "MAX_CHAIN_DEPTH",
    "MAX_LOC_COMPONENTS",
    "MAX_LOC_COMPONENT_LENGTH",
    "MAX_VALIDATION_ERRORS",
    "TERMINAL_DIAGNOSTIC_METADATA_KEY",
    "DiagnosticLink",
    "TerminalFailureDiagnostic",
    "build_terminal_failure_diagnostic",
    "citation_abstain_payload",
    "diagnostic_payload",
    "diagnostic_log_fragment",
]

# ---------------------------------------------------------------------------
# The two reserved keys below are DEFINED in ``open_world_job_metadata`` and
# re-exported here, where their rationale is documented.
#
# They cannot be defined here. The storage freeze validator and both
# offline-tuner validators must read the same key set, and one of them lives
# inside ``offline_tuner/evidence/``, from which importing anything in the
# ``open_world`` package is a hard circular-import failure. Restating the
# strings there instead would be the drift bug that stranded a real job:
# see ``open_world_job_metadata`` for the full account.
# ---------------------------------------------------------------------------

# The reserved key this diagnostic occupies inside the optimizer job's
# ``metadata_json`` object: ``TERMINAL_DIAGNOSTIC_METADATA_KEY``. Nothing else
# may write it.

# A SECOND reserved key, for a different thing entirely: an attempt that
# abstained because its citations failed verification, rather than because the
# analyst judged the evidence insufficient.
#
# Why this needs its own key, and is not a terminal reason
# --------------------------------------------------------
#
# ``_verified_discovery_envelope`` rewrites a memo to ``decision="abstain"``
# when ``verify_citations`` rejects it. The attempt then converges through the
# ordinary abstention path, so a mechanically-rewritten memo and a memo the
# analyst genuinely abstained on arrive at the same terminal outcome and read
# identically in every downstream report. That is the defect: a mechanical
# failure presented as an analyst judgement.
#
# It is NOT fixed by adding a terminal reason.
# ``AttemptTerminalFailureReason`` is a persisted contract the UI maps to copy,
# and an abstention is not a terminal *failure* anyway -- the attempt did
# converge on a decision. So the reason vocabulary stays exactly as it is, and
# the distinction is carried out of band, next to the terminal diagnostic that
# already occupies this metadata object.
#
# Content safety is the same standard the terminal diagnostic holds itself to:
# the rejected span ID and the rejected quote are MODEL OUTPUT derived from
# evidence text, so neither is recorded. Only the verifier's fixed category
# vocabulary and two integer counts are written.
#
# That key is ``CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY``, re-exported above.

# Bumped when the recorded shape changes, so a reader can tell an old record
# from a new one instead of guessing.
DIAGNOSTIC_SCHEMA_VERSION = 3

# Bounds. Each one caps a dimension an adversarial or merely pathological
# exception could otherwise grow without limit.
MAX_CHAIN_DEPTH = 5
MAX_VALIDATION_ERRORS = 10
MAX_LOC_COMPONENTS = 8
MAX_LOC_COMPONENT_LENGTH = 64

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_REDACTED = "<redacted>"


@dataclass(frozen=True)
class DiagnosticLink:
    """One exception in the chain, reduced to non-content facts."""

    relation: str
    exception_class: str
    exception_module: str
    raised_at: str | None
    validation_errors: tuple[str, ...] = ()
    # How many errors the ValidationError carried BEFORE
    # ``MAX_VALIDATION_ERRORS`` truncated ``validation_errors``. Zero when the
    # exception is not a pydantic ValidationError. An integer count is
    # content-free, so this adds no new exposure.
    validation_error_count: int = 0


@dataclass(frozen=True)
class TerminalFailureDiagnostic:
    """Why one attempt converged on its terminal reason."""

    reason: str
    chain: tuple[DiagnosticLink, ...]
    #: The :class:`OpenWorldRefusal` the raised exception carries, or ``None``
    #: for a failure that is not a refusal. Six refusals converge on
    #: ``infrastructure_failure``; this names which one without widening the
    #: persisted reason vocabulary. Added in schema version 3.
    refusal: str | None = None


def _safe_loc_component(component: object) -> str:
    """Clamp one pydantic ``loc`` component to a schema-identifier shape.

    Args:
        component (object): One element of a pydantic error's ``loc`` tuple

    Returns:
        str: The component when it is a plain identifier or index, else a
            fixed redaction marker
    """
    if isinstance(component, int):
        return str(component)
    text = str(component)
    if len(text) <= MAX_LOC_COMPONENT_LENGTH and _SAFE_TOKEN.fullmatch(text):
        return text
    return _REDACTED


def _safe_error_type(error_type: object) -> str:
    """Clamp a pydantic error tag to the same identifier shape.

    Args:
        error_type (object): The ``type`` value of one pydantic error entry

    Returns:
        str: The tag when it is a plain identifier, else a redaction marker
    """
    text = str(error_type)
    if len(text) <= MAX_LOC_COMPONENT_LENGTH and _SAFE_TOKEN.fullmatch(text):
        return text
    return _REDACTED


def _validation_errors(exc: BaseException) -> tuple[tuple[str, ...], int]:
    """Summarize a pydantic ValidationError as ``field.path:error_type`` pairs.

    Reads only ``loc`` and ``type``. The sibling ``msg``, ``input`` and ``url``
    keys carry provider output and are never touched.

    Args:
        exc (BaseException): Exception that may be a pydantic ValidationError

    The total is returned beside the bounded list because the bound is not
    cosmetic. ``DiscoveryMemo`` has nine required fields, so a response of the
    wrong SHAPE reports nine ``missing`` errors before a single model-chosen
    extra key is described -- and ``MAX_VALIDATION_ERRORS`` is ten. A reader of
    the truncated list alone sees exactly one ``extra_forbidden`` entry and
    concludes the model emitted a one-key object, when it may have emitted
    many. The count distinguishes "one extra key" from "the list ran out", and
    it is an integer, so it carries no provider content.

    Returns:
        tuple[tuple[str, ...], int]: Bounded, content-free error summaries and
            the pre-truncation total; ``((), 0)`` when the exception is not a
            pydantic ValidationError
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return (), 0
    try:
        raw = errors()
    except Exception:  # pragma: no cover - defensive; errors() is pure in pydantic
        return (), 0
    if not isinstance(raw, list):
        return (), 0
    summaries: list[str] = []
    for entry in raw[:MAX_VALIDATION_ERRORS]:
        if not isinstance(entry, dict):
            continue
        location = entry.get("loc")
        components = location if isinstance(location, tuple | list) else ()
        path = ".".join(
            _safe_loc_component(component)
            for component in tuple(components)[:MAX_LOC_COMPONENTS]
        )
        summaries.append(f"{path or '<root>'}:{_safe_error_type(entry.get('type'))}")
    return tuple(summaries), len(raw)


def _link(exc: BaseException, relation: str) -> DiagnosticLink:
    validation_errors, validation_error_count = _validation_errors(exc)
    return DiagnosticLink(
        relation=relation,
        exception_class=type(exc).__name__,
        exception_module=type(exc).__module__,
        raised_at=raised_at(exc),
        validation_errors=validation_errors,
        validation_error_count=validation_error_count,
    )


def build_terminal_failure_diagnostic(
    reason: str, cause: BaseException | None
) -> TerminalFailureDiagnostic:
    """Reduce a terminal failure and its exception chain to recordable facts.

    Follows ``__cause__`` first (an explicit ``raise ... from ...``) and falls
    back to ``__context__`` when the chain was implicit and not suppressed --
    the wrapped ``ValidationError`` that motivated this module reaches the
    runner through exactly that implicit path.

    Args:
        reason (str): Terminal reason the attempt converged on, unchanged
        cause (BaseException | None): Exception that triggered the failure, or
            None when the attempt failed on a non-exceptional decision

    Returns:
        TerminalFailureDiagnostic: Bounded, content-free attribution record
    """
    chain: list[DiagnosticLink] = []
    seen: set[int] = set()
    current = cause
    relation = "raised"
    while current is not None and len(chain) < MAX_CHAIN_DEPTH:
        if id(current) in seen:
            break
        seen.add(id(current))
        chain.append(_link(current, relation))
        if current.__cause__ is not None:
            current, relation = current.__cause__, "cause"
        elif current.__context__ is not None and not current.__suppress_context__:
            current, relation = current.__context__, "context"
        else:
            break
    refusal = cause.refusal.value if isinstance(cause, OpenWorldRefusalError) else None
    return TerminalFailureDiagnostic(reason=reason, chain=tuple(chain), refusal=refusal)


def diagnostic_log_fragment(diagnostic: TerminalFailureDiagnostic) -> str:
    """Render a diagnostic as a value-free ``key=value`` log fragment.

    The terminal reason alone cannot say which refusal or fault ended an
    attempt (most attempts in one observed run ended
    ``infrastructure_failure`` with nothing in the logs to tell them apart).
    This names the refusal, the raised class and where it surfaced, and the
    deepest wrapped cause. Same content standard as the stored record: class
    names and ``module:line`` only, never an exception message.

    Args:
        diagnostic (TerminalFailureDiagnostic): The attempt's diagnostic

    Returns:
        str: ``refusal=... exception_type=... raised_at=... cause_type=...
            cause_at=...``; ``none`` marks an absent value
    """
    head = diagnostic.chain[0] if diagnostic.chain else None
    root = diagnostic.chain[-1] if len(diagnostic.chain) > 1 else None
    return (
        f"refusal={diagnostic.refusal or 'none'} "
        f"exception_type={head.exception_class if head else 'none'} "
        f"raised_at={(head.raised_at if head else None) or 'none'} "
        f"cause_type={root.exception_class if root else 'none'} "
        f"cause_at={(root.raised_at if root else None) or 'none'}"
    )


def citation_abstain_payload(
    *, category: str, observed_citations: int, verified_citations: int
) -> dict[str, Any]:
    """Render the record that one abstention was mechanical, not judged.

    Written only when the analyst emitted ``grounded_hypothesis`` and citation
    verification downgraded it. A memo the analyst itself abstained on never
    reaches here, so the mere presence of this key under
    :data:`CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY` is the signal; the fields
    say which check rejected it and how much was lost.

    ``category`` is one of the verifier's fixed
    ``CitationVerificationCategory`` values, or ``"unclassified"`` when the
    rejection was a plain ``ValueError`` carrying no category. It is clamped to
    the same identifier shape every other recorded token is, so a future
    category cannot smuggle prose through this field.

    Args:
        category (str): Verifier category that rejected the citations
        observed_citations (int): How many citations the model emitted
        verified_citations (int): How many survived verification

    Returns:
        dict[str, Any]: JSON-safe object for the reserved metadata key
    """
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "cause": "citation_verification_failed",
        "category": _safe_error_type(category),
        "observed_citations": int(observed_citations),
        "verified_citations": int(verified_citations),
    }


def diagnostic_payload(diagnostic: TerminalFailureDiagnostic) -> dict[str, Any]:
    """Render a diagnostic as the JSON object stored on the job row.

    Args:
        diagnostic (TerminalFailureDiagnostic): Record to serialize

    Returns:
        dict[str, Any]: JSON-safe object for the reserved metadata key
    """
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "reason": diagnostic.reason,
        "refusal": diagnostic.refusal,
        "chain": [
            {
                "relation": link.relation,
                "exception_class": link.exception_class,
                "exception_module": link.exception_module,
                "raised_at": link.raised_at,
                "validation_errors": list(link.validation_errors),
                "validation_error_count": link.validation_error_count,
            }
            for link in diagnostic.chain
        ],
    }
