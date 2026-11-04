"""The one definition of which optimizer-job metadata keys are advisory.

Why this contract exists at all
-------------------------------

``playbook_optimization_jobs.metadata_json`` is read by two subsystems that
disagreed about what it is.

The open-world runner writes *advisory diagnostics* into it through
``_merge_job_metadata``, which passes only ``metadata_json`` precisely so the
write cannot touch a lifecycle column behind the lease fence. Its own docstring
says so: the record is best effort, it never changes which reason an attempt
reports, and by construction it is written when the attempt has already failed.

The storage layer, meanwhile, treats the same column as half of the job's
**immutable identity** and compares it whole, in three separate places.

So an advisory write silently mutated the job's identity.

The failure this prevents
------------------------

A diagnostic can commit while the stage advance that should follow it raises
``OpenWorldInvocationLeaseConflictError``, leaving the job ``running`` at stage
``evidence_frozen`` with one extra metadata key (for example
``open_world_terminal_failure``) as its *sole* difference from an otherwise
identical job. Because the active-job lookup falls back to
``status IN ('pending', 'running')``, every later freeze for that playbook
rediscovers the stranded job and refuses with
``OptimizationJobIdentityConflictError`` -- not for one attempt cycle but for
all of them, so the playbook can never be tuned again.

Why the definition lives HERE and not next to the diagnostic writer
--------------------------------------------------------------------

The identity validators include the host's frozen-evidence store and its
storage layer. Importing anything from the ``open_world`` package from there
would run the package ``__init__`` and can form a circular import, so the set
lives at package level, where ``offline_tuner/__init__`` is a bare docstring
and every validator and the runner can import it without a cycle.

Three hand-maintained copies of this set would be the same dual-expression
defect in a new place: the moment someone adds a third diagnostic key and
updates two lists out of three, the strand comes straight back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# The reserved key the terminal-failure diagnostic occupies inside the
# optimizer job's ``metadata_json`` object. Nothing else may write it.
TERMINAL_DIAGNOSTIC_METADATA_KEY = "open_world_terminal_failure"

# A SECOND reserved key, for a different thing entirely: an attempt that
# abstained because its citations failed verification, rather than because the
# analyst judged the evidence insufficient. See ``diagnostics.py`` for why that
# distinction is carried out of band instead of as a terminal reason.
CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY = "open_world_citation_abstain"

# THE set. Every writer of an advisory key and every validator that compares
# ``metadata_json`` as identity reads this one name.
#
# Adding a key here makes it advisory EVERYWHERE, so add one only when
# ``_merge_job_metadata`` is what writes it. An identity-bearing key such as
# ``open_world_evidence`` or ``publication_subject_epochs`` must never appear
# here: a genuine change to either one still has to refuse, and a validator
# that stopped refusing anything would be worse than the strand this cures.
ADVISORY_DIAGNOSTIC_METADATA_KEYS: frozenset[str] = frozenset(
    {
        TERMINAL_DIAGNOSTIC_METADATA_KEY,
        CITATION_ABSTAIN_DIAGNOSTIC_METADATA_KEY,
    }
)


def identity_bearing_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the advisory diagnostic keys from a decoded job metadata object.

    Every other key is preserved verbatim, including keys this module has never
    heard of. The exclusion is exactly
    :data:`ADVISORY_DIAGNOSTIC_METADATA_KEYS` and nothing wider.

    Args:
        metadata (Mapping[str, Any]): Decoded ``metadata_json`` object

    Returns:
        dict[str, Any]: The same object without the advisory diagnostic keys
    """
    return {
        key: value
        for key, value in metadata.items()
        if key not in ADVISORY_DIAGNOSTIC_METADATA_KEYS
    }


def identity_bearing_metadata_json(metadata_json: str | None) -> str:
    """Canonicalize one ``metadata_json`` string as its identity-bearing half.

    The single algorithm every whole-column ``metadata_json`` identity
    comparison runs, applied to BOTH sides. Both sides matter: a freshly built
    request carries no diagnostic today, but putting it through the same
    function is what stops the two sides drifting back into the asymmetry this
    exists to remove.

    Args:
        metadata_json (str | None): Durable or requested metadata JSON text

    Returns:
        str: RFC 8785 canonical text of the identity-bearing keys only

    Raises:
        ValueError: When the text is not a JSON object, or holds a value the
            canonical encoder refuses for a value-shaped reason -- a surrogate
            code point, an integer outside IEEE 754's exact range.
            ``json.JSONDecodeError`` subclasses ``ValueError``, so malformed
            text surfaces here rather than silently comparing equal.
        TypeError: When the object holds a non-string key or a value type the
            canonical encoder does not support. ``canonical_json_bytes`` raises
            ``TypeError`` for both, and it does NOT subclass ``ValueError``, so
            a caller that catches only ``ValueError`` lets it escape -- which
            is how a malformed durable metadata object would surface as a bare
            ``TypeError`` instead of the classified conflict error its call
            site raises. Every caller must catch both.
    """
    # Imported HERE, not at module scope, and the reason is the same cycle the
    # module docstring is about -- one step further round than it accounts for.
    #
    # `evidence/__init__` eagerly imports `open_world_store`, and
    # `open_world_store` imports THIS module. So a module-scope
    # `from ...evidence.manifests import canonical_json_bytes` runs
    # `evidence/__init__`, which re-enters this module before it has bound a
    # single name:
    #
    #     ImportError: cannot import name 'identity_bearing_metadata' from
    #     partially initialized module '...open_world_job_metadata'
    #
    # That only bites when this module is imported FIRST -- every production
    # importer happens to reach `evidence` beforehand, which is why it went
    # unnoticed -- but "importable from the runner and every validator
    # without any cycle" is a claim about the module, not about lucky import
    # order. A direct unit
    # test of this function is exactly the first-import case.
    from reflexio.server.services.offline_tuner.evidence.manifests import (
        canonical_json_bytes,
    )

    decoded = json.loads(metadata_json or "{}")
    if not isinstance(decoded, dict):
        raise ValueError("optimizer job metadata_json is not a JSON object")
    return canonical_json_bytes(identity_bearing_metadata(decoded)).decode()
