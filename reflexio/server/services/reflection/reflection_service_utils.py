"""Schemas and result types for the reflection service.

Reflection runs as its own sliding-window step inside the publish flow,
mirroring the existing profile / playbook extractor pattern: a window
of size ``window_size`` (global) advanced every ``stride_size``
interactions. When the gate passes and at least one Assistant
interaction in the window carries citations, the service asks an LLM
whether any cited user playbook / user profile rows should be replaced
in light of how they were applied across the window.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.validators import NonEmptyStr
from reflexio.models.structured_output import StrictStructuredOutput

REFLECTION_OPERATION_NAME = "reflection"


class ReflectionServiceRequest(BaseModel):
    """Input to ``ReflectionService.run``.

    The service is invoked once per publish with these scoping fields;
    it does its own bookmark / window lookup against storage and decides
    whether to fire.

    Args:
        user_id (str): User to scope the bookmark and window to.
        request_id (NonEmptyStr): The publish pass's own request id; used as the
            lineage event ``request_id`` on revise events so B3
            reconstruction can link revisions back to the triggering pass.
            Defaults to a fresh UUID hex so two passes on the same profile
            with no explicit request_id produce distinct lineage events.
            Empty strings and whitespace-only values are rejected at
            construction (``ValidationError``), before any storage write.
        agent_version (str): Agent version of the current publish; copied
            into replacement playbooks.
        source (str | None): Optional source filter for the window.
            Matches the source filter used by extractors. None = all
            sources for this user.
    """

    user_id: str
    request_id: NonEmptyStr = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_version: str = ""
    source: str | None = None


class ReflectionDecision(BaseModel):
    """A single per-citation decision returned by the LLM.

    Default expected outcome is no_change (no revision fields set).

    Attributes:
        target_kind (Literal["profile", "playbook"]): Which kind of
            cited item this decision is about.
        target_id (str): Stable id of the cited row. ``profile_id`` for
            profiles, stringified ``user_playbook_id`` for playbooks.
        new_content (str | None): Replacement content text. Setting any
            of ``new_content`` / ``new_trigger`` / ``new_rationale`` /
            ``new_profile_time_to_live`` flags this decision as a
            revision. Leave all None for no_change.

            Polarity is never declared on the decision: a playbook
            *flip* is expressed purely by rewriting ``new_content`` in
            the opposite orientation (negative wording such as
            ``Avoid`` / ``Do not`` / ``Don't`` / ``Never`` for a
            success→failure flip; affirmative wording for the reverse).
            A flip is LLM-reported via that rewritten ``new_content``
            plus a ``new_rationale`` naming the motivating failure — it
            is not derived from wording.
        new_trigger (str | None): Replacement playbook trigger.
            Optional even on revision; None falls back to the cited
            value. Ignored for profiles.
        new_rationale (str | None): Replacement playbook rationale.
            Same fallback semantics. Required whenever applying a
            ``new_content`` revision (the prompt sets it on substance
            rewrites and flips alike, so a content rewrite needs an
            audit trail naming the motivating failure/observation).
            Ignored for profiles.
        new_profile_time_to_live (ProfileTimeToLive | None): Replacement
            profile TTL. None falls back to the cited value. Ignored
            for playbooks.
        reason (str): Short justification, logged.
    """

    target_kind: Literal["profile", "playbook"]
    target_id: str
    new_content: str | None = None
    new_trigger: str | None = None
    new_rationale: str | None = None
    new_profile_time_to_live: ProfileTimeToLive | None = None
    reason: str = ""


class ReflectionOutput(StrictStructuredOutput):
    """Structured LLM output for one reflection pass."""

    decisions: list[ReflectionDecision] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """Outcome of a single reflection pass for logging / tests.

    Attributes:
        ran (bool): True iff the LLM was actually called.
        gate_open (bool): True when the stride_size bookmark gate
            permitted this run.
        cited_count (int): Distinct citations seen on Assistant
            interactions in the window.
        considered_count (int): Cited rows that were still current and
            therefore handed to the LLM after the post-horizon filter.
        skipped_count (int): Citations skipped because the target row
            was missing or already archived, or eligibility was
            ``deferred`` by the post-horizon filter.
        no_change_count (int): Decisions with no revision fields set.
        revised_count (int): Decisions with at least one revision
            field set. Flips (orientation changes) are LLM-reported via
            the rewritten ``new_content`` + ``new_rationale`` the prompt
            emits and are counted as ordinary revisions — there is no
            separate flip counter, because the prompt sets ``new_rationale``
            on both flips and non-flip content rewrites, so the two cannot
            be distinguished without re-deriving polarity (which is retired).
        trigger_revised_count (int): Subset of applied revisions where
            ``new_trigger`` was set (playbook trigger changed).
        content_revised_count (int): Subset of applied revisions where
            ``new_content`` was set (profile or playbook content changed).
        ttl_changed_count (int): Subset of applied revisions where
            ``new_profile_time_to_live`` was set (profile TTL changed).
        capped_count (int): Revision-intent decisions skipped because the
            per-pass cap (``ReflectionConfig.max_revisions_per_pass``) was
            already reached. no_change decisions never count here.
        failed_count (int): Per-decision failures, logged. Includes both
            apply-step errors and decisions rejected by ``_validate_decision``
            (e.g. a playbook ``new_content`` revision that omits
            ``new_rationale``, which the prompt requires on every playbook
            content edit).
    """

    ran: bool = False
    gate_open: bool = False
    cited_count: int = 0
    considered_count: int = 0
    skipped_count: int = 0
    no_change_count: int = 0
    revised_count: int = 0
    trigger_revised_count: int = 0
    content_revised_count: int = 0
    ttl_changed_count: int = 0
    capped_count: int = 0
    failed_count: int = 0
