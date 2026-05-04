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

from typing import Literal

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.enums import ProfileTimeToLive

REFLECTION_OPERATION_NAME = "reflection"


class ReflectionServiceRequest(BaseModel):
    """Input to ``ReflectionService.run``.

    The service is invoked once per publish with these scoping fields;
    it does its own bookmark / window lookup against storage and decides
    whether to fire.

    Args:
        user_id (str): User to scope the bookmark and window to.
        agent_version (str): Agent version of the current publish; copied
            into replacement playbooks.
        source (str | None): Optional source filter for the window.
            Matches the source filter used by extractors. None = all
            sources for this user.
    """

    user_id: str
    agent_version: str = ""
    source: str | None = None


class ReflectionDecision(BaseModel):
    """A single per-citation decision returned by the LLM.

    Default expected outcome is ``action == "no_change"``.

    Attributes:
        target_kind (Literal["profile", "playbook"]): Which kind of
            cited item this decision is about.
        target_id (str): Stable id of the cited row. ``profile_id`` for
            profiles, stringified ``user_playbook_id`` for playbooks.
        action (Literal["no_change", "replace"]): Whether to leave the
            cited item alone or archive and replace it.
        new_content (str | None): Replacement content; required on
            ``replace``.
        new_trigger (str | None): Replacement playbook trigger.
            Optional even on replace — None falls back to archived
            value. Ignored for profiles.
        new_rationale (str | None): Replacement playbook rationale.
            Same fallback semantics. Ignored for profiles.
        new_profile_time_to_live (ProfileTimeToLive | None): Replacement
            profile TTL. None falls back to archived value. Ignored for
            playbooks.
        reason (str): Short justification, logged.
    """

    target_kind: Literal["profile", "playbook"]
    target_id: str
    action: Literal["no_change", "replace"]
    new_content: str | None = None
    new_trigger: str | None = None
    new_rationale: str | None = None
    new_profile_time_to_live: ProfileTimeToLive | None = None
    reason: str = ""


class ReflectionOutput(BaseModel):
    """Structured LLM output for one reflection pass."""

    decisions: list[ReflectionDecision] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """Outcome of a single reflection pass for logging / tests.

    Attributes:
        ran (bool): True iff the LLM was actually called. False when
            reflection short-circuited (disabled, gate not yet open, no
            citations in window, no current cited rows, LLM error).
        gate_open (bool): True when the stride_size bookmark gate
            permitted this run (regardless of whether citations existed).
        cited_count (int): Distinct citations seen on Assistant
            interactions in the window.
        considered_count (int): Cited rows that were still current and
            therefore handed to the LLM.
        skipped_count (int): Citations skipped because the target row
            is missing or already archived (e.g. by deduplication
            earlier in the publish flow), plus replace decisions whose
            target was no longer current at apply time.
        no_change_count (int): Decisions where the LLM kept the row.
        replaced_count (int): Decisions where a new current row was
            inserted and the cited row was archived.
        failed_count (int): Per-decision apply failures, logged.
    """

    ran: bool = False
    gate_open: bool = False
    cited_count: int = 0
    considered_count: int = 0
    skipped_count: int = 0
    no_change_count: int = 0
    replaced_count: int = 0
    failed_count: int = 0
