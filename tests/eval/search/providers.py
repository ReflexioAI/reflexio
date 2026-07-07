"""Seeding + backend providers for the search eval harness.

A *search provider* maps a golden case (see ``golden_set/search/README.md``)
to a ``ProviderRun``: it seeds a **fresh, isolated** temp SQLite store with
the case's entities (controlled relative timestamps), runs one search
backend over the case's query, and returns the raw ``UnifiedSearchResponse``
plus the case-key → real-storage-id map the runner needs to score rankings.

Timestamps are seeded verbatim: ``add_user_profile`` writes
``last_modified_timestamp`` from the model and the playbook save paths write
``created_at`` from the model, so ``age_days`` in the YAML maps exactly to
entity age at query time — no clock patching required.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus, ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.profile.profile_generation_service_utils import (
    calculate_expiration_timestamp,
)
from reflexio.server.services.unified_search_service import run_unified_search

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import UnifiedSearchResponse
    from reflexio.server.services.storage.storage_base import BaseStorage

_SEED_REQUEST_ID = "eval-seed"
_DEFAULT_USER_ID = "u1"
_DEFAULT_AGENT_VERSION = "eval"
SECONDS_PER_DAY = 86_400


@dataclass
class ProviderRun:
    """One backend run over one case.

    Attributes:
        response: The raw ``UnifiedSearchResponse`` from the backend.
        key_to_id: Case-local seed key → real storage id (as str).
        latency_ms: Wall-clock latency of the backend call only (excludes
            seeding).
    """

    response: UnifiedSearchResponse
    key_to_id: dict[str, str]
    latency_ms: float


SearchProvider = Callable[[dict[str, Any]], ProviderRun]


def _age_to_epoch(now: int, spec: dict[str, Any]) -> int:
    return now - int(float(spec["age_days"]) * SECONDS_PER_DAY)


def seed_case_entities(
    storage: BaseStorage, case: dict[str, Any], now: int
) -> dict[str, str]:
    """Seed a case's entities into ``storage`` with controlled timestamps.

    Args:
        storage (BaseStorage): The (fresh) storage to seed.
        case (dict): The golden case dict.
        now (int): Epoch seconds the relative ``age_days`` resolve against.

    Returns:
        dict[str, str]: Case-local seed key → real storage id.
    """
    key_to_id: dict[str, str] = {}

    for spec in case.get("seeded_profiles", []):
        ttl = ProfileTimeToLive(spec.get("ttl", "infinity"))
        last_modified = _age_to_epoch(now, spec)
        profile = UserProfile(
            profile_id=spec["key"],
            user_id=spec.get("user_id", _DEFAULT_USER_ID),
            content=spec["content"],
            last_modified_timestamp=last_modified,
            generated_from_request_id=_SEED_REQUEST_ID,
            profile_time_to_live=ttl,
            expiration_timestamp=calculate_expiration_timestamp(last_modified, ttl),
        )
        storage.add_user_profile(profile.user_id, [profile])
        key_to_id[spec["key"]] = spec["key"]

    for spec in case.get("seeded_user_playbooks", []):
        playbook = UserPlaybook(
            user_id=spec.get("user_id", _DEFAULT_USER_ID),
            agent_version=spec.get("agent_version", _DEFAULT_AGENT_VERSION),
            request_id=_SEED_REQUEST_ID,
            playbook_name=spec.get("playbook_name", ""),
            trigger=spec.get("trigger"),
            content=spec["content"],
            rationale=spec.get("rationale"),
            created_at=_age_to_epoch(now, spec),
        )
        storage.save_user_playbooks([playbook])
        key_to_id[spec["key"]] = str(playbook.user_playbook_id)

    for spec in case.get("seeded_agent_playbooks", []):
        playbook = AgentPlaybook(
            agent_version=spec.get("agent_version", _DEFAULT_AGENT_VERSION),
            playbook_name=spec.get("playbook_name", ""),
            trigger=spec.get("trigger"),
            content=spec["content"],
            rationale=spec.get("rationale"),
            created_at=_age_to_epoch(now, spec),
            playbook_status=PlaybookStatus(spec.get("playbook_status", "approved")),
        )
        saved = storage.save_agent_playbooks([playbook])
        key_to_id[spec["key"]] = str(saved[0].agent_playbook_id)

    return key_to_id


def apply_config_overrides(ctx: Any, case: dict[str, Any]) -> None:
    """Apply a case's ``config_overrides`` to the org config before seeding.

    Used e.g. by ``factkey_paraphrase`` to enable write-time document
    expansion (``enable_document_expansion``) so seeded entities get
    ``expanded_terms``. The expansion flag is constructor-bound on storage,
    so it is mirrored onto the already-created instance as well.

    Args:
        ctx: The per-case ``RequestContext``.
        case (dict): The golden case dict.
    """
    overrides = case.get("config_overrides") or {}
    if not overrides:
        return
    config = ctx.configurator.get_config()
    ctx.configurator.set_config(config.model_copy(update=dict(overrides)))
    if "enable_document_expansion" in overrides and hasattr(
        ctx.storage, "_enable_document_expansion"
    ):
        ctx.storage._enable_document_expansion = bool(
            overrides["enable_document_expansion"]
        )


def build_request(
    case: dict[str, Any], *, default_search_mode: SearchMode
) -> UnifiedSearchRequest:
    """Build the ``UnifiedSearchRequest`` for a case.

    ``request_overrides`` in the case YAML win over the defaults; the
    default ``user_id`` is ``u1`` (matching the seeding default) so the
    profiles arm is always reachable.

    Args:
        case (dict): The golden case dict.
        default_search_mode (SearchMode): Mode used unless the case
            overrides it (mocked runs use FTS for keyless determinism).

    Returns:
        UnifiedSearchRequest: The request the backend runs.
    """
    overrides = dict(case.get("request_overrides") or {})
    overrides.setdefault("user_id", _DEFAULT_USER_ID)
    overrides.setdefault("search_mode", default_search_mode)
    return UnifiedSearchRequest(
        query=case["query"],
        conversation_history=case.get("conversation_history") or None,
        **overrides,
    )


def make_classic_search_provider(
    *,
    storage_base_dir: str,
    llm_client: Any,
    search_mode: SearchMode = SearchMode.FTS,
    enable_reformulation: bool = False,
    pre_retrieval_model_name: str | None = None,
) -> SearchProvider:
    """Build a classic unified search provider.

    Each case gets a fresh ``RequestContext`` (unique org id → isolated temp
    SQLite store), is seeded, then runs :func:`run_unified_search` — the same
    entry point ``/api/search`` uses, including the storage's own recency
    config, so the baseline matches production behavior.

    With ``enable_reformulation=True`` the pre-search reformulation call runs
    (real LLM required) and its temporal signals drive the time-sensitive
    behavior — this is the "temporal classic" backend under evaluation.

    Args:
        storage_base_dir: Directory the per-case SQLite stores live under.
        llm_client: LLM client passed through to the search pipeline
            (unused when reformulation is off and mode is FTS).
        search_mode (SearchMode): Default search mode for cases that don't
            override it.
        enable_reformulation: Run the reformulation (+ temporal signals)
            LLM call before retrieval.
        pre_retrieval_model_name: Optional explicit reformulation model.

    Returns:
        SearchProvider: Callable mapping a case to a :class:`ProviderRun`.
    """
    from reflexio.server.api_endpoints.request_context import RequestContext

    def provider(case: dict[str, Any]) -> ProviderRun:
        org_id = f"eval-search-{case['id']}"
        ctx = RequestContext(org_id=org_id, storage_base_dir=storage_base_dir)
        storage = ctx.storage
        if storage is None:
            raise RuntimeError(f"eval storage failed to initialize for {org_id}")
        apply_config_overrides(ctx, case)
        now = int(datetime.now(UTC).timestamp())
        key_to_id = seed_case_entities(storage, case, now)
        request = build_request(case, default_search_mode=search_mode)
        if enable_reformulation:
            request = request.model_copy(update={"enable_reformulation": True})

        start = time.monotonic()
        response = run_unified_search(
            request=request,
            org_id=org_id,
            storage=storage,
            llm_client=llm_client,
            prompt_manager=ctx.prompt_manager,
            pre_retrieval_model_name=pre_retrieval_model_name,
            recency=getattr(storage, "recency", None),
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        return ProviderRun(
            response=response, key_to_id=key_to_id, latency_ms=latency_ms
        )

    return provider
