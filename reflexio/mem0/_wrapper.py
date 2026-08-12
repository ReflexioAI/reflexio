"""Hosted mem0 client subclasses that mirror learning to Reflexio."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import httpx
from mem0 import AsyncMemoryClient as _Mem0AsyncMemoryClient
from mem0 import MemoryClient as _Mem0MemoryClient
from mem0.client.types import AddMemoryOptions, SearchMemoryOptions

from reflexio.client import ReflexioClient
from reflexio.defaults import resolve_agent_version
from reflexio.mem0._facade import AsyncReflexioFacade, ReflexioFacade

logger = logging.getLogger(__name__)

_SOURCE = "mem0"
_DEFAULT_REFLEXIO_TIMEOUT_SECONDS = 5.0
_REFLEXIO_TOP_K = 5
_ROLE_MAP = {"user": "User", "assistant": "Assistant"}
_SKIPPED_ROLES = {"system"}
_IDENTITY_NAMES = ("user_id", "agent_id", "app_id", "run_id")

ReflexioSearchStatus = Literal["ok", "skipped", "error"]
ReflexioSearchReason = Literal[
    "not_configured",
    "empty_query",
    "missing_user_id",
    "unsupported_identity_filter",
    "conflicting_identity",
    "request_failed",
    "reflexio_rejected",
]


class ReflexioSearchResult(TypedDict):
    status: ReflexioSearchStatus
    reason: ReflexioSearchReason | None
    profiles: list[dict[str, Any]]
    user_playbooks: list[dict[str, Any]]
    agent_playbooks: list[dict[str, Any]]


class ReflexioNamespaceCollisionError(RuntimeError):
    """Raised when an opted-in mem0 result already owns ``reflexio``."""


@dataclass(frozen=True)
class _Identity:
    value: str | None = None
    conflict: bool = False
    unsupported: bool = False


@dataclass(frozen=True)
class _ResolvedIdentities:
    user_id: str | None
    agent_id: str | None
    app_id: str | None
    run_id: str | None
    conflict: bool = False
    unsupported: bool = False


def _validate_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("Reflexio timeout must be a finite positive number")
    normalized = float(timeout)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("Reflexio timeout must be a finite positive number")
    return normalized


def _default_reflexio_client(
    timeout: float,
    api_key: str | None,
    url_endpoint: str | None,
) -> ReflexioClient | None:
    """Build a wrapper-owned client, or return None for pass-through."""
    if not (
        api_key
        or url_endpoint
        or os.environ.get("REFLEXIO_API_KEY")
        or os.environ.get("REFLEXIO_URL")
    ):
        logger.info(
            "REFLEXIO_API_KEY / REFLEXIO_URL not set; reflexio.mem0 runs in "
            "pass-through mode (mem0 only, no Reflexio calls)"
        )
        return None
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if url_endpoint is not None:
        client_kwargs["url_endpoint"] = url_endpoint
    try:
        return ReflexioClient(**client_kwargs)
    except Exception:  # noqa: BLE001 - construction cannot break mem0.
        logger.warning("Failed to construct ReflexioClient")
        return None


def _configure_reflexio(
    reflexio_client: ReflexioClient | None,
    reflexio_timeout: float | None,
    reflexio_api_key: str | None,
    reflexio_url_endpoint: str | None,
) -> ReflexioClient | None:
    if reflexio_client is not None:
        if any(
            value is not None
            for value in (
                reflexio_timeout,
                reflexio_api_key,
                reflexio_url_endpoint,
            )
        ):
            raise ValueError(
                "inline Reflexio configuration cannot be combined with an injected "
                "reflexio_client"
            )
        _validate_timeout(reflexio_client.timeout)
        return reflexio_client
    timeout = _validate_timeout(
        _DEFAULT_REFLEXIO_TIMEOUT_SECONDS
        if reflexio_timeout is None
        else reflexio_timeout
    )
    return _default_reflexio_client(
        timeout,
        api_key=reflexio_api_key,
        url_endpoint=reflexio_url_endpoint,
    )


def _opt(options: Any, kwargs: dict[str, Any], name: str) -> Any:
    if name in kwargs:
        return kwargs[name]
    return getattr(options, name, None) if options is not None else None


def _nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_messages(messages: Any, created_at: int | None) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        raw: list[Any] = [{"role": "user", "content": messages}]
    elif isinstance(messages, dict):
        raw = [messages]
    elif isinstance(messages, list):
        raw = messages
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"role": "user", "content": item}
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").lower()
        if role in _SKIPPED_ROLES:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        interaction: dict[str, Any] = {
            "role": _ROLE_MAP.get(role, role.capitalize()),
            "content": content,
        }
        if created_at is not None:
            interaction["created_at"] = created_at
        normalized.append(interaction)
    return normalized


def _extract_identity(filters: Any, name: str) -> _Identity:
    """Resolve one plain identity from documented top-level/one-level-AND filters."""
    if filters is None:
        return _Identity()
    if not isinstance(filters, dict):
        return _Identity(unsupported=True)
    if "OR" in filters:
        return _Identity(unsupported=True)

    values: list[str] = []

    def inspect(mapping: dict[str, Any]) -> bool:
        if name not in mapping:
            return True
        raw = mapping[name]
        value = _nonempty_str(raw)
        if value is not None:
            values.append(value)
            return True
        return raw is None or (isinstance(raw, str) and not raw.strip())

    if not inspect(filters):
        return _Identity(unsupported=True)
    clauses = filters.get("AND")
    if clauses is not None:
        if not isinstance(clauses, list):
            return _Identity(unsupported=True)
        for clause in clauses:
            if not isinstance(clause, dict) or "AND" in clause or "OR" in clause:
                return _Identity(unsupported=True)
            if not inspect(clause):
                return _Identity(unsupported=True)

    if not values:
        return _Identity()
    if any(value != values[0] for value in values[1:]):
        return _Identity(conflict=True)
    return _Identity(value=values[0])


def _resolve_add_identity(options: Any, kwargs: dict[str, Any], name: str) -> _Identity:
    if value := _nonempty_str(kwargs.get(name)):
        return _Identity(value=value)
    return _extract_identity(_opt(options, kwargs, "filters"), name)


def _resolve_identities(
    options: Any, kwargs: dict[str, Any], *, add: bool
) -> _ResolvedIdentities:
    resolver = _resolve_add_identity if add else None
    filters = _opt(options, kwargs, "filters")
    identities = {
        name: resolver(options, kwargs, name)
        if resolver
        else _extract_identity(filters, name)
        for name in _IDENTITY_NAMES
    }
    return _ResolvedIdentities(
        user_id=identities["user_id"].value,
        agent_id=identities["agent_id"].value,
        app_id=identities["app_id"].value,
        run_id=identities["run_id"].value,
        conflict=any(identity.conflict for identity in identities.values()),
        unsupported=any(identity.unsupported for identity in identities.values()),
    )


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _scope_hash(kind: str, values: dict[str, Any]) -> str:
    payload = _canonical_json({"kind": kind, **values}).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolved_user_id(user_id: str, app_id: str | None) -> str:
    user = _nonempty_str(user_id)
    if user is None:
        raise ValueError("user_id must be a non-empty string")
    app = _nonempty_str(app_id)
    if app is None:
        return user
    digest = _scope_hash("user", {"app_id": app, "user_id": user})
    return f"mem0-user-v1-{digest}"


def _resolved_agent_version(app_id: str | None, agent_id: str | None) -> str:
    app = _nonempty_str(app_id)
    agent = _nonempty_str(agent_id)
    if app is None:
        return agent or resolve_agent_version()
    digest = _scope_hash("agent", {"agent_id": agent, "app_id": app})
    return f"mem0-agent-v1-{digest}"


def _session_id(
    namespace: uuid.UUID,
    user_id: str,
    agent_version: str,
    run_id: str | None,
) -> str:
    run = _nonempty_str(run_id)
    if run is not None:
        digest = _scope_hash(
            "run",
            {
                "agent_version": agent_version,
                "run_id": run,
                "user_id": user_id,
            },
        )
        return f"mem0-run-v1-{digest}"
    payload = _canonical_json(
        {
            "agent_version": agent_version,
            "kind": "fallback",
            "user_id": user_id,
        }
    )
    return f"mem0-run-v1-{uuid.uuid5(namespace, payload).hex}"


def _empty_search_result(
    status: ReflexioSearchStatus, reason: ReflexioSearchReason | None
) -> ReflexioSearchResult:
    return {
        "status": status,
        "reason": reason,
        "profiles": [],
        "user_playbooks": [],
        "agent_playbooks": [],
    }


class _WrapperState:
    _reflexio_client: ReflexioClient | None
    _session_namespace: uuid.UUID
    _reflexio_failure_logged: bool

    def _initialize_reflexio(
        self,
        reflexio_client: ReflexioClient | None,
        reflexio_timeout: float | None,
        reflexio_api_key: str | None,
        reflexio_url_endpoint: str | None,
    ) -> None:
        self._reflexio_client = _configure_reflexio(
            reflexio_client,
            reflexio_timeout,
            reflexio_api_key,
            reflexio_url_endpoint,
        )
        self._session_namespace = uuid.uuid4()
        self._reflexio_failure_logged = False

    def _resolved_user(self, user_id: str, app_id: str | None) -> str:
        return _resolved_user_id(user_id, app_id)

    def _resolved_session(
        self,
        user_id: str,
        app_id: str | None,
        agent_id: str | None,
        run_id: str | None,
    ) -> str:
        resolved_user = _resolved_user_id(user_id, app_id)
        agent_version = _resolved_agent_version(app_id, agent_id)
        return _session_id(
            self._session_namespace, resolved_user, agent_version, run_id
        )

    def _prepare_publish(
        self, messages: Any, options: Any, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], str, str] | None:
        identities = _resolve_identities(options, kwargs, add=True)
        if identities.conflict:
            logger.debug("Skipping Reflexio publish: conflicting mem0 identities")
            return None
        if identities.unsupported:
            logger.debug("Skipping Reflexio publish: unsupported mem0 identity filter")
            return None
        if identities.user_id is None:
            logger.debug("Skipping Reflexio publish: mem0 add() has no user_id")
            return None
        resolved_user = _resolved_user_id(identities.user_id, identities.app_id)
        agent_version = _resolved_agent_version(identities.app_id, identities.agent_id)
        session_id = _session_id(
            self._session_namespace,
            resolved_user,
            agent_version,
            identities.run_id,
        )
        timestamp = _opt(options, kwargs, "timestamp")
        interactions = _normalize_messages(
            messages, timestamp if isinstance(timestamp, int) else None
        )
        if not interactions:
            logger.debug("Skipping Reflexio publish: no publishable messages")
            return None
        return resolved_user, interactions, agent_version, session_id

    def _prepare_search(
        self, query: str, options: Any, kwargs: dict[str, Any]
    ) -> tuple[str, str] | ReflexioSearchResult:
        if self._reflexio_client is None:
            return _empty_search_result("skipped", "not_configured")
        if not isinstance(query, str) or not query.strip():
            return _empty_search_result("skipped", "empty_query")
        identities = _resolve_identities(options, kwargs, add=False)
        if identities.conflict:
            return _empty_search_result("skipped", "conflicting_identity")
        if identities.unsupported:
            return _empty_search_result("skipped", "unsupported_identity_filter")
        if identities.user_id is None:
            return _empty_search_result("skipped", "missing_user_id")
        return (
            _resolved_user_id(identities.user_id, identities.app_id),
            _resolved_agent_version(identities.app_id, identities.agent_id),
        )

    def _serialize_search_response(self, response: Any) -> ReflexioSearchResult:
        if not bool(getattr(response, "success", False)):
            return _empty_search_result("error", "reflexio_rejected")
        return {
            "status": "ok",
            "reason": None,
            "profiles": [item.model_dump(mode="json") for item in response.profiles],
            "user_playbooks": [
                item.model_dump(mode="json") for item in response.user_playbooks
            ],
            "agent_playbooks": [
                item.model_dump(mode="json") for item in response.agent_playbooks
            ],
        }

    def _log_reflexio_failure(self, operation: str) -> None:
        level = logging.DEBUG if self._reflexio_failure_logged else logging.WARNING
        self._reflexio_failure_logged = True
        logger.log(level, "Best-effort Reflexio %s failed", operation)


class MemoryClient(_WrapperState, _Mem0MemoryClient):
    """Hosted mem0 client with automatic Reflexio learning and opt-in search."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        client: httpx.Client | None = None,
        *,
        reflexio_api_key: str | None = None,
        reflexio_url_endpoint: str | None = None,
        reflexio_client: ReflexioClient | None = None,
        reflexio_timeout: float | None = None,
    ) -> None:
        _Mem0MemoryClient.__init__(self, api_key=api_key, host=host, client=client)
        self._initialize_reflexio(
            reflexio_client,
            reflexio_timeout,
            reflexio_api_key,
            reflexio_url_endpoint,
        )
        self._reflexio_facade = ReflexioFacade(
            self._reflexio_client, self._resolved_user, self._resolved_session
        )

    @property
    def reflexio(self) -> ReflexioFacade:
        return self._reflexio_facade

    def add(
        self,
        messages: Any,
        options: AddMemoryOptions | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = _Mem0MemoryClient.add(self, messages, options, **kwargs)
        if self._reflexio_client is None:
            return result
        prepared = self._prepare_publish(messages, options, kwargs)
        if prepared is None:
            return result
        user_id, interactions, agent_version, session_id = prepared
        try:
            response = self._reflexio_client.publish_interaction(
                user_id=user_id,
                interactions=interactions,
                source=_SOURCE,
                agent_version=agent_version,
                session_id=session_id,
                wait_for_response=False,
            )
            if not response.success:
                self._log_reflexio_failure("publish")
        except Exception:  # noqa: BLE001 - best-effort by contract.
            self._log_reflexio_failure("publish")
        return result

    def search(
        self,
        query: str,
        options: SearchMemoryOptions | None = None,
        *,
        include_reflexio: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = _Mem0MemoryClient.search(self, query, options, **kwargs)
        if not include_reflexio:
            return result
        if "reflexio" in result:
            raise ReflexioNamespaceCollisionError(
                "mem0 search result already contains the reserved 'reflexio' key"
            )
        augmented = dict(result)
        prepared = self._prepare_search(query, options, kwargs)
        if isinstance(prepared, dict):
            augmented["reflexio"] = prepared
            return augmented
        user_id, agent_version = prepared
        try:
            response = self._reflexio_client.search(  # type: ignore[union-attr]
                query=query,
                user_id=user_id,
                agent_version=agent_version,
                top_k=_REFLEXIO_TOP_K,
            )
            augmented["reflexio"] = self._serialize_search_response(response)
        except Exception:  # noqa: BLE001 - best-effort by contract.
            self._log_reflexio_failure("search")
            augmented["reflexio"] = _empty_search_result("error", "request_failed")
        return augmented


class AsyncMemoryClient(_WrapperState, _Mem0AsyncMemoryClient):
    """Async hosted mem0 client with native-async Reflexio integration."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        reflexio_api_key: str | None = None,
        reflexio_url_endpoint: str | None = None,
        reflexio_client: ReflexioClient | None = None,
        reflexio_timeout: float | None = None,
    ) -> None:
        _Mem0AsyncMemoryClient.__init__(self, api_key=api_key, host=host, client=client)
        self._initialize_reflexio(
            reflexio_client,
            reflexio_timeout,
            reflexio_api_key,
            reflexio_url_endpoint,
        )
        self._reflexio_facade = AsyncReflexioFacade(
            self._reflexio_client, self._resolved_user, self._resolved_session
        )

    @property
    def reflexio(self) -> AsyncReflexioFacade:
        return self._reflexio_facade

    async def add(
        self,
        messages: Any,
        options: AddMemoryOptions | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await _Mem0AsyncMemoryClient.add(self, messages, options, **kwargs)
        if self._reflexio_client is None:
            return result
        prepared = self._prepare_publish(messages, options, kwargs)
        if prepared is None:
            return result
        user_id, interactions, agent_version, session_id = prepared
        try:
            response = await self._reflexio_client.publish_interaction_async(
                user_id=user_id,
                interactions=interactions,
                source=_SOURCE,
                agent_version=agent_version,
                session_id=session_id,
                wait_for_response=False,
            )
            if not response.success:
                self._log_reflexio_failure("publish")
        except Exception:  # noqa: BLE001 - cancellation remains visible.
            self._log_reflexio_failure("publish")
        return result

    async def search(
        self,
        query: str,
        options: SearchMemoryOptions | None = None,
        *,
        include_reflexio: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await _Mem0AsyncMemoryClient.search(self, query, options, **kwargs)
        if not include_reflexio:
            return result
        if "reflexio" in result:
            raise ReflexioNamespaceCollisionError(
                "mem0 search result already contains the reserved 'reflexio' key"
            )
        augmented = dict(result)
        prepared = self._prepare_search(query, options, kwargs)
        if isinstance(prepared, dict):
            augmented["reflexio"] = prepared
            return augmented
        user_id, agent_version = prepared
        try:
            response = await self._reflexio_client.search_async(  # type: ignore[union-attr]
                query=query,
                user_id=user_id,
                agent_version=agent_version,
                top_k=_REFLEXIO_TOP_K,
            )
            augmented["reflexio"] = self._serialize_search_response(response)
        except Exception:  # noqa: BLE001 - cancellation remains visible.
            self._log_reflexio_failure("search")
            augmented["reflexio"] = _empty_search_result("error", "request_failed")
        return augmented
