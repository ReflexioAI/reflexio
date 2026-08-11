"""mem0 ``MemoryClient`` subclass that mirrors traces and learnings to Reflexio.

Every Reflexio touchpoint is best-effort: failures are logged and swallowed
so Reflexio availability never breaks the customer's mem0 code path (same
contract as the openclaw ``reflexio_adapter``).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from mem0 import MemoryClient as _Mem0MemoryClient

from reflexio.client import ReflexioClient
from reflexio.defaults import resolve_agent_version

logger = logging.getLogger(__name__)

_SOURCE = "mem0"
_REFLEXIO_TIMEOUT_SECONDS = 10
_REFLEXIO_TOP_K = 5
_ROLE_MAP = {"user": "User", "assistant": "Assistant"}
_SKIPPED_ROLES = {"system"}


def _default_reflexio_client() -> ReflexioClient | None:
    """Build a ReflexioClient from env config, or None when unconfigured or broken.

    Without this guard, an unconfigured environment would fall through to the
    client's production default URL and every add()/search() would fire a
    doomed unauthenticated request at it.
    """
    if not (os.environ.get("REFLEXIO_API_KEY") or os.environ.get("REFLEXIO_URL")):
        logger.info(
            "REFLEXIO_API_KEY / REFLEXIO_URL not set; reflexio.mem0 runs in "
            "pass-through mode (mem0 only, no Reflexio calls)"
        )
        return None
    try:
        return ReflexioClient(timeout=_REFLEXIO_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 — wrapper must never break the mem0 path.
        logger.warning("Failed to construct ReflexioClient: %s", exc)
        return None


def _opt(options: Any, kwargs: dict[str, Any], name: str) -> Any:
    """Read a mem0 call parameter, kwargs first then the options model.

    Mirrors mem0's own precedence: kwargs override ``options`` fields.
    """
    if name in kwargs:
        return kwargs[name]
    return getattr(options, name, None) if options is not None else None


def _nonempty_str(value: Any) -> str | None:
    """Return the value if it is a non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _normalize_messages(messages: Any, created_at: int | None) -> list[dict[str, Any]]:
    """Convert mem0 ``add()`` messages into Reflexio interaction dicts.

    Accepts the shapes mem0 accepts (str, dict, list of dicts or strs);
    skips system messages and items without non-empty string content.
    """
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


def _extract_identity(filters: Any, name: str) -> str | None:
    """Pull a plain-string identity field from mem0 filters.

    Handles the top-level form ``{"user_id": "u1"}`` and one level of
    ``{"AND": [{"user_id": "u1"}, ...]}``. Operator forms like
    ``{"user_id": {"in": [...]}}`` identify no single value and return None.
    """
    if not isinstance(filters, dict):
        return None
    if value := _nonempty_str(filters.get(name)):
        return value
    clauses = filters.get("AND")
    if not isinstance(clauses, list):
        return None
    for clause in clauses:
        if isinstance(clause, dict) and (value := _nonempty_str(clause.get(name))):
            return value
    return None


def _resolve_add_identity(
    options: Any, kwargs: dict[str, Any], name: str
) -> str | None:
    """Resolve an add identity, preferring legacy top-level kwargs."""
    return _nonempty_str(kwargs.get(name)) or _extract_identity(
        _opt(options, kwargs, "filters"), name
    )


class MemoryClient(_Mem0MemoryClient):
    """mem0 MemoryClient that also publishes traces to Reflexio.

    Reflexio configuration comes from the ``REFLEXIO_API_KEY`` and
    ``REFLEXIO_URL`` environment variables, or a pre-built client passed as
    ``reflexio_client``. When neither env var is set and no client is passed,
    the wrapper is a pure pass-through (no Reflexio calls). All other
    behavior is inherited from mem0.
    """

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        client: Any = None,
        *,
        reflexio_client: ReflexioClient | None = None,
    ) -> None:
        super().__init__(api_key=api_key, host=host, client=client)
        self._reflexio = (
            reflexio_client
            if reflexio_client is not None
            else _default_reflexio_client()
        )
        # Namespaces deterministic fallback sessions to this client instance.
        # The derived UUID also includes user/agent identity so a long-lived
        # client cannot group unrelated users or agents into one session.
        self._session_namespace = uuid.uuid4()
        # Debounce: warn on the first Reflexio failure, DEBUG afterwards, so a
        # sustained outage doesn't spam the host application's logs.
        self._reflexio_failure_logged = False

    def add(self, messages: Any, options: Any = None, **kwargs: Any) -> Any:
        """Add memories via mem0, then best-effort publish the trace to Reflexio."""
        result = super().add(messages, options, **kwargs)
        self._publish_to_reflexio(messages, options, kwargs)
        return result

    def search(self, query: str, options: Any = None, **kwargs: Any) -> Any:
        """Search mem0, then best-effort add Reflexio learnings as sibling keys.

        On success the returned dict additionally carries
        ``reflexio_profiles``, ``reflexio_user_playbooks``, and
        ``reflexio_agent_playbooks``. On any Reflexio failure (or when no
        plain user_id is present in filters) the keys are absent and the
        payload is exactly what unwrapped mem0 returned.
        """
        result = super().search(query, options, **kwargs)
        return self._augment_search_result(result, query, options, kwargs)

    def _publish_to_reflexio(
        self, messages: Any, options: Any, kwargs: dict[str, Any]
    ) -> None:
        if self._reflexio is None:
            return
        try:
            user_id = _resolve_add_identity(options, kwargs, "user_id")
            if user_id is None:
                logger.debug("Skipping Reflexio publish: mem0 add() has no user_id")
                return
            agent_version = (
                _resolve_add_identity(options, kwargs, "agent_id")
                or resolve_agent_version()
            )
            run_id = _resolve_add_identity(options, kwargs, "run_id")
            session_identity = f"{user_id}\0{agent_version}"
            session_id = run_id or (
                f"mem0-{uuid.uuid5(self._session_namespace, session_identity).hex}"
            )
            # Only int (unix seconds) timestamps map to created_at; mem0 also
            # accepts datetime/ISO strings, which fall back to "now" here.
            timestamp = _opt(options, kwargs, "timestamp")
            interactions = _normalize_messages(
                messages, timestamp if isinstance(timestamp, int) else None
            )
            if not interactions:
                logger.debug("Skipping Reflexio publish: no publishable messages")
                return
            self._reflexio.publish_interaction(
                user_id=user_id,
                interactions=interactions,
                source=_SOURCE,
                agent_version=agent_version,
                session_id=session_id,
                wait_for_response=False,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort by contract.
            self._log_reflexio_failure("publish", exc)

    def _augment_search_result(
        self, result: Any, query: str, options: Any, kwargs: dict[str, Any]
    ) -> Any:
        if self._reflexio is None or not isinstance(result, dict):
            return result
        if not isinstance(query, str) or not query.strip():
            # Reflexio search requires a non-empty query; skip rather than
            # fail-and-warn on every blank mem0 search.
            return result
        try:
            filters = _opt(options, kwargs, "filters")
            user_id = _extract_identity(filters, "user_id")
            if user_id is None:
                logger.debug(
                    "Skipping Reflexio search augmentation: no plain user_id in filters"
                )
                return result
            agent_version = (
                _extract_identity(filters, "agent_id") or resolve_agent_version()
            )
            response = self._reflexio.search(
                query=query,
                user_id=user_id,
                agent_version=agent_version,
                top_k=_REFLEXIO_TOP_K,
            )
            # Build all three lists before assigning any key, so a failure
            # leaves the mem0 payload entirely untouched.
            profiles = [p.model_dump(mode="json") for p in response.profiles]
            user_playbooks = [
                p.model_dump(mode="json") for p in response.user_playbooks
            ]
            agent_playbooks = [
                p.model_dump(mode="json") for p in response.agent_playbooks
            ]
            result["reflexio_profiles"] = profiles
            result["reflexio_user_playbooks"] = user_playbooks
            result["reflexio_agent_playbooks"] = agent_playbooks
        except Exception as exc:  # noqa: BLE001 — best-effort by contract.
            self._log_reflexio_failure("search", exc)
        return result

    def _log_reflexio_failure(self, operation: str, exc: Exception) -> None:
        """Log a best-effort Reflexio failure, warning only on the first one."""
        level = logging.DEBUG if self._reflexio_failure_logged else logging.WARNING
        self._reflexio_failure_logged = True
        logger.log(level, "Best-effort Reflexio %s failed: %s", operation, exc)
