"""OSS FastAPI app composer (Tier3 A2).

``create_app`` is a thin composition root: it builds the FastAPI instance, wires
the five middleware (in order), registers the CORS + rate-limit + auth-override
seams, mounts the data-plane routers, and attaches the capability lifespan loop.

The data-plane handlers live in ``reflexio.server.routes.<domain>`` sub-routers.
``core_router`` aggregates every sub-router via ``include_router`` so it remains
the single data-plane router surface the enterprise composition mounts and
iterates (e.g. the QPS billable-endpoint scan over ``core_router.routes``).

``limiter`` / ``configure_rate_limiter`` are re-exported from
``reflexio.server.rate_limit`` and the middleware classes from
``reflexio.server.middleware`` for backwards-compatible imports.
"""

import inspect
import logging
import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.server.deployment_profile import DeploymentProfile
    from reflexio.server.extensions import AppContext, CapabilityRegistry

from fastapi import APIRouter, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from reflexio.server.api_endpoints import (
    health_api,
    pending_tool_call_api,
    stall_state_api,
)
from reflexio.server.auth import (
    DEFAULT_ORG_ID,
    default_billing_gate,
    default_get_caller_type,
    default_get_org_id,
)
from reflexio.server.middleware import (
    BodySizeLimitMiddleware,
    BotProtectionMiddleware,
    CorrelationIdMiddleware,
    SecurityHeadersMiddleware,
    TimeoutMiddleware,
    _resolve_cors_origins,
)
from reflexio.server.operation_limiter import log_publish_hardware_capacity
from reflexio.server.rate_limit import configure_rate_limiter, limiter
from reflexio.server.routes import (
    braintrust,
    config,
    evaluation,
    experiments,
    interactions,
    playbooks,
    profiles,
    provenance,
    search,
    system,
)

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility — callers that did
# ``from reflexio.server.api import default_get_org_id`` / ``DEFAULT_ORG_ID`` /
# ``limiter`` / ``configure_rate_limiter`` / ``core_router`` continue to work.
__all__ = [
    "DEFAULT_ORG_ID",
    "configure_rate_limiter",
    "core_router",
    "create_app",
    "default_billing_gate",
    "default_get_caller_type",
    "default_get_org_id",
    "limiter",
]


def _log_multi_worker_daemons() -> None:
    """Warn once at startup when multiple worker processes will run daemons.

    ``--workers N`` (N>1) runs the full daemon set in every worker process
    (duplicate ticking). Safe by the concurrent-tick invariant — every daemon
    tick is concurrent-safe — but worth one visible line (design D3).

    Reuses ``embedder_warmup._detected_worker_count`` (checks
    ``REFLEXIO_SERVER_WORKERS`` then falls back to ``WEB_CONCURRENCY``)
    instead of re-implementing a narrower env read here. When the count is
    undetectable (``None``), this logs nothing — it cannot verify the count,
    so it must not assume a single worker.
    """
    from reflexio.server.llm.providers.embedder_warmup import (
        _detected_worker_count,
    )

    workers = _detected_worker_count()
    if workers is not None and workers > 1:
        logger.warning(
            "event=multi_worker_daemons workers=%d — background daemons tick in "
            "every worker process (duplicate ticking; safe: ticks are "
            "concurrent-safe by design)",
            workers,
        )


# ``core_router`` stays an aggregator: it ``include_router``s every domain
# sub-router so ``core_router.routes`` still enumerates all data-plane handlers
# (enterprise QPS enforcement iterates this list). The include order is
# functionally irrelevant — all paths are distinct literals with no overlapping
# templates, so FastAPI's first-match routing is unaffected.
core_router = APIRouter()
for _domain_router in (
    system.router,
    interactions.router,
    profiles.router,
    playbooks.router,
    search.router,
    provenance.router,
    evaluation.router,
    experiments.router,
    braintrust.router,
    config.router,
):
    core_router.include_router(_domain_router)


# Paths that should remain publicly accessible (no lock icon in Swagger)
_PUBLIC_PATHS = frozenset(
    {"/", "/health", "/meta/version", "/token", "/docs", "/openapi.json"}
)
_PUBLIC_PATH_PREFIXES = ("/api/register", "/api/registration-config", "/api/auth/")


def _contains_non_finite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite_number(item) for item in value)
    return False


async def _safe_request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Keep non-standard JSON numbers from breaking FastAPI's 422 response."""
    errors = exc.errors()
    if not any(_contains_non_finite_number(error.get("input")) for error in errors):
        return await request_validation_exception_handler(request, exc)
    safe_errors = [
        {key: value for key, value in error.items() if key not in {"input", "ctx"}}
        for error in errors
    ]
    return JSONResponse(status_code=422, content={"detail": safe_errors})


def _add_openapi_security(app: FastAPI) -> None:
    """Inject Bearer auth security scheme into the OpenAPI spec.

    Overrides the default openapi() method to add a global HTTPBearer security
    requirement while exempting public endpoints (login, register, health, etc.).
    """
    original_openapi = app.openapi

    def custom_openapi() -> dict:  # type: ignore[type-arg]
        if app.openapi_schema:
            return app.openapi_schema

        schema = original_openapi()

        # Add security scheme
        schema.setdefault("components", {}).setdefault("securitySchemes", {})
        schema["components"]["securitySchemes"]["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "API key or JWT token. Pass as: Authorization: Bearer <token>",
        }

        # Apply security globally, then remove from public endpoints
        for path, methods in schema.get("paths", {}).items():
            is_public = path in _PUBLIC_PATHS or any(
                path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES
            )
            for method_detail in methods.values():
                if isinstance(method_detail, dict):
                    if is_public:
                        method_detail["security"] = []
                    else:
                        method_detail.setdefault("security", [{"BearerAuth": []}])

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


# Injection seam: enterprise registers a resolver that returns a REAL org id at
# composition time, exactly like gc_scheduler.set_org_id_provider. Needed because
# the request-scoped `get_org_id` dependency takes FastAPI parameters and so can
# never be called outside a request -- without this hook the lifespan falls
# through to the `default_get_org_id()` sentinel ("self-host-org"), which matches
# no organization row in ANY deployment mode. Every config read keyed on it then
# raises, which silently skipped the lineage-GC and extraction-resume schedulers.
_lifespan_org_id_resolver: Callable[[], str] | None = None


def set_lifespan_org_id_resolver(resolver: Callable[[], str] | None) -> None:
    """Register the resolver used to pick the bootstrap org for lifespan schedulers.

    Args:
        resolver (Callable[[], str] | None): Callable returning a real org id, or
            ``None`` to clear the hook (used by tests to restore OSS defaults).
    """
    global _lifespan_org_id_resolver  # noqa: PLW0603
    _lifespan_org_id_resolver = resolver


def _resolve_lifespan_org_id(get_org_id: Callable[..., str] | None) -> str:
    """Resolve the bootstrap org ID for lifespan schedulers without a request context.

    Args:
        get_org_id (Callable[..., str] | None): Custom org-ID dependency, or None for
            the default single-tenant mode.

    Returns:
        str: The resolved org ID string.
    """
    from reflexio.server.auth import default_get_org_id

    if _lifespan_org_id_resolver is not None:
        try:
            resolved = str(_lifespan_org_id_resolver())
            if resolved:
                return resolved
            logger.warning("Lifespan org_id resolver returned empty; using default org")
        except Exception:
            logger.exception("Lifespan org_id resolver failed; using default org")

    if get_org_id is None:
        return default_get_org_id()
    try:
        signature = inspect.signature(get_org_id)
    except (TypeError, ValueError):
        return default_get_org_id()
    if signature.parameters:
        return default_get_org_id()
    try:
        return str(get_org_id())
    except Exception:
        logger.exception("Failed to resolve lifespan org_id; using default org")
        return default_get_org_id()


def _wire_capabilities(
    app: FastAPI,
    capabilities: "CapabilityRegistry | None",
    mount_data_plane: bool,
    additional_routers: list[APIRouter] | None = None,
) -> None:
    """Wire capability routers, services, and hooks into the app at construction time.

    No-op when ``capabilities`` is None.

    The deployment role passed to each capability's ``routers(role)`` is
    sourced from ``capabilities.role`` when set (threaded in by the enterprise
    composition root via ``build_registry(deployment_role())``).  When
    ``capabilities.role`` is ``None`` the role falls back to the
    ``mount_data_plane`` derivation (``"all"`` when True, ``"control-plane"``
    when False), preserving the existing OSS-only behaviour.

    Args:
        app (FastAPI): The application instance to wire into.
        capabilities (CapabilityRegistry | None): Capabilities to install, or None.
        mount_data_plane (bool): Whether the data-plane role is active.
        additional_routers (list[APIRouter] | None): Routers already mounted via
            ``additional_routers``; used to detect and reject double-mounts at boot.

    Raises:
        ValueError: If a capability's router object is also present in
            ``additional_routers`` — each router must be mounted exactly once.
    """
    if capabilities is None:
        return
    from reflexio.server.auth import default_billing_gate
    from reflexio.server.extensions import HookRegistry

    if capabilities.role is not None:
        role = capabilities.role
    else:
        role = "all" if mount_data_plane else "control-plane"
    if capabilities.configurator_class is not None:
        from reflexio.server.services.configurator.configurator import (
            set_configurator_class,
        )

        set_configurator_class(capabilities.configurator_class)
    if capabilities.billing_gate is not None:
        for line in ("application", "learnings_generated"):
            app.dependency_overrides[default_billing_gate(line)] = (
                capabilities.billing_gate(line)
            )
    # HookRegistry is a stateless facade: its methods configure process-global
    # reporting/tracing/usage singletons; the object needn't be stored.
    hooks = HookRegistry()
    additional = additional_routers or []
    for cap in capabilities.capabilities:
        cap.install_services()
        cap.install_hooks(hooks)
        for r in cap.routers(role):
            if any(r is a for a in additional):
                raise ValueError(
                    f"router for capability {cap.name!r} is mounted both via the "
                    f"registry and additional_routers; mount it exactly once"
                )
            app.include_router(r)


def _resolve_app_profile(
    profile: "DeploymentProfile | None",
    *,
    mount_data_plane: bool,
    require_auth: bool,
) -> "DeploymentProfile":
    """Return the caller's profile, or derive one from the legacy knobs.

    Args:
        profile (DeploymentProfile | None): Explicit profile, or None to derive.
        mount_data_plane (bool): Legacy knob — whether data-plane routers/lifespan run.
        require_auth (bool): Legacy knob — whether auth is declared.

    Returns:
        DeploymentProfile: ``profile`` when provided, else the derived profile.
    """
    from reflexio.server.deployment_profile import resolve_profile

    if profile is not None:
        return profile
    return resolve_profile(mount_data_plane=mount_data_plane, require_auth=require_auth)


def create_app(
    get_org_id: Callable[..., str] | None = None,
    additional_routers: list[APIRouter] | None = None,
    middleware_config: dict | None = None,
    require_auth: bool = False,
    get_caller_type: Callable[..., str] | None = None,
    get_billing_gate: Callable[[str], Callable[..., None]] | None = None,
    mount_data_plane: bool = True,
    capabilities: "CapabilityRegistry | None" = None,
    app_context_factory: "Callable[[], AppContext] | None" = None,
    profile: "DeploymentProfile | None" = None,
    durable_org_ids_provider: Callable[[], Iterable[str]] | None = None,
) -> FastAPI:
    """Factory to create a FastAPI app.

    Args:
        get_org_id: Custom dependency for resolving org_id (e.g., from JWT auth).
            When provided, overrides the default_get_org_id dependency globally.
        additional_routers: Extra APIRouter instances (e.g., enterprise login/oauth).
        middleware_config: Optional middleware overrides (not used yet, reserved for future).
        require_auth: When True, declares a Bearer security scheme in the OpenAPI spec
            so Swagger UI shows lock icons and the Authorize button works.
        get_caller_type: Custom dependency for classifying the caller (e.g., production
            agent vs dashboard).  When provided, overrides the default_get_caller_type
            dependency globally, exactly mirroring the get_org_id override.
        get_billing_gate: Optional factory ``(line: str) -> FastAPI dependency`` that
            replaces the default no-op gate for each billable billing line.  When
            provided, for every line used in the app (``"application"`` and
            ``"learnings_generated"``) the returned dependency overrides the
            ``default_billing_gate(line)`` sentinel in ``dependency_overrides``,
            exactly mirroring the ``get_caller_type`` override pattern.
        mount_data_plane: When True (default), include the data-plane routers
            (core, stall-state, pending-tool-call) and run the data-plane
            lifespan work (LLM availability check, cross-encoder prewarm,
            resume scheduler). When False, skip both so a control-plane host
            can build an app without requiring LLM/storage or starting the
            scheduler, while keeping all other scaffolding (middleware, CORS,
            auth overrides, OpenAPI security, health, ``/meta/version``,
            ``additional_routers``).
        capabilities: Optional registry of enterprise capabilities. When provided,
            each capability's routers, services, hooks, and lifecycle methods are
            wired into the app at construction time. Behavior is unchanged when
            ``None``.
        app_context_factory: Optional callable returning the AppContext passed to
            each capability's on_startup. When None, an empty AppContext() is used
            (local OSS / tests). Enterprise binds this to supply self_host_org_id /
            activated computed during its own startup.
        profile: Optional declarative :class:`DeploymentProfile`. When None
            (default), it is DERIVED from the ``mount_data_plane`` / ``require_auth``
            knobs so behaviour is identical to before this parameter existed. When
            provided it is the single source for router-group mounting, auth, and
            data-plane lifespan gating (the legacy knobs still populate the derived
            profile, so callers may pass either — they express the same thing).
        durable_org_ids_provider: Optional zero-arg callable returning the org_ids
            with actionable durable-learning work, threaded straight through to
            ``maybe_start_durable_learning(org_ids_provider=...)``. When None
            (default) the single-ref bootstrap default is used, so behaviour is
            byte-identical to before this parameter existed. It is a plain
            injectable seam so a deployment (e.g. enterprise cross-ref fan-out) can
            supply its own discovery WITHOUT this factory importing deployment
            logic. It is only consulted when ``REFLEXIO_DURABLE_LEARNING_QUEUE`` is
            on — when the flag is off ``maybe_start_durable_learning`` returns None
            without ever calling the provider.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If both ``get_billing_gate`` and ``capabilities.billing_gate``
            are provided — pass billing_gate through exactly one path.
    """
    if (
        get_billing_gate is not None
        and capabilities is not None
        and capabilities.billing_gate is not None
    ):
        raise ValueError(
            "pass billing_gate via either get_billing_gate or capabilities, not both"
        )
    # The profile is a cleaner representation of the two existing knobs. When the
    # caller does not pass one, derive it from ``mount_data_plane`` / ``require_auth``
    # so behaviour is byte-identical to before this parameter existed.
    profile = _resolve_app_profile(
        profile, mount_data_plane=mount_data_plane, require_auth=require_auth
    )
    mounts_data_plane = profile.mounts_data_plane
    auth_required = profile.auth_required
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.auth import (
        default_billing_gate,
        default_get_caller_type,
        default_get_org_id,
    )
    from reflexio.server.extensions import AppContext
    from reflexio.server.llm.model_defaults import validate_llm_availability
    from reflexio.server.services.durable_learning import (
        maybe_start_durable_learning,
    )
    from reflexio.server.services.extraction.resume_scheduler import (
        maybe_start_resume_scheduler,
    )
    from reflexio.server.services.lineage.gc_scheduler import (
        maybe_start_lineage_gc,
    )
    from reflexio.server.services.lineage.vector_backfill_sweep import (
        install_missing_vector_backfill_sweep,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        scheduler = None
        gc_scheduler = None
        durable_learning_scheduler = None
        started_caps: list = []
        # D8 config guards + D5 warm-before-ready. Both are dormant unless the
        # deployment has flipped to the in-process local embedder
        # (REFLEXIO_EMBEDDING_PROVIDER=inprocess + local/* default); pre-flip
        # this is a no-op and /health stays byte-for-byte unchanged. Run before
        # the data-plane block so a gated deployment warms regardless of the
        # mount profile (otherwise /health could 503 forever).
        from reflexio.server.llm.providers.embedder_warmup import (
            maybe_start_embedder_warmup,
            run_startup_config_guards,
        )

        run_startup_config_guards()
        maybe_start_embedder_warmup()
        if mounts_data_plane:
            _log_multi_worker_daemons()
            log_publish_hardware_capacity()
            validate_llm_availability()
            from reflexio.server.llm.rerank import prewarm as _prewarm_cross_encoder

            _prewarm_cross_encoder()
            # The scheduler discovers every org with resumable work each tick and
            # drives a per-org worker with org-scoped claims, so it is not limited
            # to the bootstrap org. The bootstrap org is only used to read config
            # and to seed cross-org discovery.
            bootstrap_org_id = _resolve_lifespan_org_id(get_org_id)
            scheduler = maybe_start_resume_scheduler(
                lambda org_id: RequestContext(org_id=org_id),
                bootstrap_org_id=bootstrap_org_id,
            )
            # Register the missing-vector backfill sweep (opt-in via
            # REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED) BEFORE starting the GC
            # scheduler, so its per-org hook is visible when maybe_start_lineage_gc
            # evaluates its start conditions. No-op when the flag is off.
            install_missing_vector_backfill_sweep()
            gc_scheduler = maybe_start_lineage_gc(
                lambda org_id: RequestContext(org_id=org_id),
                bootstrap_org_id=bootstrap_org_id,
            )
            # Durable learning drains the learning_jobs queue per org. Gated on
            # REFLEXIO_DURABLE_LEARNING_QUEUE; the default provider discovers
            # orgs-with-work via the bootstrap storage (single-ref). A deployment
            # may inject its own discovery via ``durable_org_ids_provider`` (e.g.
            # enterprise cross-ref fan-out); when None the single-ref default runs.
            durable_learning_scheduler = maybe_start_durable_learning(
                lambda org_id: RequestContext(org_id=org_id),
                bootstrap_org_id=bootstrap_org_id,
                org_ids_provider=durable_org_ids_provider,
            )
        try:
            if capabilities is not None:
                ctx = (
                    app_context_factory()
                    if app_context_factory is not None
                    else AppContext()
                )
                for cap in capabilities.capabilities:
                    await cap.on_startup(ctx)
                    started_caps.append(cap)
            yield
        finally:
            for cap in reversed(started_caps):
                try:
                    await cap.on_shutdown()
                except Exception:
                    logger.warning(
                        "capability %r on_shutdown raised; continuing cleanup",
                        cap,
                        exc_info=True,
                    )
            for sched in (scheduler, gc_scheduler, durable_learning_scheduler):
                if sched is not None:
                    sched.stop()
            from reflexio.server.services.publish_learning_worker import (
                stop_publish_learning_worker,
            )

            stop_publish_learning_worker(timeout=5.0)

    app = FastAPI(docs_url="/docs", lifespan=lifespan)
    app.add_exception_handler(
        RequestValidationError,
        _safe_request_validation_exception_handler,  # type: ignore[reportArgumentType]
    )

    if auth_required:
        _add_openapi_security(app)

    @app.get("/meta/version")
    def get_version_info() -> dict[str, str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            server_version = version("reflexio")
        except PackageNotFoundError:
            server_version = "0.0.0-dev"
        return {
            "server_version": server_version,
            "api_version": "v1",
            "min_client_version": "0.1.0",
        }

    # Configure rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[reportArgumentType]

    # CORS
    # The locked-down, credentialed allowlist is an enterprise concern: only
    # hosts that wire in auth (``require_auth=True``) restrict browser origins.
    # OSS/local runs have no auth and bundle their own docs playground on a
    # separate port, so they allow any origin (no credentials needed).
    if auth_required:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_resolve_cors_origins(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Reject oversized requests before they reach endpoint handlers.
    app.add_middleware(BodySizeLimitMiddleware)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Timeout middleware
    app.add_middleware(TimeoutMiddleware)

    # Bot protection
    app.add_middleware(BotProtectionMiddleware)

    # Correlation ID — added last so it runs outermost (Starlette reverses order)
    app.add_middleware(CorrelationIdMiddleware)

    # Override get_org_id dependency if custom one provided
    if get_org_id is not None:
        app.dependency_overrides[default_get_org_id] = get_org_id

    # Override get_caller_type dependency if custom one provided
    if get_caller_type is not None:
        app.dependency_overrides[default_get_caller_type] = get_caller_type

    # Override billing gate dependencies if a custom gate factory is provided.
    # Each billing line needs its own override because dependency_overrides is
    # keyed by callable identity.  ``default_billing_gate`` uses lru_cache so
    # the same sentinel object is returned for the same line — which is why
    # the overrides reliably fire at request time.
    if get_billing_gate is not None:
        for line in ("application", "learnings_generated"):
            app.dependency_overrides[default_billing_gate(line)] = get_billing_gate(
                line
            )

    # When a custom get_org_id is provided together with require_auth,
    # auth is enforced on every route — mark this app instance so the
    # token-gated my_config endpoint becomes reachable. Using
    # ``app.state`` instead of a module-level global keeps the gate
    # scoped to this FastAPI instance, so multiple apps (e.g. tests,
    # multi-tenant embeddings) can coexist without leaking state.
    app.state.my_config_enabled = bool(get_org_id is not None and auth_required)

    # Include data-plane routes (core, stall-state, pending-tool-call). A
    # control-plane host sets mount_data_plane=False to skip these while
    # keeping every other piece of scaffolding below.
    if mounts_data_plane:
        # Include core routes
        app.include_router(core_router)

        # Include stall_state routes
        app.include_router(stall_state_api.router)

        # Include pending tool call routes
        app.include_router(pending_tool_call_api.router)

    # Include additional routers
    for router in additional_routers or []:
        app.include_router(router)

    # Wire capability routers, services, and hooks (composition-root only)
    _wire_capabilities(app, capabilities, mounts_data_plane, additional_routers)

    # Health/observability endpoint (per-worker metrics for recycling)
    health_api.install(app)

    return app


# Default standalone app (no auth)
app = create_app()
