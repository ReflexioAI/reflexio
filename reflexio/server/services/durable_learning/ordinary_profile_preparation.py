"""Native-text source/runtime-version/recipe witness; no planner or provider execution.

This is deliberately conservative: unrelated installed SDK source/template drift
holds selective continuation. It does not establish cross-version compatibility.
Python source and runtime/version metadata are process-cached under an immutable
installation assumption; prompt resources, configuration and resolved request settings are always reread.
No incumbent profile payload or credential descriptor is persisted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import re
import struct
import sys
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import distributions
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.env_utils import env_str
from reflexio.server.llm.input_budget import (
    ExtractionInputIntegrityError,
    declared_context_limits,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.extraction.profile_authoring_protocol import (
    PROFILE_AUTHORING_POLICY,
)
from reflexio.server.services.extraction.profile_package_preparation import (
    build_profile_package_request,
)
from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
)
from reflexio.server.services.profile.components.extractor import (
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.service import ProfileGenerationService

if TYPE_CHECKING:
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.storage.storage_base import AgentRunRecord
    from reflexio.server.services.storage.storage_base._extraction_stream import Window

KEY = "ordinary_profile_preparation"


class _NativeProfileStorage(Protocol):
    llm_client: LiteLLMClient
    embedding_model_name: str
    embedding_dimensions: int
    _expansion_model_name: str

    def _should_expand_documents(self) -> bool: ...


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _files_digest(root: Path, *, python_only: bool) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ExtractionInputIntegrityError("Ordinary consumer resources unavailable")
    files = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ExtractionInputIntegrityError(
                "Ordinary consumer resource layout ambiguous"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExtractionInputIntegrityError(
                "Ordinary consumer resource is not a file"
            )
        if python_only and path.suffix != ".py":
            continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if not files:
        raise ExtractionInputIntegrityError("Ordinary consumer resources empty")
    return _digest(files)


@lru_cache(maxsize=1)
def _installed_source_digest() -> str:
    root = Path(__file__).parents[3]
    if not (root / "__init__.py").is_file():
        raise ExtractionInputIntegrityError(
            "Ordinary consumer requires installed Python source"
        )
    return _files_digest(root, python_only=True)


@lru_cache(maxsize=1)
def _installed_native_source_digest() -> str:
    package = importlib.import_module("reflexio_ext")
    location = getattr(package, "__file__", None)
    if not isinstance(location, str) or not Path(location).is_file():
        raise ExtractionInputIntegrityError(
            "Native ordinary consumer requires installed Enterprise source"
        )
    return _files_digest(Path(location).parent, python_only=True)


@lru_cache(maxsize=1)
def _installed_runtime_digest() -> str:
    """Bind local runtime/version metadata, not dependency bytes or an image.

    No environment, paths, credentials, package source or external lookup enters
    this fingerprint. Duplicate/missing package metadata cannot prove an import
    environment and therefore holds selective continuation.
    """
    installed: dict[str, str] = {}
    for distribution in distributions():
        metadata = distribution.metadata
        names = metadata.get_all("Name", [])
        versions = metadata.get_all("Version", [])
        if len(names) != 1 or len(versions) != 1:
            raise ExtractionInputIntegrityError(
                "Ordinary runtime distribution metadata ambiguous or missing"
            )
        name, version = names[0], versions[0]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name) is None
            or not isinstance(version, str)
            or re.fullmatch(r"[vV]?[0-9][A-Za-z0-9.!+_-]{0,255}", version) is None
        ):
            raise ExtractionInputIntegrityError(
                "Ordinary runtime distribution identity invalid"
            )
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in installed:
            raise ExtractionInputIntegrityError(
                "Ordinary runtime distribution identity duplicated"
            )
        installed[canonical] = version
    if not installed:
        raise ExtractionInputIntegrityError(
            "Ordinary runtime distribution metadata unavailable"
        )
    return _digest(
        {
            "distributions": installed,
            "python": {
                "implementation": sys.implementation.name,
                "version": list(sys.version_info),
                "cache_tag": sys.implementation.cache_tag,
                "abi_flags": getattr(sys, "abiflags", ""),
                "soabi": sysconfig.get_config_var("SOABI"),
                "platform": sys.platform,
                "machine": platform.machine(),
                "byteorder": sys.byteorder,
                "pointer_bits": struct.calcsize("P") * 8,
            },
        }
    )


def ordinary_profile_preparation(
    executor: WindowExecutor,
    window: Window,
    service: ProfileGenerationService,
    config: ProfileExtractorConfig,
) -> dict[str, Any]:
    """Read actual native contracts without claims, writes, calls or v2 planning.

    The retained source/runtime/settings must reproduce this descriptor. Equality
    does not authenticate an immutable image or same-version dependency bytes.
    Custom services/clients and non-text/legacy adapters are not qualified here.
    Dynamic incumbents stay at their ordinary native lookup time.
    """
    if (
        window.kind != "profile"
        or type(service) is not ProfileGenerationService
        or type(executor.client) is not LiteLLMClient
        or service.client is not executor.client
    ):
        raise ExtractionInputIntegrityError("Unsupported ordinary profile consumer")
    manager = executor.context.prompt_manager
    banks = [Path(path) for path in manager.prompt_bank_paths]
    if not banks or len({str(path.resolve()) for path in banks}) != len(banks):
        raise ExtractionInputIntegrityError("Ordinary prompt resource layout ambiguous")
    templates = {
        prompt: manager.get_prompt_template_identity(prompt)
        if manager.get_active_version(prompt) is not None
        else None
        for prompt in sorted(manager.get_all_prompt_ids())
    }
    if not templates:
        raise ExtractionInputIntegrityError("Ordinary native templates unavailable")
    root_config = executor.context.configurator.get_config()
    settings = root_config.model_dump(
        mode="json",
        exclude={
            "storage_config": True,
            "storage_config_test": True,
            "api_key_config": True,
            "pending_tool_call_config": {"hmac_secrets"},
        },
    )
    client = executor.client
    dedup = ProfileConsolidator(executor.context, client)
    options = {
        "native": {},
        "gate": {"model": service._resolve_should_run_model()},
        "agent": {
            "model_role": ModelRole.EXTRACTION_AGENT,
            "response_format": StructuredProfilesOutput,
        },
        "dedup": {
            "model": dedup.model_name,
            "response_format": ProfileDeduplicationOutput,
        },
    }
    stage_clients = dict.fromkeys(options, client)
    if (type(executor.storage).__module__, type(executor.storage).__name__) not in {
        ("reflexio.server.services.storage.sqlite_storage", "SQLiteStorage"),
        ("reflexio_ext.server.services.storage.supabase_storage", "SupabaseStorage"),
        ("reflexio_ext.server.services.storage.postgres_storage", "PostgresStorage"),
    }:
        raise ExtractionInputIntegrityError(
            "Unsupported ordinary storage implementation"
        )
    storage = cast(_NativeProfileStorage, executor.storage)
    expansion_enabled = storage._should_expand_documents()
    if expansion_enabled:
        expansion_client = storage.llm_client
        if type(expansion_client) is not LiteLLMClient:
            raise ExtractionInputIntegrityError(
                "Unsupported ordinary storage completion adapter"
            )
        options["expansion"] = {"model": storage._expansion_model_name}
        stage_clients["expansion"] = expansion_client
    requests = {}
    for stage, selected in options.items():
        stage_client = stage_clients[stage]
        models = stage_client.get_completion_model_ladder(**selected)
        if not models or any(
            model.partition("/")[0] in {"claude-code", "openclaw"} for model in models
        ):
            raise ExtractionInputIntegrityError(
                "Unsupported ordinary completion adapter"
            )
        per_rung = {
            key: value for key, value in selected.items() if key != "model_role"
        }
        requests[stage] = [
            _digest(
                stage_client._completion_request_contract_from_params(
                    build_profile_package_request(
                        stage_client,
                        [],
                        **{**per_rung, "model": model, "fallback_models": []},
                    )
                )
            )
            for model in models
        ]
    # Only non-secret primitive transport settings; native request descriptors
    # above are hashed before persistence (URLs may themselves contain secrets).
    transport = {
        key: getattr(client.config, key)
        for key in (
            "model",
            "temperature",
            "max_tokens",
            "timeout",
            "max_retries",
            "retry_delay",
            "top_p",
            "fallback_models",
        )
    }
    return {
        "version": 1,
        "consumer": "native-ordinary-text-profile-v1",
        "source_sha256": _digest(
            {
                "sdk": _installed_source_digest(),
                "runtime_versions": _installed_runtime_digest(),
                "native": _installed_native_source_digest()
                if type(executor.storage).__module__.startswith("reflexio_ext.")
                else None,
            }
        ),
        "templates_sha256": _digest(
            {
                "resources": [_files_digest(path, python_only=False) for path in banks],
                "active": templates,
            }
        ),
        "instructions_sha256": _digest(
            {
                "settings": settings,
                "extractor": config.model_dump(mode="json"),
                "policy": window.policy,
                "force": window.force,
                "skip_aggregation": window.skip_aggregation,
                "authoring": PROFILE_AUTHORING_POLICY,
            }
        ),
        "context_sha256": _digest(executor.context.configurator.get_agent_context()),
        "requests_sha256": _digest(
            {
                "requests": requests,
                "transport": transport,
                "storage": {
                    "expansion_enabled": expansion_enabled,
                    "embedding_model": storage.embedding_model_name,
                    "embedding_dimensions": storage.embedding_dimensions,
                },
                "limits": declared_context_limits(
                    env_str("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", default="{}")
                ),
            }
        ),
        "schemas_sha256": _digest(
            {
                "agent": StructuredProfilesOutput.model_json_schema(),
                "dedup": ProfileDeduplicationOutput.model_json_schema(),
            }
        ),
    }


def validate_ordinary_profile_preparation(
    executor: WindowExecutor,
    window: Window,
    service: ProfileGenerationService,
    config: ProfileExtractorConfig,
    parent: AgentRunRecord,
) -> dict[str, Any]:
    saved = parent.generation_request_snapshot.get(KEY)
    current = ordinary_profile_preparation(executor, window, service, config)
    if (
        not isinstance(saved, dict)
        or type(saved.get("version")) is not int
        or saved != current
    ):
        raise ExtractionInputIntegrityError(
            "Original ordinary profile consumer contract unavailable or changed"
        )
    return current


@dataclass(frozen=True)
class _OriginalOrdinaryInstructions:
    executor: WindowExecutor
    window: Window
    service: ProfileGenerationService
    config: ProfileExtractorConfig
    parent: AgentRunRecord


_original: ContextVar[_OriginalOrdinaryInstructions | None] = ContextVar(
    "original_ordinary_profile_instructions", default=None
)


def require_original_ordinary_profile_preparation(
    executor: WindowExecutor,
    window: Window,
    service: ProfileGenerationService,
    config: ProfileExtractorConfig,
    parent: AgentRunRecord,
) -> dict[str, Any]:
    """Selected-only frozen-instruction refusal is an operational hold.

    Preview keeps its configuration blocker. Do not reclassify arbitrary normal
    configuration errors or discard observed output/usage as failed generation.
    """
    from reflexio.server.services.storage.storage_base.learning_transition import (
        LearningTransitionBlockedError,
    )

    try:
        return validate_ordinary_profile_preparation(
            executor, window, service, config, parent
        )
    except (ExtractionInputIntegrityError, OSError, ValueError, TypeError) as error:
        raise LearningTransitionBlockedError(
            "Original selected ordinary instructions unavailable or changed"
        ) from error


def check_original_ordinary_instructions() -> None:
    original = _original.get()
    if original is not None:
        require_original_ordinary_profile_preparation(
            original.executor,
            original.window,
            original.service,
            original.config,
            original.parent,
        )


@contextmanager
def original_ordinary_instruction_scope(
    executor: WindowExecutor,
    coordinator: OverflowCoordinator,
    service: ProfileGenerationService,
    config: ProfileExtractorConfig,
) -> Iterator[None]:
    from reflexio.server.services.durable_learning.selected_learning import (
        current_selected_learning,
    )

    if (
        current_selected_learning() is None
        or coordinator.saved_run is None
        or coordinator.saved_run.generation_request_snapshot.get("overflow", {}).get(
            "version"
        )
        != 1
    ):
        yield
        return
    token = _original.set(
        _OriginalOrdinaryInstructions(
            executor, coordinator.window, service, config, coordinator.saved_run
        )
    )
    try:
        check_original_ordinary_instructions()
        yield
    finally:
        _original.reset(token)
