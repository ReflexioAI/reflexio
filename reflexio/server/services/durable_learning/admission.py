"""Resolve extraction eligibility once, when interactions are accepted."""

from typing import Any, Literal

from reflexio.models.api_schema.domain.entities import PublishUserInteractionRequest
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.extraction.agent_run_records import (
    snapshot_extractor_config,
)
from reflexio.server.services.extractor_interaction_utils import (
    get_extractor_window_params,
)


def extraction_admission(
    config: Config, request: PublishUserInteractionRequest, *, stalled: bool = False
) -> dict[str, Any]:
    admission: dict[str, Any] = {
        "force": request.force_extraction,
        "session_finished": request.session_finished,
        "idle_timeout_seconds": config.extraction_idle_timeout_seconds,
        "max_batch_wait_seconds": config.extraction_max_batch_wait_seconds,
        "skip_aggregation": request.skip_aggregation,
    }
    extractors: dict[
        Literal["profile", "playbook"],
        ProfileExtractorConfig | UserPlaybookExtractorConfig | None,
    ] = {
        "profile": config.profile_extractor_config,
        "playbook": config.user_playbook_extractor_config,
    }
    for kind, extractor in extractors.items():
        eligible = extractor is not None and not request.evaluation_only and not stalled
        if extractor is None:
            admission[kind] = {"eligible": False}
            continue
        sources = extractor.request_sources_enabled
        window = get_extractor_window_params(config, kind)
        eligible = (
            eligible and window.automatic and (not sources or request.source in sources)
        )
        if not eligible:
            admission[kind] = {"eligible": False}
            continue
        window.require_automatic()
        width, stride = window
        admission[kind] = {
            "eligible": eligible,
            "window_size": width,
            "stride_size": stride,
            # Extractor policy contains prompts and window controls, never API keys.
            "extractor": snapshot_extractor_config(extractor),
            "skip_should_run_check": config.skip_should_run_check,
        }
    return admission
