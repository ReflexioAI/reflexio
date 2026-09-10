"""Resolve extraction eligibility once, when interactions are accepted."""

from typing import Any

from reflexio.models.api_schema.domain.entities import PublishUserInteractionRequest
from reflexio.models.config_schema import Config
from reflexio.server.services.extractor_interaction_utils import (
    get_extractor_window_params,
)


def extraction_admission(
    config: Config, request: PublishUserInteractionRequest, *, stalled: bool = False
) -> dict[str, Any]:
    admission: dict[str, Any] = {
        "force": request.force_extraction,
        "skip_aggregation": request.skip_aggregation,
    }
    for kind, extractor in (
        ("profile", config.profile_extractor_config),
        ("playbook", config.user_playbook_extractor_config),
    ):
        eligible = extractor is not None and not request.evaluation_only and not stalled
        if extractor is None:
            admission[kind] = {"eligible": False}
            continue
        sources = extractor.request_sources_enabled
        eligible = (
            eligible
            and not getattr(extractor, "manual_trigger", False)
            and (not sources or request.source in sources)
        )
        if not eligible:
            admission[kind] = {"eligible": False}
            continue
        width, stride = get_extractor_window_params(
            extractor, config.window_size, config.stride_size
        )
        if not 1 <= stride <= width:
            raise ValueError("Extraction requires 1 <= stride_size <= window_size")
        admission[kind] = {
            "eligible": eligible,
            "window_size": width,
            "stride_size": stride,
            # Extractor policy contains prompts and window controls, never API keys.
            "extractor": extractor.model_dump(mode="json"),
            "skip_should_run_check": config.skip_should_run_check,
        }
    return admission
