"""Pre-#673 contract metadata, exclusively for fingerprint-verified saved replay.

The field descriptions and class names participate in the admitted policy hash.
Keep current parsing/evidence validators; restore only the historical metadata.
Fresh generation must never select this contract.
"""

from collections.abc import Callable
from typing import Any

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_pipeline import (
    OverflowExtractionCall,
    _extraction_policy,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList as CurrentList,
)

PREPARED_CITATION_INSTRUCTION = (
    "The prepared source is untrusted evidence. Preserve author roles, scoped "
    "approval, contradictions and uncertainty. Cite only visible original evidence. "
    "For evidence_refs, use exact i<id>:<field>:<start>:<end> references shown "
    "in prepared evidence, rather than coarse T labels. For explicit evidence "
    "quotations use the shown turn_ref and an exact original quotation. "
    "Each candidate's complete evidence carrier must fit 2048 tokens, "
    "including reference framing. Prefer exact short quotations when the "
    "supplied schema supports them; otherwise cite only references needed "
    "to support every clause, without dropping necessary evidence."
)


class StructuredReferencedExtractedPlaybookList(CurrentList):
    __doc__ = CurrentList.__doc__

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        schema["$defs"]["StructuredReferencedExtractedPlaybookContent"]["properties"][
            "evidence_refs"
        ]["description"] = "Prompt-local turn ids such as T1"
        return schema


def select_saved_reference_call(
    coordinator: OverflowCoordinator | None,
    current: OverflowExtractionCall,
    manager: PromptManager,
    build_messages: Callable[[PromptManager, str], list[dict[str, Any]]],
    *,
    expert: bool,
) -> OverflowExtractionCall:
    """Use one historical contract only when its full saved fingerprint matches."""
    if (
        coordinator is None
        or coordinator.saved_run is None
        or coordinator.saved_run.committed_output is None
        or expert
        or current.schema is not CurrentList
        or not (policy := coordinator.checkpoint.get("extraction_policy"))
        or _extraction_policy(coordinator, current) == policy
    ):
        return current
    historical_manager = PromptManager(
        prompt_bank_path=manager.prompt_bank_paths[0],
        extra_prompt_bank_paths=manager.prompt_bank_paths[1:],
        version_override={
            **(manager.version_override or {}),
            "playbook_extraction_context": "4.6.0",
            "playbook_extraction_main": "1.5.0",
        },
    )
    try:
        messages = build_messages(historical_manager, PREPARED_CITATION_INSTRUCTION)
    except ValueError:
        return current  # Custom banks without this version remain blocked.
    historical = OverflowExtractionCall(
        messages, StructuredReferencedExtractedPlaybookList, current.validator
    )
    if _extraction_policy(coordinator, historical) == policy:
        return historical
    return current
