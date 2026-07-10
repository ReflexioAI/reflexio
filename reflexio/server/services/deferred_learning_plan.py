"""Write-plan dataclasses for the durable-learning compute/persist split (gate b).

Compute resolves these in-memory plans issuing **no** learning DB write; persist
applies them inside one short fenced ``commit_scope``. This module starts with the
extractor stride-bookmark advance and grows additional write-plan dataclasses in the
later gate-(b) tasks (profile/playbook/reflection/generation/deferred plans).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.models.api_schema.service_schemas import Interaction


@dataclass(frozen=True)
class ExtractorBookmarkAdvance:
    """The extractor stride-bookmark advance, deferred out of the extractor (F1).

    The extractor no longer self-advances its bookmark inside ``run()`` (a DB
    write). It emits this on the ``ExtractionOutcome`` instead, so the advance is
    applied later — inside the persist fence on the durable path, or right after
    result-processing on the synchronous ``.run()`` path — keeping the bookmark
    advance atomic with the row writes it corresponds to.
    """

    extractor_name: str
    processed_interactions: list[Interaction]
    # str | None (not str): the playbook extractor runs org-level with
    # user_id=None (PlaybookGenerationServiceConfig.user_id is Optional), which
    # maps to an unscoped bookmark key. update_extractor_bookmark accepts None.
    user_id: str | None
