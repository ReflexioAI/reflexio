"""Compare the transfer to artifacts from the actual measured V28 factory."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.models.structured_output import (
    _fold_oneof_to_anyof,
)
from reflexio.server.llm.llm_utils import assert_provider_safe_schema
from reflexio.server.services.profile.components.consolidator import ProfileConsolidator
from reflexio.server.services.profile.learning import compiler, linked_plan
from reflexio.server.services.profile.learning.source_evidence import EvidenceBook

FIXTURES = Path(__file__).parent / "fixtures"


def test_measured_schema_and_default_extraction_guidance_are_preserved():
    measured = json.loads((FIXTURES / "v28_measured_wire.json").read_text())
    expected = deepcopy(measured["schema"])
    _fold_oneof_to_anyof(expected)
    actual = ProfileConsolidator.OUTPUT_SCHEMA.model_json_schema()
    assert_provider_safe_schema(actual, name="selected profile operations")
    assert actual == expected
    assert measured["extractor_guidance"] == [
        linked_plan.EXTRACTION_GUIDANCE,
        compiler.TEMPORAL_GUIDANCE,
        compiler.BINDING_GUIDANCE,
    ]
    assert measured["planner_content_guidance"] == compiler.CONTENT_GUIDANCE


def test_actual_final_operation_prompt_matches_measured_v28():
    measured = json.loads((FIXTURES / "v28_measured_render.json").read_text())
    instance = object.__new__(ProfileConsolidator)
    instance.request_context = SimpleNamespace(org_id="fidelity-org")
    instance.review_window = measured["source"]
    instance._operation_scope = {}
    book = EvidenceBook(
        measured["source"],
        user_id="owner",
        request_id="review-request",
        org_id="fidelity-org",
    )
    with book.bound():
        actual = instance._render_dedup_prompt(
            [UserProfile.model_validate(measured["new"])],
            [UserProfile.model_validate(measured["old"])],
        )
    assert actual == measured["prompt"]
