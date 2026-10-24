"""Org admission survives rollout retirement without silently opening a veto."""

import json
from types import SimpleNamespace
from typing import cast

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_disabled_reason,
    pending_tool_calls_enabled,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager


@pytest.fixture
def context(monkeypatch, tmp_path):
    original_init = SiteVarManager.__init__
    monkeypatch.setattr(
        SiteVarManager,
        "__init__",
        lambda self: original_init(self, source_dir=str(tmp_path)),
    )
    config = SimpleNamespace(
        pending_tool_call_config=PendingToolCallConfig(enabled=True)
    )
    return cast(
        RequestContext,
        SimpleNamespace(
            org_id="org_1",
            storage=object(),
            configurator=SimpleNamespace(get_config=lambda: config),
        ),
    )


@pytest.mark.parametrize(
    "legacy",
    [
        {},
        {"resumable_extraction_agent": None},
        {
            "resumable_extraction_agent": {
                "enabled": True,
                "enabled_org_ids": ["old-cohort"],
            }
        },
    ],
)
def test_verified_neutral_json_requires_no_rewrite(context, tmp_path, legacy):
    path = tmp_path / "feature_flags.json"
    original = json.dumps(legacy)
    path.write_text(original)
    assert pending_tool_calls_enabled(context)
    assert path.read_text() == original


def test_missing_legacy_file_preserves_org_and_storage_requirements(context):
    assert pending_tool_calls_enabled(context)
    context.configurator.get_config().pending_tool_call_config.enabled = False
    assert not pending_tool_calls_enabled(context)
    context.configurator.get_config().pending_tool_call_config.enabled = True
    context.storage = None
    assert (
        pending_tool_calls_disabled_reason(context)
        == "the request context has no storage"
    )


def test_org_opt_out_remains_off_with_a_saved_cohort_veto(context, tmp_path):
    (tmp_path / "feature_flags.json").write_text(
        json.dumps({"resumable_extraction_agent": {"enabled": False}})
    )
    context.configurator.get_config().pending_tool_call_config.enabled = False
    assert (
        pending_tool_calls_disabled_reason(context)
        == "pending_tool_call_config.enabled is false"
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"secret-value-not-for-errors":',
        "[]",
        '{"resumable_extraction_agent":"secret-value-not-for-errors"}',
    ],
)
def test_ambiguous_legacy_json_never_becomes_an_enablement(context, tmp_path, raw):
    (tmp_path / "feature_flags.json").write_text(raw)
    with pytest.raises(ValueError, match="pending_tool_call_config.enabled") as error:
        pending_tool_calls_enabled(context)
    assert "secret-value-not-for-errors" not in str(error.value)


def test_json_priority_and_text_only_ambiguity_are_preserved(context, tmp_path):
    (tmp_path / "feature_flags.txt").write_text("unverified legacy policy")
    neutral = tmp_path / "feature_flags.json"
    neutral.write_text('{"resumable_extraction_agent":{"enabled":true}}')
    assert pending_tool_calls_enabled(context)
    neutral.unlink()
    with pytest.raises(ValueError, match="pending_tool_call_config.enabled"):
        pending_tool_calls_enabled(context)
