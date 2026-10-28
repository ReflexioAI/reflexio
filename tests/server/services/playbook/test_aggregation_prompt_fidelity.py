"""Aggregation must receive each selected source's complete, associated evidence."""

import json

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.playbook.components.aggregator_prompt_formatting import (
    format_structured_cluster_input,
)


def _rows(rendered: str) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in rendered.splitlines()
        if line.lstrip().startswith("{")
    ]


@pytest.mark.parametrize("threshold", [0.0, 1.0])
@pytest.mark.parametrize("trigger", [None, 'When the user says "deploy"\n再试一次'])
@pytest.mark.parametrize("rationale", [None, "Protect access\nDo not expose secrets"])
def test_prompt_preserves_source_fields_and_association(threshold, trigger, rationale):
    sources = [
        UserPlaybook(
            user_playbook_id=index,
            agent_version="v1",
            request_id=f"request-{index}",
            playbook_name="test",
            content=content,
            trigger=trigger,
            rationale=rationale,
        )
        for index, content in enumerate(
            [
                "Always ask for approval before deploying",
                "Never disclose access tokens",
            ],
            start=1,
        )
    ]
    originals = [source.model_dump() for source in sources]

    rendered = format_structured_cluster_input(sources, threshold)

    assert _rows(rendered) == [
        {name: value for name, value in row.items() if value}
        for row in (
            {"content": source.content, "trigger": trigger, "rationale": rationale}
            for source in sources
        )
    ]
    assert [source.model_dump() for source in sources] == originals
    assert ("Group 1" in rendered) is (threshold == 1.0)


def test_large_cluster_fallback_preserves_every_source_without_grouping(monkeypatch):
    from reflexio.server.services.playbook.components import (
        aggregator_prompt_formatting,
    )

    sources = [
        UserPlaybook(
            user_playbook_id=index,
            agent_version="v1",
            request_id=f"request-{index}",
            playbook_name="test",
            content=f"instruction-{index}",
            trigger=f"condition-{index}",
            rationale=f"reason-{index}",
        )
        for index in range(257)
    ]

    def forbidden_grouping(*_args, **_kwargs):
        pytest.fail("large clusters must retain the linear formatting fallback")

    monkeypatch.setattr(
        aggregator_prompt_formatting, "group_playbooks_by_direction", forbidden_grouping
    )

    assert _rows(format_structured_cluster_input(sources)) == [
        {
            "content": source.content,
            "trigger": source.trigger,
            "rationale": source.rationale,
        }
        for source in sources
    ]
