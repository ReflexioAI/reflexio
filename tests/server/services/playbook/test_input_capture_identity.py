"""A prompt-only policy change must change the retained runtime identity."""

from reflexio.server.services.playbook.input_capture import (
    build_consolidator_input_snapshot,
)


def test_policy_identity_covers_all_three_prompt_stages(tmp_path):
    runtime = tmp_path / "reconciliation.py"
    runtime.write_text("# unchanged runtime")
    prompts = tmp_path / "reconciliation_prompts"
    prompts.mkdir()
    stages = ("reconcile-method", "reconcile-audit", "reconcile-revise")
    for stage in stages:
        (prompts / f"{stage}.md").write_text(stage)

    def snapshot():
        return build_consolidator_input_snapshot(
            payload={"new": [], "existing": [], "sessions": []},
            interactions={},
            existing_rows=[],
            user_id="owner",
            agent_version="v1",
            generation_request_id="request",
            model_name="scripted",
            runtime_path=str(runtime),
        )

    original = snapshot()
    for stage in stages:
        path = prompts / f"{stage}.md"
        path.write_text(stage + " changed")
        changed = snapshot()
        assert changed.policy_sha256 != original.policy_sha256
        assert changed.runtime_sha256 == original.runtime_sha256
        assert changed.payload_sha256 == original.payload_sha256
        path.write_text(stage)
    assert snapshot() == original
