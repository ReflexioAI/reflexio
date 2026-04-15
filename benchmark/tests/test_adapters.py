"""Unit tests for the GDPVal benchmark adapters and reflexio bridge.

These use mocks for `AIAgent`, `OpenSpace`, and `ReflexioClient` so the test
suite runs without any real model/LLM/network access. Snapshot/restore is
exercised against a temporary directory to verify the filesystem contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.memory.injection import render_memory_block

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test task fixture
# ---------------------------------------------------------------------------

_EXAMPLE_TASK: dict = {
    "task_id": "test-task-001",
    "occupation": "Compliance Officers",
    "sector": "Professional Services",
    "prompt": "Draft a quarterly compliance report summary.",
    "reference_files": [],
    "task_value_usd": 25.0,
}


# ---------------------------------------------------------------------------
# OpenSpaceAdapter
# ---------------------------------------------------------------------------


def test_openspace_adapter_memory_prepend(tmp_path: Path) -> None:
    """OpenSpace adapter wraps reflexio memory into a <memory>…</memory> block."""
    from benchmark.adapters.openspace_adapter import OpenSpaceAdapter

    adapter = OpenSpaceAdapter(model="openrouter/minimax/MiniMax-M2.7")

    fake_os = MagicMock()
    fake_os.execute = AsyncMock(
        return_value={
            "status": "success",
            "iterations": 3,
            "tool_executions": [{"tool": "shell", "output": "ok"}],
            "skills_used": ["compliance-report"],
            "evolved_skills": [],
        }
    )
    adapter._cs = fake_os
    adapter._workspace_root = tmp_path
    adapter._tracker = MagicMock()
    adapter._tracker.begin_task = MagicMock(return_value="ctx-token-mock")
    adapter._tracker.end_task = MagicMock(
        return_value=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            llm_calls=1,
            wall_time_sec=1.0,
        )
    )

    ws = tmp_path / "ws"

    with patch(
        "benchmark.task_loader.prepare_task_workspace",
        return_value=_EXAMPLE_TASK["prompt"],
    ):
        result = asyncio.run(adapter.run(_EXAMPLE_TASK, ws, memory="USE PLAYBOOK: foo"))

    assert result.status == "success"
    assert result.tokens.total_tokens == 150
    # Concurrent-mode contract: begin_task/end_task must each be called
    # exactly once per adapter.run(), with the end_task ctx_token from
    # begin_task round-tripped.
    adapter._tracker.begin_task.assert_called_once()
    adapter._tracker.end_task.assert_called_once()
    end_args = adapter._tracker.end_task.call_args
    assert end_args[0][1] == "ctx-token-mock"  # second positional arg = ctx_token
    assert result.iterations == 3
    assert result.tool_calls == 1

    # Verify the task prompt was prepended with the memory block.
    called_task = fake_os.execute.call_args.kwargs["task"]
    assert called_task.startswith("<memory>\nUSE PLAYBOOK: foo\n</memory>")
    assert _EXAMPLE_TASK["prompt"] in called_task


def test_openspace_adapter_no_memory(tmp_path: Path) -> None:
    """Without memory, the prompt is unmodified."""
    from benchmark.adapters.openspace_adapter import OpenSpaceAdapter

    adapter = OpenSpaceAdapter(model="openrouter/minimax/MiniMax-M2.7")

    fake_os = MagicMock()
    fake_os.execute = AsyncMock(
        return_value={"status": "success", "iterations": 1, "tool_executions": []}
    )
    adapter._cs = fake_os
    adapter._workspace_root = tmp_path
    adapter._tracker = MagicMock()
    adapter._tracker.begin_task = MagicMock(return_value="ctx")
    adapter._tracker.end_task = MagicMock(
        return_value=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            llm_calls=0,
            wall_time_sec=0.1,
        )
    )

    ws = tmp_path / "ws"
    with patch(
        "benchmark.task_loader.prepare_task_workspace",
        return_value=_EXAMPLE_TASK["prompt"],
    ):
        asyncio.run(adapter.run(_EXAMPLE_TASK, ws, memory=None))

    called_task = fake_os.execute.call_args.kwargs["task"]
    assert "<memory>" not in called_task
    assert called_task == _EXAMPLE_TASK["prompt"]


# ---------------------------------------------------------------------------
# HermesAdapter
# ---------------------------------------------------------------------------


def test_hermes_adapter_system_message_wiring(tmp_path: Path) -> None:
    """Hermes adapter passes reflexio memory via `system_message=`."""
    from benchmark.adapters.hermes_adapter import HermesAdapter

    adapter = HermesAdapter(model="minimax/MiniMax-M2.7", api_key="fake-key")

    fake_agent = MagicMock()
    fake_agent.run_conversation = MagicMock(
        return_value={
            "completed": True,
            "final_response": "done",
            "messages": [
                {"role": "user", "content": _EXAMPLE_TASK["prompt"]},
                {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [{"name": "write"}],
                },
            ],
            "api_calls": 2,
            "input_tokens": 200,
            "output_tokens": 80,
            # Hermes reports an inflated cumulative here; our adapter ignores
            # it and derives `total_tokens = input + output` on its own.
            "total_tokens": 2000,
            "estimated_cost_usd": 0.02,
        }
    )
    # Mock the fresh-agent factory so run() doesn't try to import Hermes.
    adapter._ai_agent_cls = lambda **kw: fake_agent  # type: ignore[assignment]
    adapter._build_fresh_agent = lambda: fake_agent  # type: ignore[assignment]

    ws = tmp_path / "ws"
    with patch(
        "benchmark.task_loader.prepare_task_workspace",
        return_value=_EXAMPLE_TASK["prompt"],
    ):
        result = asyncio.run(adapter.run(_EXAMPLE_TASK, ws, memory="USE PLAYBOOK: foo"))

    fake_agent.run_conversation.assert_called_once()
    kwargs = fake_agent.run_conversation.call_args.kwargs
    assert kwargs["system_message"] == "USE PLAYBOOK: foo"
    assert kwargs["user_message"] == _EXAMPLE_TASK["prompt"]
    assert result.tokens.total_tokens == 280  # 200 + 80, derived not read
    assert result.tool_calls == 1
    assert result.status == "success"


def test_hermes_adapter_strips_litellm_prefix() -> None:
    """The adapter strips `openrouter/` prefixes so Hermes gets the bare ID."""
    from benchmark.adapters.hermes_adapter import HermesAdapter

    adapter = HermesAdapter(model="openrouter/minimax/MiniMax-M2.7", api_key="k")
    assert adapter._model == "minimax/MiniMax-M2.7"

    adapter_bare = HermesAdapter(model="minimax/MiniMax-M2.7", api_key="k")
    assert adapter_bare._model == "minimax/MiniMax-M2.7"


# ---------------------------------------------------------------------------
# Reflexio bridge
# ---------------------------------------------------------------------------


def test_injection_empty_response_returns_empty_string() -> None:
    """render_memory_block returns '' when the response has no hits."""
    response = SimpleNamespace(profiles=[], agent_playbooks=[], user_playbooks=[])
    assert render_memory_block(response) == ""


def test_injection_renders_sections() -> None:
    """Non-empty hits are rendered as a CACHED SOLUTION recipe block.

    The extractor emits a single concrete recipe per task, and the
    renderer wraps it in a strong trust header telling the agent to
    re-run the cached steps. Profiles are dropped (they are org-wide
    and leak across tasks), so only playbook content should appear.
    """
    response = SimpleNamespace(
        profiles=[SimpleNamespace(content="Workspace uses Python 3.11")],
        agent_playbooks=[
            SimpleNamespace(
                playbook_name="compliance-report",
                content="TASK: SOAP notes. INPUT: patient.txt. STEPS: write four-section output. OUTPUT: soap.md.",
            )
        ],
        user_playbooks=[],
    )
    block = render_memory_block(response)
    assert "CACHED SOLUTION" in block
    assert "Recipe" in block
    assert "TASK: SOAP notes" in block
    assert "STEPS: write four-section output" in block
    # Profiles are intentionally dropped to prevent cross-task leakage.
    assert "Workspace uses Python 3.11" not in block


def test_injection_drops_profiles_even_when_only_hit() -> None:
    """If the only hit is a profile, the renderer returns an empty string.

    Regression guard for the cross-task profile leak: profiles are
    org-wide in the reflexio backend, so they would otherwise bleed
    facts from task A's trajectory into task B's P3 memory block.
    """
    response = SimpleNamespace(
        profiles=[SimpleNamespace(content="Prior task wrote out.csv with 142 rows")],
        agent_playbooks=[],
        user_playbooks=[],
    )
    assert render_memory_block(response) == ""


def test_injection_dedupes_whitespace_variants() -> None:
    """Near-duplicate playbooks differing only in whitespace render once.

    Regression guard for a prior benchmark failure in which the extractor
    emitted three copies of the same 3715-char recipe, producing a
    ~12k-char block that pushed a Hermes task over its token budget.
    """
    recipe = "Step 1: load data\nStep 2: aggregate\nFinal: 61 rows"
    response = SimpleNamespace(
        profiles=[],
        agent_playbooks=[],
        user_playbooks=[
            SimpleNamespace(content=recipe),
            SimpleNamespace(content=recipe + "\n\n"),
            SimpleNamespace(content="  " + recipe.replace("\n", "  ") + "  "),
            SimpleNamespace(content=recipe.upper()),
        ],
    )
    block = render_memory_block(response)
    assert block.count("Step 1: load data") == 1
    assert block.count("Final: 61 rows") == 1


def test_injection_renders_plain_recipe_body() -> None:
    """A recipe-style body is passed through without any rule wrapping."""
    response = SimpleNamespace(
        profiles=[],
        agent_playbooks=[
            SimpleNamespace(
                playbook_name="pb",
                content="Step 1: pd.read_excel('data.xlsx'). Step 2: df.to_csv('out.csv'). Final: 142 rows.",
            ),
        ],
        user_playbooks=[],
    )
    block = render_memory_block(response)
    assert "CACHED SOLUTION" in block
    assert "pd.read_excel('data.xlsx')" in block
    assert "Final: 142 rows" in block


def test_reflexio_bridge_fetch_empty_returns_none(tmp_path: Path) -> None:
    """fetch_for_task returns None when the client yields no hits."""
    from benchmark.memory.reflexio_bridge import ReflexioMemory

    bridge = ReflexioMemory(user_id_prefix="bench_test")
    empty_response = SimpleNamespace(profiles=[], agent_playbooks=[], user_playbooks=[])
    bridge._client.search = MagicMock(return_value=empty_response)

    result = asyncio.run(bridge.fetch_for_task(_EXAMPLE_TASK))
    assert result is None


def test_reflexio_bridge_fetch_renders_hits() -> None:
    """fetch_for_task returns a rendered block when hits are present."""
    from benchmark.memory.reflexio_bridge import ReflexioMemory

    bridge = ReflexioMemory(user_id_prefix="bench_test")
    response = SimpleNamespace(
        profiles=[],
        agent_playbooks=[
            SimpleNamespace(
                playbook_name="pb1",
                content="TASK: audit sample. INPUT: Population.xlsx. FINAL: 61 rows.",
            )
        ],
        user_playbooks=[],
    )
    bridge._client.search = MagicMock(return_value=response)

    result = asyncio.run(bridge.fetch_for_task(_EXAMPLE_TASK))
    assert result is not None
    # Recipe body is passed through verbatim
    assert "audit sample" in result
    assert "FINAL: 61 rows" in result
    # And the CACHED SOLUTION framing header is present
    assert "CACHED SOLUTION" in result


def test_reflexio_bridge_publish_swallows_errors() -> None:
    """Publishing never raises even if the backend errors out."""
    from benchmark.memory.reflexio_bridge import ReflexioMemory

    bridge = ReflexioMemory(user_id_prefix="bench_test")
    bridge._client.publish_interaction = MagicMock(side_effect=RuntimeError("boom"))

    asyncio.run(
        bridge.publish_trajectory(
            _EXAMPLE_TASK,
            [{"role": "User", "content": "hi"}, {"role": "Agent", "content": "hello"}],
        )
    )
    bridge._client.publish_interaction.assert_called_once()


# ---------------------------------------------------------------------------
# Snapshot round-trip
# ---------------------------------------------------------------------------


def test_copy_tree_round_trip(tmp_path: Path) -> None:
    """_copy_tree in the OpenSpace adapter reproduces contents byte-for-byte."""
    from benchmark.adapters.openspace_adapter import _copy_tree

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    (src / "nested").mkdir()
    (src / "nested" / "b.txt").write_text("world")

    dest = tmp_path / "dest"
    _copy_tree(src, dest)

    assert (dest / "a.txt").read_text() == "hello"
    assert (dest / "nested" / "b.txt").read_text() == "world"

    # Re-copying overwrites cleanly.
    (src / "a.txt").write_text("updated")
    _copy_tree(src, dest)
    assert (dest / "a.txt").read_text() == "updated"


# ---------------------------------------------------------------------------
# run_benchmark CLI helpers
# ---------------------------------------------------------------------------


def _mk_tasks(n: int) -> list[dict]:
    """Build `n` stub task dicts for slice tests."""
    return [{"task_id": f"t{i:03d}"} for i in range(n)]


def test_apply_task_slice_offset_and_cap() -> None:
    """_apply_task_slice skips `offset` tasks, then caps at `max_tasks`."""
    from benchmark.run_benchmark import _apply_task_slice

    tasks = _mk_tasks(10)
    sliced = _apply_task_slice(tasks, offset=5, max_tasks=3)
    assert [t["task_id"] for t in sliced] == ["t005", "t006", "t007"]


def test_apply_task_slice_offset_only() -> None:
    """max_tasks=None means "no cap" — offset still applies alone."""
    from benchmark.run_benchmark import _apply_task_slice

    tasks = _mk_tasks(4)
    sliced = _apply_task_slice(tasks, offset=2, max_tasks=None)
    assert [t["task_id"] for t in sliced] == ["t002", "t003"]


def test_apply_task_slice_offset_beyond_length_returns_empty() -> None:
    """offset >= len(tasks) yields an empty list instead of wrapping."""
    from benchmark.run_benchmark import _apply_task_slice

    tasks = _mk_tasks(3)
    assert _apply_task_slice(tasks, offset=5, max_tasks=10) == []


def test_apply_task_slice_negative_offset_clamped() -> None:
    """Negative offsets are clamped to 0 so callers can't accidentally
    reverse-index from the end of the list."""
    from benchmark.run_benchmark import _apply_task_slice

    tasks = _mk_tasks(3)
    sliced = _apply_task_slice(tasks, offset=-2, max_tasks=2)
    assert [t["task_id"] for t in sliced] == ["t000", "t001"]


def test_resolve_task_ids_csv_overrides_task_list(tmp_path: Path) -> None:
    """Explicit --task-ids takes precedence over --task-list (JSON file)."""
    from benchmark.run_benchmark import _resolve_task_ids

    list_file = tmp_path / "tasks.json"
    list_file.write_text('["from-file-001", "from-file-002"]')

    resolved = _resolve_task_ids(str(list_file), "from-cli-a, from-cli-b ,")
    assert resolved == ["from-cli-a", "from-cli-b"]


def test_resolve_task_ids_falls_back_to_list(tmp_path: Path) -> None:
    """With no CSV, _resolve_task_ids reads the JSON file."""
    from benchmark.run_benchmark import _resolve_task_ids

    list_file = tmp_path / "tasks.json"
    list_file.write_text('["a", "b"]')

    assert _resolve_task_ids(str(list_file), None) == ["a", "b"]


def test_resolve_task_ids_returns_none_when_both_absent() -> None:
    """No filter flags → no task-ID filter."""
    from benchmark.run_benchmark import _resolve_task_ids

    assert _resolve_task_ids(None, None) is None
    assert _resolve_task_ids("", "") is None
