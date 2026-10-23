"""Integration: LiteLLM dispatches to the openclaw provider end-to-end.

Exercises ``register_if_enabled`` + ``litellm.completion`` together so a
regression in the registration glue (provider key, handler installation,
or kwargs routing) is caught even when each layer's unit tests pass.

The ``subprocess.run`` call inside the provider is patched so the test
never spawns a real openclaw CLI.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import litellm
import pytest

from reflexio.server.llm.providers import openclaw_provider as ocp

pytestmark = pytest.mark.integration


def _fake_completed_process(stdout: str, *, stderr: str = "", returncode: int = 0):
    """Build a CompletedProcess look-alike for ``patch('subprocess.run')``."""
    from subprocess import CompletedProcess

    return CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


@pytest.fixture(autouse=True)
def _reset_provider_state(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with the provider unregistered + a clean LiteLLM map."""
    monkeypatch.setattr(ocp, "_REGISTERED", False, raising=False)
    monkeypatch.setattr(ocp, "_HANDLER", None, raising=False)
    monkeypatch.setattr(litellm, "custom_provider_map", [], raising=False)
    yield


def test_litellm_dispatches_to_openclaw_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """End-to-end: env opt-in → register → provider handler completes via fake CLI.

    Calls the registered handler directly rather than going through
    ``litellm.completion`` because the root conftest installs a global
    completion mock for unit/integration tests. The handler-side wiring is
    exactly what LiteLLM would route to in production.
    """
    fake_cli = tmp_path / "openclaw"
    fake_cli.write_text("#!/bin/sh\nexit 0\n")
    fake_cli.chmod(0o755)
    monkeypatch.setenv(ocp.ENV_ENABLE, "1")
    monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake_cli))

    assert ocp.register_if_enabled() is True

    # Registration installed our handler under the openclaw provider key.
    entry = next(
        item
        for item in getattr(litellm, "custom_provider_map", [])
        if item.get("provider") == ocp.PROVIDER_KEY
    )
    handler = entry["custom_handler"]
    assert isinstance(handler, ocp.OpenClawLLM)

    fake_proc = _fake_completed_process(
        json.dumps({"ok": True, "outputs": [{"text": "PONG"}]})
    )
    with patch("subprocess.run", return_value=fake_proc) as run:
        resp = handler.completion(
            model="openclaw/anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "ping"}],
        )

    assert resp.choices[0].message.content == "PONG"

    # Provider unwrapped the openclaw/ prefix and forwarded the inner model id.
    argv = run.call_args[0][0]
    assert "--model" in argv
    assert "anthropic/claude-sonnet-4-6" in argv
    # Recursion guard env is set on the spawned process.
    spawn_env = run.call_args[1]["env"]
    assert spawn_env.get("OPENCLAW_SMART_INTERNAL") == "1"


def test_register_noop_when_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``OPENCLAW_SMART_USE_LOCAL_CLI``, registration is skipped."""
    monkeypatch.delenv(ocp.ENV_ENABLE, raising=False)
    assert ocp.register_if_enabled() is False
    assert litellm.custom_provider_map == []


def test_register_noop_when_cli_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Env opt-in without a resolvable CLI path is a logged no-op, not a crash."""
    monkeypatch.setenv(ocp.ENV_ENABLE, "1")
    # Point to a path that doesn't exist *and* override PATH so the
    # ``shutil.which`` fallback also misses.
    monkeypatch.setenv(ocp.ENV_CLI_PATH, str(tmp_path / "does-not-exist"))
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ocp.register_if_enabled() is False


def test_spawned_completion_worker_can_reach_the_openclaw_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A ``spawn`` child must register the provider itself.

    uvicorn workers are spawned, so their hard-timeout completion child is too,
    and it does not inherit the parent's ``custom_provider_map``. Run the real
    worker in a spawned process against a fake ``openclaw`` binary.
    """
    import multiprocessing

    from reflexio.server.llm._litellm_subprocess import _litellm_completion_worker

    fake = tmp_path / "openclaw"
    fake.write_text('#!/bin/sh\necho \'{"ok": true, "outputs": [{"text": "PONG"}]}\'\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENCLAW_SMART_USE_LOCAL_CLI", "1")
    monkeypatch.setenv("OPENCLAW_BIN", str(fake))

    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_litellm_completion_worker,
        args=(
            {
                "model": "openclaw/default",
                "messages": [{"role": "user", "content": "hi"}],
            },
            queue,
        ),
    )
    process.start()
    status, result = queue.get(timeout=120)
    process.join(30)
    assert status == "ok", result
    assert result.choices[0].message.content == "PONG"


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.6-luna",  # as LiteLLM hands it to the handler (prefix removed)
        "openclaw/openai/gpt-5.6-luna",  # as Reflexio config names it
    ],
)
def test_provider_qualified_model_reaches_the_cli_intact(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenClaw needs ``--model <provider>/<model>``.

    The handler used to drop the first path segment unconditionally, so the
    LiteLLM-stripped ``openai/gpt-x`` became ``gpt-x`` and OpenClaw refused it
    with "Model overrides must use the form <provider/model>".
    """
    monkeypatch.delenv(ocp.ENV_DEFAULT_MODEL, raising=False)
    with patch.object(ocp, "_call_cli", return_value="ok") as call:
        ocp.OpenClawLLM().completion(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )
    assert call.call_args.kwargs["model"] == "openai/gpt-5.6-luna"
