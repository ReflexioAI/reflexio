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

    return CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _reset_provider_state(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with the provider unregistered + a clean LiteLLM map."""
    monkeypatch.setattr(ocp, "_REGISTERED", False, raising=False)
    monkeypatch.setattr(ocp, "_HANDLER", None, raising=False)
    monkeypatch.setattr(
        litellm, "custom_provider_map", [], raising=False
    )
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

    fake_proc = _fake_completed_process(json.dumps({"text": "PONG"}))
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
