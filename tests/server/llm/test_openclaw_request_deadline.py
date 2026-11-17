"""One selected budget through real LiteLLM and owned, keyless CLI processes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import litellm
import pytest
from litellm.types.utils import ModelResponse

from reflexio.server.llm.providers import openclaw_provider as provider


@pytest.mark.unit
@pytest.mark.parametrize(
    "timeout", [None, True, "2", 0, -1, float("inf"), float("nan"), 2**1024]
)
def test_invalid_or_missing_request_allowance_refuses_before_cli(monkeypatch, timeout):
    dispatch = Mock()
    monkeypatch.setattr(provider, "_call_cli", dispatch)
    with pytest.raises(ValueError, match="finite positive timeout"):
        provider.OpenClawLLM().completion(model="openclaw/default", timeout=timeout)
    dispatch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("legacy", ["0.1", "180", "private-invalid-duration"])
def test_retired_timeout_is_not_an_alias_or_silently_ignored(monkeypatch, legacy):
    monkeypatch.setenv("OPENCLAW_CLI_TIMEOUT", legacy)
    dispatch = Mock()
    monkeypatch.setattr(provider, "_call_cli", dispatch)
    with pytest.raises(ValueError, match="OPENCLAW_CLI_TIMEOUT") as failure:
        provider.OpenClawLLM().completion(model="openclaw/default", timeout=1)
    assert legacy not in str(failure.value)
    dispatch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("legacy", [None, "", "  "])
def test_empty_old_assignment_has_no_authority(monkeypatch, legacy):
    if legacy is None:
        monkeypatch.delenv("OPENCLAW_CLI_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("OPENCLAW_CLI_TIMEOUT", legacy)
    dispatch = Mock(return_value="ok")
    monkeypatch.setattr(provider, "_call_cli", dispatch)
    provider.OpenClawLLM().completion(model="openclaw/default", timeout=0.25)
    assert dispatch.call_args.kwargs["timeout_s"] == 0.25


def _prepare_owned_cli(monkeypatch, tmp_path):
    """Retain real dispatch/launch; only executable and capture destinations vary."""
    monkeypatch.delenv("OPENCLAW_CLI_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENCLAW_DEFAULT_MODEL", raising=False)
    calls = tmp_path / "calls.jsonl"
    script = tmp_path / "openclaw"
    monkeypatch.setenv("OPENCLAW_BIN", str(script))
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": "openclaw", "custom_handler": provider.OpenClawLLM()}],
    )
    dispatch = Mock(wraps=provider.run_native_cli)
    monkeypatch.setattr(provider, "run_native_cli", dispatch)

    def prepare(body):
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, time\nfrom pathlib import Path\n"
            f"calls = Path({str(calls)!r})\n"
            "previous = calls.read_text().splitlines() if calls.exists() else []\n"
            "with calls.open('a') as out: out.write(json.dumps({'pid': os.getpid(), 'time': time.monotonic()}) + '\\n')\n"
            + body
        )
        script.chmod(0o700)
        return calls, dispatch

    return prepare


def _complete(timeout):
    return litellm.completion(
        model="openclaw/default",
        messages=[{"role": "user", "content": "owned synthetic deadline test"}],
        timeout=timeout,
        num_retries=0,
    )


def _assert_reaped(calls: Path):
    for line in calls.read_text().splitlines():
        pid = json.loads(line)["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _probe_fractional_timeout(owned_cli):
    calls, dispatch = owned_cli('print(json.dumps({"outputs": [{"text": "ok"}]}))\n')
    response = _complete(1.25)
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "ok"
    assert dispatch.call_count == 1
    assert 0 < dispatch.call_args.kwargs["timeout"] <= 1.25
    assert dispatch.call_args.kwargs["timeout"] > 1.0
    _assert_reaped(calls)


def _probe_timeout_reaping(owned_cli):
    calls, dispatch = owned_cli("time.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(Exception, match="openclaw timed out"):
        _complete(0.3)
    assert time.monotonic() - started < 2
    assert dispatch.call_count == 1
    _assert_reaped(calls)


def _probe_busy_retries(owned_cli, monkeypatch):
    monkeypatch.setattr(provider, "_BUSY_BACKOFF_SECONDS", 0.05)
    calls, dispatch = owned_cli(
        "if not previous:\n"
        " time.sleep(0.3)\n"
        " print(json.dumps({'ok': False, 'error': 'retry when it finishes'}))\n"
        "else: time.sleep(30)\n"
    )
    with pytest.raises(Exception, match="openclaw timed out"):
        _complete(1.0)
    assert dispatch.call_count == 2
    first, second = [call.kwargs["timeout"] for call in dispatch.call_args_list]
    assert 0 < second < first - 0.3
    assert first <= 1.0
    _assert_reaped(calls)


def _probe_busy_backoff(owned_cli):
    calls, dispatch = owned_cli(
        "print(json.dumps({'ok': False, 'error': 'retry when it finishes'}))\n"
    )
    started = time.monotonic()
    with pytest.raises(Exception, match="deadline cannot accommodate busy retry"):
        _complete(0.5)
    assert time.monotonic() - started < 1.5
    assert dispatch.call_count == 1
    _assert_reaped(calls)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["fractional", "reaping", "retry", "backoff"])
@pytest.mark.skipif(
    os.name == "nt", reason="POSIX executable fixture; native ownership covers Windows"
)
def test_unpatched_litellm_and_owned_native_process(tmp_path, mode):
    # A fresh interpreter bypasses the repo-wide paid-provider mocks. Pass no
    # user credentials, telemetry configuration or host CLI configuration.
    env = {
        key: os.environ[key]
        for key in ("PATH", "TMPDIR", "LANG", "LC_ALL")
        if key in os.environ
    }
    home = tmp_path / "home"
    home.mkdir()
    empty = home / "empty.env"
    empty.touch()
    env.update(
        HOME=str(home),
        REFLEXIO_ENV_FILE=str(empty),
        PYTHON_DOTENV_DISABLED="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    result = subprocess.run(  # noqa: S603 — owned interpreter and test script.
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), mode],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
    assert "OWNED_OPENCLAW_DEADLINE_OK" in result.stdout


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        prepare = _prepare_owned_cli(patch, Path(sys.argv[1]))
        mode = sys.argv[2]
        if mode == "fractional":
            _probe_fractional_timeout(prepare)
        elif mode == "reaping":
            _probe_timeout_reaping(prepare)
        elif mode == "retry":
            _probe_busy_retries(prepare, patch)
        elif mode == "backoff":
            _probe_busy_backoff(prepare)
        else:
            raise AssertionError(mode)
    print("OWNED_OPENCLAW_DEADLINE_OK")
