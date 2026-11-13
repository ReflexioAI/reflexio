"""Integration: LiteLLM dispatches to the openclaw provider end-to-end.

Exercises ``register_if_enabled`` + ``litellm.completion`` together so a
regression in the registration glue (provider key, handler installation,
or kwargs routing) is caught even when each layer's unit tests pass.

The ``subprocess.run`` call inside the provider is patched so the test
never spawns a real openclaw CLI.
"""

from __future__ import annotations

import json
import time
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
            time.monotonic() + 120,
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


def test_fresh_process_uses_real_litellm_and_owned_native_cli(tmp_path, monkeypatch):
    """Prove transport selection without the pytest completion mock or a paid CLI."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setenv("BRAINTRUST_API_KEY", "synthetic-parent-telemetry-canary")
    monkeypatch.setenv("REFLEXIO_TEST_ENV_CANARY", "parent-only")
    record = tmp_path / "native-calls.jsonl"
    standin = tmp_path / "native.py"
    standin.write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        f"record=Path({str(record)!r})\n"
        "args=sys.argv[1:]\n"
        "prompt=args[args.index('--prompt')+1]\n"
        "entry={'argv':args,'pid':os.getpid(),'internal':os.environ.get('OPENCLAW_SMART_INTERNAL')}\n"
        "with record.open('a') as stream: stream.write(json.dumps(entry)+'\\n')\n"
        "text=(json.dumps({'tool':'finish_test','args':{'answer':'ok'}}) if 'tool-synthetic' in prompt else json.dumps({'answer':'ok'}) if 'structured-synthetic' in prompt else 'from-owned-cli')\n"
        "print(json.dumps({'ok':True,'outputs':[{'text':text}]}))\n"
    )
    if os.name == "nt":
        cli = tmp_path / "openclaw.cmd"
        cli.write_text(f'@"{sys.executable}" "{standin}" %*\r\n')
    else:
        cli = tmp_path / "openclaw"
        cli.write_text(f"#!{sys.executable}\n" + standin.read_text())
        cli.chmod(0o700)
    program = tmp_path / "probe.py"
    program.write_text(
        """import json,os,threading
assert 'BRAINTRUST_API_KEY' not in os.environ
assert 'REFLEXIO_TEST_ENV_CANARY' not in os.environ
from http.server import BaseHTTPRequestHandler,HTTPServer
from pydantic import BaseModel
from reflexio.server.llm.litellm_client import LiteLLMClient,LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
class Answer(BaseModel):
    answer:str
def local_only(params,*_args):
    assert params['model'].startswith('openclaw/'), 'Unintended cloud route'
    endpoint=params.get('api_base') or params.get('base_url')
    assert not endpoint or endpoint.startswith('http://127.0.0.1:'), 'Unowned endpoint'
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.bodies.append(body)
        result={'id':'synthetic-http','object':'chat.completion','created':1,'model':'loopback-model','choices':[{'index':0,'message':{'role':'assistant','content':'from-owned-http'},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        payload=json.dumps(result).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def log_message(self,*_args): pass
if __name__=='__main__':
    target='openclaw/anthropic/claude-sonnet-4-6'
    client=LiteLLMClient(LiteLLMConfig(model=target,max_tokens=32,timeout=20,fallback_models=[]))
    os.environ['OPENCLAW_DEFAULT_MODEL']='private-unused-model'
    native=client.generate_chat_response_with_provenance([{'role':'user','content':'explicit-synthetic'}],model_role=ModelRole.EXTRACTION_AGENT,provider_request_guard=local_only)
    assert native.value=='from-owned-cli' and native.provenance.model_name is None and native.provenance.provider is None
    tools=[{'type':'function','function':{'name':'finish_test','description':'Synthetic test finish','parameters':{'type':'object','properties':{'answer':{'type':'string'}},'required':['answer']}}}]
    native_tool=client.generate_chat_response_with_provenance([{'role':'user','content':'tool-synthetic'}],tools=tools,provider_request_guard=local_only)
    assert native_tool.provenance.model_name is None and native_tool.provenance.provider is None
    output=native_tool.value
    assert output.tool_calls[0].function.name=='finish_test'
    assert json.loads(output.tool_calls[0].function.arguments)=={'answer':'ok'}
    output=client.generate_chat_response([{'role':'user','content':'structured-synthetic'}],response_format=Answer,provider_request_guard=local_only)
    assert output.answer=='ok'
    os.environ.pop('OPENCLAW_DEFAULT_MODEL')
    assert client.generate_chat_response([{'role':'user','content':'default-synthetic'}],model='openclaw/default',model_role=ModelRole.EXTRACTION_AGENT,provider_request_guard=local_only)=='from-owned-cli'
    server=HTTPServer(('127.0.0.1',0),Handler);server.bodies=[]
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        os.environ['OPENCLAW_DEFAULT_MODEL']='private-unused-model'
        http=client.generate_chat_response_with_provenance([{'role':'user','content':'http-synthetic'}],model='openclaw/default',custom_llm_provider='openai',api_base=f'http://127.0.0.1:{server.server_port}/v1',provider_request_guard=local_only)
        assert http.value=='from-owned-http' and http.provenance.model_name=='loopback-model'
        assert len(server.bodies)==1 and server.bodies[0]['model']=='openclaw/default'
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
    print('OWNED_TRANSPORT_OK')
"""
    )
    # Only process-launch essentials cross the boundary: provider/telemetry
    # credentials, callbacks, proxy settings and user configuration do not.
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
        )
        if key in os.environ
    }
    env.update(
        OPENAI_API_KEY="sk-test-not-real",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        OPENCLAW_SMART_USE_LOCAL_CLI="1",
        OPENCLAW_BIN=str(cli),
        PYTHON_DOTENV_DISABLED="1",
        REFLEXIO_ENV_FILE=str(tmp_path / "empty.env"),
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        APPDATA=str(tmp_path / "home" / "appdata"),
        LOCALAPPDATA=str(tmp_path / "home" / "localappdata"),
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
    )
    Path(env["HOME"]).mkdir()
    Path(env["REFLEXIO_ENV_FILE"]).touch()
    result = subprocess.run(  # noqa: S603 — fixed interpreter and test-owned script.
        [sys.executable, str(program)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "OWNED_TRANSPORT_OK" in result.stdout
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == 4
    assert all(call["internal"] == "1" for call in calls)
    for call in calls[:3]:
        argv = call["argv"]
        assert argv[argv.index("--model") + 1] == "anthropic/claude-sonnet-4-6"
    assert "--model" not in calls[3]["argv"]
    assert "finish_test" in calls[1]["argv"][calls[1]["argv"].index("--prompt") + 1]
    assert "answer" in calls[2]["argv"][calls[2]["argv"].index("--prompt") + 1]
    assert "private-unused-model" not in result.stdout + result.stderr
