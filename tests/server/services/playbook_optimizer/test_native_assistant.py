"""The local assistant compiler never owns the native model process."""

import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from reflexio.server.llm.providers import claude_code_provider as provider
from reflexio.server.llm.providers.claude_code_stream_parser import ParseResult
from reflexio.server.services.playbook_optimizer.assistant_webhook import (
    LocalScriptAssistant,
)

HEADER = "_REFLEXIO_ASSISTANT_INVOCATION"


def _compiler(tmp_path: Path, source: str, timeout: int = 2) -> LocalScriptAssistant:
    script = tmp_path / "compiler.py"
    script.write_text(source, encoding="utf-8")
    return LocalScriptAssistant(sys.executable, [str(script)], timeout)


def test_compiled_native_request_is_executed_by_sdk(tmp_path, monkeypatch):
    request = {
        "protocol": "codex",
        "executable": sys.executable,
        "model": "owned-model",
        "system_prompt": "synthetic system",
        "dialogue": "synthetic task",
    }
    assistant = _compiler(
        tmp_path, "print(" + repr(json.dumps({"native_cli_v1": request})) + ")"
    )
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == "owned-model"
        assert kwargs["optimizer"] is True
        assert 0 < kwargs["deadline"] - time.monotonic() <= 2
        return ParseResult(True, "owned result")

    monkeypatch.setattr(provider, "_run_codex_stream", launch)
    assert assistant([], []) == "owned result"
    assert len(calls) == 1


def test_private_capability_overwrites_parent_value_without_changing_payload(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(HEADER, "private-untrusted-parent-value")
    assistant = _compiler(
        tmp_path,
        "import os,json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "print(json.dumps({'content':json.dumps({'header':json.loads(os.environ["
        + repr(HEADER)
        + "]), 'keys':sorted(payload)})}))\n",
    )
    result = json.loads(assistant([], []))
    assert result["keys"] == ["messages", "playbooks"]
    assert result["header"]["protocol"] == "native-cli/v1"
    assert result["header"]["deadline_monotonic"] > 0
    assert os.environ[HEADER] == "private-untrusted-parent-value"


def _request(**changes):
    return {
        "protocol": "codex",
        "executable": sys.executable,
        "model": None,
        "system_prompt": "synthetic system",
        "dialogue": "synthetic task",
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol": "unknown-private"},
        {"executable": ""},
        {"model": True},
        {"timeout": 600},
        {"env": {"private": "private-value"}},
        {"permissions": "allow"},
        {"argv": ["private-value"]},
    ],
)
def test_invalid_native_request_never_dispatches(
    tmp_path, monkeypatch, caplog, changes
):
    from reflexio.server.services.playbook_optimizer import assistant_webhook as adapter
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    launch = Mock()
    monkeypatch.setattr(adapter, "run_native_assistant", launch)
    assistant = _compiler(
        tmp_path,
        "print(" + repr(json.dumps({"native_cli_v1": _request(**changes)})) + ")",
    )
    with pytest.raises(
        LocalScriptFailedError, match="invalid native request"
    ) as failure:
        assistant([], [])
    assert "private" not in str(failure.value) + caplog.text
    launch.assert_not_called()


def test_ambiguous_native_and_content_result_never_dispatches(tmp_path, monkeypatch):
    from reflexio.server.services.playbook_optimizer import assistant_webhook as adapter
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    launch = Mock()
    monkeypatch.setattr(adapter, "run_native_assistant", launch)
    assistant = _compiler(
        tmp_path,
        "print("
        + repr(json.dumps({"native_cli_v1": _request(), "content": "private"}))
        + ")",
    )
    with pytest.raises(LocalScriptFailedError, match="ambiguous"):
        assistant([], [])
    launch.assert_not_called()


def test_parsing_time_cannot_extend_native_deadline(monkeypatch):
    from reflexio.server.services.playbook_optimizer import assistant_webhook as adapter
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    now = [0.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: now[0])

    def compile_request(*_args, **_kwargs):
        now[0] = 3.0
        return subprocess.CompletedProcess(
            [], 0, json.dumps({"native_cli_v1": _request()}), ""
        )

    monkeypatch.setattr(adapter, "run_native_cli", compile_request)
    launch = Mock()
    monkeypatch.setattr(provider, "_run_codex_stream", launch)
    with pytest.raises(LocalScriptFailedError, match="deadline expired"):
        LocalScriptAssistant("owned-compiler", [], 2)([], [])
    launch.assert_not_called()


def test_caller_allowance_above_old_300_seconds_survives(tmp_path, monkeypatch):
    calls = []

    def launch(**kwargs):
        calls.append(kwargs)
        return ParseResult(True, "owned")

    monkeypatch.setattr(provider, "_run_codex_stream", launch)
    assistant = _compiler(
        tmp_path,
        "print(" + repr(json.dumps({"native_cli_v1": _request()})) + ")",
        timeout=600,
    )
    assert assistant([], []) == "owned"
    assert 300 < calls[0]["deadline"] - time.monotonic() <= 600


@pytest.mark.parametrize("field", ["result", "response"])
@pytest.mark.parametrize("model", [None, "opaque model"])
def test_custom_claude_json_contract_and_readonly_policy(
    tmp_path, monkeypatch, field, model
):
    from reflexio.server.services.playbook_optimizer import assistant_native as native

    calls = []

    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({field: "owned reply"}), ""
        )

    monkeypatch.setattr(native, "run_native_cli", launch)
    monkeypatch.setenv("CLAUDE_SMART_CLI_MODEL", "unused-private-model")
    request = _request(protocol="claude-json", model=model)
    assistant = _compiler(
        tmp_path, "print(" + repr(json.dumps({"native_cli_v1": request})) + ")"
    )
    assert assistant([], []) == "owned reply"
    argv, kwargs = calls[0]
    assert argv[:4] == [sys.executable, "-p", "--output-format", "json"]
    assert "stream-json" not in argv
    assert argv[argv.index("--permission-mode") + 1] == "plan"
    assert argv[argv.index("--tools") + 1] == "Read,Grep,Glob,LS"
    assert (
        argv[argv.index("--disallowedTools") + 1]
        == "Bash,Edit,Write,MultiEdit,NotebookEdit"
    )
    assert "--no-session-persistence" in argv and "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert argv[argv.index("--append-system-prompt") + 1] == request["system_prompt"]
    assert kwargs["input"] == request["dialogue"]
    assert kwargs["env"]["CLAUDE_SMART_INTERNAL"] == "1"
    assert kwargs["env"]["CLAUDE_CODE_ENTRYPOINT"] == "optimizer"
    assert "unused-private-model" not in argv
    if model is None:
        assert "--model" not in argv
    else:
        assert argv[argv.index("--model") + 1] == model


@pytest.mark.parametrize(
    "body,code",
    [
        ("private-native-output", 0),
        ('{"result":null}', 0),
        ("{}", 0),
        ('{"result":"private-native-output"}', 1),
    ],
)
def test_native_errors_do_not_replay_or_expose_output(
    tmp_path, monkeypatch, caplog, body, code
):
    from reflexio.server.services.playbook_optimizer import assistant_native as native
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    effect = tmp_path / "compile-effects"
    request = _request(protocol="claude-json")
    compiler = (
        "from pathlib import Path\nwith Path("
        + repr(str(effect))
        + ").open('a') as f:f.write('compile\\n')\nprint("
        + repr(json.dumps({"native_cli_v1": request}))
        + ")"
    )
    assistant = _compiler(tmp_path, compiler)
    launch = Mock(
        return_value=subprocess.CompletedProcess([], code, body, "private-native-error")
    )
    monkeypatch.setattr(native, "run_native_cli", launch)
    with pytest.raises(LocalScriptFailedError) as failure:
        assistant([], [])
    assert "private-native" not in str(failure.value) + caplog.text
    assert launch.call_count == 1
    assert effect.read_text() == "compile\n"


def _owned_native_request(tmp_path, monkeypatch, model, delay):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    record = tmp_path / "native.json"
    (hooks / "sitecustomize.py").write_text(
        "import sys,os,json,time\nfrom pathlib import Path\n"
        "if Path(sys.argv[0]).name == 'exec':\n"
        " data={'pid':os.getpid(),'ppid':os.getppid(),'argv':sys.argv,'stdin':sys.stdin.read()}\n"
        " data['env']={key:os.environ.get(key) for key in ('CLAUDE_SMART_INTERNAL','CLAUDE_CODE_ENTRYPOINT')}\n"
        f" Path({str(record)!r}).write_text(json.dumps(data),encoding='utf-8')\n"
        f" time.sleep({delay})\n"
        " Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('owned reply',encoding='utf-8')\n"
        " os._exit(0)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(hooks) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    monkeypatch.setenv("CLAUDE_SMART_CLI_MODEL", "unused-private-model")
    executable = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    request = _request(executable=executable, model=model)
    return request, record


@pytest.mark.parametrize("model", [None, "owned/native-model"])
@pytest.mark.parametrize("delay", [0, 30])
def test_real_compiler_native_process_and_timeout_reaping(
    tmp_path, monkeypatch, model, delay
):
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    request, record = _owned_native_request(tmp_path, monkeypatch, model, delay)
    assistant = _compiler(
        tmp_path,
        "print(" + repr(json.dumps({"native_cli_v1": request})) + ")",
        timeout=2 if delay else 5,
    )
    children = []
    original_popen = subprocess.Popen

    def observed(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", observed)
    started = time.monotonic()
    if delay:
        with pytest.raises(LocalScriptFailedError, match="timed out"):
            assistant([], [])
        assert time.monotonic() - started < 4
    else:
        assert assistant([], []) == "owned reply"
    entry = json.loads(record.read_text())
    assert entry["ppid"] == os.getpid(), "the compiler launched the model CLI"
    owned = [child for child in children if child.pid == entry["pid"]]
    assert len(owned) == 1 and owned[0].poll() is not None
    assert entry["env"] == {
        "CLAUDE_SMART_INTERNAL": "1",
        "CLAUDE_CODE_ENTRYPOINT": "optimizer",
    }
    assert entry["argv"][entry["argv"].index("--sandbox") + 1] == "read-only"
    assert all(
        flag in entry["argv"]
        for flag in ("--ephemeral", "--ignore-rules", "--skip-git-repo-check")
    )
    assert "unused-private-model" not in repr(entry["argv"])
    if model is None:
        assert "--model" not in entry["argv"]
    else:
        assert entry["argv"][entry["argv"].index("--model") + 1] == model
    assert (
        entry["stdin"]
        == request["system_prompt"] + "\n\n## Task\n" + request["dialogue"]
    )
    assert not Path(
        entry["argv"][entry["argv"].index("--output-last-message") + 1]
    ).exists()
    if os.name != "nt":
        with pytest.raises(ProcessLookupError):
            os.kill(entry["pid"], 0)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("private-path"),
        ValueError("private-argument"),
        UnicodeDecodeError("utf8", b"private-bytes", 0, 1, "private-reason"),
    ],
)
def test_native_runtime_failures_are_metadata_only(
    tmp_path, monkeypatch, caplog, failure
):
    from reflexio.server.services.playbook_optimizer.assistant_webhook import (
        LocalScriptFailedError,
    )

    launch = Mock(side_effect=failure)
    monkeypatch.setattr(provider, "_run_codex_stream", launch)
    assistant = _compiler(
        tmp_path, "print(" + repr(json.dumps({"native_cli_v1": _request()})) + ")"
    )
    with pytest.raises(LocalScriptFailedError) as caught:
        assistant([], [])
    assert "private" not in str(caught.value) + caplog.text
    assert launch.call_count == 1


def _compiled_completion(params):
    result = LocalScriptAssistant(params["compiler"], [params["script"]], 15)([], [])
    return {"reply": result, "worker_pid": os.getpid()}


def _compiled_worker(params, result_queue, deadline, *native_context):
    from reflexio.server.llm import _litellm_subprocess as worker

    worker.model_completion = _compiled_completion
    worker._register_local_cli_providers = lambda: None
    worker._litellm_completion_worker(params, result_queue, deadline, *native_context)


@pytest.mark.parametrize(
    "method",
    ["spawn", *(["fork"] if "fork" in multiprocessing.get_all_start_methods() else [])],
)
@pytest.mark.parametrize("delay", [0, 30])
def test_compiler_and_native_calls_reuse_retained_owner(
    tmp_path, monkeypatch, method, delay
):
    import threading

    import psutil

    from reflexio.server.llm import _litellm_text_generation as generation
    from reflexio.server.llm import _provider_concurrency as admission
    from reflexio.server.llm._litellm_types import LLMHardTimeoutError
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
    )

    request, record = _owned_native_request(tmp_path, monkeypatch, "owned/model", delay)
    script = tmp_path / "compiler.py"
    compiler_record = tmp_path / "compiler.json"
    script.write_text(
        "import os,json\nfrom pathlib import Path\nPath("
        + repr(str(compiler_record))
        + ").write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid()}))\nprint("
        + repr(json.dumps({"native_cli_v1": request}))
        + ")"
    )
    context = multiprocessing.get_context(method)
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(generation, "_litellm_completion_worker", _compiled_worker)
    client = LiteLLMClient(LiteLLMConfig(model="openai/gpt-4o", fallback_models=[]))
    monkeypatch.setattr(client, "_run_completion", generation._run_isolated_completion)
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(admission, "_get_semaphore", lambda _name: semaphore)
    with admission.named_provider_slot("native-assistant-test"):
        options = {
            "model": "openai/gpt-4o",
            "compiler": sys.executable,
            "script": str(script),
        }
        if delay:
            with pytest.raises(
                (LiteLLMClientError, LLMHardTimeoutError), match="timeout|timed out"
            ):
                client._completion_with_hard_timeout(
                    options, 10 if method == "spawn" else 3
                )
        else:
            result = client._completion_with_hard_timeout(options, 30)
            assert result["reply"] == "owned reply"
    assert semaphore.acquire(timeout=5), "native admission was not retired"
    semaphore.release()
    entry = json.loads(record.read_text())
    compiled = json.loads(compiler_record.read_text())
    assert entry["ppid"] not in {os.getpid(), compiled["pid"]}, (
        "compiler or worker owns native CLI"
    )
    assert compiled["ppid"] != os.getpid(), "compiler bypassed the native owner"
    assert not psutil.pid_exists(entry["pid"])
    assert not psutil.pid_exists(compiled["pid"])
    if not delay:
        assert entry["ppid"] != result["worker_pid"]
        assert compiled["ppid"] != result["worker_pid"]


def test_codex_setup_cannot_start_after_absolute_deadline(tmp_path, monkeypatch):
    from reflexio.server.services.playbook_optimizer import assistant_native as native

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    output = tmp_path / "native-output"

    def slow_setup():
        output.touch()
        now[0] = 3.0
        return output

    monkeypatch.setattr(provider, "_temporary_output_path", slow_setup)
    launch = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(provider, "run_native_cli", launch)
    with pytest.raises(native.NativeAssistantError, match="timed out|expired"):
        native.run_native_assistant(native.NativeAssistantRequest(**_request()), 2)
    launch.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("protocol", ["codex", "claude-json"])
def test_native_decoding_cannot_accept_expired_content(tmp_path, monkeypatch, protocol):
    from reflexio.server.services.playbook_optimizer import assistant_native as native

    now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    output = tmp_path / "native-output"
    output.touch()
    if protocol == "codex":
        monkeypatch.setattr(provider, "_temporary_output_path", lambda: output)
        original_read = Path.read_text

        def slow_read(path, *args, **kwargs):
            if path == output:
                now[0] = 3.0
                return "owned reply"
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", slow_read)
        launch = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        monkeypatch.setattr(provider, "run_native_cli", launch)
    else:
        body = '{"result":"owned reply"}'
        original_loads = json.loads

        def slow_decode(value, *args, **kwargs):
            if value == body:
                now[0] = 3.0
            return original_loads(value, *args, **kwargs)

        monkeypatch.setattr(json, "loads", slow_decode)
        launch = Mock(return_value=subprocess.CompletedProcess([], 0, body, ""))
        monkeypatch.setattr(native, "run_native_cli", launch)
    with pytest.raises(native.NativeAssistantError, match="timed out|expired"):
        native.run_native_assistant(
            native.NativeAssistantRequest(**_request(protocol=protocol)), 2
        )
    assert launch.call_count == 1, "expired completion was replayed"
