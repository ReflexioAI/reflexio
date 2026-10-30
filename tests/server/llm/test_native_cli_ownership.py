"""An isolated SDK worker must not own the lifetime of a native provider CLI."""

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from reflexio.server.llm import _litellm_subprocess as worker_module
from reflexio.server.llm import _litellm_text_generation as generation
from reflexio.server.llm import _provider_concurrency as admission
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers import claude_code_provider


def _synthetic_completion(params):
    os.environ["SDK_WORKER_PID"] = str(os.getpid())
    return claude_code_provider._run_claude_stream(
        cli_path=params["native_cli_script"],
        system_prompt="",
        dialogue="synthetic local CLI only",
        timeout_seconds=30,
    )


def _synthetic_worker(params, result_queue, deadline, *native_context):
    worker_module.model_completion = _synthetic_completion
    worker_module._register_local_cli_providers = lambda: None
    worker_module._litellm_completion_worker(
        params, result_queue, deadline, *native_context
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable baseline repro")
def test_native_cli_is_parent_owned_and_reaped_after_worker_expiry(
    monkeypatch, tmp_path
):
    brokers = _capture_brokers(monkeypatch)
    pid_file = tmp_path / "native-pid"
    release_file = tmp_path / "release-native"
    script = tmp_path / "synthetic-cli"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os,time\n"
        "from pathlib import Path\n"
        "worker_pid=os.environ['SDK_WORKER_PID']\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{os.getppid()}} {{worker_pid}}')\n"
        "end=time.monotonic()+5\n"
        f"while not Path({str(release_file)!r}).exists() and time.monotonic()<end: time.sleep(.02)\n"
    )
    script.chmod(0o700)
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(generation, "_litellm_completion_worker", _synthetic_worker)
    client = LiteLLMClient(LiteLLMConfig(model="openai/gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    semaphore = threading.BoundedSemaphore(1)
    released = threading.Event()
    original_release = semaphore.release

    def release():
        original_release()
        released.set()

    monkeypatch.setattr(semaphore, "release", release)
    monkeypatch.setattr(admission, "_provider_key", lambda _: "native-cli-test")
    monkeypatch.setattr(admission, "_get_semaphore", lambda _: semaphore)
    started = time.monotonic()
    try:
        with (
            admission.provider_slot("openai/gpt-4o"),
            pytest.raises(Exception, match="hard timeout|timed out"),
        ):
            client._completion_with_hard_timeout(
                {"model": "openai/gpt-4o", "native_cli_script": str(script)}, 1.0
            )
        assert time.monotonic() - started < 3
        assert released.wait(3), "Native process ownership was never retired"
        pid, parent_pid, worker_pid = map(int, pid_file.read_text().split())
        assert parent_pid not in {os.getpid(), worker_pid}
        assert any(b.pid == parent_pid and b.returncode == 0 for b in brokers)
        assert pid != worker_pid
    finally:
        release_file.touch()


def _broker_completion(params):
    from reflexio.server.llm._native_cli import native_cli_directory, run_native_cli

    if params.get("native_cwd"):
        os.chdir(params["native_cwd"])
    environment = params.get(
        "native_env", {**os.environ, "SDK_WORKER_PID": str(os.getpid())}
    )
    output = (
        claude_code_provider._temporary_output_path()
        if params.get("native_codex_temp")
        else None
    )
    if output is not None:
        environment["NATIVE_OUTPUT_FILE"] = str(output)
    try:
        results = []
        for index in range(params.get("native_calls", 1)):
            result = run_native_cli(
                # Exercise one actual native executable, not Windows' venv
                # redirector plus a descendant. The synthetic CLI speaks UTF-8
                # independently of the host console's default encoding.
                [
                    vars(sys)["_base_executable"],
                    "-X",
                    "utf8",
                    "-c",
                    params["native_code"],
                ],
                input=params.get("native_input", str(index)),
                env=environment,
                timeout=params.get("native_timeout", 10),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            results.append(result.stdout)
        return "".join(results)
    finally:
        if output is not None and native_cli_directory() is None:
            output.unlink(missing_ok=True)


def _broker_worker(params, result_queue, deadline, native_context=None):
    worker_module.model_completion = _broker_completion
    worker_module._register_local_cli_providers = lambda: None
    worker_module._litellm_completion_worker(
        params, result_queue, deadline, native_context
    )


def _early_deadline_worker(params, result_queue, deadline, native_context=None):
    # Make the worker's real socket expiry win the parent queue-read race.
    _broker_worker(params, result_queue, deadline - 0.5, native_context)


def _broker_client(monkeypatch, method):
    context = multiprocessing.get_context(method)
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(generation, "_litellm_completion_worker", _broker_worker)
    client = LiteLLMClient(LiteLLMConfig(model="openai/gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    return client


def _permit(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    released = threading.Event()
    release = semaphore.release

    def note_release():
        release()
        released.set()

    monkeypatch.setattr(semaphore, "release", note_release)
    monkeypatch.setattr(admission, "_provider_key", lambda _: "native-cli-test")
    monkeypatch.setattr(admission, "_get_semaphore", lambda _: semaphore)
    return semaphore, released


def _capture_brokers(monkeypatch):
    import subprocess

    brokers = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        brokers.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    return brokers


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_broker_preserves_large_output_input_and_parent_identity(monkeypatch, method):
    import json

    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not supported on this platform")
    client = _broker_client(monkeypatch, method)
    processes = _capture_brokers(monkeypatch)
    _, released = _permit(monkeypatch)
    text = "input-λ" * 10000
    code = "import os,sys,json;print(json.dumps({'parent':os.getppid(),'input':sys.stdin.read(),'output':'x'*200000}))"
    with admission.provider_slot("openai/gpt-4o"):
        response = client._completion_with_hard_timeout(
            {"model": "openai/gpt-4o", "native_code": code, "native_input": text}, 30
        )
    payload = json.loads(response)
    assert released.wait(5)
    assert payload == {
        "parent": processes[0].pid,
        "input": text,
        "output": "x" * 200000,
    }
    assert payload["parent"] != os.getpid()
    assert released.wait(5)
    assert len(processes) == 1 and processes[0].returncode == 0


def test_broker_serial_requests_do_not_reuse_a_previous_reply(monkeypatch):
    client = _broker_client(monkeypatch, "spawn")
    response = client._completion_with_hard_timeout(
        {
            "model": "openai/gpt-4o",
            "native_code": "import sys;print(sys.stdin.read())",
            "native_calls": 3,
        },
        30,
    )
    assert response == "0\n1\n2\n"


def test_native_shorter_cap_preserves_timeout_error_and_reaps_child(monkeypatch):
    from reflexio.server.llm.litellm_client import LiteLLMClientError

    client = _broker_client(monkeypatch, "spawn")
    processes = _capture_brokers(monkeypatch)
    _, released = _permit(monkeypatch)
    with (
        admission.provider_slot("openai/gpt-4o"),
        pytest.raises(LiteLLMClientError) as caught,
    ):
        client._completion_with_hard_timeout(
            {
                "model": "openai/gpt-4o",
                "native_code": "import time;time.sleep(5)",
                "native_timeout": 0.1,
            },
            30,
        )
    assert caught.value.upstream_error_type == "TimeoutExpired"
    assert released.wait(5)
    assert len(processes) == 1 and processes[0].returncode is not None


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Controlled startup race uses a fast forked worker",
)
@pytest.mark.parametrize("interrupt_after_start", [False, True])
def test_inflight_native_startup_keeps_capacity_after_caller_finishes(
    monkeypatch, interrupt_after_start
):
    client = _broker_client(monkeypatch, "fork")
    processes = _capture_brokers(monkeypatch)
    semaphore, released = _permit(monkeypatch)
    native_started, return_handle = threading.Event(), threading.Event()
    import subprocess

    popen = subprocess.Popen

    def delayed_handle(*args, **kwargs):
        process = popen(*args, **kwargs)
        native_started.set()
        assert return_handle.wait(5), "Test did not release launch owner startup"
        return process

    monkeypatch.setattr(subprocess, "Popen", delayed_handle)
    if interrupt_after_start:
        start = threading.Thread.start

        def interrupted_start(thread):
            start(thread)
            if thread.name == "reflexio-native-cli":
                assert native_started.wait(3)
                raise KeyboardInterrupt("Synthetic interruption after thread startup")

        monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    try:
        expected = KeyboardInterrupt if interrupt_after_start else Exception
        with admission.provider_slot("openai/gpt-4o"), pytest.raises(expected):
            client._completion_with_hard_timeout(
                {"model": "openai/gpt-4o", "native_code": "import time;time.sleep(5)"},
                1,
            )
        assert native_started.is_set()
        assert not released.is_set(), "Worker exit released an unretired native process"
        assert not semaphore.acquire(blocking=False)
    finally:
        return_handle.set()
    assert released.wait(5)
    assert len(processes) == 1 and processes[0].returncode is not None


def test_native_thread_start_failure_releases_admission(monkeypatch):
    client = _broker_client(monkeypatch, "spawn")
    processes = _capture_brokers(monkeypatch)
    _, released = _permit(monkeypatch)
    start = threading.Thread.start

    def fail_native_start(thread):
        if thread.name == "reflexio-native-cli":
            raise RuntimeError("Synthetic native-thread start failure")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_native_start)
    with (
        admission.provider_slot("openai/gpt-4o"),
        pytest.raises(RuntimeError, match="native-thread start failure"),
    ):
        client._completion_with_hard_timeout(
            {"model": "openai/gpt-4o", "native_code": "print('must not start')"}, 30
        )
    assert released.wait(5)
    assert processes == []


def test_worker_death_does_not_abandon_its_running_cli(monkeypatch):
    client = _broker_client(monkeypatch, "spawn")
    processes = _capture_brokers(monkeypatch)
    _, released = _permit(monkeypatch)
    code = "import os,signal,time;os.kill(int(os.environ['SDK_WORKER_PID']),signal.SIGTERM);time.sleep(5)"
    with (
        admission.provider_slot("openai/gpt-4o"),
        pytest.raises(Exception, match="exited without returning"),
    ):
        client._completion_with_hard_timeout(
            {"model": "openai/gpt-4o", "native_code": code, "native_timeout": 0.5}, 30
        )
    assert released.wait(5)
    assert len(processes) == 1 and processes[0].returncode is not None


def test_worker_environment_and_cwd_are_preserved(monkeypatch, tmp_path):
    import json

    monkeypatch.setenv("NATIVE_PARENT_MARKER", "must not leak into empty child env")
    client = _broker_client(monkeypatch, "spawn")
    response = client._completion_with_hard_timeout(
        {
            "model": "openai/gpt-4o",
            "native_code": "import os,json;print(json.dumps([os.getcwd(),os.environ.get('NATIVE_PARENT_MARKER')]))",
            "native_cwd": str(tmp_path),
            "native_env": {},
        },
        30,
    )
    assert json.loads(response) == [str(tmp_path), None]


def _fault_owner_script(monkeypatch, tmp_path, mode):
    from reflexio.server.llm import _native_cli

    real_script = _native_cli._NATIVE_OWNER_SCRIPT
    script = tmp_path / "fault-owner.py"
    script.write_text(
        "import os,pickle,runpy,subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"api=runpy.run_path({str(real_script)!r})\n"
        "directory,counter,deadline=sys.argv[1],int(sys.argv[2]),float(sys.argv[3])\n"
        + (
            "os._exit(3)\n"
            if mode == "crash"
            else "api['_publish'](Path(directory)/f'retired-{counter}.pickle',(counter+1,'error',RuntimeError('foreign')))\n"
            if mode == "foreign"
            else "with (Path(directory)/f'request-{counter}.pickle').open('rb') as f: environment=pickle.load(f)['kwargs']['env']\n"
            "marker=Path(environment['TEST_REAP_MARKER']); release=Path(environment['TEST_REAP_RELEASE']); wait=subprocess.Popen.wait\n"
            "def controlled_wait(process,*args,**kwargs):\n"
            " if kwargs.get('timeout') is None:\n"
            "  if environment.get('TEST_REAP_FAIL'):\n"
            "   wait(process,*args,**kwargs); marker.write_text('real native exit confirmed by test'); raise OSError('Synthetic loss of native exit evidence')\n"
            "  marker.touch(); end=time.monotonic()+25\n"
            "  while not release.exists() and time.monotonic()<end:time.sleep(.01)\n"
            " return wait(process,*args,**kwargs)\n"
            "subprocess.Popen.wait=controlled_wait\n"
            "api['native_cli_main'](directory,counter,deadline)\n"
        )
    )
    monkeypatch.setattr(_native_cli, "_NATIVE_OWNER_SCRIPT", script)


def test_codex_scratch_and_permit_survive_until_native_reaping(monkeypatch, tmp_path):
    from reflexio.server.llm._litellm_types import LLMHardTimeoutError

    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    client = _broker_client(monkeypatch, method)
    _, released = _permit(monkeypatch)
    monkeypatch.setattr(
        generation, "_litellm_completion_worker", _early_deadline_worker
    )
    _fault_owner_script(monkeypatch, tmp_path, "reap")
    reaping, allow_reap = tmp_path / "reaping", tmp_path / "allow-reap"
    record = tmp_path / "codex-path"
    code = (
        "import os,time;from pathlib import Path;"
        "p=Path(os.environ['NATIVE_OUTPUT_FILE']);p.write_text('native output');"
        f"Path({str(record)!r}).write_text(str(p));time.sleep(30)"
    )
    try:
        with (
            admission.provider_slot("openai/gpt-4o"),
            pytest.raises(LLMHardTimeoutError, match="hard timeout"),
        ):
            client._completion_with_hard_timeout(
                {
                    "model": "openai/gpt-4o",
                    "native_code": code,
                    "native_timeout": 1,
                    "native_codex_temp": True,
                    "native_env": {
                        **os.environ,
                        "TEST_REAP_MARKER": str(reaping),
                        "TEST_REAP_RELEASE": str(allow_reap),
                    },
                },
                3 if method == "fork" else 15,
            )
        assert reaping.exists()
        output = Path(record.read_text())
        assert output.read_text() == "native output"
        assert not released.is_set()
    finally:
        allow_reap.touch()
    assert released.wait(10)
    end = time.monotonic() + 5
    while output.parent.exists() and time.monotonic() < end:
        time.sleep(0.01)
    assert not output.parent.exists()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Controlled unknown-owner GC test uses a fast forked SDK worker",
)
@pytest.mark.parametrize("foreign_receipt", [False, True])
def test_unexpected_launch_owner_exit_retains_admission_and_scratch(
    monkeypatch, tmp_path, foreign_receipt
):
    import gc
    import weakref

    client = _broker_client(monkeypatch, "fork")
    semaphore, released = _permit(monkeypatch)
    owner_refs = []
    run_finished = threading.Event()
    owner_class = generation.NativeCLIOwner
    run = owner_class._execute

    def capture_owner(deadline):
        owner = owner_class(deadline)
        owner_refs.append(weakref.ref(owner))
        return owner

    def note_run_finished(owner, counter, task):
        try:
            return run(owner, counter, task)
        finally:
            run_finished.set()

    monkeypatch.setattr(generation, "NativeCLIOwner", capture_owner)
    monkeypatch.setattr(owner_class, "_execute", note_run_finished)
    _fault_owner_script(
        monkeypatch, tmp_path, "foreign" if foreign_receipt else "crash"
    )
    owner = None
    try:
        with admission.provider_slot("openai/gpt-4o"), pytest.raises(TimeoutError):
            client._completion_with_hard_timeout(
                {
                    "model": "openai/gpt-4o",
                    "native_code": "raise AssertionError('must not launch')",
                },
                1,
            )
        assert run_finished.wait(10)
        gc.collect()
        owner = owner_refs[0]()
        assert owner is not None, "Unresolved launch owner was garbage collected"
        assert Path(owner.context.directory).is_dir()
        assert not released.is_set()
        assert not semaphore.acquire(blocking=False)
        task = next(iter(owner._tasks.values()))
        assert task.process.returncode == (0 if foreign_receipt else 3)
    finally:
        if owner is not None:
            for counter, task in list(owner._tasks.items()):
                # This exact synthetic target never launches a native process.
                task.exit_confirmed = True
                owner._finish(counter, task)
    assert released.wait(5)
    end = time.monotonic() + 5
    while Path(owner.context.directory).exists() and time.monotonic() < end:
        time.sleep(0.01)
    assert not Path(owner.context.directory).exists()


def test_delegated_no_input_uses_null_stdin(monkeypatch):
    client = _broker_client(monkeypatch, "spawn")
    response = client._completion_with_hard_timeout(
        {
            "model": "openai/gpt-4o",
            "native_code": "import sys;print(repr(sys.stdin.read()))",
            "native_input": None,
        },
        30,
    )
    assert response == "''\n"


def test_real_response_socket_expiry_is_a_typed_hard_timeout(monkeypatch, tmp_path):
    import queue
    import socket

    from reflexio.server.llm import _native_cli
    from reflexio.server.llm._litellm_types import LLMHardTimeoutError

    monkeypatch.setattr(_native_cli, "_worker_context", None)
    sender, receiver = socket.socketpair()
    try:
        _native_cli.native_cli_worker(
            _native_cli.NativeCLIContext(str(tmp_path), receiver),
            queue.SimpleQueue(),
            time.monotonic() + 0.5,
        )
        with pytest.raises(LLMHardTimeoutError, match="hard timeout"):
            _native_cli.run_native_cli(
                [vars(sys)["_base_executable"], "-c", "print('never launched')"],
                timeout=10,
                capture_output=True,
                text=True,
                check=False,
            )
    finally:
        sender.close()
        receiver.close()


def _hold_unrelated_worker(params, result_queue, deadline, native_context):
    release = Path(params["release"])
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.01)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Regression requires a second forked SDK worker",
)
def test_concurrent_worker_cannot_inherit_native_stdin_writer(monkeypatch, tmp_path):
    from reflexio.server.llm import _native_cli

    other_context = multiprocessing.get_context("fork")
    client = _broker_client(monkeypatch, "fork")
    created = tmp_path / "native-ready"
    consume = tmp_path / "consume-input"
    release_other = tmp_path / "release-other-worker"
    text = "payload-λ" * 300000  # Exceeds native pipe capacity before reads begin.
    code = (
        "import sys,time;from pathlib import Path;"
        f"Path({str(created)!r}).touch();"
        f"release=Path({str(consume)!r});end=time.monotonic()+15\n"
        "while not release.exists() and time.monotonic()<end:time.sleep(.01)\n"
        "print(len(sys.stdin.read()))"
    )
    results = []

    def complete():
        try:
            results.append(
                client._completion_with_hard_timeout(
                    {
                        "model": "openai/gpt-4o",
                        "native_code": code,
                        "native_input": text,
                        "native_timeout": 20,
                    },
                    25,
                )
            )
        except BaseException as exc:
            results.append(exc)

    call = threading.Thread(target=complete)
    other_owner = _native_cli.NativeCLIOwner(time.monotonic() + 25)
    other = result_queue = None
    try:
        call.start()
        end = time.monotonic() + 10
        while not created.exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert created.exists(), repr(results)
        other, result_queue = other_owner.create_worker(
            other_context,
            _hold_unrelated_worker,
            {"release": str(release_other)},
        )
        other.start()
        other_owner.worker_started()
        consume.touch()
        call.join(3)
        assert not call.is_alive(), "Unrelated SDK worker kept native stdin open"
        assert results == [f"{len(text)}\n"]
        assert other.is_alive(), "Regression did not overlap the two workers"
    finally:
        consume.touch()
        release_other.touch()
        call.join(25)
        if other is not None:
            other.join(5)
            if other.is_alive():
                other.terminate()
                other.join(5)
            other.close()
        if result_queue is not None:
            result_queue.close()
        other_owner.close()
        other_owner.worker_exited()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Controlled reaping-evidence failure uses a fast SDK worker",
)
def test_unknown_native_exit_keeps_launch_owner_alive(monkeypatch, tmp_path):
    import weakref

    client = _broker_client(monkeypatch, "fork")
    semaphore, released = _permit(monkeypatch)
    owner_refs = []
    owner_class = generation.NativeCLIOwner
    run_finished = threading.Event()
    run = owner_class._execute

    def capture_owner(deadline):
        owner = owner_class(deadline)
        owner_refs.append(weakref.ref(owner))
        return owner

    def note_finished(owner, counter, task):
        try:
            return run(owner, counter, task)
        finally:
            run_finished.set()

    monkeypatch.setattr(generation, "NativeCLIOwner", capture_owner)
    monkeypatch.setattr(owner_class, "_execute", note_finished)
    _fault_owner_script(monkeypatch, tmp_path, "reap")
    marker = tmp_path / "native-really-reaped"
    owner = None
    try:
        with admission.provider_slot("openai/gpt-4o"), pytest.raises(TimeoutError):
            client._completion_with_hard_timeout(
                {
                    "model": "openai/gpt-4o",
                    "native_code": "import time;time.sleep(30)",
                    "native_timeout": 1,
                    "native_env": {
                        **os.environ,
                        "TEST_REAP_MARKER": str(marker),
                        "TEST_REAP_RELEASE": str(tmp_path / "unused-release"),
                        "TEST_REAP_FAIL": "1",
                    },
                },
                3,
            )
        owner = owner_refs[0]()
        assert marker.exists(), "Synthetic failure did not establish safe teardown"
        task = next(iter(owner._tasks.values()))
        assert task.process.poll() is None, "Owner abandoned unconfirmed native exit"
        assert not (Path(owner.context.directory) / "retired-1.pickle").exists()
        assert not released.is_set()
        assert not semaphore.acquire(blocking=False)
    finally:
        if owner is not None and marker.exists():
            for counter, task in list(owner._tasks.items()):
                # Our synthetic target confirmed the real child exited before
                # raising. Only that evidence permits terminating this test owner.
                task.process.terminate()
                assert run_finished.wait(10)
                task.exit_confirmed = True
                owner._finish(counter, task)
    assert released.wait(5)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Compatibility contract exercises inherited programmatic SDK hooks",
)
def test_native_owner_preserves_programmatic_sdk_state_and_local_schema(monkeypatch):
    import litellm
    from pydantic import BaseModel

    from reflexio.server.llm._native_cli import run_native_cli

    class LocalSchema(BaseModel):
        answer: str

    handler = object()
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [{"provider": "programmatic", "custom_handler": handler}],
    )
    monkeypatch.setattr(litellm, "callbacks", ["inherited-observability"])
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    captured = "local provider hook"

    def completion(params):
        assert params["response_format"] is LocalSchema
        assert litellm.custom_provider_map[0]["custom_handler"] is handler
        assert litellm.callbacks == ["inherited-observability"]
        result = run_native_cli(
            [vars(sys)["_base_executable"], "-c", "print('native result')"],
            input="",
            timeout=20,
            text=True,
            capture_output=True,
            check=False,
        )
        return {"hook": captured, "native": result.stdout}

    monkeypatch.setattr(worker_module, "model_completion", completion)
    client = LiteLLMClient(LiteLLMConfig(model="openai/gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    result = client._completion_with_hard_timeout(
        {"model": "programmatic/example", "response_format": LocalSchema}, 25
    )
    assert result == {"hook": captured, "native": "native result\n"}


@pytest.mark.skipif(os.name == "nt", reason="POSIX live fork worker regression")
def test_cancellation_marker_failure_still_retires_live_sdk_worker(monkeypatch):
    from reflexio.server.llm import _native_cli as native

    context = multiprocessing.get_context("fork")
    processes = []
    real_factory = context.Process

    def factory(*args, **kwargs):
        process = real_factory(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(context, "Process", factory)
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)

    def blocked_worker(*_args):
        time.sleep(10)

    monkeypatch.setattr(generation, "_litellm_completion_worker", blocked_worker)
    real_touch = Path.touch

    def fail_marker(path, *args, **kwargs):
        if path.name == "cancelled":
            raise OSError("synthetic disk failure")
        return real_touch(path, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fail_marker)
    client = LiteLLMClient(LiteLLMConfig(model="openai/gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    owners = set(native._live_owners)
    try:
        with pytest.raises(Exception, match="hard timeout|timed out"):
            client._completion_with_hard_timeout({"model": "openai/gpt-4o"}, 0.2)
        deadline = time.monotonic() + 3
        while native._live_owners != owners and time.monotonic() < deadline:
            time.sleep(0.01)
        assert processes and all(not process.is_alive() for process in processes)
        assert native._live_owners == owners
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
