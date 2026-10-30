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
    pid_file = tmp_path / "native-pid"
    release_file = tmp_path / "release-native"
    script = tmp_path / "synthetic-cli"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os,time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{os.getppid()}}')\n"
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
            pytest.raises(Exception, match="hard timeout"),
        ):
            client._completion_with_hard_timeout(
                {"model": "openai/gpt-4o", "native_cli_script": str(script)}, 1.0
            )
        assert time.monotonic() - started < 3
        assert released.wait(3), "Native process ownership was never retired"
        pid, parent_pid = map(int, pid_file.read_text().split())
        assert parent_pid == os.getpid(), "The killable SDK worker still owns the CLI"
        # Only this parent may reap the native child; the broker must finish it.
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, 0)
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


def _capture_native_processes(monkeypatch):
    from reflexio.server.llm import _native_cli

    processes = []
    popen = _native_cli.subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(_native_cli.subprocess, "Popen", capture)
    return processes


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_broker_preserves_large_output_input_and_parent_identity(monkeypatch, method):
    import json

    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not supported on this platform")
    client = _broker_client(monkeypatch, method)
    processes = _capture_native_processes(monkeypatch)
    _, released = _permit(monkeypatch)
    text = "input-λ" * 10000
    code = "import os,sys,json;print(json.dumps({'parent':os.getppid(),'input':sys.stdin.read(),'output':'x'*200000}))"
    with admission.provider_slot("openai/gpt-4o"):
        response = client._completion_with_hard_timeout(
            {"model": "openai/gpt-4o", "native_code": code, "native_input": text}, 30
        )
    payload = json.loads(response)
    assert payload == {"parent": os.getpid(), "input": text, "output": "x" * 200000}
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
    processes = _capture_native_processes(monkeypatch)
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
    from reflexio.server.llm import _native_cli

    client = _broker_client(monkeypatch, "fork")
    processes = _capture_native_processes(monkeypatch)
    semaphore, released = _permit(monkeypatch)
    native_started, return_handle = threading.Event(), threading.Event()
    popen = _native_cli.subprocess.Popen

    def delayed_handle(*args, **kwargs):
        process = popen(*args, **kwargs)
        native_started.set()
        assert return_handle.wait(5), "Test did not release native startup"
        return process

    monkeypatch.setattr(_native_cli.subprocess, "Popen", delayed_handle)
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
    processes = _capture_native_processes(monkeypatch)
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
    processes = _capture_native_processes(monkeypatch)
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


def test_codex_scratch_and_permit_survive_until_native_reaping(monkeypatch, tmp_path):
    import subprocess

    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    client = _broker_client(monkeypatch, method)
    _, released = _permit(monkeypatch)
    reaping, allow_reap = threading.Event(), threading.Event()
    wait = subprocess.Popen.wait

    def delayed_wait(process, *args, **kwargs):
        if threading.current_thread().name == "reflexio-native-cli":
            reaping.set()
            assert allow_reap.wait(20), "Test did not release native reaping"
        return wait(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", delayed_wait)
    record = tmp_path / "codex-path"
    code = (
        "import os,time;from pathlib import Path;"
        "p=Path(os.environ['NATIVE_OUTPUT_FILE']);p.write_text('native output');"
        f"Path({str(record)!r}).write_text(str(p));time.sleep(5)"
    )
    try:
        with (
            admission.provider_slot("openai/gpt-4o"),
            pytest.raises(Exception, match="hard timeout"),
        ):
            client._completion_with_hard_timeout(
                {
                    "model": "openai/gpt-4o",
                    "native_code": code,
                    "native_timeout": 0.2,
                    "native_codex_temp": True,
                },
                1 if method == "fork" else 15,
            )
        assert reaping.is_set()
        output = Path(record.read_text())
        assert output.read_text() == "native output"
        assert not released.is_set()
    finally:
        allow_reap.set()
    assert released.wait(5)
    deadline = time.monotonic() + 5
    while output.parent.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not output.parent.exists()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Controlled unconfirmed-reap test uses a fast forked worker",
)
def test_unconfirmed_native_exit_retains_owner_and_scratch_after_gc(monkeypatch):
    import gc
    import subprocess
    import weakref

    client = _broker_client(monkeypatch, "fork")
    semaphore, released = _permit(monkeypatch)
    owner_refs = []
    worker_retired = threading.Event()
    owner_class = generation.NativeCLIOwner
    wait = subprocess.Popen.wait
    worker_exited = owner_class.worker_exited

    def capture_owner(deadline):
        owner = owner_class(deadline)
        owner_refs.append(weakref.ref(owner))
        return owner

    def note_worker_exit(owner):
        worker_exited(owner)
        worker_retired.set()

    def unconfirmed_wait(process, *args, **kwargs):
        result = wait(process, *args, **kwargs)
        if threading.current_thread().name == "reflexio-native-cli":
            # The test knows the harmless child exited; production cannot infer
            # that from a wait operation which raises instead of confirming it.
            raise OSError("Synthetic loss of native exit evidence")
        return result

    monkeypatch.setattr(generation, "NativeCLIOwner", capture_owner)
    monkeypatch.setattr(owner_class, "worker_exited", note_worker_exit)
    monkeypatch.setattr(subprocess.Popen, "wait", unconfirmed_wait)
    owner = None
    try:
        with (
            admission.provider_slot("openai/gpt-4o"),
            pytest.raises(Exception, match="hard timeout"),
        ):
            client._completion_with_hard_timeout(
                {
                    "model": "openai/gpt-4o",
                    "native_code": "import time;time.sleep(5)",
                    "native_timeout": 0.1,
                },
                1,
            )
        assert worker_retired.wait(5)
        gc.collect()
        owner = owner_refs[0]()
        assert owner is not None, "Unconfirmed native owner was garbage collected"
        assert Path(owner.context.directory).is_dir()
        assert not released.is_set()
        assert not semaphore.acquire(blocking=False)
        task = next(iter(owner._tasks.values()))
        assert task.process is not None and task.process.returncode is not None
    finally:
        # Only this test's real wait confirmed exit before raising. Restore the
        # evidence and retire the synthetic resource without a force-release API.
        if owner is not None:
            for counter, task in list(owner._tasks.items()):
                task.exit_confirmed = True
                owner._finish(counter, task)
    assert released.wait(5)
    assert not Path(owner.context.directory).exists()


def test_delegated_no_input_uses_null_stdin(monkeypatch):
    import subprocess

    client = _broker_client(monkeypatch, "spawn")
    popen = subprocess.Popen
    observed = []

    def capture_stdin(*args, **kwargs):
        observed.append(kwargs.get("stdin"))
        return popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", capture_stdin)
    response = client._completion_with_hard_timeout(
        {
            "model": "openai/gpt-4o",
            "native_code": "import sys;print(repr(sys.stdin.read()))",
            "native_input": None,
        },
        30,
    )
    assert response == "''\n"
    assert observed == [subprocess.DEVNULL]
