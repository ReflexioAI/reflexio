"""Real keyless tokenizer ownership: bounded pools, reloads and fork cleanup."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROGRAM = r"""
import json, os, sys, threading
import psutil
import reflexio.native_compute as owner

mode = sys.argv[1]
n = int(os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"])
if mode == "concurrent":
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(4)
    ready = threading.Barrier(5)
    release = threading.Event()
    before = psutil.Process().num_threads()
    def work(_):
        barrier.wait(3)
        owner.configure_tokenizer_compute()
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        t = Tokenizer(WordLevel({"word":0}, unk_token="word"))
        assert [item.ids for item in t.encode_batch(["word"] * 128)] == [[0]] * 128
        ready.wait(3)
        assert release.wait(3)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(work, i) for i in range(4)]
        try:
            ready.wait(3)
            # Keep Python workers alive during measurement. Thread.join can
            # precede the OS retiring its native thread on macOS.
            added = psutil.Process().num_threads() - before - 4
            assert added == n, added
        finally:
            release.set()
        for future in futures:
            future.result(timeout=3)
    assert owner._STATE.bound == n and not owner._STATE.serial
    print(json.dumps({"mode":mode, "added":n}))
    sys.exit(0)
if mode.startswith("entrypoint_"):
    import importlib
    name = "reflexio.server.api" if mode == "entrypoint_api" else "reflexio.server.llm.embedding_service"
    importlib.import_module(name)
    assert owner._STATE.bound == n and not owner._STATE.serial
elif mode != "unknown":
    owner.configure_tokenizer_compute()
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
tokenizer = Tokenizer(WordLevel({"[UNK]":0, "alpha":1, "beta":2}, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
texts = ["alpha beta"] * 128
def encode():
    assert [value.ids for value in tokenizer.encode_batch(texts)] == [[1,2]] * 128
before = psutil.Process().num_threads()
encode()
added = psutil.Process().num_threads() - before
if mode == "unknown":
    assert added == 3
    os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "2"
    owner.configure_tokenizer_compute()
    assert owner._STATE.bound is None
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    encode()
elif mode == "cold" or mode.startswith("entrypoint_"):
    assert added == (0 if n == 1 else n), added
    assert os.environ["RAYON_NUM_THREADS"] == str(n)
    assert os.environ.get("TOKENIZERS_PARALLELISM") == ("false" if n == 1 else None)
elif mode == "lower":
    assert added == n
    for budget in (1, 5):
        os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = str(budget)
        owner.configure_tokenizer_compute()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        encode()
elif mode in {"fork", "locked_fork", "fork_generations"}:
    import multiprocessing
    assert added == n
    if mode in {"fork", "fork_generations"}:
        os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "1"
        owner.configure_tokenizer_compute()
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    entered, release = threading.Event(), threading.Event()
    def hold():
        with owner._STATE.lock:
            entered.set()
            release.wait(10)
    thread = None
    if mode == "locked_fork":
        thread = threading.Thread(target=hold)
        thread.start()
        assert entered.wait(3)
    def child(send):
        try:
            os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "5"
            owner.configure_tokenizer_compute()
            assert owner._STATE.bound is None
            assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
            encode()
            if mode == "fork_generations":
                import signal, time
                grandchild = os.fork()
                if grandchild == 0:
                    try:
                        owner.configure_tokenizer_compute()
                        assert owner._STATE.bound is None
                        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
                        encode()
                    except BaseException:
                        os._exit(1)
                    os._exit(0)
                reaped = False
                try:
                    limit = time.monotonic() + 1
                    while time.monotonic() < limit:
                        pid, status = os.waitpid(grandchild, os.WNOHANG)
                        if pid:
                            reaped = True
                            assert status == 0
                            break
                        time.sleep(0.01)
                    assert reaped, "grandchild hung"
                finally:
                    if not reaped:
                        os.kill(grandchild, signal.SIGKILL)
                        os.waitpid(grandchild, 0)
            send.send("encoded")
        finally:
            send.close()
    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=child, args=(send,))
    try:
        process.start()
        send.close()
        process.join(3)
        assert not process.is_alive(), "owned child hung"
        assert process.exitcode == 0
        assert receive.poll() and receive.recv() == "encoded"
    finally:
        if process.is_alive():
            process.kill()
            process.join(3)
        assert not process.is_alive()
        process.close()
        receive.close()
        release.set()
        if thread:
            thread.join(3)
            assert not thread.is_alive()
else:
    raise AssertionError(mode)
print(json.dumps({"mode":mode, "added":added}))
"""


def _run(tmp_path: Path, mode: str, budget: int, old: str = "true") -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "TMPDIR", "LANG", "SYSTEMROOT", "WINDIR"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        REFLEXIO_NATIVE_COMPUTE_THREADS=str(budget),
        RAYON_NUM_THREADS="3" if mode == "unknown" else "19",
        TOKENIZERS_PARALLELISM=old,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHON_DOTENV_DISABLED="1",
        OPENAI_API_KEY="sk-test-not-real",
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and owned source
        [sys.executable, "-c", _PROGRAM, mode],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize("old", ["false", "true", "", "unrecognized"])
def test_real_cold_pool_ignores_vendor_inputs(tmp_path, budget, old):
    assert _run(tmp_path, "cold", budget, old)["mode"] == "cold"


@pytest.mark.parametrize("mode", ["unknown", "lower"])
def test_warm_pool_never_claims_reconciliation(tmp_path, mode):
    assert _run(tmp_path, mode, 3)["mode"] == mode


@pytest.mark.skipif(os.name != "posix", reason="actual POSIX fork contract")
@pytest.mark.parametrize("mode", ["fork", "locked_fork", "fork_generations"])
def test_real_fork_serializes_and_replaces_owner_mutex(tmp_path, mode):
    assert _run(tmp_path, mode, 3)["mode"] == mode


@pytest.mark.parametrize("mode", ["entrypoint_api", "entrypoint_inference"])
@pytest.mark.parametrize("budget", [2, 3])
def test_real_serving_entrypoint_claims_cold_parallel_pool(tmp_path, mode, budget):
    assert _run(tmp_path, mode, budget, "false")["added"] == budget


def test_concurrent_first_use_keeps_one_real_pool_bound(tmp_path):
    assert _run(tmp_path, "concurrent", 3)["added"] == 3
