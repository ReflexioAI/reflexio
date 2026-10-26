"""A single budget owns cold and already-loaded inference libraries."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from reflexio.native_compute import native_compute_env, native_compute_threads

BUDGET = "REFLEXIO_NATIVE_COMPUTE_THREADS"
LEGACY = (
    "REFLEXIO_INFERENCE_TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


@pytest.mark.parametrize(
    "value, expected", [(None, 2), ("", 2), ("  ", 2), ("1", 1), (" 3 ", 3)]
)
def test_budget_and_derived_environment(value, expected):
    env = {"TOKENIZERS_PARALLELISM": "false", "SECRET_KEY": "untouched"}
    if value is not None:
        env[BUDGET] = value
    before = dict(env)
    result = native_compute_env(env)
    assert env == before
    assert native_compute_threads(env) == expected
    assert result[BUDGET] == str(expected)
    assert all(result[name] == str(expected) for name in LEGACY[1:])
    assert LEGACY[0] not in result
    assert result["TOKENIZERS_PARALLELISM"] == "false"
    assert result["SECRET_KEY"] == "untouched"


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "private-value"])
def test_invalid_budget_does_not_mutate_or_echo_value(value):
    env = {BUDGET: value}
    with pytest.raises(ValueError, match=BUDGET) as error:
        native_compute_env(env)
    assert value not in str(error.value)
    assert env == {BUDGET: value}


@pytest.mark.parametrize("name", LEGACY)
@pytest.mark.parametrize("value", ["", "2", " 2 "])
def test_legacy_defaults_do_not_select_budget(name, value):
    assert native_compute_threads({name: value}) == 2


@pytest.mark.parametrize("name", LEGACY)
@pytest.mark.parametrize("value", ["1", "3", "private-value"])
def test_custom_legacy_requires_explicit_adoption(name, value):
    env = {name: value}
    with pytest.raises(ValueError, match=name) as error:
        native_compute_env(env)
    assert value not in str(error.value)
    assert env == {name: value}
    env[BUDGET] = "2"
    assert native_compute_env(env)[BUDGET] == "2"


@pytest.mark.parametrize("value", ["", "2", "private-value"])
def test_previously_removed_alias_stays_rejected(value):
    with pytest.raises(ValueError, match="REFLEXIO_EMBED_TORCH_THREADS"):
        native_compute_env({BUDGET: "3", "REFLEXIO_EMBED_TORCH_THREADS": value})


def test_rejection_precedes_process_or_library_mutation(monkeypatch):
    from reflexio.native_compute import configure_native_compute

    monkeypatch.delenv(BUDGET, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "private-value")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        configure_native_compute(torch=True)
    assert dict(os.environ) == before


def test_import_is_inert_in_a_client_process():
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test source
        [
            sys.executable,
            "-c",
            "import os, sys; before=dict(os.environ); import reflexio.native_compute; assert before == dict(os.environ); assert not {'torch', 'numpy', 'threadpoolctl', 'reflexio.removed_env'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("preloaded", [False, True])
def test_real_loaded_and_cold_pools_and_worker_results(count, preloaded):
    env = dict(os.environ)
    env.pop("REFLEXIO_EMBED_TORCH_THREADS", None)
    env[BUDGET] = str(count)
    # Deliberately conflicting native inputs: only the explicit owner can win.
    env.update(dict.fromkeys(LEGACY, "2"))
    source = """
import json, os
from concurrent.futures import ThreadPoolExecutor
from reflexio.native_compute import configure_native_compute
if PRELOADED:
    import numpy as np
    import torch
n = configure_native_compute(torch=True)
import numpy as np
import torch
from threadpoolctl import threadpool_info
assert torch.get_num_threads() == n
pools = threadpool_info()
assert pools, 'Torch must expose a controllable native pool'
assert all(pool['num_threads'] == n for pool in pools), pools
assert os.environ['OMP_NUM_THREADS'] == str(n)
assert os.environ['MKL_NUM_THREADS'] == str(n)
assert os.environ['OPENBLAS_NUM_THREADS'] == str(n)
assert 'REFLEXIO_INFERENCE_TORCH_THREADS' not in os.environ
# Actual inference-worker style dispatch, including a library operation.
def compute(_):
    assert torch.get_num_threads() == n
    assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8
    assert np.array_equal(np.ones((2, 2)) @ np.ones((2, 2)), np.full((2, 2), 2))
with ThreadPoolExecutor(max_workers=1) as pool:
    list(pool.map(compute, [0, 1]))
print(json.dumps(pools))
""".replace("PRELOADED", str(preloaded))
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test source
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    pools = json.loads(result.stdout)
    assert all(pool["num_threads"] == count for pool in pools)


@pytest.mark.parametrize("count", [1, 3])
def test_real_onnx_adapter_session_obeys_budget(monkeypatch, tmp_path, count):
    import numpy as np
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from reflexio.server.llm.providers.local_embedding_provider import ONNXMiniLM

    for name in (BUDGET, *LEGACY[1:]):
        monkeypatch.setenv(name, str(count))
    monkeypatch.delenv(LEGACY[0], raising=False)
    cache = tmp_path / "onnx"
    cache.mkdir()
    for name in (
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (cache / name).write_text("{}")
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tokenizer.save(str(cache / "tokenizer.json"))
    # A real, keyless ONNX Identity graph: int64 input_ids[1,2] -> output[1,2],
    # IR8/opset13. Serialized fixture avoids adding the onnx compiler dependency.
    (cache / "model.onnx").write_bytes(
        bytes.fromhex(
            "08083a600a1d0a09696e7075745f69647312066f757470757422084964656e74697479"
            "12086964656e746974795a1b0a09696e7075745f696473120e0a0c080712080a020801"
            "0a02080262180a066f7574707574120e0a0c080712080a0208010a0208024202100d"
        )
    )
    monkeypatch.setattr(ONNXMiniLM, "DOWNLOAD_PATH", tmp_path)
    adapter = ONNXMiniLM()
    adapter._load_model()
    options = adapter._session.get_session_options()
    assert options.intra_op_num_threads == count
    assert options.inter_op_num_threads == 1
    inputs = np.array([[2, 3]], dtype=np.int64)
    assert np.array_equal(adapter._session.run(None, {"input_ids": inputs})[0], inputs)


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX exec contract; SDK supervision uses direct Popen"
)
def test_exec_wrapper_preserves_pid_argv_and_parent_environment():
    env = native_compute_env({**os.environ, BUDGET: "3"})
    source = """
import json, os, subprocess, sys
before = dict(os.environ)
child = subprocess.Popen([sys.executable, '-m', 'reflexio.native_compute', '--', sys.executable, '-c', 'import json, os, sys; print(json.dumps([os.getpid(), sys.argv[1:], os.environ["OMP_NUM_THREADS"], os.environ["MKL_NUM_THREADS"], os.environ["OPENBLAS_NUM_THREADS"]]))', 'literal argument'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = child.communicate(timeout=30)
assert child.returncode == 0, err
pid, argv, *limits = json.loads(out)
assert pid == child.pid
assert argv == ['literal argument']
assert limits == ['3', '3', '3']
assert before == dict(os.environ)
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test source
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("service", ["embedding", "backend"])
def test_supervisor_prepares_only_inference_child_at_launch(
    monkeypatch, tmp_path, service
):
    from unittest.mock import Mock

    from reflexio.cli.utils import ServiceConfig, _ServiceSupervisor

    monkeypatch.setenv(BUDGET, "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    supervisor = _ServiceSupervisor(
        service_configs={}, pidfile=tmp_path / "pids.json", stop_request_paths={}
    )
    process = Mock()
    process.pid = 1234
    process.stdout = None
    popen = Mock(return_value=process)
    monkeypatch.setattr("reflexio.cli.utils.subprocess.Popen", popen)
    monkeypatch.setattr("reflexio.cli.utils.threading.Thread", Mock())
    monkeypatch.setattr(supervisor, "write_current_pidfile", Mock())
    before = dict(os.environ)
    command = [sys.executable, "-m", "uvicorn", "example:app"]
    config = ServiceConfig(name=service, command=command)
    supervisor.start_service(config)
    assert config.command == command
    assert popen.call_args.args[0] == command
    assert supervisor.processes[service] is process
    child = popen.call_args.kwargs["env"]
    assert child["OMP_NUM_THREADS"] == ("3" if service == "embedding" else "2")
    assert dict(os.environ) == before
