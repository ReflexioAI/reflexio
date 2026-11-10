"""Real completion worker entrypoint retains safe tokenizer policy."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SOURCE = r"""
import multiprocessing, os, sys, time
from reflexio.native_compute import configure_tokenizer_compute

def execute(queue, method):
    import litellm
    from reflexio.server.llm._litellm_subprocess import _litellm_completion_worker
    os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "3"
    # A forked pool stays serial. A spawned child owns a cold pool only while
    # nothing has imported tokenizers yet (litellm>=1.108.4 imports it lazily);
    # a pool of unknown provenance stays serial.
    serial = method == "fork" or "tokenizers" in sys.modules
    def synthetic(**kwargs):
        assert os.environ["RAYON_NUM_THREADS"] == "3"
        expected = "false" if serial else None
        assert os.environ.get("TOKENIZERS_PARALLELISM") == expected
        return {"synthetic": True}
    litellm.completion = synthetic
    _litellm_completion_worker(
        {"model": "synthetic", "messages": []}, queue, time.monotonic() + 8
    )

if __name__ == "__main__":
    configure_tokenizer_compute()
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    tokenizer = Tokenizer(WordLevel({"word": 0}, unk_token="word"))
    assert len(tokenizer.encode_batch(["word"] * 32)) == 32
    os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "1"
    configure_tokenizer_compute()
    context = multiprocessing.get_context(sys.argv[1])
    queue = context.Queue()
    process = context.Process(target=execute, args=(queue, sys.argv[1]))
    try:
        process.start()
        packet = queue.get(timeout=12)
        process.join(3)
        assert not process.is_alive(), "completion worker did not retire"
        assert process.exitcode == 0
        assert packet == ("ok", {"synthetic": True}), packet
    finally:
        if process.is_alive():
            process.kill()
            process.join(3)
        assert not process.is_alive()
        process.close()
        queue.close()
        queue.join_thread()
"""


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_actual_completion_worker_without_client_constructor(tmp_path, method):
    if method == "fork" and os.name != "posix":
        pytest.skip("actual POSIX fork contract")
    script = tmp_path / "owned_worker.py"
    script.write_text(_SOURCE)
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "TMPDIR", "LANG", "SYSTEMROOT", "WINDIR"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
        PYTHON_DOTENV_DISABLED="1",
        REFLEXIO_NATIVE_COMPUTE_THREADS="3",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OPENAI_API_KEY="sk-test-not-real",
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        HF_HUB_OFFLINE="1",
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter, owned fixture
        [sys.executable, str(script), method],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stderr
