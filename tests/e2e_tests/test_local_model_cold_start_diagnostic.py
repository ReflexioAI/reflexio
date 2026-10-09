"""Opt-in, keyless real-model probe for the startup slice of Windows #482.

Downloads the ~79MB MiniLM archive into a throwaway cache. No paid LLM calls.
The backend child is a launch marker, not the full extraction pipeline.
"""

import os
import sys
from pathlib import Path

import pytest

from reflexio.cli import utils

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("RUN_COLD_MODEL_PROBE") != "1",
        reason="opt-in real MiniLM download: set RUN_COLD_MODEL_PROBE=1",
    ),
]


def test_real_cold_model_ready_before_backend_launch(monkeypatch, tmp_path: Path):
    cache = tmp_path / "cold-model"
    ready = tmp_path / "model-ready"
    backend = tmp_path / "backend-started"
    assert not cache.exists()
    monkeypatch.setattr(utils, "ensure_requested_ports_available", lambda _: None)
    monkeypatch.setattr(
        utils, "get_pidfile_path", lambda *_: tmp_path / "services.json"
    )
    monkeypatch.setattr(
        utils,
        "get_stop_request_path",
        lambda name, port: tmp_path / f"{name}-{port}.stop",
    )
    # Keep the shipped readiness deadlines. Only paths and the non-listening
    # diagnostic children's port check differ from normal supervision.
    embedding_code = (
        "import faulthandler, time; from pathlib import Path; "
        "faulthandler.enable(); faulthandler.dump_traceback_later(60, repeat=True); "
        "from reflexio.server.llm.providers.local_embedding_provider import ONNXMiniLM; "
        f"ONNXMiniLM.DOWNLOAD_PATH = Path({str(cache)!r}); "
        "vectors = ONNXMiniLM()(['Cold-start diagnostic.']); "
        "assert len(vectors) == 1 and len(vectors[0]) == 384; "
        f"Path({str(ready)!r}).write_text('real 384-dimension embedding ready'); "
        "print('Application startup complete.', flush=True); time.sleep(30)"
    )
    backend_code = (
        "from pathlib import Path; "
        f"assert Path({str(ready)!r}).exists(), 'backend started before model readiness'; "
        f"Path({str(backend)!r}).write_text('started after model readiness'); "
        "print('Application startup complete.', flush=True)"
    )
    utils.run_services(
        [
            utils.ServiceConfig(
                name="embedding", command=[sys.executable, "-u", "-c", embedding_code]
            ),
            utils.ServiceConfig(
                name="backend", command=[sys.executable, "-u", "-c", backend_code]
            ),
        ],
        {"embedding": 8072, "backend": 8071},
    )
    assert ready.read_text() == "real 384-dimension embedding ready"
    assert backend.read_text() == "started after model readiness"
    assert (cache / "onnx" / "model.onnx").is_file()
    assert not (tmp_path / "services.json").exists()
