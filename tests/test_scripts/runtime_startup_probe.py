"""Fresh-process runtime/control import contract, with an owned HOME only."""

import logging
import logging.handlers
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from reflexio.cli import env_loader

mode = sys.argv[1]
load = env_loader.load_existing_reflexio_env
calls: list[None] = []


def tracked_load(*args, **kwargs):
    calls.append(None)
    return load(*args, **kwargs)


env_loader.load_existing_reflexio_env = tracked_load

if mode == "passive":

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Passive metadata import started runtime or storage")

    env_loader.load_existing_reflexio_env = forbidden
    import sqlite3

    sqlite3.connect = forbidden
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    assert SQLiteStreamSQL.__name__ == "SQLiteStreamSQL"
    assert "litellm" not in sys.modules
    assert "reflexio.server.llm.litellm_client" not in sys.modules
    assert not calls
    assert not (Path.home() / ".reflexio").exists()
    print("PASS passive")
    raise SystemExit(0)

# Provider registration must see the selected environment already loaded.
import importlib.abc


class ProviderOrder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "reflexio.server.llm.litellm_client":
            assert calls, "Provider imported before explicit runtime startup"


sys.meta_path.insert(0, ProviderOrder())
expected_calls = 1
expected_key = "sk-owned-startup"
expected_storage = str(Path.home() / "data")

if mode in {"api", "inference"}:
    from reflexio import native_compute

    configure_compute = native_compute.configure_tokenizer_compute

    def require_environment_before_native_compute():
        assert len(calls) == 1, "Native setup preceded runtime environment loading"
        return configure_compute()

    native_compute.configure_tokenizer_compute = (
        require_environment_before_native_compute
    )

if mode == "api":
    from reflexio.server.api import app

    assert app.openapi()["openapi"]
elif mode == "inference":
    from reflexio.server.llm.embedding_service import app

    assert app.openapi()["openapi"]
elif mode == "server_entry":
    from reflexio.server.__main__ import main

    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("Server help did not return through its parser")
elif mode == "library":
    from reflexio.lib import Reflexio

    assert Reflexio.__name__ == "Reflexio"
elif mode == "storage":
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(
        org_id="owned-startup", db_path=str(Path.home() / "state.db")
    )
    assert storage.read_learning_policy_authority().phase == "open"
    storage.conn.close()
elif mode == "failure":

    def interrupted_load(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("Owned startup interruption")
        return load(*args, **kwargs)

    env_loader.load_existing_reflexio_env = interrupted_load
    import reflexio.server as runtime

    try:
        runtime.initialize_server_runtime()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Initialization failure was swallowed")
    assert "LOGGING_REPORT" not in vars(runtime)
    assert expected_key == runtime.OPENAI_API_KEY
    expected_calls = 2
elif mode == "concurrent":

    def slow_load(*args, **kwargs):
        calls.append(None)
        time.sleep(0.03)
        return load(*args, **kwargs)

    env_loader.load_existing_reflexio_env = slow_load
    import reflexio.server as runtime

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(
            pool.map(lambda _: runtime.initialize_server_runtime(), range(4))
        )
    assert all(report is reports[0] for report in reports)
elif mode == "reentrant":
    import reflexio.server as runtime

    class Reenter(logging.Handler):
        def emit(self, record):
            _ = runtime.LOCAL_STORAGE_PATH

    logger = logging.getLogger("reflexio.pending_tool_policy")
    handler = Reenter()
    logger.addHandler(handler)
    os.environ["REFLEXIO_RESUME_POLL_INTERVAL_SECONDS"] = "1"
    try:
        try:
            runtime.initialize_server_runtime()
        except RuntimeError as error:
            assert str(error) == "Server runtime initialization cannot reenter"
        else:
            raise AssertionError("Logging callback did not refuse reentrant startup")
        assert (
            not {"OPENAI_API_KEY", "LOCAL_STORAGE_PATH", "LOGGING_REPORT"}
            & vars(runtime).keys()
        )
    finally:
        logger.removeHandler(handler)
        os.environ.pop("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS")
    runtime.initialize_server_runtime()
    expected_calls = 1  # Runtime retries keep the already loaded launch source.
elif mode == "logging_retry":
    import reflexio.server as runtime

    original_handler = logging.handlers.RotatingFileHandler
    attempts = 0

    def fail_second_file(*args, **kwargs):
        global attempts
        attempts += 1
        if attempts == 2:
            raise OSError("Owned second file-handler interruption")
        return original_handler(*args, **kwargs)

    with patch.object(logging.handlers, "RotatingFileHandler", fail_second_file):
        try:
            runtime.initialize_server_runtime()
        except OSError:
            pass
        else:
            raise AssertionError("File-handler failure was swallowed")
        assert "LOGGING_REPORT" not in vars(runtime)
        runtime.initialize_server_runtime()
    roles = Counter(
        getattr(handler, "_reflexio_managed_handler_role", None)
        for handler in logging.getLogger().handlers
    )
    assert roles["console"] == roles["devlog"] == roles["llmio"] == 1
    assert attempts == 3
    expected_calls = 1  # Runtime retries keep the already loaded launch source.
elif mode == "reload":
    import importlib

    import reflexio.server as runtime

    runtime.initialize_server_runtime()
    assert len(calls) == 1
    runtime.OPENAI_API_KEY = "sk-owned-stale"
    runtime.LOCAL_STORAGE_PATH = "owned-stale-path"
    runtime.LOGGING_REPORT = object()
    os.environ["OPENAI_API_KEY"] = "sk-owned-reloaded"
    expected_key = "sk-owned-reloaded"
    expected_storage = str(Path.home() / "reloaded-data")
    os.environ["LOCAL_STORAGE_PATH"] = expected_storage
    importlib.reload(runtime)
    assert len(calls) == 1
    assert (
        not {"OPENAI_API_KEY", "LOCAL_STORAGE_PATH", "LOGGING_REPORT"}
        & vars(runtime).keys()
    )
    assert expected_key == runtime.OPENAI_API_KEY
    expected_calls = 1  # Runtime retries keep the already loaded launch source.
elif mode == "legacy_attributes":
    import reflexio.server as runtime

    assert expected_key == runtime.OPENAI_API_KEY
    original_report = runtime.LOGGING_REPORT
    with patch.object(runtime, "LOCAL_STORAGE_PATH", "owned-context-override"):
        assert runtime.LOCAL_STORAGE_PATH == "owned-context-override"
        assert runtime.initialize_server_runtime() is original_report
    assert expected_storage == runtime.LOCAL_STORAGE_PATH
    os.environ["OPENAI_API_KEY"] = "sk-different-process-value"
    assert expected_key == runtime.OPENAI_API_KEY
    runtime.LOCAL_STORAGE_PATH = "owned-post-startup-override"
    expected_storage = "owned-post-startup-override"
    assert runtime.initialize_server_runtime() is original_report
else:
    raise AssertionError(f"Unknown probe: {mode}")

import reflexio.server as runtime

assert len(calls) == expected_calls, (mode, len(calls))
assert expected_key == runtime.OPENAI_API_KEY
assert expected_storage == runtime.LOCAL_STORAGE_PATH
assert runtime.initialize_server_runtime() is runtime.LOGGING_REPORT
assert len(calls) == expected_calls
print(f"PASS {mode}")
