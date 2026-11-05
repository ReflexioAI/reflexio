"""Fresh interpreter lifecycle witness; no provider/network calls."""

import importlib
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock


def _installed_litellm() -> None:
    from reflexio.test_support import llm_mock as helper

    config = SimpleNamespace(
        args=["tests/guard.py"], option=SimpleNamespace(collectonly=True)
    )
    assert "litellm" not in sys.modules
    helper.configure_llm_mock(config)
    assert "litellm" not in sys.modules
    import litellm
    from litellm import completion

    assert isinstance(completion, Mock), "refuse to call a real completion"
    result = completion(
        model="openai/mock",
        messages=[{"role": "user", "content": "Output just a boolean value"}],
    )
    assert result.choices[0].message.content == "true"
    with helper.unpatched_litellm():
        helper.assert_litellm_unpatched()
        assert not isinstance(litellm.completion, Mock)
    assert isinstance(litellm.completion, Mock)
    helper.cleanup_llm_mock(config)
    assert not isinstance(litellm.completion, Mock)


def _import_cleanup_order(helper: Any, collect: SimpleNamespace, mode: str) -> None:
    if mode == "immediate-cleanup-order":
        importlib.import_module("litellm")
    if mode == "import-cleanup-order":
        helper.configure_llm_mock(collect)
        assert "litellm" not in sys.modules
    entered, release, done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_patch = helper.patch

    class Gate:
        @staticmethod
        def object(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original_patch.object(*args, **kwargs)

    helper.patch = Gate
    errors = []

    def importing():
        try:
            if mode == "immediate-cleanup-order":
                helper.configure_llm_mock(collect)
            else:
                importlib.import_module("litellm")
        except BaseException as error:
            errors.append(error)

    def cleaning():
        try:
            helper.cleanup_llm_mock(collect)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    importer = threading.Thread(target=importing)
    cleaner = threading.Thread(target=cleaning)
    importer.start()
    try:
        assert entered.wait(5)
        cleaner.start()
        assert not done.wait(0.05), (
            "cleanup returned before pending activation finished"
        )
    finally:
        release.set()
        importer.join(5)
        if cleaner.ident is not None:
            cleaner.join(5)
    assert not importer.is_alive() and not cleaner.is_alive()
    assert not errors, errors
    assert importlib.import_module("litellm").completion() == "REAL"


def _fork_lock_reset(helper: Any) -> None:
    entered, release = threading.Event(), threading.Event()

    def holding() -> None:
        with helper._litellm_mock_lock:
            entered.set()
            assert release.wait(5)

    thread = threading.Thread(target=holding)
    thread.start()
    try:
        assert entered.wait(5)
        child = os.fork()
        if child == 0:
            # A child cannot wait for a vanished parent's thread to release it.
            acquired = helper._litellm_mock_lock.acquire(blocking=False)
            os._exit(0 if acquired else 1)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()


def main() -> None:
    mode = sys.argv[1]
    if mode == "installed":
        _installed_litellm()
        return
    with tempfile.TemporaryDirectory() as directory:
        stub = Path(directory)
        (stub / "litellm.py").write_text(
            'def completion(*args, **kwargs):\n    return "REAL"\n'
        )
        sys.path.insert(0, str(stub))
        os.environ.pop("MOCK_LLM_RESPONSE", None)
        assert "litellm" not in sys.modules
        from reflexio.test_support import llm_mock as helper

        helper._mock_completion = Mock(return_value="MOCK")
        collect = SimpleNamespace(
            args=["tests/guard.py"], option=SimpleNamespace(collectonly=True)
        )
        runtime = SimpleNamespace(
            args=["tests/unit.py"], option=SimpleNamespace(collectonly=False)
        )
        if mode == "fork-lock-reset":
            _fork_lock_reset(helper)
        elif mode == "e2e":
            helper.configure_llm_mock(
                SimpleNamespace(
                    args=["tests/e2e_tests/provider.py"],
                    option=SimpleNamespace(collectonly=True),
                )
            )
            assert "litellm" not in sys.modules
            module = importlib.import_module("litellm")
            assert module.completion() == "REAL"
            helper.assert_litellm_unpatched()
        elif mode == "runtime":
            helper.configure_llm_mock(runtime)
            assert "litellm" in sys.modules
            assert helper.litellm_is_patched()
            helper.cleanup_llm_mock(runtime)
            assert importlib.import_module("litellm").completion() == "REAL"
        elif mode == "already-imported":
            module = importlib.import_module("litellm")
            original = module.completion
            helper.configure_llm_mock(collect)
            assert module.completion() == "MOCK"
            helper.cleanup_llm_mock(collect)
            assert module.completion is original
        elif mode == "cleanup-before-import":
            helper.configure_llm_mock(collect)
            assert "litellm" not in sys.modules
            helper.cleanup_llm_mock(collect)
            module = importlib.import_module("litellm")
            assert module.completion() == "REAL"
            with helper.unpatched_litellm():
                helper.assert_litellm_unpatched()
        elif mode == "temporary-before-import":
            helper.configure_llm_mock(collect)
            assert "litellm" not in sys.modules
            with helper.unpatched_litellm():
                module = importlib.import_module("litellm")
                assert module.completion() == "REAL"
                helper.assert_litellm_unpatched()
            assert module.completion() == "MOCK"
            helper.cleanup_llm_mock(collect)
            assert module.completion() == "REAL"
        elif mode == "bounded-pending":
            import wrapt

            original_register = wrapt.register_post_import_hook
            registrations = []

            def register(*args):
                registrations.append(args[1])
                return original_register(*args)

            wrapt.register_post_import_hook = register
            for _ in range(100):
                helper.configure_llm_mock(collect)
                helper.configure_llm_mock(collect)
                helper.cleanup_llm_mock(collect)
            helper.configure_llm_mock(collect)
            assert "litellm" not in sys.modules
            assert registrations == ["litellm"]
            module = importlib.import_module("litellm")
            assert module.completion() == "MOCK"
            helper.cleanup_llm_mock(collect)
            assert module.completion() == "REAL"
        elif mode in ("import-cleanup-order", "immediate-cleanup-order"):
            _import_cleanup_order(helper, collect, mode)
        else:
            assert mode == "alias-and-restore"
            helper.configure_llm_mock(collect)
            assert "litellm" not in sys.modules
            from litellm import completion

            module = importlib.import_module("litellm")
            assert completion() == "MOCK"
            assert helper.litellm_is_patched()
            try:
                with helper.unpatched_litellm():
                    helper.assert_litellm_unpatched()
                    assert module.completion() == "REAL"
                    raise RuntimeError("body failure witness")
            except RuntimeError:
                pass
            assert module.completion() == "MOCK"
            assert os.environ["MOCK_LLM_RESPONSE"] == "true"
            helper.cleanup_llm_mock(collect)
            assert module.completion() == "REAL"
        print("PASS", mode)


if __name__ == "__main__":
    main()
