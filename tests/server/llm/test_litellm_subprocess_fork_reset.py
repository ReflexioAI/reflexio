"""Fork-safety of the isolated-completion worker.

A forked ``_litellm_completion_worker`` child inherits the parent's litellm
module-level HTTP client cache (``litellm.in_memory_llm_clients_cache``). If the
parent ever holds a warm client, the child would reuse an inherited — and now
cross-process-shared — socket, corrupting the very first completion in that
child (observed as ``[SSL] record layer failure`` / ``Server disconnected`` /
a garbled request the provider rejects as missing auth). The worker must reset
that inherited client state in the child *before* any completion runs, without
touching the parent's cache.

These tests fork real children (default ``fork`` start method on Linux) and
report the child-observed cache state back over a ``Queue``.
"""

import multiprocessing

import litellm

from reflexio.server.llm._litellm_subprocess import (
    _litellm_completion_worker,
    _reset_llm_client_state_after_fork,
)

_SENTINEL_KEY = "reflexio-fork-reset-sentinel"


def _child_report_cache_state(result_queue: multiprocessing.Queue) -> None:
    """Run the child-side reset, then report whether the inherited cache is
    empty (from the child's point of view)."""
    _reset_llm_client_state_after_fork()
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    cache_dict = getattr(cache, "cache_dict", None)
    result_queue.put(
        ("child_cache_size", len(cache_dict) if cache_dict is not None else -1)
    )


def _child_report_via_worker(result_queue: multiprocessing.Queue) -> None:
    """Drive the real worker entrypoint (with ``litellm.completion`` stubbed so
    no network happens) and report the inherited cache size the child sees
    *after* the worker's built-in reset but as observed via a side channel."""

    def _fake_completion(**_params):
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        cache_dict = getattr(cache, "cache_dict", None)
        # Report the cache size the worker left us with, then return a minimal
        # picklable object so the worker's "ok" path succeeds.
        result_queue.put(
            ("worker_saw_cache_size", len(cache_dict) if cache_dict is not None else -1)
        )

        class _Resp:
            choices: list = []

        return _Resp()

    litellm.completion = _fake_completion  # type: ignore[assignment]
    worker_queue: multiprocessing.Queue = multiprocessing.get_context().Queue(maxsize=1)
    _litellm_completion_worker({"model": "gpt-5-mini"}, worker_queue)


def test_forked_child_resets_inherited_llm_client_cache():
    """The child clears the inherited cache; the PARENT keeps its entry."""
    cache = litellm.in_memory_llm_clients_cache
    cache.cache_dict[_SENTINEL_KEY] = "warm-client-placeholder"
    try:
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_child_report_cache_state, args=(result_queue,))
        proc.start()
        try:
            label, child_cache_size = result_queue.get(timeout=10.0)
        finally:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
            result_queue.close()
            result_queue.join_thread()

        # Child exited cleanly — a crash or hung child must not pass or leak.
        assert proc.exitcode == 0
        # Child ran the reset -> inherited cache is empty in the child.
        assert label == "child_cache_size"
        assert child_cache_size == 0
        # Parent is unaffected -> its sentinel survives.
        assert _SENTINEL_KEY in cache.cache_dict
    finally:
        cache.cache_dict.pop(_SENTINEL_KEY, None)


def test_worker_entry_resets_cache_before_completion():
    """The reset runs inside ``_litellm_completion_worker`` itself, before the
    completion call — so a stubbed completion in the child sees an empty cache
    even though the parent seeded a sentinel."""
    cache = litellm.in_memory_llm_clients_cache
    cache.cache_dict[_SENTINEL_KEY] = "warm-client-placeholder"
    try:
        ctx = multiprocessing.get_context("fork")
        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_child_report_via_worker, args=(result_queue,))
        proc.start()
        try:
            label, worker_saw = result_queue.get(timeout=10.0)
        finally:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
            result_queue.close()
            result_queue.join_thread()

        # Child exited cleanly — a crash or hung child must not pass or leak.
        assert proc.exitcode == 0
        assert label == "worker_saw_cache_size"
        assert worker_saw == 0
        assert _SENTINEL_KEY in cache.cache_dict
    finally:
        cache.cache_dict.pop(_SENTINEL_KEY, None)
