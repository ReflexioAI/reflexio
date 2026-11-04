"""One CPU numerical-library budget for an owned inference process.

Importing this module never imports a numerical framework or changes process
state. Client processes should only prepare their child's environment.
"""

from __future__ import annotations

import os
import sys
import threading
from _thread import LockType
from collections.abc import Mapping
from dataclasses import dataclass, field

_BUDGET = "REFLEXIO_NATIVE_COMPUTE_THREADS"
TOKENIZER_ENV_OUTPUTS = frozenset({"RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM"})


@dataclass
class _ComputeState:
    lock: LockType = field(default_factory=threading.Lock)
    initialized: bool = False
    bound: int | None = None
    serial: bool = False


_STATE = _ComputeState()
_FORK_REGISTERED = False


def _after_fork() -> None:
    global _STATE
    # Never acquire a mutex possibly held by a vanished parent thread.
    _STATE = _ComputeState(initialized=True, serial=True)
    os.environ.update(TOKENIZERS_PARALLELISM="false")


def _process_state() -> _ComputeState:
    global _FORK_REGISTERED
    if not _FORK_REGISTERED and hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork)
        _FORK_REGISTERED = True
    return _STATE


def _canonical_threads(env: Mapping[str, str]) -> int:
    raw = env.get(_BUDGET, "").strip()
    if not raw:
        return 2
    try:
        threads = int(raw)
    except ValueError:
        raise ValueError(f"{_BUDGET} must be a positive integer") from None
    if threads <= 0:
        raise ValueError(f"{_BUDGET} must be a positive integer")
    return threads


def _configure_tokenizers(state: _ComputeState, threads: int) -> None:
    if not state.initialized:
        # Publish before checking imports: a concurrent import either sees the
        # bound or makes provenance unknown. Never infer a warm pool's size.
        os.environ.update(RAYON_NUM_THREADS=str(threads))
        state.bound = threads if "tokenizers" not in sys.modules else None
        state.initialized = True
    elif state.bound is not None:
        state.bound = max(state.bound, threads)
    state.serial |= state.bound is None or threads == 1 or threads < state.bound
    os.environ.update(RAYON_NUM_THREADS=str(threads))
    if state.serial:
        os.environ.update(TOKENIZERS_PARALLELISM="false")
    else:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)


def configure_tokenizer_compute() -> int:
    """Apply the native budget to subsequent tokenizer work, without BLAS.

    A cold owner bounds Rayon. Unknown, forked or lowered-budget pools remain
    serial until process restart: Rayon cannot resize its global pool. This
    does not interrupt in-flight work or make inherited model locks fork-safe.
    """
    threads = _canonical_threads(os.environ)
    state = _process_state()
    with state.lock:
        _configure_tokenizers(state, threads)
    return threads


def native_compute_threads(env: Mapping[str, str]) -> int:
    """Resolve the explicit budget; legacy inputs never select its value.

    Old default-equivalent assignments are inert. Custom legacy limits need an
    explicit new budget rather than a guessed conversion across runtimes.
    Diagnostics contain names only, since environment values may be sensitive.
    """
    # Keep model-only builders independent of the application-wide migration
    # registry. This boundary owns only its already-retired inference spelling.
    if "REFLEXIO_EMBED_TORCH_THREADS" in env:
        raise ValueError(
            "Removed environment name: REFLEXIO_EMBED_TORCH_THREADS -> "
            "REFLEXIO_NATIVE_COMPUTE_THREADS. Remove the obsolete assignment; "
            "legacy values are never copied."
        )
    raw = env.get(_BUDGET, "").strip()
    if raw:
        return _canonical_threads(env)
    legacy = {
        "REFLEXIO_INFERENCE_TORCH_THREADS": env.get(
            "REFLEXIO_INFERENCE_TORCH_THREADS", ""
        ),
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", ""),
    }
    custom = [name for name, value in legacy.items() if value.strip() not in {"", "2"}]
    if custom:
        raise ValueError(
            f"Retired inference thread inputs: {', '.join(custom)}. "
            f"Choose {_BUDGET} explicitly and remove the old assignments; "
            "legacy values are never copied."
        )
    return 2


def native_compute_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a validated child environment without mutating the parent.

    The standard library variables are derived outputs for cold native imports;
    they are no longer independent Reflexio inference configuration choices.
    """
    threads = str(native_compute_threads(env))
    prepared = dict(env)
    prepared.pop("REFLEXIO_INFERENCE_TORCH_THREADS", None)
    prepared[_BUDGET] = threads
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        prepared[name] = threads
    prepared["RAYON_NUM_THREADS"] = threads
    prepared.pop("TOKENIZERS_PARALLELISM", None)
    if threads == "1":
        prepared["TOKENIZERS_PARALLELISM"] = "false"
    return prepared


def configure_native_compute(*, torch: bool = False) -> int:
    """Apply the budget before loading a model in its owned serving process.

    Loaded BLAS/OpenMP pools need runtime control as well as cold-import env.
    Limits persist for the process lifetime; temporary restoration contexts
    would race concurrent inference. This is a per-backend limit, not an
    aggregate CPU quota. ONNX callers also pass the returned limit to sessions.
    """
    state = _process_state()
    with state.lock:
        prepared = native_compute_env(os.environ)
        threads = int(prepared[_BUDGET])
        _configure_tokenizers(state, threads)
        os.environ.update(
            {
                name: prepared[name]
                for name in (
                    _BUDGET,
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            }
        )
        os.environ.pop("REFLEXIO_INFERENCE_TORCH_THREADS", None)
        from threadpoolctl import threadpool_limits

        if torch:
            import torch as torch_runtime

            if torch_runtime.get_num_threads() != threads:
                torch_runtime.set_num_threads(threads)
        # Rescan after framework import: controllers only see loaded libraries.
        threadpool_limits(limits=threads)
        return threads


def main() -> None:
    """Exec a POSIX inference command with derived cold-start library limits."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("the exec wrapper requires POSIX; use reflexio embeddings serve")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an inference command is required after --")
    # The operator supplies the command; exec directly, without a shell.
    os.execvpe(command[0], command, native_compute_env(os.environ))  # noqa: S606


if __name__ == "__main__":
    main()
