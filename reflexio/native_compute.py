"""One CPU numerical-library budget for an owned inference process.

Importing this module never imports a numerical framework or changes process
state. Client processes should only prepare their child's environment.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping

from reflexio.removed_env import reject_removed_env

_LOCK = threading.Lock()
_BUDGET = "REFLEXIO_NATIVE_COMPUTE_THREADS"


def native_compute_threads(env: Mapping[str, str]) -> int:
    """Resolve the explicit budget; legacy inputs never select its value.

    Old default-equivalent assignments are inert. Custom legacy limits need an
    explicit new budget rather than a guessed conversion across runtimes.
    Diagnostics contain names only, since environment values may be sensitive.
    """
    reject_removed_env(env, canonical="REFLEXIO_NATIVE_COMPUTE_THREADS")
    raw = env.get(_BUDGET, "").strip()
    if raw:
        try:
            threads = int(raw)
        except ValueError:
            raise ValueError(f"{_BUDGET} must be a positive integer") from None
        if threads <= 0:
            raise ValueError(f"{_BUDGET} must be a positive integer")
        return threads
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
    return prepared


def configure_native_compute(*, torch: bool = False) -> int:
    """Apply the budget before loading a model in its owned serving process.

    Loaded BLAS/OpenMP pools need runtime control as well as cold-import env.
    Limits persist for the process lifetime; temporary restoration contexts
    would race concurrent inference. This is a per-backend limit, not an
    aggregate CPU quota. ONNX callers also pass the returned limit to sessions.
    """
    with _LOCK:
        prepared = native_compute_env(os.environ)
        threads = int(prepared[_BUDGET])
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
