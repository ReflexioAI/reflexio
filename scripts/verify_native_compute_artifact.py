"""Keyless installed-wheel qualification of real inference CPU library pools."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_PROBE = """
import os, sys
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
if sys.platform == 'linux':
    assert any(pool['user_api'] == 'blas' for pool in pools), pools
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    assert os.environ[name] == str(n)
assert 'REFLEXIO_INFERENCE_TORCH_THREADS' not in os.environ
def compute(_):
    assert torch.get_num_threads() == n
    assert (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8
    assert np.array_equal(np.ones((2, 2)) @ np.ones((2, 2)), np.full((2, 2), 2))
with ThreadPoolExecutor(max_workers=1) as pool:
    list(pool.map(compute, [0, 1]))
print('qualified', n, PRELOADED, pools)
"""


def main() -> None:
    """Keep probes sequential and isolated from credentials and user config."""
    for count in (1, 3):
        for preloaded in (False, True):
            with tempfile.TemporaryDirectory(
                prefix="reflexio-native-artifact-"
            ) as home:
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
                }
                env.update(
                    HOME=home,
                    USERPROFILE=home,
                    PYTHON_DOTENV_DISABLED="true",
                    REFLEXIO_NATIVE_COMPUTE_THREADS=str(count),
                    REFLEXIO_INFERENCE_TORCH_THREADS="2",
                    OMP_NUM_THREADS="2",
                    MKL_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                )
                subprocess.run(  # noqa: S603 - fixed interpreter and literal probe
                    [sys.executable, "-c", _PROBE.replace("PRELOADED", str(preloaded))],
                    env=env,
                    check=True,
                    timeout=60,
                )


if __name__ == "__main__":
    main()
