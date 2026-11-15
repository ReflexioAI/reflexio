"""Policy identity imports without initializing application or environment code."""

import subprocess
import sys
from pathlib import Path


def test_policy_constants_have_no_runtime_import_side_effects():
    package_root = Path(__file__).resolve().parents[1]
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])

class RefuseRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "reflexio.server" or fullname.startswith("reflexio.server."):
            raise AssertionError("Policy constants must not initialize the server")
        if fullname == "reflexio.models" or fullname.startswith("reflexio.models."):
            raise AssertionError("Policy constants must not initialize application models")
        if fullname == "dotenv" or fullname.startswith("dotenv."):
            raise AssertionError("Policy constants must not load environment files")

sys.meta_path.insert(0, RefuseRuntime())
from reflexio import profile_authoring_protocol as protocol
assert protocol.LEARNING_ADMISSION_PROTOCOL == "learning-transition-v1"
assert protocol.PROFILE_AUTHORING_POLICY in protocol.AUTHORING_POLICIES
assert set(protocol.AUTHORING_POLICIES) == {"estimated-lifetime-v1", "source-retention-v1"}
assert "reflexio.server" not in sys.modules
assert "reflexio.models" not in sys.modules
"""
    result = subprocess.run(  # noqa: S603 -- Fixed interpreter and owned import-only source.
        [sys.executable, "-I", "-c", script, str(package_root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
