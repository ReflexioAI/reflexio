"""Bootstrap policy imports must work in stripped inference distributions."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_resume_policy_bootstrap_does_not_require_model_schemas():
    code = """
import sys
class RejectModelSchemas:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'reflexio.models' or fullname.startswith('reflexio.models.'):
            raise AssertionError('Bootstrap attempted model-schema import: ' + fullname)
sys.meta_path.insert(0, RejectModelSchemas())
from reflexio.pending_tool_policy import resolve_resume_timing
policy = resolve_resume_timing({})
assert policy.claim_ttl_seconds > 0
assert not any(name.startswith('reflexio.models') for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 - fixed owned Python program
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
